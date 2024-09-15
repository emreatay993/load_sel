import cupy
import numpy as np
import psutil
import gc
from numba import njit
from pyyeti.cyclecount import rainflow
from pyyeti import cyclecount
import time
import cupy as cp

# Enable CuPy if CUDA is installed
CUDA_ENABLED = True

# Global flag to control disk writing
WRITE_TO_DISK = False  # Set to False to compute fatigue damage directly

# Set the float size (directly affects RAM usage, computation speed)
DTYPE = np.float32
# Determine the byte size of the data type
ELEMENT_SIZE = np.dtype(DTYPE).itemsize

RAM_PERCENT = 0.5

# Memory management: get total and available memory
total_memory = psutil.virtual_memory().total / (1024 ** 3)
available_memory = psutil.virtual_memory().available * RAM_PERCENT / (1024 ** 3)  # Use only 80% of available RAM

print(f"Total system RAM: {total_memory :.2f} GB")
print(f"Available system RAM: {available_memory :.2f} GB")


def get_chunk_size(num_nodes, num_time_points, num_modes, element_size=ELEMENT_SIZE):
    """Calculate the optimal chunk size for processing based on available memory."""
    available_memory = psutil.virtual_memory().available * RAM_PERCENT  # Use only 80% of available RAM

    # Calculate memory needed for intermediate stress matrices (6 stress matrices + von Mises matrix)
    memory_per_node = (7 * num_time_points * element_size) + (num_time_points * element_size)

    # Calculate number of nodes that can fit within the available memory
    max_nodes_per_iteration = available_memory // memory_per_node
    return max(1, int(max_nodes_per_iteration))  # Ensure at least one node per chunk

if not CUDA_ENABLED:
    @njit
    def compute_von_mises(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
        """Compute von Mises stress in a single-line vectorized manner to minimize memory usage."""
        sigma_vm = np.sqrt(
            0.5 * ((actual_sx - actual_sy) ** 2 + (actual_sy - actual_sz) ** 2 + (actual_sz - actual_sx) ** 2) +
            6 * (actual_sxy ** 2 + actual_syz ** 2 + actual_sxz ** 2)
        )
        return sigma_vm

if CUDA_ENABLED:
    def compute_von_mises(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
        """Compute von Mises stress in a single-line vectorized manner to minimize memory usage."""
        sigma_vm = cp.sqrt(
            0.5 * ((actual_sx - actual_sy) ** 2 + (actual_sy - actual_sz) ** 2 + (actual_sz - actual_sx) ** 2) +
            6 * (actual_sxy ** 2 + actual_syz ** 2 + actual_sxz ** 2)
        )
        return sigma_vm


'''
def vectorized_rainflow(series):
    """Perform optimized rainflow counting on a stress-time series using the three-point method with Numba."""
    n = len(series)

    # Calculate the differences and sign changes
    diff = np.diff(series)
    sign_change = np.diff(np.sign(diff))

    # Peaks and valleys are where sign changes occur
    peaks_indices_temp = np.where(sign_change != 0)[0] + 1
    peaks_indices = np.empty(len(peaks_indices_temp) + 2, dtype=np.int64)
    peaks_indices[0] = 0
    peaks_indices[1:-1] = peaks_indices_temp
    peaks_indices[-1] = n - 1

    peaks_and_valleys = series[peaks_indices]

    # Initialize fixed-size arrays for stack, ranges, and counts
    max_cycles = len(peaks_and_valleys)  # Maximum possible number of cycles
    stack = np.empty(max_cycles, dtype=np.float64)
    ranges = np.empty(max_cycles, dtype=np.float64)
    counts = np.empty(max_cycles, dtype=np.float64)

    stack_size = 0
    range_count = 0

    # Use a loop to find cycles
    for i in range(len(peaks_and_valleys)):
        stack[stack_size] = peaks_and_valleys[i]
        stack_size += 1

        while stack_size >= 3:
            S0, S1, S2 = stack[stack_size - 3], stack[stack_size - 2], stack[stack_size - 1]
            R1 = abs(S1 - S0)
            R2 = abs(S2 - S1)

            if R1 >= R2:
                # Count half cycle
                ranges[range_count] = R2
                counts[range_count] = 0.5
                range_count += 1

                # Remove the middle point (shift elements)
                stack[stack_size - 2] = stack[stack_size - 1]
                stack_size -= 1
            else:
                break

    # Handle remaining half-cycles
    for i in range(stack_size - 1):
        R = abs(stack[i + 1] - stack[i])
        ranges[range_count] = R
        counts[range_count] = 0.5
        range_count += 1

    return ranges[:range_count], counts[:range_count]
'''


def vectorized_rainflow(series):
    series=cupy.asnumpy(series)
    rf = rainflow(cyclecount.findap(series, tol=1e-6), use_pandas=False)
    return rf[:,0], rf[:,2]

@njit
def vectorized_calculate_damage(stress_ranges, counts, A, m):
    """
    Calculate the total damage using the S-N curve parameters.
    """
    # Add epsilon to avoid division by zero
    epsilon = 1e-10
    Nf = A / ((stress_ranges + epsilon) ** m)  # Basquin's equation
    damage = np.sum(counts / Nf)
    return damage

@njit
def estimate_ram_required_per_iteration(chunk_size, num_time_points):
    """Estimate the total RAM required per iteration to compute von Mises stress."""
    # Memory required for each actual stress matrix (actual_sx, actual_sy, etc.)
    size_per_stress_matrix = chunk_size * num_time_points * ELEMENT_SIZE
    # Total memory for all six stress matrices
    total_stress_memory = 7 * size_per_stress_matrix

    return total_stress_memory / (1024**3)  # Convert to GB


def process_stress_results(modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord):
    """Process stress results to compute von Mises stresses or fatigue damage."""
    # Determine matrix shapes
    num_nodes, num_modes = modal_sx.shape
    num_time_points = modal_coord.shape[1]

    # Calculate optimal chunk size
    chunk_size = get_chunk_size(num_nodes, num_time_points, num_modes)
    num_iterations = (num_nodes + chunk_size - 1) // chunk_size  # Calculate the total number of iterations

    print(f"Estimated number of iterations to avoid memory overflow: {num_iterations}")

    # Calculate and display memory requirements per iteration
    memory_required_per_iteration = estimate_ram_required_per_iteration(chunk_size, num_time_points)
    print(f"Estimated RAM required per iteration: {memory_required_per_iteration:.2f} GB")

    # Initialize memmap file if writing to disk
    if WRITE_TO_DISK:
        vm_memmap = np.memmap('von_mises_stresses.dat', dtype='float32', mode='w+', shape=(num_nodes, num_time_points))

    for start_idx in range(0, num_nodes, chunk_size):
        end_idx = min(start_idx + chunk_size, num_nodes)

        # Perform matrix multiplications for the chunk
        if not CUDA_ENABLED:
            actual_sx = np.dot(modal_sx[start_idx:end_idx, :], modal_coord)
            actual_sy = np.dot(modal_sy[start_idx:end_idx, :], modal_coord)
            actual_sz = np.dot(modal_sz[start_idx:end_idx, :], modal_coord)
            actual_sxy = np.dot(modal_sxy[start_idx:end_idx, :], modal_coord)
            actual_syz = np.dot(modal_syz[start_idx:end_idx, :], modal_coord)
            actual_sxz = np.dot(modal_sxz[start_idx:end_idx, :], modal_coord)

        if CUDA_ENABLED:
            actual_sx = cp.dot(modal_sx[start_idx:end_idx, :], modal_coord)
            actual_sy = cp.dot(modal_sy[start_idx:end_idx, :], modal_coord)
            actual_sz = cp.dot(modal_sz[start_idx:end_idx, :], modal_coord)
            actual_sxy = cp.dot(modal_sxy[start_idx:end_idx, :], modal_coord)
            actual_syz = cp.dot(modal_syz[start_idx:end_idx, :], modal_coord)
            actual_sxz = cp.dot(modal_sxz[start_idx:end_idx, :], modal_coord)

        # Compute von Mises stresses
        sigma_vm = compute_von_mises(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz)

        if WRITE_TO_DISK:
            # Write to disk
            vm_memmap[start_idx:end_idx, :] = sigma_vm
        else:
            start_time = time.time()
            # Compute fatigue damage directly for each node
            for node_idx in range(sigma_vm.shape[0]):
                ranges, counts = vectorized_rainflow(sigma_vm[node_idx, :])
                damage = vectorized_calculate_damage(ranges, counts, A=1.0, m=3.0)  # Example S-N curve parameters
                print(f"Node {start_idx + node_idx} damage: {damage:.4f}")


        # Inform user of memory status after each iteration
        current_available_memory = psutil.virtual_memory().available * RAM_PERCENT
        print(f"Iteration completed for nodes {start_idx} to {end_idx}. Available RAM: {current_available_memory / (1024**3):.2f} GB")

        fatigue_calc_endtime = time.time() - start_time
        print("Elapsed time: " + str(fatigue_calc_endtime))

    if WRITE_TO_DISK:
        # Flush memory-mapped data to disk
        vm_memmap.flush()
        del vm_memmap

# Modal Inputs
modal_sx = np.random.randn(20000, 40).astype(DTYPE)
modal_sy = np.random.randn(20000, 40).astype(DTYPE)
modal_sz = np.random.randn(20000, 40).astype(DTYPE)
modal_sxy = np.random.randn(20000, 40).astype(DTYPE)
modal_syz = np.random.randn(20000, 40).astype(DTYPE)
modal_sxz = np.random.randn(20000, 40).astype(DTYPE)
modal_coord = np.random.randn(40, 100000).astype(DTYPE)

if CUDA_ENABLED:
    # Convert Modal Inputs into CuPy arrays
    modal_sx = cp.asarray(modal_sx)
    modal_sy = cp.asarray(modal_sy)
    modal_sz = cp.asarray(modal_sz)
    modal_sxy = cp.asarray(modal_sxy)
    modal_syz = cp.asarray(modal_syz)
    modal_sxz = cp.asarray(modal_sxz)
    modal_coord = cp.asarray(modal_coord)
# endregion

start_time = time.time()
process_stress_results(modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord)
endtime_main_calc = time.time() - start_time

print("Main calculation routine completed in: " + str(endtime_main_calc) + " seconds")
