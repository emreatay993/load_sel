# region Import libraries
import numpy as np
import torch
import psutil
import gc
from numba import njit
from pyyeti.cyclecount import rainflow
import time
# endregion

# region Global variables
IS_CUDA_ENABLED = True
IS_WRITE_TO_DISK = False
# endregion

# region Define global class & functions
class PotentialDamageSolverByMSUP:
    def __init__(self, modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord, cuda_enabled=IS_CUDA_ENABLED,
                 write_to_disk=IS_WRITE_TO_DISK):
        # Initialize modal inputs
        self.device = torch.device("cuda" if cuda_enabled and torch.cuda.is_available() else "cpu")
        self.modal_sx = torch.tensor(modal_sx, dtype=torch.float32).to(self.device)
        self.modal_sy = torch.tensor(modal_sy, dtype=torch.float32).to(self.device)
        self.modal_sz = torch.tensor(modal_sz, dtype=torch.float32).to(self.device)
        self.modal_sxy = torch.tensor(modal_sxy, dtype=torch.float32).to(self.device)
        self.modal_syz = torch.tensor(modal_syz, dtype=torch.float32).to(self.device)
        self.modal_sxz = torch.tensor(modal_sxz, dtype=torch.float32).to(self.device)
        self.modal_coord = torch.tensor(modal_coord, dtype=torch.float32).to(self.device)

        # Global settings
        self.cuda_enabled = cuda_enabled
        self.write_to_disk = write_to_disk
        self.DTYPE = np.float32
        self.ELEMENT_SIZE = np.dtype(self.DTYPE).itemsize
        self.RAM_PERCENT = 0.7

        # Memory details
        self.total_memory = psutil.virtual_memory().total / (1024 ** 3)
        self.available_memory = psutil.virtual_memory().available * self.RAM_PERCENT / (1024 ** 3)
        print(f"Total system RAM: {self.total_memory:.2f} GB")
        print(f"Available system RAM: {self.available_memory:.2f} GB")

    def get_chunk_size(self, num_nodes, num_time_points):
        """Calculate the optimal chunk size for processing based on available memory."""
        available_memory = psutil.virtual_memory().available * self.RAM_PERCENT  # Use only 80% of available RAM
        memory_per_node = (7 * num_time_points * self.ELEMENT_SIZE) + (num_time_points * self.ELEMENT_SIZE)
        max_nodes_per_iteration = available_memory // memory_per_node
        return max(1, int(max_nodes_per_iteration))  # Ensure at least one node per chunk

    @staticmethod
    @njit
    def estimate_ram_required_per_iteration(chunk_size, num_time_points, element_size):
        """Estimate the total RAM required per iteration to compute von Mises stress."""
        size_per_stress_matrix = chunk_size * num_time_points * element_size
        total_stress_memory = 8 * size_per_stress_matrix
        return total_stress_memory / (1024 ** 3)  # Convert to GB

    @staticmethod
    def compute_principal_stresses(modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord, start_idx,
                                   end_idx):
        """Compute actual stresses using matrix multiplication."""
        actual_sx = torch.matmul(modal_sx[start_idx:end_idx, :], modal_coord)
        actual_sy = torch.matmul(modal_sy[start_idx:end_idx, :], modal_coord)
        actual_sz = torch.matmul(modal_sz[start_idx:end_idx, :], modal_coord)
        actual_sxy = torch.matmul(modal_sxy[start_idx:end_idx, :], modal_coord)
        actual_syz = torch.matmul(modal_syz[start_idx:end_idx, :], modal_coord)
        actual_sxz = torch.matmul(modal_sxz[start_idx:end_idx, :], modal_coord)

        return actual_sx.cpu().numpy(), actual_sy.cpu().numpy(), actual_sz.cpu().numpy(), actual_sxy.cpu().numpy(), actual_syz.cpu().numpy(), actual_sxz.cpu().numpy()

    @staticmethod
    @njit(parallel=True)
    def compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
        """Compute von Mises stress."""
        sigma_vm = np.sqrt(
            0.5 * ((actual_sx - actual_sy) ** 2 + (actual_sy - actual_sz) ** 2 + (actual_sz - actual_sx) ** 2) +
            6 * (actual_sxy ** 2 + actual_syz ** 2 + actual_sxz ** 2)
        )
        return sigma_vm

    def process_stress_results(self):
        """Process stress results to compute von Mises stresses or fatigue damage."""
        num_nodes, num_modes = self.modal_sx.shape
        num_time_points = self.modal_coord.shape[1]

        chunk_size = self.get_chunk_size(num_nodes, num_time_points)
        num_iterations = (num_nodes + chunk_size - 1) // chunk_size
        print(f"Estimated number of iterations to avoid memory overflow: {num_iterations}")

        memory_required_per_iteration = self.estimate_ram_required_per_iteration(chunk_size, num_time_points,
                                                                                 self.ELEMENT_SIZE)
        print(f"Estimated RAM required per iteration: {memory_required_per_iteration:.2f} GB")

        if self.write_to_disk:
            vm_memmap = np.memmap('von_mises_stresses.dat', dtype='float32', mode='w+',
                                  shape=(num_nodes, num_time_points))

        for start_idx in range(0, num_nodes, chunk_size):
            end_idx = min(start_idx + chunk_size, num_nodes)

            # region Calculate principal stresses
            start_time = time.time()
            actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
                self.compute_principal_stresses(self.modal_sx, self.modal_sy, self.modal_sz, self.modal_sxy,
                                                self.modal_syz, self.modal_sxz, self.modal_coord, start_idx, end_idx)
            print("Elapsed time for principal stresses: " + str(time.time() - start_time))
            # endregion

            # region Calculate von-Mises stresses
            start_time = time.time()
            sigma_vm = self.compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz)
            print("Elapsed time for Von-Mises stresses: " + str(time.time() - start_time))
            # endregion

            # region Free up some memory
            start_time = time.time()
            del actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz
            #gc.collect()  # Force garbage collection to free memory
            print("Elapsed time for garbage collection: " + str(time.time() - start_time))
            # endregion

            if self.write_to_disk:
                vm_memmap[start_idx:end_idx, :] = sigma_vm
            else:
                start_time = time.time()
                A = 1
                m = -3
                for node_idx in range(sigma_vm.shape[0]):
                    ranges, counts = self.calculate_rainflow(sigma_vm[node_idx, :])
                    damage = self.calculate_potential_damage_index(ranges, counts, A, m)

            current_available_memory = psutil.virtual_memory().available * self.RAM_PERCENT
            print(
                f"Iteration completed for nodes {start_idx} to {end_idx}. Available RAM: {current_available_memory / (1024 ** 3):.2f} GB")
            print("Elapsed time for calculating cycle counts and damage index: " + str(time.time() - start_time))

        if self.write_to_disk:
            vm_memmap.flush()
            del vm_memmap

    @staticmethod
    def calculate_rainflow(series):
        rf = rainflow(series, use_pandas=False)
        return rf[:, 0], rf[:, 2]

    @staticmethod
    @njit(parallel=True)
    def calculate_potential_damage_index(stress_ranges, counts, A, m):
        """Calculate the total damage using the S-N curve parameters."""
        damage = np.sum(counts / (A / ((stress_ranges + 1e-10) ** m)))
        return damage
# endregion

# region Define modal inputs
modal_sx = np.random.randn(30000, 40).astype(np.float32)
modal_sy = np.random.randn(30000, 40).astype(np.float32)
modal_sz = np.random.randn(30000, 40).astype(np.float32)
modal_sxy = np.random.randn(30000, 40).astype(np.float32)
modal_syz = np.random.randn(30000, 40).astype(np.float32)
modal_sxz = np.random.randn(30000, 40).astype(np.float32)
modal_coord = np.random.randn(40, 500000).astype(np.float32)
# endregion

# region Run the main processor
potential_damage_processor = PotentialDamageSolverByMSUP(modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord)

start_time = time.time()
potential_damage_processor.process_stress_results()
endtime_main_calc = time.time() - start_time
print("Main calculation routine completed in: " + str(endtime_main_calc) + " seconds")
# endregion
