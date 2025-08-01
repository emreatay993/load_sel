# ---- Standard Library Imports ----
import gc
import math
import os
import time

# ---- Third-Party Imports ----
import psutil
from numba import njit, prange
import numpy as np
import pandas as pd
import torch
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

# region Define global variables
# --- Solver Configuration ---
RAM_PERCENT = 0.9  # Default RAM allocation percentage based on available memory.
DEFAULT_PRECISION = 'Double'  # 'Single' or 'Double'. Double is more precise but uses more memory.
IS_GPU_ACCELERATION_ENABLED = False  # Set to True to use GPU (if a compatible NVIDIA GPU and CUDA is available).

# --- Data Type Configuration ---
# Dynamically set NumPy and Torch data types based on the selected precision.
if DEFAULT_PRECISION == 'Single':
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32
    RESULT_DTYPE = 'float32'
elif DEFAULT_PRECISION == 'Double':
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64
    RESULT_DTYPE = 'float64'

# --- Environment Configuration ---
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
# endregion

class MSUPSmartSolverTransient(QObject):
    progress_signal = pyqtSignal(int)

    def __init__(self, modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord, time_values,
                 steady_sx=None, steady_sy=None, steady_sz=None, steady_sxy=None, steady_syz=None, steady_sxz=None,
                 steady_node_ids=None, modal_node_ids=None, output_directory=None, modal_deformations=None):
        super().__init__()

        # Initializing class attributes used
        self.total_memory = None
        self.available_memory = None
        self.allocated_memory = None

        self.max_over_time_s1 = None
        self.min_over_time_s3 = None
        self.max_over_time_svm = None
        self.max_over_time_def = None
        self.max_over_time_vel = None
        self.max_over_time_acc = None

        self.fatigue_A = None
        self.fatigue_m = None


        # Use selected output directory or fallback to script location
        self.output_directory = output_directory if output_directory else os.path.dirname(os.path.abspath(__file__))

        # Global settings
        self.NP_DTYPE = NP_DTYPE
        self.TORCH_DTYPE = TORCH_DTYPE
        self.RESULT_DTYPE = RESULT_DTYPE
        self.ELEMENT_SIZE = np.dtype(self.NP_DTYPE).itemsize
        self.RAM_PERCENT = RAM_PERCENT
        self.device = torch.device("cuda" if IS_GPU_ACCELERATION_ENABLED and torch.cuda.is_available() else "cpu")

        self.modal_coord = torch.tensor(modal_coord, dtype=self.TORCH_DTYPE).to(self.device)

        if modal_deformations is not None:
            self.modal_deformations_ux = torch.tensor(modal_deformations[0], dtype=self.TORCH_DTYPE).to(self.device)
            self.modal_deformations_uy = torch.tensor(modal_deformations[1], dtype=self.TORCH_DTYPE).to(self.device)
            self.modal_deformations_uz = torch.tensor(modal_deformations[2], dtype=self.TORCH_DTYPE).to(self.device)
        else:
            self.modal_deformations_ux = None
            self.modal_deformations_uy = None
            self.modal_deformations_uz = None

        # Initialize modal inputs
        self.modal_sx = torch.tensor(modal_sx, dtype=TORCH_DTYPE).to(self.device)
        self.modal_sy = torch.tensor(modal_sy, dtype=TORCH_DTYPE).to(self.device)
        self.modal_sz = torch.tensor(modal_sz, dtype=TORCH_DTYPE).to(self.device)
        self.modal_sxy = torch.tensor(modal_sxy, dtype=TORCH_DTYPE).to(self.device)
        self.modal_syz = torch.tensor(modal_syz, dtype=TORCH_DTYPE).to(self.device)
        self.modal_sxz = torch.tensor(modal_sxz, dtype=TORCH_DTYPE).to(self.device)
        self.modal_coord = torch.tensor(modal_coord, dtype=TORCH_DTYPE).to(self.device)

        # Store modal node IDs
        self.modal_node_ids = modal_node_ids

        # If steady-state stress data is provided, process it
        if steady_sx is not None and steady_node_ids is not None:
            self.is_steady_state_included = True
            # Map steady-state stresses to modal nodes
            self.steady_sx = self.map_steady_state_stresses(steady_sx, steady_node_ids, modal_node_ids)
            self.steady_sy = self.map_steady_state_stresses(steady_sy, steady_node_ids, modal_node_ids)
            self.steady_sz = self.map_steady_state_stresses(steady_sz, steady_node_ids, modal_node_ids)
            self.steady_sxy = self.map_steady_state_stresses(steady_sxy, steady_node_ids, modal_node_ids)
            self.steady_syz = self.map_steady_state_stresses(steady_syz, steady_node_ids, modal_node_ids)
            self.steady_sxz = self.map_steady_state_stresses(steady_sxz, steady_node_ids, modal_node_ids)

            # Convert to torch tensors and move to device
            self.steady_sx = torch.tensor(self.steady_sx, dtype=TORCH_DTYPE).to(self.device)
            self.steady_sy = torch.tensor(self.steady_sy, dtype=TORCH_DTYPE).to(self.device)
            self.steady_sz = torch.tensor(self.steady_sz, dtype=TORCH_DTYPE).to(self.device)
            self.steady_sxy = torch.tensor(self.steady_sxy, dtype=TORCH_DTYPE).to(self.device)
            self.steady_syz = torch.tensor(self.steady_syz, dtype=TORCH_DTYPE).to(self.device)
            self.steady_sxz = torch.tensor(self.steady_sxz, dtype=TORCH_DTYPE).to(self.device)
        else:
            self.is_steady_state_included = False

        # Store time axis once for gradient calcs
        self.time_values = time_values.astype(self.NP_DTYPE)

    # region Memory Management
    def _estimate_chunk_size(self, num_time_points, calculate_von_mises, calculate_max_principal_stress,
                             calculate_damage, calculate_deformation=False,
                             calculate_velocity=False, calculate_acceleration=False):
        """Calculate the optimal chunk size for processing based on available memory."""
        available_memory = psutil.virtual_memory().available * self.RAM_PERCENT
        memory_per_node = self._get_memory_per_node(num_time_points,
                                                    calculate_von_mises,
                                                    calculate_max_principal_stress,
                                                    calculate_damage,
                                                    calculate_deformation,
                                                    calculate_velocity,
                                                    calculate_acceleration)
        max_nodes_per_iteration = available_memory // memory_per_node
        return max(1, int(max_nodes_per_iteration))  # Ensure at least one node per chunk

    def _estimate_ram_required_per_iteration(self, chunk_size, memory_per_node):
        """Estimate the total RAM required per iteration to compute stresses."""
        total_memory = chunk_size * memory_per_node
        return total_memory / (1024 ** 3)  # Convert bytes to GB

    def _get_memory_per_node(self, num_time_points, calculate_von_mises, calculate_max_principal_stress,
                             calculate_damage, calculate_deformation=False,
                             calculate_velocity=False, calculate_acceleration=False):
        num_arrays = 6  # For actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz

        if calculate_von_mises:
            num_arrays += 1  # For sigma_vm
        if calculate_max_principal_stress:
            num_arrays += 3  # For s1, s2, s3
        if calculate_damage:
            num_arrays += 1  # For signed_von_mises

        # First, check if the base displacement components (ux, uy, uz) need to be computed at all.
        if calculate_deformation or calculate_velocity or calculate_acceleration:
            num_arrays += 3  # For the initial ux, uy, uz arrays

        # Now, add memory for the final deformation magnitude array ONLY if requested.
        if calculate_deformation:
            num_arrays += 1  # For def_mag

        # Finally, add memory for ALL velocity/acceleration arrays ONLY if one of them is requested.
        if calculate_velocity or calculate_acceleration:
            # This accounts for vel_x/y/z, acc_x/y/z, vel_mag, and acc_mag.
            num_arrays += 8

        memory_per_node = num_arrays * num_time_points * self.ELEMENT_SIZE
        return memory_per_node
    # endregion

    # region JIT Compiled Kernels (for heavy numerical operations)
    @staticmethod
    @njit(parallel=True)
    def compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
        """Compute von Mises stress."""
        sigma_vm = np.sqrt(
            0.5 * ((actual_sx - actual_sy) ** 2 + (actual_sy - actual_sz) ** 2 + (actual_sz - actual_sx) ** 2) +
            3 * (actual_sxy ** 2 + actual_syz ** 2 + actual_sxz ** 2)
        )
        return sigma_vm

    @staticmethod
    @njit(parallel=True)
    def _vel_acc_from_disp(ux, uy, uz, dt):
        """
        Compute velocity and acceleration using 4th-order central differences
        on a uniform grid dt. Endpoints use lower-order one-sided formulas.
        NumPy arrays are created with np.empty_like to avoid dtype issues.
        """
        n_nodes, n_times = ux.shape

        # Preallocate arrays without explicit dtype arguments
        vel_x = np.empty_like(ux)
        vel_y = np.empty_like(ux)
        vel_z = np.empty_like(ux)
        acc_x = np.empty_like(ux)
        acc_y = np.empty_like(ux)
        acc_z = np.empty_like(ux)

        # Uniform step size
        h = dt[1] - dt[0]

        for i in prange(n_nodes):
            # --- Velocity (first derivative) interior points ---
            for j in range(2, n_times - 2):
                # Coefficients as float literals for type inference
                vel_x[i, j] = (-ux[i, j + 2] + 8.0 * ux[i, j + 1]
                               - 8.0 * ux[i, j - 1] + ux[i, j - 2]) / (12.0 * h)
                vel_y[i, j] = (-uy[i, j + 2] + 8.0 * uy[i, j + 1]
                               - 8.0 * uy[i, j - 1] + uy[i, j - 2]) / (12.0 * h)
                vel_z[i, j] = (-uz[i, j + 2] + 8.0 * uz[i, j + 1]
                               - 8.0 * uz[i, j - 1] + uz[i, j - 2]) / (12.0 * h)

            # Fallback (second-order) at boundaries
            # j = 0, 1
            for j in (0, 1):
                vel_x[i, j] = (ux[i, j + 1] - ux[i, j]) / (dt[j + 1] - dt[j])
                vel_y[i, j] = (uy[i, j + 1] - uy[i, j]) / (dt[j + 1] - dt[j])
                vel_z[i, j] = (uz[i, j + 1] - uz[i, j]) / (dt[j + 1] - dt[j])
            # j = n_times-2, n_times-1
            for j in (n_times - 2, n_times - 1):
                vel_x[i, j] = (ux[i, j] - ux[i, j - 1]) / (dt[j] - dt[j - 1])
                vel_y[i, j] = (uy[i, j] - uy[i, j - 1]) / (dt[j] - dt[j - 1])
                vel_z[i, j] = (uz[i, j] - uz[i, j - 1]) / (dt[j] - dt[j - 1])

            # --- Acceleration (second derivative) interior points ---
            for j in range(2, n_times - 2):
                acc_x[i, j] = (-ux[i, j + 2] + 16.0 * ux[i, j + 1]
                               - 30.0 * ux[i, j] + 16.0 * ux[i, j - 1]
                               - ux[i, j - 2]) / (12.0 * h * h)
                acc_y[i, j] = (-uy[i, j + 2] + 16.0 * uy[i, j + 1]
                               - 30.0 * uy[i, j] + 16.0 * uy[i, j - 1]
                               - uy[i, j - 2]) / (12.0 * h * h)
                acc_z[i, j] = (-uz[i, j + 2] + 16.0 * uz[i, j + 1]
                               - 30.0 * uz[i, j] + 16.0 * uz[i, j - 1]
                               - uz[i, j - 2]) / (12.0 * h * h)

            # Fallback (second-order) at boundaries
            for j in (0, 1, n_times - 2, n_times - 1):
                if 0 < j < n_times - 1:
                    acc_x[i, j] = (ux[i, j + 1] - 2.0 * ux[i, j] + ux[i, j - 1]) / (h * h)
                    acc_y[i, j] = (uy[i, j + 1] - 2.0 * uy[i, j] + uy[i, j - 1]) / (h * h)
                    acc_z[i, j] = (uz[i, j + 1] - 2.0 * uz[i, j] + uz[i, j - 1]) / (h * h)
                else:
                    # One-sided second-order
                    k0 = 0 if j == 0 else n_times - 3
                    acc_x[i, j] = (ux[i, k0] - 2.0 * ux[i, k0 + 1] + ux[i, k0 + 2]) / (h * h)
                    acc_y[i, j] = (uy[i, k0] - 2.0 * uy[i, k0 + 1] + uy[i, k0 + 2]) / (h * h)
                    acc_z[i, j] = (uz[i, k0] - 2.0 * uz[i, k0 + 1] + uz[i, k0 + 2]) / (h * h)

        # Compute magnitudes with supported ufuncs
        vel_mag = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2)
        acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        return vel_mag, acc_mag

    @staticmethod
    @njit(parallel=True)
    def _vel_acc_from_disp(ux, uy, uz, dt):
        """
        Compute velocity and acceleration using 6th-order central differences
        on a uniform grid dt. Endpoints use lower-order one-sided formulas.
        """
        n_nodes, n_times = ux.shape
        # Preallocate output arrays
        vel_x = np.empty_like(ux)
        vel_y = np.empty_like(ux)
        vel_z = np.empty_like(ux)
        acc_x = np.empty_like(ux)
        acc_y = np.empty_like(ux)
        acc_z = np.empty_like(ux)

        # Uniform step size
        h = dt[1] - dt[0]

        for i in prange(n_nodes):
            # --- Velocity (first derivative) interior points (6th-order) ---
            for j in range(3, n_times - 3):
                vel_x[i, j] = (-ux[i, j + 3]
                               + 9.0 * ux[i, j + 2]
                               - 45.0 * ux[i, j + 1]
                               + 45.0 * ux[i, j - 1]
                               - 9.0 * ux[i, j - 2]
                               + ux[i, j - 3]
                               ) / (60.0 * h)
                vel_y[i, j] = (-uy[i, j + 3]
                               + 9.0 * uy[i, j + 2]
                               - 45.0 * uy[i, j + 1]
                               + 45.0 * uy[i, j - 1]
                               - 9.0 * uy[i, j - 2]
                               + uy[i, j - 3]
                               ) / (60.0 * h)
                vel_z[i, j] = (-uz[i, j + 3]
                               + 9.0 * uz[i, j + 2]
                               - 45.0 * uz[i, j + 1]
                               + 45.0 * uz[i, j - 1]
                               - 9.0 * uz[i, j - 2]
                               + uz[i, j - 3]
                               ) / (60.0 * h)

            # Fallback (lower-order) at boundaries for velocity
            # j = 0,1,2
            for j in (0, 1, 2):
                vel_x[i, j] = (ux[i, j + 1] - ux[i, j]) / (dt[j + 1] - dt[j])
                vel_y[i, j] = (uy[i, j + 1] - uy[i, j]) / (dt[j + 1] - dt[j])
                vel_z[i, j] = (uz[i, j + 1] - uz[i, j]) / (dt[j + 1] - dt[j])
            # j = n_times-3, n_times-2, n_times-1
            for j in (n_times - 3, n_times - 2, n_times - 1):
                vel_x[i, j] = (ux[i, j] - ux[i, j - 1]) / (dt[j] - dt[j - 1])
                vel_y[i, j] = (uy[i, j] - uy[i, j - 1]) / (dt[j] - dt[j - 1])
                vel_z[i, j] = (uz[i, j] - uz[i, j - 1]) / (dt[j] - dt[j - 1])

            # --- Acceleration (second derivative) interior points (6th-order) ---
            for j in range(3, n_times - 3):
                acc_x[i, j] = (2.0 * ux[i, j + 3]
                               - 27.0 * ux[i, j + 2]
                               + 270.0 * ux[i, j + 1]
                               - 490.0 * ux[i, j]
                               + 270.0 * ux[i, j - 1]
                               - 27.0 * ux[i, j - 2]
                               + 2.0 * ux[i, j - 3]
                               ) / (180.0 * h * h)
                acc_y[i, j] = (2.0 * uy[i, j + 3]
                               - 27.0 * uy[i, j + 2]
                               + 270.0 * uy[i, j + 1]
                               - 490.0 * uy[i, j]
                               + 270.0 * uy[i, j - 1]
                               - 27.0 * uy[i, j - 2]
                               + 2.0 * uy[i, j - 3]
                               ) / (180.0 * h * h)
                acc_z[i, j] = (2.0 * uz[i, j + 3]
                               - 27.0 * uz[i, j + 2]
                               + 270.0 * uz[i, j + 1]
                               - 490.0 * uz[i, j]
                               + 270.0 * uz[i, j - 1]
                               - 27.0 * uz[i, j - 2]
                               + 2.0 * uz[i, j - 3]
                               ) / (180.0 * h * h)

            # Fallback (lower-order) at boundaries for acceleration
            for j in (0, 1, 2, n_times - 3, n_times - 2, n_times - 1):
                if 0 < j < n_times - 1:
                    # central 2nd-order
                    acc_x[i, j] = (ux[i, j + 1] - 2.0 * ux[i, j] + ux[i, j - 1]) / (h * h)
                    acc_y[i, j] = (uy[i, j + 1] - 2.0 * uy[i, j] + uy[i, j - 1]) / (h * h)
                    acc_z[i, j] = (uz[i, j + 1] - 2.0 * uz[i, j] + uz[i, j - 1]) / (h * h)
                else:
                    # one-sided 2nd-order
                    k0 = 0 if j < 3 else n_times - 3
                    acc_x[i, j] = (ux[i, k0] - 2.0 * ux[i, k0 + 1] + ux[i, k0 + 2]) / (h * h)
                    acc_y[i, j] = (uy[i, k0] - 2.0 * uy[i, k0 + 1] + uy[i, k0 + 2]) / (h * h)
                    acc_z[i, j] = (uz[i, k0] - 2.0 * uz[i, k0 + 1] + uz[i, k0 + 2]) / (h * h)

        # Compute magnitudes
        vel_mag = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2)
        acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        return vel_mag, acc_mag, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z

    @staticmethod
    @njit(parallel=True)
    def compute_signed_von_mises_stress(sigma_vm, actual_sx, actual_sy, actual_sz):
        """
        Compute the signed von Mises stress by assigning a sign to the existing von Mises stress.
        Signed von Mises = sigma_vm * ((sx + sy + sz + 1e-6) / |sx + sy + sz + 1e-6|)
        """
        # Calculate the sum of normal stresses
        normal_stress_sum = actual_sx + actual_sy + actual_sz

        # Add a small value (1e-6) to avoid division by zero
        signed_von_mises = sigma_vm * (normal_stress_sum + 1e-6) / np.abs(normal_stress_sum + 1e-6)

        return signed_von_mises

    @staticmethod
    @njit(parallel=True)
    def compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
        """
        Calculates the three principal stresses from the six components of stress.

        -- How This Function Works --
        This function takes 2D arrays of the six standard stress components as input.
        For each point in these arrays, it uses an analytical method (Cardano's formula for
        solving cubic equations) to calculate the three principal stresses.

        Args:
            sx (np.ndarray): 2D array of normal stresses in the X-direction. Shape is (num_nodes, num_time_points).
            sy (np.ndarray): 2D array of normal stresses in the Y-direction.
            sz (np.ndarray): 2D array of normal stresses in the Z-direction.
            sxy (np.ndarray): 2D array of XY shear stresses.
            syz (np.ndarray): 2D array of YZ shear stresses.
            sxz (np.ndarray): 2D array of XZ shear stresses.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three 2D arrays (s1, s2, s3).
            - s1: The maximum (most positive) principal stress at each point.
            - s2: The intermediate principal stress at each point.
            - s3: The minimum (most negative) principal stress at each point.
        """
        # Read the dimensions (e.g., height and width) from the input stress array.
        # This tells us how many nodes and time points we need to process.
        num_nodes, num_time_points = actual_sx.shape

        # Create empty 2D arrays filled with zeros to hold our final results.
        # Pre-allocating memory like this is more efficient than building the arrays on the fly.
        s1_out = np.zeros((num_nodes, num_time_points), dtype=NP_DTYPE)
        s2_out = np.zeros_like(s1_out)
        s3_out = np.zeros_like(s1_out)

        # --- Pre-calculate mathematical constants to avoid recalculating them inside the loop ---
        two_pi_3 = 2.0943951023931953  # This is 2 * pi / 3, used in the trigonometric solution.
        tiny_p = 1.0e-12  # A very small number used as a tolerance for a special case.

        # These nested loops ensure we perform the calculation for every node at every time step.
        # `prange` is Numba's "parallel range," which splits the work of this outer loop
        # across multiple CPU cores automatically.
        for i in prange(num_nodes):
            for j in range(num_time_points):
                # For the current point (node `i`, time `j`), get the six stress values.
                s_x = actual_sx[i, j]
                s_y = actual_sy[i, j]
                s_z = actual_sz[i, j]
                s_xy = actual_sxy[i, j]
                s_yz = actual_syz[i, j]
                s_xz = actual_sxz[i, j]

                # --- Step 1: Calculate Stress Invariants (I1, I2, I3) ---
                # To find the principal stresses, we first calculate three special values called
                # "invariants." They are called this because their values don't change even if you
                # rotate the object. They are fundamental properties of the stress state.
                I1 = s_x + s_y + s_z
                I2 = (s_x * s_y + s_y * s_z + s_z * s_x
                      - s_xy ** 2 - s_yz ** 2 - s_xz ** 2)
                I3 = (s_x * s_y * s_z
                      + 2 * s_xy * s_yz * s_xz
                      - s_x * s_yz ** 2
                      - s_y * s_xz ** 2
                      - s_z * s_xy ** 2)

                # --- Step 2: Formulate the Cubic Equation ---
                # The three principal stresses are the mathematical roots of a cubic polynomial equation
                # defined by the invariants. To solve it, we convert it into a simpler "depressed"
                # form: y³ + p*y + q = 0. The variables `p` and `q` are the coefficients for this equation.
                p = I2 - I1 ** 2 / 3.0
                q = (2.0 * I1 ** 3) / 27.0 - (I1 * I2) / 3.0 + I3

                # --- Step 3: Check for the special "Hydrostatic" case ---
                # This `if` statement checks for a simple case where an object is stressed equally
                # in all directions (like being deep underwater). In this case, `p` and `q` are
                # effectively zero, and all three principal stresses are equal.
                # This check handles that case directly and avoids division-by-zero errors in the
                # more complex calculations below, making the function more robust.
                if abs(p) < tiny_p and abs(q) < tiny_p:
                    s_hydro = I1 / 3.0
                    s1_out[i, j] = s_hydro
                    s2_out[i, j] = s_hydro
                    s3_out[i, j] = s_hydro
                    continue  # Skip to the next point in the loop

                # --- Step 4: Solve the Cubic Equation using the Trigonometric Method ---
                # For the general case, the roots are found using a reliable and stable trigonometric
                # formula (related to Viète's formulas).
                minus_p_over_3 = -p / 3.0
                # For real stresses, minus_p_over_3 must be non-negative, so sqrt is safe.
                sqrt_m = math.sqrt(minus_p_over_3)
                cos_arg = q / (2.0 * sqrt_m ** 3)

                # This is a numerical safety check. Due to tiny computer precision errors, `cos_arg`
                # might be slightly outside the valid [-1, 1] range for the `acos` function.
                # This code nudges it back into range to prevent a crash.
                if cos_arg > 1.0:
                    cos_arg = 1.0
                elif cos_arg < -1.0:
                    cos_arg = -1.0

                # These lines are the core of the trigonometric solution formula.
                phi = math.acos(cos_arg) / 3.0
                amp = 2.0 * sqrt_m

                # The final formulas give us the three roots, which are our principal stresses.
                s1 = I1 / 3.0 + amp * math.cos(phi)
                s2 = I1 / 3.0 + amp * math.cos(phi - two_pi_3)
                s3 = I1 / 3.0 + amp * math.cos(phi + two_pi_3)

                # --- Step 5: Sort the Results ---
                # The formulas don't guarantee which of s1, s2, or s3 is the largest.
                # This block of code performs a simple sort to ensure that s1 is always the
                # maximum value, s2 is the middle, and s3 is the minimum. This is the standard
                # engineering convention.
                if s1 < s2:
                    s1, s2 = s2, s1
                if s2 < s3:
                    s2, s3 = s3, s2
                if s1 < s2:
                    s1, s2 = s2, s1

                # --- Step 6: Store the Final Results ---
                # Assign the sorted principal stresses to their correct place in our output arrays.
                s1_out[i, j] = s1
                s2_out[i, j] = s2
                s3_out[i, j] = s3

        # After the loops have finished, return the three complete 2D arrays of results.
        return s1_out, s2_out, s3_out

    @staticmethod
    @njit
    def rainflow_counter(series):
        n = len(series)
        cycles = []
        stack = []
        for i in range(n):
            s = series[i]
            stack.append(s)
            while len(stack) >= 3:
                s0, s1, s2 = stack[-3], stack[-2], stack[-1]
                if (s1 - s0) * (s1 - s2) >= 0:
                    stack.pop(-2)
                else:
                    break
            if len(stack) >= 4:
                s0, s1, s2, s3 = stack[-4], stack[-3], stack[-2], stack[-1]
                if abs(s1 - s2) <= abs(s0 - s1):
                    cycles.append((abs(s1 - s2), 0.5))
                    stack.pop(-3)
        # Count residuals
        for i in range(len(stack) - 1):
            cycles.append((abs(stack[i] - stack[i + 1]), 0.5))
        # Convert cycles to ranges and counts
        ranges = np.array([c[0] for c in cycles])
        counts = np.array([c[1] for c in cycles])
        return ranges, counts

    @staticmethod
    @njit(parallel=True)
    def compute_potential_damage_for_all_nodes(sigma_vm, coeff_A, coeff_m):
        num_nodes = sigma_vm.shape[0]
        damages = np.zeros(num_nodes, dtype=NP_DTYPE)
        for i in prange(num_nodes):
            series = sigma_vm[i, :]
            ranges, counts = rainflow_counter(series)
            # Compute damage
            damage = np.sum(counts / (coeff_A / ((ranges + 1e-10) ** coeff_m)))
            damages[i] = damage
        return damages
    # endregion

    # region Core Computations (PyTorch/Numpy)
    @staticmethod
    def map_steady_state_stresses(steady_stress, steady_node_ids, modal_node_ids):
        """Map steady-state stress data to modal node IDs."""
        # Create a mapping from steady_node_ids to steady_stress
        steady_node_dict = dict(zip(steady_node_ids.flatten(), steady_stress.flatten()))
        # Create an array for mapped steady stress
        mapped_steady_stress = np.array([steady_node_dict.get(node_id, 0.0) for node_id in modal_node_ids],
                                        dtype=NP_DTYPE)
        return mapped_steady_stress

    def compute_normal_stresses(self, start_idx, end_idx):
        """Compute actual stresses using matrix multiplication."""
        actual_sx = torch.matmul(self.modal_sx[start_idx:end_idx, :], self.modal_coord)
        actual_sy = torch.matmul(self.modal_sy[start_idx:end_idx, :], self.modal_coord)
        actual_sz = torch.matmul(self.modal_sz[start_idx:end_idx, :], self.modal_coord)
        actual_sxy = torch.matmul(self.modal_sxy[start_idx:end_idx, :], self.modal_coord)
        actual_syz = torch.matmul(self.modal_syz[start_idx:end_idx, :], self.modal_coord)
        actual_sxz = torch.matmul(self.modal_sxz[start_idx:end_idx, :], self.modal_coord)

        # Add steady-state stresses if included
        if self.is_steady_state_included:
            actual_sx += self.steady_sx[start_idx:end_idx].unsqueeze(1)
            actual_sy += self.steady_sy[start_idx:end_idx].unsqueeze(1)
            actual_sz += self.steady_sz[start_idx:end_idx].unsqueeze(1)
            actual_sxy += self.steady_sxy[start_idx:end_idx].unsqueeze(1)
            actual_syz += self.steady_syz[start_idx:end_idx].unsqueeze(1)
            actual_sxz += self.steady_sxz[start_idx:end_idx].unsqueeze(1)

        return actual_sx.cpu().numpy(), actual_sy.cpu().numpy(), actual_sz.cpu().numpy(), \
            actual_sxy.cpu().numpy(), actual_syz.cpu().numpy(), actual_sxz.cpu().numpy()

    def compute_normal_stresses_for_a_single_node(self, selected_node_idx):
        """Compute actual stresses using matrix multiplication."""
        actual_sx = torch.matmul(self.modal_sx[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_sy = torch.matmul(self.modal_sy[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_sz = torch.matmul(self.modal_sz[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_sxy = torch.matmul(self.modal_sxy[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_syz = torch.matmul(self.modal_syz[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_sxz = torch.matmul(self.modal_sxz[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)

        # Add steady-state stresses if included
        if self.is_steady_state_included:
            actual_sx += self.steady_sx[selected_node_idx].unsqueeze(0)
            actual_sy += self.steady_sy[selected_node_idx].unsqueeze(0)
            actual_sz += self.steady_sz[selected_node_idx].unsqueeze(0)
            actual_sxy += self.steady_sxy[selected_node_idx].unsqueeze(0)
            actual_syz += self.steady_syz[selected_node_idx].unsqueeze(0)
            actual_sxz += self.steady_sxz[selected_node_idx].unsqueeze(0)

        return actual_sx.cpu().numpy(), actual_sy.cpu().numpy(), actual_sz.cpu().numpy(), actual_sxy.cpu().numpy(), actual_syz.cpu().numpy(), actual_sxz.cpu().numpy()

    def compute_deformations(self, start_idx, end_idx):
        """
        Compute actual nodal displacements (deformations) for all nodes in [start_idx, end_idx].
        This method multiplies the modal deformation modes with the modal coordinate matrix.
        """
        if self.modal_deformations_ux is None:
            return None  # No deformations data available

        # Compute displacements
        actual_ux = torch.matmul(self.modal_deformations_ux[start_idx:end_idx, :], self.modal_coord)
        actual_uy = torch.matmul(self.modal_deformations_uy[start_idx:end_idx, :], self.modal_coord)
        actual_uz = torch.matmul(self.modal_deformations_uz[start_idx:end_idx, :], self.modal_coord)
        return (actual_ux.cpu().numpy(), actual_uy.cpu().numpy(), actual_uz.cpu().numpy())
    # endregion

    # region Internal Batch Processing Helpers
    def _setup_calculation_jobs(self, calculate_von_mises, calculate_max_principal_stress,
                                calculate_min_principal_stress, calculate_deformation,
                                calculate_velocity, calculate_acceleration, calculate_damage):
        """
        Initializes a dictionary of calculation jobs, their memmap files, and result metadata.
        This centralizes the configuration for all possible calculations.
        """
        jobs = {}

        if calculate_von_mises:
            self.max_over_time_svm = -np.inf * np.ones(self.modal_coord.shape[1], dtype=self.NP_DTYPE)
            jobs['von_mises'] = {
                'max_memmap': np.memmap(os.path.join(self.output_directory, 'max_von_mises_stress.dat'),
                                        dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'time_memmap': np.memmap(os.path.join(self.output_directory, 'time_of_max_von_mises_stress.dat'),
                                         dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'csv_header_val': "SVM_Max",
                'csv_header_time': "Time_of_SVM_Max"
            }

        if calculate_max_principal_stress:
            self.max_over_time_s1 = -np.inf * np.ones(self.modal_coord.shape[1], dtype=self.NP_DTYPE)
            jobs['s1_max'] = {
                'max_memmap': np.memmap(os.path.join(self.output_directory, 'max_s1_stress.dat'),
                                        dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'time_memmap': np.memmap(os.path.join(self.output_directory, 'time_of_max_s1_stress.dat'),
                                         dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'csv_header_val': "S1_Max",
                'csv_header_time': "Time_of_S1_Max"
            }

        if calculate_min_principal_stress:
            self.min_over_time_s3 = np.inf * np.ones(self.modal_coord.shape[1], dtype=self.NP_DTYPE)
            jobs['s3_min'] = {
                'min_memmap': np.memmap(os.path.join(self.output_directory, 'min_s3_stress.dat'),
                                        dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'time_memmap': np.memmap(os.path.join(self.output_directory, 'time_of_min_s3_stress.dat'),
                                         dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'csv_header_val': "S3_Min",
                'csv_header_time': "Time_of_S3_Min"
            }

        if calculate_deformation:
            self.max_over_time_def = -np.inf * np.ones(self.modal_coord.shape[1], dtype=self.NP_DTYPE)
            jobs['deformation'] = {
                'max_memmap': np.memmap(os.path.join(self.output_directory, 'max_deformation.dat'),
                                        dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'time_memmap': np.memmap(os.path.join(self.output_directory, 'time_of_max_deformation.dat'),
                                         dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'csv_header_val': "DEF_Max",
                'csv_header_time': "Time_of_DEF_Max"
            }

        if calculate_velocity:
            self.max_over_time_vel = -np.inf * np.ones(self.modal_coord.shape[1], dtype=self.NP_DTYPE)
            jobs['velocity'] = {
                'max_memmap': np.memmap(os.path.join(self.output_directory, 'max_velocity.dat'),
                                        dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'time_memmap': np.memmap(os.path.join(self.output_directory, 'time_of_max_velocity.dat'),
                                         dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'csv_header_val': "VEL_Max",
                'csv_header_time': "Time_of_VEL_Max"
            }

        if calculate_acceleration:
            self.max_over_time_acc = -np.inf * np.ones(self.modal_coord.shape[1], dtype=self.NP_DTYPE)
            jobs['acceleration'] = {
                'max_memmap': np.memmap(os.path.join(self.output_directory, 'max_acceleration.dat'),
                                        dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'time_memmap': np.memmap(os.path.join(self.output_directory, 'time_of_max_acceleration.dat'),
                                         dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'csv_header_val': "ACC_Max",
                'csv_header_time': "Time_of_ACC_Max"
            }

        if calculate_damage:
            jobs['damage'] = {
                'damage_memmap': np.memmap(os.path.join(self.output_directory, 'potential_damage_results.dat'),
                                           dtype=self.RESULT_DTYPE, mode='w+', shape=(self.modal_sx.shape[0],)),
                'csv_header_val': "Potential Damage (Damage Index)"
            }

        return jobs

    def _process_stress_chunk(self, jobs, time_values, start_idx, end_idx, actual_sx, actual_sy, actual_sz, actual_sxy,
                              actual_syz, actual_sxz):
        """Processes all stress-derived calculations for a given chunk of nodes."""
        # --- Von Mises Stress Calculation ---
        if 'von_mises' in jobs or 'damage' in jobs:
            start_time = time.time()
            sigma_vm = self.compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                     actual_sxz)
            print(f"Elapsed time for von Mises stresses: {(time.time() - start_time):.3f} seconds")

            if 'von_mises' in jobs:
                job = jobs['von_mises']
                self.max_over_time_svm = np.maximum(self.max_over_time_svm, np.max(sigma_vm, axis=0))
                job['max_memmap'][start_idx:end_idx] = np.max(sigma_vm, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(sigma_vm, axis=1)]

        # --- Principal Stress Calculation ---
        if 's1_max' in jobs or 's3_min' in jobs:
            start_time = time.time()
            s1, _, s3 = self.compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                        actual_sxz)
            print(f"Elapsed time for principal stresses: {(time.time() - start_time):.3f} seconds")

            if 's1_max' in jobs:
                job = jobs['s1_max']
                self.max_over_time_s1 = np.maximum(self.max_over_time_s1, np.max(s1, axis=0))
                job['max_memmap'][start_idx:end_idx] = np.max(s1, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(s1, axis=1)]

            if 's3_min' in jobs:
                job = jobs['s3_min']
                self.min_over_time_s3 = np.minimum(self.min_over_time_s3, np.min(s3, axis=0))
                job['min_memmap'][start_idx:end_idx] = np.min(s3, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmin(s3, axis=1)]

        # --- Damage Calculation ---
        if 'damage' in jobs:
            start_time = time.time()
            job = jobs['damage']
            signed_von_mises = self.compute_signed_von_mises_stress(sigma_vm, actual_sx, actual_sy, actual_sz)
            # Use fatigue parameters if they have been set; otherwise, fall back to defaults.
            coeff_A = getattr(self, 'fatigue_A', 1)
            coeff_m = getattr(self, 'fatigue_m', -3)
            potential_damages = self.compute_potential_damage_for_all_nodes(signed_von_mises, coeff_A, coeff_m)
            job['damage_memmap'][start_idx:end_idx] = potential_damages
            print(f"Elapsed time for damage index calculation: {(time.time() - start_time):.3f} seconds")

    def _process_kinematics_chunk(self, jobs, time_values, start_idx, end_idx):
        """Processes deformation, velocity, and acceleration for a given chunk."""
        if self.modal_deformations_ux is None:
            return

        ux, uy, uz = self.compute_deformations(start_idx, end_idx)

        # --- Deformation ---
        if 'deformation' in jobs:
            start_time = time.time()
            job = jobs['deformation']
            def_mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
            self.max_over_time_def = np.maximum(self.max_over_time_def, np.max(def_mag, axis=0))
            job['max_memmap'][start_idx:end_idx] = np.max(def_mag, axis=1)
            job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(def_mag, axis=1)]
            print(f"Elapsed time for deformation magnitude and time: {(time.time() - start_time):.3f} seconds")

        # --- Velocity & Acceleration ---
        if 'velocity' in jobs or 'acceleration' in jobs:
            start_time = time.time()
            vel_mag, acc_mag, _, _, _, _, _, _ = self._vel_acc_from_disp(ux, uy, uz, self.time_values)
            print(
                f"Elapsed time for calculation of velocity/acceleration components: {(time.time() - start_time):.3f} seconds")

            if 'velocity' in jobs:
                start_time = time.time()
                job = jobs['velocity']
                self.max_over_time_vel = np.maximum(self.max_over_time_vel, np.max(vel_mag, axis=0))
                job['max_memmap'][start_idx:end_idx] = np.max(vel_mag, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(vel_mag, axis=1)]
                print(f"Elapsed time for velocity magnitude and time: {(time.time() - start_time):.3f} seconds")

            if 'acceleration' in jobs:
                start_time = time.time()
                job = jobs['acceleration']
                self.max_over_time_acc = np.maximum(self.max_over_time_acc, np.max(acc_mag, axis=0))
                job['max_memmap'][start_idx:end_idx] = np.max(acc_mag, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(acc_mag, axis=1)]
                print(f"Elapsed time for acceleration magnitude and time: {(time.time() - start_time):.3f} seconds")
    # endregion

    # region Main Methods
    def process_results_in_batch(self,
                                 time_values,
                                 df_node_ids,
                                 node_coords,
                                 calculate_damage=False,
                                 calculate_von_mises=False,
                                 calculate_max_principal_stress=False,
                                 calculate_min_principal_stress=False,
                                 calculate_deformation=False,
                                 calculate_velocity=False,
                                 calculate_acceleration=False):
        """
        Processes stress and deformation results in batches to manage memory usage.
        This method coordinates the setup, execution, and finalization of calculations.
        """
        # --- 1. Initialization and Memory Estimation ---
        print("--- Starting Batch Processing ---")
        num_nodes, _ = self.modal_sx.shape
        num_time_points = self.modal_coord.shape[1]

        my_virtual_memory = psutil.virtual_memory()
        self.total_memory = my_virtual_memory.total / (1024 ** 3)
        self.available_memory = my_virtual_memory.available / (1024 ** 3)
        self.allocated_memory = my_virtual_memory.available * self.RAM_PERCENT / (1024 ** 3)
        print(f"Total system RAM: {self.total_memory:.2f} GB")
        print(f"Available system RAM: {self.available_memory:.2f} GB")
        print(f"Allocated system RAM: {self.allocated_memory:.2f} GB")

        chunk_size = self._estimate_chunk_size(
            num_time_points, calculate_von_mises, calculate_max_principal_stress, calculate_damage,
            calculate_deformation, calculate_velocity, calculate_acceleration)
        num_iterations = (num_nodes + chunk_size - 1) // chunk_size

        memory_per_node = self._get_memory_per_node(
            num_time_points, calculate_von_mises, calculate_max_principal_stress, calculate_damage,
            calculate_deformation, calculate_velocity, calculate_acceleration)
        memory_required_per_iteration = self._estimate_ram_required_per_iteration(chunk_size, memory_per_node)

        print(f"Processing {num_nodes} nodes in {num_iterations} iterations (chunk size: {chunk_size}).")
        print(f"Estimated RAM required per iteration: {memory_required_per_iteration:.2f} GB\n")

        # --- 2. Setup Calculation Jobs and Memmap Files ---
        calculation_jobs = self._setup_calculation_jobs(
            calculate_von_mises, calculate_max_principal_stress, calculate_min_principal_stress,
            calculate_deformation, calculate_velocity, calculate_acceleration, calculate_damage
        )

        is_stress_needed = any(k in calculation_jobs for k in ['von_mises', 's1_max', 's3_min', 'damage'])
        is_kinematics_needed = any(k in calculation_jobs for k in ['deformation', 'velocity', 'acceleration'])

        # --- 3. Main Processing Loop ---
        for i, start_idx in enumerate(range(0, num_nodes, chunk_size)):
            end_idx = min(start_idx + chunk_size, num_nodes)
            print(f"\n--- Iteration {i + 1}/{num_iterations} (Nodes {start_idx}-{end_idx - 1}) ---")

            actual_stresses = None
            if is_stress_needed:
                start_time = time.time()
                actual_stresses = self.compute_normal_stresses(start_idx, end_idx)
                print(f"Elapsed time for normal stresses: {(time.time() - start_time):.3f} seconds")

            if is_stress_needed:
                self._process_stress_chunk(calculation_jobs, time_values, start_idx, end_idx, *actual_stresses)

            if is_kinematics_needed:
                self._process_kinematics_chunk(calculation_jobs, time_values, start_idx, end_idx)

            # --- Memory Management and Progress Update ---
            start_time = time.time()
            del actual_stresses
            gc.collect()
            print(f"Elapsed time for garbage collection: {(time.time() - start_time):.3f} seconds")

            progress_percentage = ((i + 1) / num_iterations) * 100
            self.progress_signal.emit(int(progress_percentage))
            QApplication.processEvents()

            current_available_memory = psutil.virtual_memory().available
            print(
                f"Iteration {i + 1} complete. Available RAM: {current_available_memory / (1024 ** 3):.2f} GB. Progress: {progress_percentage:.1f}%")

        # --- 4. Finalization ---
        print("\n--- Finalizing Results ---")
        self._finalize_and_convert_results(calculation_jobs, df_node_ids, node_coords)
        print("--- Batch Processing Finished ---")

    def process_results_for_a_single_node(self,
                                          selected_node_idx,
                                          selected_node_id,
                                          _df_node_ids, # will be used in future for multiple node plots
                                          calculate_von_mises=False,
                                          calculate_max_principal_stress=False,
                                          calculate_min_principal_stress=False,
                                          calculate_deformation=False,
                                          calculate_velocity=False,
                                          calculate_acceleration=False):
        """
        Process results for a single node and return the stress data for plotting.

        Parameters:
        - selected_node_idx: The index of the node to process.
        - calculate_von_mises: Boolean flag to compute Von Mises stress.
        - calculate_max_principal_stress: Boolean flag to compute Max Principal Stress.
        - calculate_min_principal_stress: Boolean flag to compute Min Principal Stress.
        - calculate deformation,velocity,acceleration flag are also used for computing the related parameters

        Returns:
        - time_points: Array of time points for the selected node.
        - stress_values: Array of stress values (either Von Mises or Max/Min Principal Stress).
        """

        is_stress_calc_needed = calculate_von_mises or calculate_max_principal_stress or calculate_min_principal_stress
        if is_stress_calc_needed:
            # region Compute normal stresses for the selected node
            actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
                self.compute_normal_stresses_for_a_single_node(selected_node_idx)
            # endregion

            if calculate_von_mises:
                # Compute Von Mises stress for the selected node
                sigma_vm = self.compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                         actual_sxz)
                print(f"Von Mises Stress calculated for Node {selected_node_id}\n")

                return np.arange(sigma_vm.shape[1]), sigma_vm[0, :]  # time_points, stress_values

            if calculate_max_principal_stress or calculate_min_principal_stress:
                s1, _, s3 = self.compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                             actual_sxz)
                if calculate_max_principal_stress:
                    # Compute Principal Stresses for the selected node
                    print(f"Max Principal Stresses calculated for Node {selected_node_id}\n")
                    return np.arange(s1.shape[1]), s1[0, :]  # time_indices, stress_values

                if calculate_min_principal_stress:
                    print(f"Min Principal Stresses calculated for Node {selected_node_id}\n")
                    return np.arange(s3.shape[1]), s3[0, :]  # S₃ min history

        if calculate_deformation or calculate_velocity or calculate_acceleration:
            if self.modal_deformations_ux is None:
                print("Deformation data missing – velocity/acceleration/deformation calculations are skipped.")
            else:
                ux, uy, uz = self.compute_deformations(selected_node_idx, selected_node_idx+1)

                if calculate_deformation:
                    def_mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
                    deformation_data = {
                        'Magnitude': def_mag[0, :],
                        'X': ux[0, :],
                        'Y': uy[0, :],
                        'Z': uz[0, :]
                    }
                    print(f"Deformation calculated for Node {selected_node_id}\n")
                    return np.arange(def_mag.shape[1]), deformation_data

                if calculate_velocity or calculate_acceleration:
                    vel_mag, acc_mag, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z = \
                        self._vel_acc_from_disp(ux, uy, uz, self.time_values)
                    if calculate_velocity:
                        velocity_data = {
                            'Magnitude': vel_mag[0, :],
                            'X': vel_x[0, :],
                            'Y': vel_y[0, :],
                            'Z': vel_z[0, :]
                        }
                        print(f"Velocity calculated for Node {selected_node_id}\n")
                        return np.arange(vel_mag.shape[1]), velocity_data
                    if calculate_acceleration:
                        acceleration_data = {
                            'Magnitude': acc_mag[0, :],
                            'X': acc_x[0, :],
                            'Y': acc_y[0, :],
                            'Z': acc_z[0, :]
                        }
                        print(f"Acceleration calculated for Node {selected_node_id}\n")
                        return np.arange(acc_mag.shape[1]), acceleration_data

        # Return none if no output is requested
        return None, None
    # endregion

    # region File I/O Utilities
    def _convert_dat_to_csv(self, node_ids, node_coords, dat_filename, csv_filename, header):
        """Converts a .dat file to a .csv file with NodeID and, if available, X,Y,Z coordinates."""
        try:
            # Read the memmap file as a NumPy array
            data = np.memmap(dat_filename, dtype=RESULT_DTYPE, mode='r', shape=(len(node_ids),))
            # Create a DataFrame for NodeID and the computed stress data
            df_out = pd.DataFrame({
                'NodeID': node_ids,
                header: data
            })
            # If node_coords is available, include the X, Y, Z coordinates
            if node_coords is not None:
                df_coords = pd.DataFrame(node_coords, columns=['X', 'Y', 'Z'])
                df_out = pd.concat([df_out, df_coords], axis=1)
            # Save to CSV
            df_out.to_csv(csv_filename, index=False)
            print(f"Successfully converted {dat_filename} to {csv_filename}.")
        except Exception as e:
            print(f"Error converting {dat_filename} to {csv_filename}: {e}")

    def _finalize_and_convert_results(self, jobs, df_node_ids, node_coords):
        """Flushes all memmap files and converts them to CSV."""
        for job_name, job_data in jobs.items():
            if job_name == 'damage':
                memmap_file = job_data['damage_memmap']
                memmap_file.flush()
                self._convert_dat_to_csv(df_node_ids, node_coords,
                                         memmap_file.filename,
                                         memmap_file.filename.replace('.dat', '.csv'),
                                         job_data['csv_header_val'])
            elif job_name == 's3_min':
                min_memmap = job_data['min_memmap']
                time_memmap = job_data['time_memmap']
                min_memmap.flush()
                time_memmap.flush()
                self._convert_dat_to_csv(df_node_ids, node_coords,
                                         min_memmap.filename,
                                         min_memmap.filename.replace('.dat', '.csv'),
                                         job_data['csv_header_val'])
                self._convert_dat_to_csv(df_node_ids, node_coords,
                                         time_memmap.filename,
                                         time_memmap.filename.replace('.dat', '.csv'),
                                         job_data['csv_header_time'])
            else:  # Handles max value cases (s1, svm, def, vel, acc)
                max_memmap = job_data['max_memmap']
                time_memmap = job_data['time_memmap']
                max_memmap.flush()
                time_memmap.flush()
                self._convert_dat_to_csv(df_node_ids, node_coords,
                                         max_memmap.filename,
                                         max_memmap.filename.replace('.dat', '.csv'),
                                         job_data['csv_header_val'])
                self._convert_dat_to_csv(df_node_ids, node_coords,
                                         time_memmap.filename,
                                         time_memmap.filename.replace('.dat', '.csv'),
                                         job_data['csv_header_time'])
    # endregion
