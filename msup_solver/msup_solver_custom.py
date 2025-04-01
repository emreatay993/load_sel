# region Import libraries
import math
import threading
import numpy as np
import pandas as pd
import torch
import psutil
import gc
from numba import njit, prange
import time
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QMainWindow, QCheckBox, QProgressBar, QFileDialog, QGroupBox, QGridLayout, QSizePolicy,
                             QTextEdit, QTabWidget, QComboBox, QMenuBar, QAction, QDockWidget, QTreeView,
                             QFileSystemModel, QMessageBox, QSpinBox, QDoubleSpinBox, QShortcut, QSplitter)
from PyQt5.QtGui import QPalette, QColor, QFont, QTextCursor, QKeySequence
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QUrl, QDir, QStandardPaths, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineView
import sys
from io import StringIO
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly_resampler import FigureResampler
import cProfile
# endregion

# region Define global variables
NP_DTYPE = np.float32
TORCH_DTYPE = torch.float32
RESULT_DTYPE = 'float32'

IS_GPU_ACCELERATION_ENABLED = False

IS_WRITE_TO_DISK_STRESS_S1_MAX_AT_ALL_TIME_POINTS = True
IS_WRITE_TO_DISK_TIMES_OF_MAX_STRESS_S1 = True

IS_WRITE_TO_DISK_VON_MISES_MAX_AT_ALL_TIME_POINTS = True
IS_WRITE_TO_DISK_TIMES_OF_MAX_VON_STRESS_AT_EACH_NODE = True

# Set OpenBLAS to use all available CPU cores
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
# endregion

# region Define global functions
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
        cycles.append((abs(stack[i] - stack[i+1]), 0.5))
    # Convert cycles to ranges and counts
    ranges = np.array([c[0] for c in cycles])
    counts = np.array([c[1] for c in cycles])
    return ranges, counts

@njit(parallel=True)
def compute_potential_damage_for_all_nodes(sigma_vm, A, m):
    num_nodes = sigma_vm.shape[0]
    damages = np.zeros(num_nodes, dtype=NP_DTYPE)
    for i in prange(num_nodes):
        series = sigma_vm[i, :]
        ranges, counts = rainflow_counter(series)
        # Compute damage
        damage = np.sum(counts / (A / ((ranges + 1e-10) ** m)))
        damages[i] = damage
    return damages

def get_node_index_from_id(node_id, node_ids):
    """
    Map the given node_id to its corresponding index in the modal stress array.

    Parameters:
    - node_id: The node ID to map.
    - node_ids: The array of node IDs.

    Returns:
    - The index of the node ID in the modal stress array.
    """
    try:
        # Find the index of the node ID in the list of node IDs
        return np.where(node_ids == node_id)[0][0]
    except IndexError:
        print(f"Node ID {node_id} not found in the list of nodes.")
        return None

def unwrap_mcf_file(input_file, output_file):
    """
    Unwraps a file that has a header section and then data lines.
    After the header line (the one starting with "Number of Modes"),
    some records are wrapped. Additionally, there is a header line
    (e.g. "      Time          Coordinates...") in the data block that should
    remain separate. The algorithm:

    1. Keeps all lines up to and including the line that starts (after stripping)
       with "Number of Modes".
    2. For the remaining lines, if a line (after stripping) contains both "Time"
       and "Coordinates", it is treated as a header line and is preserved as its own record.
    3. For other lines, the minimum indentation among them is determined (the base indent).
       Lines with exactly that indentation start new records, while lines with extra
       indentation are treated as continuations (wrapped lines) and appended to the previous record.
    """
    # Read all lines
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Separate header (everything up to and including the line that starts with "Number of Modes")
    header_end = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Number of Modes"):
            header_end = i
            break
    if header_end is None:
        header_lines = []
        data_lines = lines
    else:
        header_lines = lines[:header_end + 1]
        data_lines = lines[header_end + 1:]

    # For base indentation calculation, skip any line that (after stripping) is a header line
    # like the one with "Time" and "Coordinates".
    data_non_header = []
    for line in data_lines:
        stripped = line.strip()
        if stripped and ("Time" in stripped and "Coordinates" in stripped):
            continue  # skip header lines for indent calculation
        if stripped:
            data_non_header.append(line)
    base_indent = None
    for line in data_non_header:
        indent = len(line) - len(line.lstrip(' '))
        if base_indent is None or indent < base_indent:
            base_indent = indent
    if base_indent is None:
        base_indent = 0

    # Process data lines:
    unwrapped_data = []
    current_line = ""
    for line in data_lines:
        stripped = line.strip()
        if not stripped:
            continue  # skip empty lines

        # If this line is the special header (e.g., "Time          Coordinates...")
        if "Time" in stripped and "Coordinates" in stripped:
            if current_line:
                unwrapped_data.append(current_line)
                current_line = ""
            unwrapped_data.append(stripped)
            continue

        # Determine indentation of the current line.
        indent = len(line) - len(line.lstrip(' '))
        if indent == base_indent:
            # New record.
            if current_line:
                unwrapped_data.append(current_line)
            current_line = stripped
        else:
            # Wrapped (continuation) line.
            current_line = current_line.rstrip('\n') + " " + stripped

    if current_line:
        unwrapped_data.append(current_line)

    # Combine header and unwrapped data.
    final_lines = [h.rstrip('\n') for h in header_lines] + unwrapped_data

    # Write final result to output file.
    with open(output_file, 'w') as f:
        for line in final_lines:
            f.write(line + "\n")

    return final_lines
# endregion

# region Define global class & functions
class MSUPSmartSolverTransient(QObject):
    progress_signal = pyqtSignal(int)

    def __init__(self, modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord,
                 steady_sx=None, steady_sy=None, steady_sz=None, steady_sxy=None, steady_syz=None, steady_sxz=None,
                 steady_node_ids=None, modal_node_ids=None, output_directory=None, modal_deformations=None):
        super().__init__()

        # Use selected output directory or fallback to script location
        self.output_directory = output_directory if output_directory else os.path.dirname(os.path.abspath(__file__))

        # Global settings
        self.NP_DTYPE = NP_DTYPE
        self.TORCH_DTYPE = TORCH_DTYPE
        self.RESULT_DTYPE = RESULT_DTYPE
        self.ELEMENT_SIZE = np.dtype(self.NP_DTYPE).itemsize
        self.RAM_PERCENT = 0.1
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

        # Memory details
        self.total_memory = psutil.virtual_memory().total / (1024 ** 3)
        self.available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.allocated_memory = psutil.virtual_memory().available * self.RAM_PERCENT / (1024 ** 3)
        print(f"Total system RAM: {self.total_memory:.2f} GB")
        print(f"Available system RAM: {self.available_memory:.2f} GB")
        print(f"Allocated system RAM: {self.allocated_memory:.2f} GB")

    def get_chunk_size(self, num_nodes, num_time_points, calculate_von_mises, calculate_principal_stress,
                       calculate_damage):
        """Calculate the optimal chunk size for processing based on available memory."""
        available_memory = psutil.virtual_memory().available * self.RAM_PERCENT
        memory_per_node = self.get_memory_per_node(num_time_points, calculate_von_mises, calculate_principal_stress,
                                                   calculate_damage)
        max_nodes_per_iteration = available_memory // memory_per_node
        return max(1, int(max_nodes_per_iteration))  # Ensure at least one node per chunk

    def estimate_ram_required_per_iteration(self, chunk_size, memory_per_node):
        """Estimate the total RAM required per iteration to compute stresses."""
        total_memory = chunk_size * memory_per_node
        return total_memory / (1024 ** 3)  # Convert bytes to GB

    def get_memory_per_node(self, num_time_points, calculate_von_mises, calculate_principal_stress, calculate_damage):
        num_arrays = 6  # For actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz
        if calculate_von_mises:
            num_arrays += 1  # For sigma_vm
        if calculate_principal_stress:
            num_arrays += 3  # For s1, s2, s3
        if calculate_damage:
            num_arrays += 1  # For signed_von_mises
        # Add any additional arrays used during calculations
        memory_per_node = num_arrays * num_time_points * self.ELEMENT_SIZE
        return memory_per_node

    def map_steady_state_stresses(self, steady_stress, steady_node_ids, modal_node_ids):
        """Map steady-state stress data to modal node IDs."""
        # Create a mapping from steady_node_ids to steady_stress
        steady_node_dict = dict(zip(steady_node_ids.flatten(), steady_stress.flatten()))
        # Create an array for mapped steady stress
        mapped_steady_stress = np.array([steady_node_dict.get(node_id, 0.0) for node_id in modal_node_ids], dtype=NP_DTYPE)
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

        return actual_sx.cpu().numpy(), actual_sy.cpu().numpy(), actual_sz.cpu().numpy(), actual_sxy.cpu().numpy(), actual_syz.cpu().numpy(), actual_sxz.cpu().numpy()

    def compute_normal_stresses_for_a_single_node(self, selected_node_idx):
        """Compute actual stresses using matrix multiplication."""
        actual_sx = torch.matmul(self.modal_sx[selected_node_idx: selected_node_idx +1, :], self.modal_coord)
        actual_sy = torch.matmul(self.modal_sy[selected_node_idx: selected_node_idx +1, :], self.modal_coord)
        actual_sz = torch.matmul(self.modal_sz[selected_node_idx: selected_node_idx +1, :], self.modal_coord)
        actual_sxy = torch.matmul(self.modal_sxy[selected_node_idx: selected_node_idx +1, :], self.modal_coord)
        actual_syz = torch.matmul(self.modal_syz[selected_node_idx: selected_node_idx +1, :], self.modal_coord)
        actual_sxz = torch.matmul(self.modal_sxz[selected_node_idx: selected_node_idx +1, :], self.modal_coord)

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

        # Compute displacements (you can adjust the math if you have a different formulation)
        actual_ux = torch.matmul(self.modal_deformations_ux[start_idx:end_idx, :], self.modal_coord)
        actual_uy = torch.matmul(self.modal_deformations_uy[start_idx:end_idx, :], self.modal_coord)
        actual_uz = torch.matmul(self.modal_deformations_uz[start_idx:end_idx, :], self.modal_coord)
        return (actual_ux.cpu().numpy(), actual_uy.cpu().numpy(), actual_uz.cpu().numpy())

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
    def compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
        """Compute principal stresses using NumPy vectorized operations."""
        num_nodes, num_time_points = actual_sx.shape

        # Construct the stress tensors
        stress_tensors = np.zeros((num_nodes, num_time_points, 3, 3), dtype=actual_sx.dtype)
        stress_tensors[:, :, 0, 0] = actual_sx
        stress_tensors[:, :, 1, 1] = actual_sy
        stress_tensors[:, :, 2, 2] = actual_sz
        stress_tensors[:, :, 0, 1] = actual_sxy
        stress_tensors[:, :, 1, 0] = actual_sxy
        stress_tensors[:, :, 0, 2] = actual_sxz
        stress_tensors[:, :, 2, 0] = actual_sxz
        stress_tensors[:, :, 1, 2] = actual_syz
        stress_tensors[:, :, 2, 1] = actual_syz

        # Reshape to combine nodes and time points
        stress_tensors_reshaped = stress_tensors.reshape(-1, 3, 3)

        # Compute eigenvalues
        eigvals = np.linalg.eigh(stress_tensors_reshaped)[0]  # Eigenvalues in ascending order

        # Reshape back to original dimensions
        eigvals = eigvals.reshape(num_nodes, num_time_points, 3)

        # Extract principal stresses
        s1 = eigvals[:, :, 2]  # Largest eigenvalue
        s2 = eigvals[:, :, 1]  # Middle eigenvalue
        s3 = eigvals[:, :, 0]  # Smallest eigenvalue

        return s1, s2, s3

    @staticmethod
    @njit(parallel=True)
    def compute_principal_stresses_yedek(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
        """Compute principal stresses using analytical methods without constructing full stress tensors."""
        num_nodes, num_time_points = actual_sx.shape
        s1 = np.zeros((num_nodes, num_time_points), dtype=actual_sx.dtype)
        s2 = np.zeros_like(s1)
        s3 = np.zeros_like(s1)

        for i in prange(num_nodes):
            for j in range(num_time_points):
                # Extract stress components
                sx = actual_sx[i, j]
                sy = actual_sy[i, j]
                sz = actual_sz[i, j]
                sxy = actual_sxy[i, j]
                syz = actual_syz[i, j]
                sxz = actual_sxz[i, j]

                # Compute the stress invariants
                I1 = sx + sy + sz  # First invariant
                I2 = (sx * sy + sy * sz + sz * sx) - (sxy ** 2 + syz ** 2 + sxz ** 2)  # Second invariant
                I3 = (sx * sy * sz + 2 * sxy * syz * sxz) - (
                            sx * syz ** 2 + sy * sxz ** 2 + sz * sxy ** 2)  # Third invariant

                # Compute coefficients of the characteristic equation: λ^3 - I1*λ^2 + I2*λ - I3 = 0
                a = -1
                b = I1
                c = -I2
                d = I3

                # Normalize the cubic equation to the depressed cubic: t^3 + pt + q = 0
                p = - (b ** 2) / (3 * a ** 2) + c / a
                q = (2 * b ** 3) / (27 * a ** 3) - (b * c) / (3 * a ** 2) + d / a

                # Compute discriminant
                discriminant = (q / 2) ** 2 + (p / 3) ** 3

                # Compute roots based on the discriminant
                if discriminant > 0:
                    # One real root and two complex conjugate roots
                    A = (-q / 2 + np.sqrt(discriminant)) ** (1 / 3)
                    B = (-q / 2 - np.sqrt(discriminant)) ** (1 / 3)
                    root1 = (A + B) - b / (3 * a)
                    root2 = np.nan  # Complex root
                    root3 = np.nan  # Complex root
                elif discriminant == 0:
                    # All roots real, at least two are equal
                    A = (-q / 2) ** (1 / 3)
                    root1 = 2 * A - b / (3 * a)
                    root2 = -A - b / (3 * a)
                    root3 = root2
                else:
                    # Three real roots
                    phi = np.arccos(-q / (2 * np.sqrt(-(p / 3) ** 3)))
                    t = 2 * np.sqrt(-p / 3)
                    root1 = t * np.cos(phi / 3) - b / (3 * a)
                    root2 = t * np.cos((phi + 2 * np.pi) / 3) - b / (3 * a)
                    root3 = t * np.cos((phi + 4 * np.pi) / 3) - b / (3 * a)

                # Collect real roots and sort them in descending order
                roots = np.array([root for root in [root1, root2, root3] if not np.isnan(root)])
                roots.sort()
                s1[i, j] = roots[-1]  # Largest principal stress
                s2[i, j] = roots[-2] if len(roots) > 1 else roots[-1]
                s3[i, j] = roots[0] if len(roots) > 2 else roots[0]

        return s1, s2, s3

    def process_results(self, calculate_damage=False, calculate_von_mises=False, calculate_principal_stress=False):
        """Process stress results to compute potential (relative) fatigue damage."""
        num_nodes, num_modes = self.modal_sx.shape
        num_time_points = self.modal_coord.shape[1]

        # Initialize max over time vectors
        self.max_over_time_svm = -np.inf * np.ones(num_time_points, dtype=NP_DTYPE)
        self.max_over_time_s1 = -np.inf * np.ones(num_time_points, dtype=NP_DTYPE)

        # Get the chunk size based on selected options
        chunk_size = self.get_chunk_size(
            num_nodes, num_time_points,
            calculate_von_mises, calculate_principal_stress, calculate_damage
        )

        num_iterations = (num_nodes + chunk_size - 1) // chunk_size
        print(f"Estimated number of iterations to avoid memory overflow: {num_iterations}")

        memory_per_node = self.get_memory_per_node(
            num_time_points, calculate_von_mises, calculate_principal_stress, calculate_damage
        )
        memory_required_per_iteration = self.estimate_ram_required_per_iteration(chunk_size, memory_per_node)
        print(f"Estimated RAM required per iteration: {memory_required_per_iteration:.2f} GB\n")

        if calculate_principal_stress:
            # Create memmap files for storing the maximum principal stresses per node (s1)
            s1_max_memmap = np.memmap(os.path.join(self.output_directory, 'max_s1_stress.dat'),
                                      dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))
            s1_max_time_memmap = np.memmap(os.path.join(self.output_directory, 'time_of_max_s1_stress.dat'),
                                           dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))

        if calculate_von_mises:
            # Create memmap files for storing the maximum von Mises stresses per node
            von_mises_max_memmap = np.memmap(os.path.join(self.output_directory, 'max_von_mises_stress.dat'),
                                             dtype=RESULT_DTYPE, mode='w+',shape=(num_nodes,))
            von_mises_max_time_memmap = np.memmap(os.path.join(self.output_directory, 'time_of_max_von_mises_stress.dat'),
                                                  dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))

        if calculate_damage:
            potential_damage_memmap = np.memmap(os.path.join(self.output_directory,'potential_damage_results.dat'),
                                                dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))

        for start_idx in range(0, num_nodes, chunk_size):
            end_idx = min(start_idx + chunk_size, num_nodes)

            # Calculate normal stresses
            start_time = time.time()
            actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
                self.compute_normal_stresses(start_idx, end_idx)
            print(f"Elapsed time for normal stresses: {(time.time() - start_time):.3f} seconds")

            if calculate_von_mises:
                # Calculate von Mises stresses
                start_time = time.time()
                sigma_vm = self.compute_von_mises_stress(actual_sx, actual_sy, actual_sz,
                                                         actual_sxy, actual_syz, actual_sxz)
                print(f"Elapsed time for von Mises stresses: {(time.time() - start_time):.3f} seconds")

                # Update max_over_time using the maximum from this chunk (axis=0 for time points)
                chunk_max = np.max(sigma_vm, axis=0)
                self.max_over_time_svm = np.maximum(self.max_over_time_svm, chunk_max)

                # Calculate the maximum von Mises stress and its time index for each node
                start_time = time.time()
                max_von_mises_stress_per_node = np.max(sigma_vm, axis=1)
                time_indices = np.argmax(sigma_vm, axis=1)
                time_of_max_von_mises_stress_per_node = time_values[time_indices]
                von_mises_max_memmap[start_idx:end_idx] = max_von_mises_stress_per_node
                von_mises_max_time_memmap[start_idx:end_idx] = time_of_max_von_mises_stress_per_node
                print(f"Elapsed time for max von Mises stress and time: {(time.time() - start_time):.3f} seconds")

            if calculate_principal_stress:
                # Calculate principal stresses
                start_time = time.time()
                s1, s2, s3 = self.compute_principal_stresses(actual_sx, actual_sy, actual_sz,
                                                             actual_sxy, actual_syz, actual_sxz)
                print(f"Elapsed time for principal stresses: {(time.time() - start_time):.3f} seconds")

                # Update global max for s1 (maximum principal stress)
                chunk_max = np.max(s1, axis=0)
                self.max_over_time_s1 = np.maximum(self.max_over_time_s1, chunk_max)

                # Calculate the maximum principal stress (s1) and its time index for each node
                start_time = time.time()
                max_s1_per_node = np.max(s1, axis=1)
                time_indices = np.argmax(s1, axis=1)
                time_of_max_s1_per_node = time_values[time_indices]
                s1_max_memmap[start_idx:end_idx] = max_s1_per_node
                s1_max_time_memmap[start_idx:end_idx] = time_of_max_s1_per_node
                print(f"Elapsed time for max principal stress (s1) and time: {(time.time() - start_time):.3f} seconds")

            # Calculate potential damage index for all nodes in the chunk
            if calculate_damage:
                if calculate_damage:
                    start_time = time.time()

                    # Compute the signed von Mises stress using the existing von Mises results
                    signed_von_mises = self.compute_signed_von_mises_stress(sigma_vm, actual_sx, actual_sy, actual_sz)

                    # Use the signed von Mises stress for damage calculation
                    A = 1  # Material constant (example value)
                    m = -3  # Material exponent (example value)
                    potential_damages = compute_potential_damage_for_all_nodes(signed_von_mises, A,m)
                    potential_damage_memmap[start_idx:end_idx] = potential_damages
                    print(f"Elapsed time for damage index calculation: {(time.time() - start_time):.3f} seconds")

            # Free up some memory
            start_time = time.time()
            del actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz

            if calculate_von_mises:
                del sigma_vm
            if calculate_principal_stress:
                del s1, s2, s3

            gc.collect()
            print(f"Elapsed time for garbage collection: {(time.time() - start_time):.3f} seconds")

            current_available_memory = psutil.virtual_memory().available * self.RAM_PERCENT

            # Emit progress signal as a percentage of the total iterations
            current_iteration = (start_idx // chunk_size) + 1
            progress_percentage = (current_iteration / num_iterations) * 100
            self.progress_signal.emit(int(progress_percentage))
            QApplication.processEvents()  # Keep UI responsive

            print(f"Iteration completed for nodes {start_idx} to {end_idx}. "
                  f"Allocated system RAM: {current_available_memory / (1024 ** 3):.2f} GB\n")

        # Ensure all memmap files are flushed to disk
        if calculate_von_mises:
            von_mises_max_memmap.flush()
            von_mises_max_time_memmap.flush()
        if calculate_principal_stress:
            s1_max_memmap.flush()
            s1_max_time_memmap.flush()
        if calculate_damage:
            potential_damage_memmap.flush()

        # region Convert the .dat files to .csv
        if calculate_von_mises:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "max_von_mises_stress.dat"),
                                    os.path.join(self.output_directory, "max_von_mises_stress.csv"),
                                    "SVM_Max")
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "time_of_max_von_mises_stress.dat"),
                                    os.path.join(self.output_directory, "time_of_max_von_mises_stress.csv"),
                                    "Time_of_SVM_Max")
        if calculate_principal_stress:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "max_s1_stress.dat"),
                                    os.path.join(self.output_directory, "max_s1_stress.csv"),
                                    "S1_Max")
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "time_of_max_s1_stress.dat"),
                                    os.path.join(self.output_directory, "time_of_max_s1_stress.csv"),
                                    "Time_of_S1_Max")
        if calculate_damage:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "potential_damage_results.dat"),
                                    os.path.join(self.output_directory, "potential_damage_results.csv"),
                                    "Potential Damage (Damage Index)")
        # endregion

    def process_results_for_a_single_node(self, selected_node_idx, calculate_von_mises=False, calculate_principal_stress=False):
        """
        Process results for a single node and return the stress data for plotting.

        Parameters:
        - selected_node_idx: The index of the node to process.
        - calculate_von_mises: Boolean flag to compute Von Mises stress.
        - calculate_principal_stress: Boolean flag to compute Principal Stress.

        Returns:
        - time_points: Array of time points for the selected node.
        - stress_values: Array of stress values (either Von Mises or Principal Stress).
        """
        # Fetch stress data for the selected node
        actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
            self.compute_normal_stresses_for_a_single_node(selected_node_idx)

        selected_node_id = df_node_ids[selected_node_idx]

        if calculate_von_mises:
            # Compute Von Mises stress for the selected node
            sigma_vm = self.compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                     actual_sxz)
            print(f"Von Mises Stress calculated for Node {selected_node_id}\n")

            return np.arange(sigma_vm.shape[1]), sigma_vm[0, :]  # time_points, stress_values

        if calculate_principal_stress:
            # Compute Principal Stresses for the selected node
            s1, s2, s3 = self.compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                         actual_sxz)
            print(f"Principal Stresses calculated for Node {selected_node_id}\n")

            return np.arange(s1.shape[1]), s1[0, :]  # time_indices, stress_values

        return None, None

    def convert_dat_to_csv(self, node_ids, num_nodes, dat_filename, csv_filename, header):
        """Converts a .dat file to a .csv file with NodeID and, if available, X,Y,Z coordinates."""
        try:
            # Read the memmap file as a NumPy array
            data = np.memmap(dat_filename, dtype=RESULT_DTYPE, mode='r', shape=(num_nodes,))
            # Create a DataFrame for NodeID and the computed stress data
            df_out = pd.DataFrame({
                'NodeID': node_ids,
                header: data
            })
            # If node_coords is available, include the X, Y, Z coordinates
            if 'node_coords' in globals() and node_coords is not None:
                df_coords = pd.DataFrame(node_coords, columns=['X', 'Y', 'Z'])
                df_out = pd.concat([df_out, df_coords], axis=1)
            # Save to CSV
            df_out.to_csv(csv_filename, index=False)
            print(f"Successfully converted {dat_filename} to {csv_filename}.")
        except Exception as e:
            print(f"Error converting {dat_filename} to {csv_filename}: {e}")

class Logger(QObject):
    def __init__(self, text_edit, flush_interval=200):
        super().__init__()
        self.text_edit = text_edit
        self.terminal = sys.stdout
        self.log_buffer = ""  # Buffer for messages
        self.flush_interval = flush_interval  # in milliseconds

        # Set up a QTimer to flush the buffer periodically
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.flush_buffer)
        self.timer.start(self.flush_interval)

    def write(self, message):
        # Write to the original terminal
        self.terminal.write(message)
        # Append the message to the buffer
        self.log_buffer += message

    def flush_buffer(self):
        if self.log_buffer:
            # Append the buffered messages to the text edit in one update
            self.text_edit.moveCursor(QTextCursor.End)
            self.text_edit.insertPlainText(self.log_buffer)
            self.text_edit.moveCursor(QTextCursor.End)
            self.text_edit.ensureCursorVisible()
            self.log_buffer = ""

    def flush(self):
        self.flush_buffer()

class DisplayTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mesh = None  # Track current mesh for memory management
        self.current_actor = None  # Track the current actor for scalar range updates
        self.camera_state = None
        self.camera_widget = None
        self.last_hover_time = 0  # For frame rate throttling
        self.hover_annotation = None
        self.hover_observer = None  # Track hover callback observer
        self.anim_timer = None       # timer for animation
        self.time_text_actor = None
        self.current_anim_time = 0.0  # current time in the animation
        self.temp_solver = None
        self.init_ui()

    def init_ui(self):
        # Style settings
        button_style = """
            QPushButton {
                background-color: #e7f0fd;
                border: 1px solid #5b9bd5;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #cce4ff;
            }
        """

        # Create file selection components
        self.file_button = QPushButton('Load Visualization File')
        self.file_button.setStyleSheet(button_style)
        self.file_button.clicked.connect(self.load_file)

        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setStyleSheet("background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;")

        # Visualization controls
        self.point_size = QSpinBox()
        self.point_size.setRange(1, 100)
        self.point_size.setValue(40)
        self.point_size.setPrefix("Size: ")
        self.point_size.valueChanged.connect(self.update_point_size)

        # Scalar range controls
        self.scalar_min_spin = QDoubleSpinBox()
        self.scalar_max_spin = QDoubleSpinBox()
        self.scalar_min_spin.setPrefix("Min: ")
        self.scalar_max_spin.setPrefix("Max: ")
        self.scalar_min_spin.setDecimals(3)
        self.scalar_max_spin.setDecimals(3)
        self.scalar_min_spin.valueChanged.connect(self.update_scalar_range)
        self.scalar_max_spin.valueChanged.connect(self.update_scalar_range)
        self.scalar_min_spin.valueChanged.connect(lambda v: self.scalar_max_spin.setMinimum(v))
        self.scalar_max_spin.valueChanged.connect(lambda v: self.scalar_min_spin.setMaximum(v))

        # Create PyVista widget
        self.plotter = QtInteractor(parent=self)
        self.plotter.set_background('#FFFFFF')

        # Layout
        layout = QVBoxLayout()

        # File controls
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_path)

        # Visualization controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Node Point Size:"))
        control_layout.addWidget(self.point_size)
        control_layout.addWidget(QLabel("Scalar Range:"))
        control_layout.addWidget(self.scalar_min_spin)
        control_layout.addWidget(self.scalar_max_spin)
        control_layout.addStretch()

        # Contour Time-point controls layout:
        self.selected_time_checkbox = QCheckBox("Display results for a selected time point")
        self.selected_time_checkbox.setStyleSheet("margin: 10px;")
        self.selected_time_checkbox.toggled.connect(self.toggle_time_point_controls)
        # Initially hide the checkbox, and show it once the required files are loaded.
        self.selected_time_checkbox.setVisible(False)

        self.time_point_spinbox = QDoubleSpinBox()
        self.time_point_spinbox.setDecimals(3)
        # Range will be updated later from the modal coordinate file's time values.
        self.time_point_spinbox.setRange(0, 0)
        self.time_point_spinbox.setVisible(False)

        self.update_time_button = QPushButton("Update")
        self.update_time_button.setVisible(False)
        self.update_time_button.clicked.connect(self.update_time_point_results)

        self.save_time_button = QPushButton("Save Time Point")
        self.save_time_button.setVisible(False)
        self.save_time_button.clicked.connect(self.save_time_point_results)

        # Put the new widgets in a horizontal layout
        time_point_layout = QHBoxLayout()
        time_point_layout.addWidget(self.selected_time_checkbox)
        time_point_layout.addWidget(self.time_point_spinbox)
        time_point_layout.addWidget(self.update_time_button)
        time_point_layout.addWidget(self.save_time_button)

        # Animation Control Layout
        self.anim_layout = QHBoxLayout()
        # Add a spin box for animation frame interval (in milliseconds)
        self.anim_interval_spin = QSpinBox()
        self.anim_interval_spin.setRange(10, 10000)  # Allow between 10 ms and 10,000 ms delay
        self.anim_interval_spin.setValue(100)  # Default delay is 100 ms
        self.anim_interval_spin.setPrefix("Interval (ms): ")
        self.anim_layout.addWidget(self.anim_interval_spin)
        # Label and spinbox for animation start time
        self.anim_start_label = QLabel("Time Range:")
        self.anim_start_spin = QDoubleSpinBox()
        self.anim_start_spin.setPrefix("Start: ")
        self.anim_start_spin.setDecimals(3)
        self.anim_start_spin.setMinimum(0)
        self.anim_start_spin.setValue(0)
        # Label and spinbox for animation end time
        self.anim_end_spin = QDoubleSpinBox()
        self.anim_end_spin.setPrefix("End: ")
        self.anim_end_spin.setDecimals(3)
        self.anim_end_spin.setMinimum(0)
        self.anim_end_spin.setValue(1)
        # Ensure valid range by connecting valueChanged signals
        self.anim_start_spin.valueChanged.connect(self.update_anim_range_min)
        self.anim_end_spin.valueChanged.connect(self.update_anim_range_max)
        # Play and Stop buttons
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.play_button.clicked.connect(self.start_animation)
        self.pause_button.clicked.connect(self.pause_animation)
        self.stop_button.clicked.connect(self.reset_animation)
        # Add Time Step Mode ComboBox and Custom Step SpinBox
        self.time_step_mode_combo = QComboBox()
        self.time_step_mode_combo.addItems(["Custom Time Step", "Actual Data Time Steps"])
        self.custom_step_spin = QDoubleSpinBox()
        self.custom_step_spin.setDecimals(5)
        self.custom_step_spin.setRange(0.000001, 10)
        self.custom_step_spin.setValue(0.05)
        self.custom_step_spin.setPrefix("Step (seconds): ")

        # Connect the combo box's text change signal
        self.time_step_mode_combo.currentTextChanged.connect(self.update_step_spinbox_state)

        # Add widgets to the animation layout
        self.anim_layout.addWidget(self.time_step_mode_combo)
        self.anim_layout.addWidget(self.custom_step_spin)
        self.anim_layout.addWidget(self.anim_interval_spin)
        self.anim_layout.addWidget(self.anim_start_label)
        self.anim_layout.addWidget(self.anim_start_spin)
        self.anim_layout.addWidget(self.anim_end_spin)
        self.anim_layout.addWidget(self.play_button)
        self.anim_layout.addWidget(self.pause_button)
        self.anim_layout.addWidget(self.stop_button)
        self.anim_layout.addStretch()

        # Wrap the animation layout in a container widget and hide it initially
        self.anim_controls_widget = QWidget()
        self.anim_controls_widget.setLayout(self.anim_layout)
        self.anim_controls_widget.setVisible(False)  # initially hidden

        layout.addLayout(file_layout)
        layout.addLayout(control_layout)
        layout.addLayout(time_point_layout)
        layout.addWidget(self.anim_controls_widget)
        layout.addWidget(self.plotter)
        self.setLayout(layout)

    def toggle_time_point_controls(self, checked):
        self.time_point_spinbox.setVisible(checked)
        self.update_time_button.setVisible(checked)
        self.save_time_button.setVisible(checked)

    def update_time_point_range(self):
        """
        Check whether both the modal coordinate file (MCF) and the modal stress file have been loaded.
        If so, update the range of the time_point_spinbox (using the global time_values array),
        set its singleStep to the average sampling interval, and make the
        'Display results for a selected time point' checkbox visible.
        """
        if "modal_coord" in globals() and "modal_sx" in globals() and "time_values" in globals():
            min_time = np.min(time_values)
            max_time = np.max(time_values)
            self.time_point_spinbox.setRange(min_time, max_time)
            self.time_point_spinbox.setValue(min_time)
            # Compute the average sampling interval (dt)
            if len(time_values) > 1:
                avg_dt = np.mean(np.diff(time_values))
            else:
                avg_dt = 1.0  # Fallback if only one time value exists
            self.time_point_spinbox.setSingleStep(avg_dt)
            self.selected_time_checkbox.setVisible(True)

            # Update animation time range controls
            self.anim_start_spin.setRange(min_time, max_time)
            self.anim_end_spin.setRange(min_time, max_time)
            self.anim_start_spin.setValue(min_time)
            self.anim_end_spin.setValue(max_time)

            self.anim_controls_widget.setVisible(True)
        else:
            # Hide the controls if the required data is not available.
            self.selected_time_checkbox.setVisible(False)

    def update_time_point_results(self):
        """
        When the Update button is clicked:
          - Retrieve the selected time value from the spinbox.
          - Find the closest matching time index from the global time_values array.
          - Slice the modal coordinate matrix for that time point.
          - Create a temporary instance of MSUPSmartSolverTransient using the sliced modal coordinate tensor.
          - Depending on the main window’s selection:
                * If von Mises is selected, compute normal stresses and then von Mises stress.
                * If principal stress is selected, compute normal stresses and then principal stresses (selecting s₁).
          - Update the PyVista plot with the computed scalar field.
        """
        # Verify that required global variables exist.
        required_vars = ["modal_coord", "time_values", "modal_sx", "modal_sy", "modal_sz",
                         "modal_sxy", "modal_syz", "modal_sxz"]
        if not all(var in globals() for var in required_vars):
            QMessageBox.warning(self, "Missing Data", "Modal stress and coordinate files are not loaded.")
            return

        # Access the main window's solver tab (assumed to be stored in batch_solver_tab)
        main_tab = self.main_window.batch_solver_tab

        # Ensure that damage index and time history mode are not selected (invalid for time point mode)
        if main_tab.damage_index_checkbox.isChecked() or main_tab.time_history_checkbox.isChecked():
            QMessageBox.warning(self, "Invalid Selection",
                                "Damage Index and Time History Mode are not valid for single time point visualization.")
            return

        # Determine which stress type is selected.
        compute_von = main_tab.von_mises_checkbox.isChecked()
        compute_principal = main_tab.principal_stress_checkbox.isChecked()

        if not (compute_von or compute_principal):
            QMessageBox.warning(self, "No Selection",
                                "No output is selected. Please select a valid output type to compute the results.")
            return

        # Get the selected time value and find the closest index in the global time_values array.
        selected_time = self.time_point_spinbox.value()
        time_index = np.argmin(np.abs(time_values - selected_time))

        # Slice the modal coordinate matrix for the chosen time point.
        # Assuming modal_coord shape is [num_modes, num_time_points], we extract a column vector.
        selected_modal_coord = modal_coord[:, time_index:time_index + 1]

        # Create a temporary solver instance using the sliced modal coordinate matrix.
        include_steady = self.main_window.batch_solver_tab.steady_state_checkbox.isChecked()
        try:
            if include_steady and "steady_sx" in globals() and steady_sx is not None and "steady_node_ids" in globals():
                temp_solver = MSUPSmartSolverTransient(
                    modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                    selected_modal_coord,
                    steady_sx, steady_sy, steady_sz, steady_sxy, steady_syz, steady_sxz,
                    steady_node_ids, modal_node_ids=df_node_ids
                )
            else:
                temp_solver = MSUPSmartSolverTransient(
                    modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                    selected_modal_coord,
                    modal_node_ids=df_node_ids
                )
        except Exception as e:
            QMessageBox.warning(self, "Solver Error", f"Failed to create solver instance: {e}")
            return

        # Determine the number of nodes (assumed from the shape of the modal stress arrays).
        num_nodes = modal_sx.shape[0]

        # Compute normal stresses for all nodes using the temporary solver.
        actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
            temp_solver.compute_normal_stresses(0, num_nodes)

        # Based on the selected stress type, compute the scalar field.
        if compute_von:
            scalar_field = temp_solver.compute_von_mises_stress(actual_sx, actual_sy, actual_sz,
                                                                actual_sxy, actual_syz, actual_sxz)
            field_name = "SVM"
            display_name = "SVM (MPa)"
        elif compute_principal:
            s1, s2, s3 = temp_solver.compute_principal_stresses(actual_sx, actual_sy, actual_sz,
                                                                actual_sxy, actual_syz, actual_sxz)
            scalar_field = s1  # Choose the maximum principal stress
            field_name = "S1"
            display_name = "S1 (MPa)"
        else:
            QMessageBox.warning(self, "Selection Error", "No valid stress type selected.")
            return

        # Calculate min and max of the scalar field
        data_min = np.min(scalar_field)
        data_max = np.max(scalar_field)

        # Update the scalar range spin boxes
        self.scalar_min_spin.blockSignals(True)
        self.scalar_max_spin.blockSignals(True)
        self.scalar_min_spin.setRange(data_min, data_max)
        self.scalar_max_spin.setRange(data_min, 1e30)
        self.scalar_min_spin.setValue(data_min)
        self.scalar_max_spin.setValue(data_max)
        self.scalar_min_spin.blockSignals(False)
        self.scalar_max_spin.blockSignals(False)

        # Update the visualization.
        if "node_coords" in globals() and node_coords is not None:
            mesh = pv.PolyData(node_coords)

            # Add NodeID array if available
            if 'df_node_ids' in globals() and df_node_ids is not None:
                mesh["NodeID"] = df_node_ids.astype(int)

            mesh[field_name] = scalar_field  # Use field-specific name
            mesh.set_active_scalars(field_name)  # Set active scalars

            self.current_mesh = mesh
            self.data_column = display_name  # For visualization labels

            # Remove existing time text actor if present
            if hasattr(self, 'time_text_actor') and self.time_text_actor is not None:
                self.plotter.remove_actor(self.time_text_actor)
                self.time_text_actor = None

            self.plotter.reset_camera()
            self.plotter.render()
            self.update_visualization()

        else:
            QMessageBox.warning(self, "Missing Data", "Node coordinates are not available.")

    def save_time_point_results(self):
        """
        When the Save Time Point button is clicked, save the currently displayed results (node coordinates
        and the computed stress field) into a CSV file so that they can later be reloaded via the Load Visualization File button.
        """
        if self.current_mesh is None:
            QMessageBox.warning(self, "No Data", "No visualization data to save.")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Save Time Point Results", "", "CSV Files (*.csv)")
        if file_name:
            if "Stress" in self.current_mesh.array_names:
                stress_data = self.current_mesh["Stress"]
                coords = self.current_mesh.points
                df_out = pd.DataFrame(coords, columns=["X", "Y", "Z"])
                df_out["Stress"] = stress_data
                df_out.to_csv(file_name, index=False)
                QMessageBox.information(self, "Saved", "Time point results saved successfully.")
            else:
                QMessageBox.warning(self, "Missing Data", "The current mesh does not contain a 'Stress' field.")

    def update_anim_range_min(self, value):
        # Ensure that the animation end spin box cannot be set to a value less than the start
        self.anim_end_spin.setMinimum(value)

    def update_anim_range_max(self, value):
        # Ensure that the animation start spin box cannot exceed the end value
        self.anim_start_spin.setMaximum(value)

    def start_animation(self):
        """Start animating through the time range."""
        if self.current_mesh is None:
            QMessageBox.warning(self, "No Data", "Please load the mesh before animating by initializing and solving at "
                                                 "least a single time point")
            return
        if hasattr(self, 'time_text_actor'):
            self.plotter.remove_actor(self.time_text_actor)
            self.time_text_actor = None  # clear the reference
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        # Initialize current animation time with the start value
        self.current_anim_time = self.anim_start_spin.value()
        # Create a QTimer to update the animation frame periodically
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.animate_frame)
        self.anim_timer.start(self.anim_interval_spin.value())  # Update every 100 ms (adjust as needed)

    def pause_animation(self):
        """Pause the animation (resumes from the current time when Play is clicked)."""
        if self.anim_timer is not None:
            self.anim_timer.stop()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def reset_animation(self):
        """Stop the animation and reset the time to the start value."""
        if self.anim_timer is not None:
            self.anim_timer.stop()
        self.current_anim_time = self.anim_start_spin.value()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def animate_frame(self):
        # profiler = cProfile.Profile()
        # profiler.enable()

        global time_values, modal_coord, modal_sx
        try:
            # Increment the current animation time
            start_time = self.anim_start_spin.value()
            end_time = self.anim_end_spin.value()

            # Update current_anim_time based on user selection ---
            if self.time_step_mode_combo.currentText() == "Custom Time Step":
                step = self.custom_step_spin.value()
                self.current_anim_time += step
            else:
                idx = np.argmin(np.abs(time_values - self.current_anim_time))
                if idx < len(time_values) - 1:
                    self.current_anim_time = time_values[idx + 1]
                else:
                    self.current_anim_time = start_time

            if self.current_anim_time > end_time:
                self.current_anim_time = start_time

            # Update the scalar field on your mesh (replace with your actual computation)
            new_scalars = self.get_scalar_field_for_time(self.current_anim_time)
            self.current_mesh[self.data_column] = new_scalars

            # Update node positions if modal deformations are loaded
            if 'modal_ux' in globals() and modal_ux is not None:
                # Find the time index corresponding to the current animation time
                time_index = np.argmin(np.abs(time_values - self.current_anim_time))
                # Slice the modal coordinate for the current time point
                selected_modal_coord = modal_coord[:, time_index:time_index + 1]
                # Create a temporary solver that now also receives the modal deformations tuple
                # (Assuming that the other required globals like modal_sx, etc., are available)
                try:
                    temp_solver = MSUPSmartSolverTransient(
                        modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                        selected_modal_coord,
                        modal_node_ids=df_node_ids,
                        modal_deformations=(modal_ux, modal_uy, modal_uz)
                    )
                    num_nodes = modal_sx.shape[0]
                    # Compute deformations for all nodes at this time
                    deformations = temp_solver.compute_deformations(0, num_nodes)
                    if deformations is not None:
                        ux, uy, uz = deformations  # Each with shape (n_nodes, 1)
                        # Update mesh node positions: add computed displacements to original coordinates
                        # (Assuming global variable node_coords holds the original positions)
                        new_coords = node_coords + np.hstack((ux, uy, uz))
                        self.current_mesh.points = new_coords
                except Exception as e:
                    print(f"Deformation update error: {e}")

            # Update the scalar bar range:
            fixed_min = self.scalar_min_spin.value()
            fixed_max = self.scalar_max_spin.value()
            current_data_min = np.min(new_scalars)
            current_data_max = np.max(new_scalars)
            if abs(fixed_min - current_data_min) < 1e-6 and abs(fixed_max - current_data_max) < 1e-6:
                self.current_actor.mapper.SetScalarRange(current_data_min, current_data_max)
            else:
                self.current_actor.mapper.SetScalarRange(fixed_min, fixed_max)

            # Update (or create) the free-floating text actor
            current_text = f"Time: {self.current_anim_time:.5f}"

            # Remove existing time text actor if present
            if hasattr(self, 'time_text_actor') and self.time_text_actor is not None:
                self.plotter.remove_actor(self.time_text_actor)
                self.time_text_actor = None

            # Create a free-floating text actor with normalized viewport coordinates.
            # For example, (0.8, 0.9) positions it near the upper-right.
            self.time_text_actor = self.plotter.add_text(current_text,
                                                         position=(0.8, 0.9),
                                                         viewport=True,
                                                         font_size=10)

            self.plotter.render()

        except Exception as e:
            QMessageBox.critical(self, "Animation Error", f"Failed to animate the results: {str(e)}")

        # profiler.disable()
        # profiler.dump_stats("animation.prof")

    def get_scalar_field_for_time(self, time_val):
        """
        Computes the actual stress results (von Mises or principal stress) at a given time_val
        for all nodes, returning a 1D NumPy array of length n_points.

        This replaces the dummy sinusoidal code with real computations:
          1) Identify the user’s selection (von Mises or principal).
          2) Find the nearest time index from global time_values.
          3) Slice modal_coord to a single column for that time.
          4) Create a temporary solver to compute normal stresses, then compute the final result.
        """
        # 1) Check which output is selected in the main GUI:
        main_tab = self.main_window.batch_solver_tab
        compute_von = main_tab.von_mises_checkbox.isChecked()
        compute_principal = main_tab.principal_stress_checkbox.isChecked()
        # If neither is selected, return zeros (or you could raise an error).
        if not (compute_von or compute_principal):
            return np.zeros(self.current_mesh.n_points, dtype=np.float32)

        # 2) Ensure that global data is loaded:
        required_vars = ["modal_coord", "time_values", "modal_sx", "modal_sy", "modal_sz",
                         "modal_sxy", "modal_syz", "modal_sxz", "df_node_ids"]
        if not all(var in globals() for var in required_vars):
            # Missing data => return zeros or raise an error
            return np.zeros(self.current_mesh.n_points, dtype=np.float32)

        # 3) Find the closest time index to time_val:
        global time_values
        time_index = np.argmin(np.abs(time_values - time_val))
        # Slice out a single column from modal_coord:
        selected_modal_coord = modal_coord[:, time_index: time_index + 1]

        # 4) Create a small “temporary” solver for that single time slice:
        try:
            # Check if steady-state stress is included
            global steady_sx, steady_sy, steady_sz, steady_sxy, steady_syz, steady_sxz, steady_node_ids
            if (
                    "steady_sx" in globals() and steady_sx is not None
                    and "steady_node_ids" in globals() and steady_node_ids is not None
            ):
                temp_solver = MSUPSmartSolverTransient(
                    modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                    selected_modal_coord,
                    steady_sx, steady_sy, steady_sz, steady_sxy, steady_syz, steady_sxz,
                    steady_node_ids, modal_node_ids=df_node_ids
                )
            else:
                temp_solver = MSUPSmartSolverTransient(
                    modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                    selected_modal_coord,
                    modal_node_ids=df_node_ids
                )
        except Exception as e:
            print(f"[Animation] Error creating temp solver: {e}")
            return np.zeros(self.current_mesh.n_points, dtype=np.float32)

        # 5) Compute the normal stresses for all nodes:
        num_nodes = modal_sx.shape[0]
        actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
            temp_solver.compute_normal_stresses(0, num_nodes)

        # 6) Depending on the selection, compute von Mises or principal stress:
        if compute_von:
            sigma_vm = temp_solver.compute_von_mises_stress(
                actual_sx, actual_sy, actual_sz,
                actual_sxy, actual_syz, actual_sxz
            )
            # sigma_vm has shape (n_nodes, 1) => flatten to 1D
            return sigma_vm[:, 0]

        elif compute_principal:
            s1, s2, s3 = temp_solver.compute_principal_stresses(
                actual_sx, actual_sy, actual_sz,
                actual_sxy, actual_syz, actual_sxz
            )
            # s1 shape is (n_nodes, 1) => flatten to 1D
            return s1[:, 0]

        # If we get here somehow, return zeros
        return np.zeros(self.current_mesh.n_points, dtype=np.float32)

    def load_file(self):
        """Load and visualize new data file"""
        try:
            # Clear previous data
            self.clear_visualization()

            file_name, _ = QFileDialog.getOpenFileName(
                self, 'Open Visualization File', '', 'CSV Files (*.csv)'
            )
            if not file_name:
                return

            self.file_path.setText(file_name)
            self.visualize_data(file_name)

        except Exception as e:
            QMessageBox.critical(self, "Loading Error", f"Failed to load file: {str(e)}")

    def visualize_data(self, filename):
        """Handle data visualization with validation"""
        try:
            # Load and validate data
            df = pd.read_csv(filename)

            # Recommendation 1: Data validation
            if df.empty:
                raise ValueError("Selected file is empty")

            if len(df) > 1e6:
                QMessageBox.warning(
                    self,
                    "Performance Warning",
                    "Large dataset (>{:,} points) may affect performance".format(len(df))
                )

            # Validate coordinates
            required_cols = {'X', 'Y', 'Z'}
            if not required_cols.issubset(df.columns.str.strip()):
                raise ValueError("CSV file must contain x, y, z columns (case insensitive)")

            # Get coordinates and values
            coords = df[['X', 'Y', 'Z']].values
            self.data_column = df.columns[1]

            # Create and store mesh
            self.current_mesh = pv.PolyData(coords)
            if self.data_column:
                self.current_mesh[self.data_column] = df[self.data_column].values
                # After creating mesh:
                self.current_mesh.set_active_scalars(self.data_column)

                # Initialize scalar range spin boxes
                data_min = np.min(self.current_mesh[self.data_column])
                data_max = np.max(self.current_mesh[self.data_column])
                self.scalar_min_spin.blockSignals(True)
                self.scalar_max_spin.blockSignals(True)
                self.scalar_min_spin.setRange(data_min, data_max)
                self.scalar_min_spin.setValue(data_min)
                self.scalar_max_spin.setRange(data_min, 1e30)
                self.scalar_max_spin.setValue(data_max)
                self.scalar_min_spin.blockSignals(False)
                self.scalar_max_spin.blockSignals(False)

            # Store NodeID if available
            if 'NodeID' in df.columns:
                self.current_mesh['NodeID'] = df['NodeID'].values

            # Initial visualization
            if not self.camera_widget:
                self.camera_widget = self.plotter.add_camera_orientation_widget()
                self.camera_widget.EnabledOn()
            #self.plotter.enable_point_picking(point_size=self.point_size.value())
            self.update_visualization()
            self.plotter.reset_camera()
            self.plotter.camera.zoom(1)

        except Exception as e:
            self.clear_visualization()
            QMessageBox.critical(self, "Visualization Error", f"Error visualizing data: {str(e)}")

    def update_scalar_range(self):
        """Update the scalar range of the current visualization based on spin box values."""
        if self.current_actor is None:
            return
        min_val = self.scalar_min_spin.value()
        max_val = self.scalar_max_spin.value()
        self.current_actor.mapper.SetScalarRange(min_val, max_val)
        self.plotter.render()

    def update_step_spinbox_state(self, text):
        """Enable/disable the step spinbox based on the selected time step mode."""
        if text == "Actual Data Time Steps":
            self.custom_step_spin.setEnabled(False)
        else:
            self.custom_step_spin.setEnabled(True)

    def update_visualization(self):
        """Update plotter with current settings"""
        if not self.current_mesh:
            return

        # Store current camera state before clearing
        self.camera_state = {
            'position': self.plotter.camera.position,
            'focal_point': self.plotter.camera.focal_point,
            'view_up': self.plotter.camera.up,
            'view_angle': self.plotter.camera.view_angle
        }

        self.plotter.clear()
        self.data_column = self.current_mesh.array_names[0] if self.current_mesh.array_names else None

        self.current_actor = self.plotter.add_mesh(
            self.current_mesh,
            scalars=self.data_column,
            cmap='jet',  # Changed colormap to 'jet' to mimic ANSYS Mechanical
            point_size=self.point_size.value(),
            render_points_as_spheres=True,
            below_color='gray',
            above_color='magenta',
            scalar_bar_args={
                'title': self.data_column,
                'fmt': '%.4f',
                'position_x': 0.04,  # Left edge (5% from left)
                'position_y': 0.35,  # Vertical position (35% from bottom)
                'width': 0.05,  # Width of the scalar bar (5% of window)
                'height': 0.5,  # Height of the scalar bar (50% of window)
                'vertical': True,  # Force vertical orientation
                'title_font_size': 10,
                'label_font_size': 8,
                'shadow': True,  # Optional: Add shadow for readability
                'n_labels': 10  # Number of labels to display
            }
        )
        self.setup_hover_annotation()

        # Restore camera state if available
        if self.camera_state:
            self.plotter.camera.position = self.camera_state['position']
            self.plotter.camera.focal_point = self.camera_state['focal_point']
            self.plotter.camera.up = self.camera_state['view_up']
            self.plotter.camera.view_angle = self.camera_state['view_angle']

        # self.plotter.add_axes(
        #     line_width=2,  # Reduced complexity
        #     color='black',  # Simpler color
        #     xlabel='X',
        #     ylabel='Y',
        #     zlabel='Z',
        #     interactive=False  # Disable dynamic updates
        # )
        #self.plotter.reset_camera()

    def setup_hover_annotation(self):
        """Set up hover callback to display node ID and value"""
        if not self.current_mesh or 'NodeID' not in self.current_mesh.array_names:
            return

        # Clean up previous hover elements
        self.clear_hover_elements()

        # Create new annotation
        self.hover_annotation = self.plotter.add_text(
            "", position='upper_right', font_size=8,
            color='black', name='hover_annotation'
        )

        # Create picker and callback with throttling
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.01)

        def hover_callback(obj, event):
            now = time.time()
            if (now - self.last_hover_time) < 0.033:  # 30 FPS throttle
                return

            iren = obj
            pos = iren.GetEventPosition()
            picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
            point_id = picker.GetPointId()

            if point_id != -1 and point_id < self.current_mesh.n_points:
                node_id = self.current_mesh['NodeID'][point_id]
                value = self.current_mesh[self.data_column][point_id]
                self.hover_annotation.SetText(2, f"Node ID: {node_id}\n{self.data_column}: {value:.2f}")
            else:
                self.hover_annotation.SetText(2, "")

            iren.GetRenderWindow().Render()
            self.last_hover_time = now

        # Add and track new observer
        self.hover_observer = self.plotter.iren.add_observer('MouseMoveEvent', hover_callback)

    def clear_hover_elements(self):
        """Dedicated hover element cleanup"""
        if self.hover_annotation:
            self.plotter.remove_actor(self.hover_annotation)
            self.hover_annotation = None

        if self.hover_observer:
            self.plotter.iren.remove_observer(self.hover_observer)
            self.hover_observer = None

    def update_point_size(self):
        """Handle dynamic point size updates"""
        if self.current_mesh:
            self.clear_hover_elements()  # Clean up before update
            self.update_visualization()
            self.setup_hover_annotation()  # Reinitialize hover

    def clear_visualization(self):
        """Properly clear existing visualization"""
        self.clear_hover_elements()

        if self.camera_widget:
            self.camera_widget.EnabledOff()
            self.camera_widget = None

        self.plotter.clear()
        if self.current_mesh:
            self.current_mesh.clear_data()
            self.current_mesh = None

        self.current_actor = None
        self.scalar_min_spin.clear()
        self.scalar_max_spin.clear()

    def __del__(self):
        """Ensure proper cleanup"""
        self.clear_visualization()

class MSUPSmartSolverGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

        # Ensure project_directory exists
        self.project_directory = None  # Default to None if not set

        # Initialize solver attribute
        self.solver = None

        # Track whether the Plot(Modal Coordinates) tab is currently maximized
        self.modal_plot_window = None

        # Create a keyboard shortcut for "M"
        self.shortcut_m = QShortcut(QKeySequence("M"), self)
        self.shortcut_m.activated.connect(self.toggle_modal_coord_fullscreen_plot)

        # Set up a single logger instance
        self.logger = Logger(self.console_textbox)
        sys.stdout = self.logger  # Redirect stdout to the logger

        # Enable drag-and-drop for file selection buttons and text fields
        self.setAcceptDrops(True)

    def init_ui(self):
        # Set window background color
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(230, 230, 230))  # Light gray background
        self.setPalette(palette)

        # Common stylesheets
        button_style = """
            QPushButton {
                background-color: #e7f0fd;
                border: 1px solid #5b9bd5;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #cce4ff;
            }
        """

        group_box_style = """
            QGroupBox {
                border: 1px solid #5b9bd5;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
        """

        tab_style = """
            QTabBar::tab {
                background-color: #d6e4f5;  /* Pale blue background for inactive tabs */
                border: 1px solid #5b9bd5;   /* Default border for tabs */
                padding: 3px;
                border-top-left-radius: 5px;  /* Upper left corner rounded */
                border-top-right-radius: 5px; /* Upper right corner rounded */
                margin: 2px;
            }
            QTabBar::tab:hover {
                background-color: #cce4ff;  /* Background color when hovering over tabs */
            }
            QTabBar::tab:selected {
                background-color: #e7f0fd;  /* Active tab has your blue theme color */
                border: 2px solid #5b9bd5;  /* Thicker border for the active tab */
                color: #000000;  /* Active tab text color */
            }
            QTabBar::tab:!selected {
                background-color: #d6e4f5;  /* Paler blue for unselected tabs */
                color: #808080;  /* Gray text for inactive tabs */
                margin-top: 3px;  /* Make the unselected tabs slightly smaller */
            }
        """

        # Create UI elements
        # Modal Coordinate File Section
        self.coord_file_button = QPushButton('Read Modal Coordinate File (.mcf)')
        self.coord_file_button.setStyleSheet(button_style)
        self.coord_file_button.setFont(QFont('Arial', 8))
        self.coord_file_path = QLineEdit()
        self.coord_file_path.setReadOnly(True)
        self.coord_file_path.setStyleSheet("background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;")

        # Modal Stress File Section
        self.stress_file_button = QPushButton('Read Modal Stress File (.csv)')
        self.stress_file_button.setStyleSheet(button_style)
        self.stress_file_button.setFont(QFont('Arial', 8))
        self.stress_file_path = QLineEdit()
        self.stress_file_path.setReadOnly(True)
        self.stress_file_path.setStyleSheet("background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;")

        # Checkbox for "Include Steady-State Stress Field"
        self.steady_state_checkbox = QCheckBox("Include Steady-State Stress Field")
        self.steady_state_checkbox.setStyleSheet("margin: 10px 0;")
        self.steady_state_checkbox.toggled.connect(self.toggle_steady_state_stress_inputs)

        # Button and text box for steady-state stress file
        self.steady_state_file_button = QPushButton('Read Full Stress Tensor File (.txt)')
        self.steady_state_file_button.setStyleSheet(button_style)
        self.steady_state_file_button.setFont(QFont('Arial', 8))
        self.steady_state_file_button.clicked.connect(self.select_steady_state_file)
        self.steady_state_file_button.setVisible(False)  # Initially hidden

        self.steady_state_file_path = QLineEdit()
        self.steady_state_file_path.setReadOnly(True)
        self.steady_state_file_path.setStyleSheet(
            "background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;")
        self.steady_state_file_path.setVisible(False)  # Initially hidden

        # Checkbox for including deformations
        self.deformations_checkbox = QCheckBox("Include Deformations for Animation")
        self.deformations_checkbox.setStyleSheet("margin: 10px 0;")
        self.deformations_checkbox.toggled.connect(self.toggle_deformations_inputs)

        self.deformations_file_button = QPushButton('Read Modal Deformations File (.csv)')
        self.deformations_file_button.setStyleSheet("/* use your button style */")
        self.deformations_file_button.setFont(QFont('Arial', 8))
        self.deformations_file_path = QLineEdit()
        self.deformations_file_button.setStyleSheet(button_style)
        self.deformations_file_path.setReadOnly(True)
        self.deformations_file_path.setStyleSheet("background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;")
        self.deformations_file_button.clicked.connect(self.select_deformations_file)

        # Initially hide the deformations file controls until the checkbox is checked.
        self.deformations_file_button.setVisible(False)
        self.deformations_file_path.setVisible(False)

        # Checkbox for Time History Mode (Single Node)
        self.time_history_checkbox = QCheckBox('Time History Mode (Single Node)')
        self.time_history_checkbox.setStyleSheet("margin: 10px 0;")
        self.time_history_checkbox.toggled.connect(self.toggle_single_node_solution_group)

        # Checkbox for Calculate Principal Stress
        self.principal_stress_checkbox = QCheckBox('Max Principal Stress')
        self.principal_stress_checkbox.setStyleSheet("margin: 10px 0;")
        self.principal_stress_checkbox.toggled.connect(self.update_single_node_plot_based_on_checkboxes)

        # Checkbox for Calculate Von-Mises Stress
        self.von_mises_checkbox = QCheckBox('Von-Mises Stress')
        self.von_mises_checkbox.setStyleSheet("margin: 10px 0;")
        self.von_mises_checkbox.toggled.connect(self.update_single_node_plot_based_on_checkboxes)

        # Checkbox for Calculate Damage Index
        self.damage_index_checkbox = QCheckBox('Damage Index')
        self.damage_index_checkbox.setStyleSheet("margin: 10px 0;")

        # Connect checkbox signal to the method for controlling the visibility of the damage index checkbox
        self.von_mises_checkbox.toggled.connect(self.toggle_damage_index_checkbox_visibility)

        # LineEdit for Node ID input
        self.node_line_edit = QLineEdit()
        self.node_line_edit.setPlaceholderText("Enter Node ID")
        self.node_line_edit.setStyleSheet(button_style)
        self.node_line_edit.setMaximumWidth(150)
        self.node_line_edit.setMinimumWidth(100)
        self.node_line_edit.returnPressed.connect(self.on_node_entered)

        # Solve Button
        self.solve_button = QPushButton('Solve')
        self.solve_button.setStyleSheet(button_style)
        self.solve_button.setFont(QFont('Arial', 8, QFont.Bold))
        self.solve_button.clicked.connect(self.solve)

        # Read-only Output Console
        self.console_textbox = QTextEdit()
        self.console_textbox.setReadOnly(True)
        self.console_textbox.setStyleSheet("background-color: #ffffff; border: 1px solid #5b9bd5")
        self.console_textbox.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.console_textbox.setText('Console Output:\n')

        # Set monospaced font for log terminal
        terminal_font = QFont("Consolas", 7)
        terminal_font.setStyleHint(QFont.Monospace)  # For a more console-like textbox
        self.console_textbox.setFont(terminal_font)

        # Create a QTabWidget for the Log Terminal etc.
        self.show_output_tab_widget = QTabWidget()
        self.show_output_tab_widget.setStyleSheet(tab_style)
        self.show_output_tab_widget.addTab(self.console_textbox, "Console")

        # Initialize matplotlib plot
        self.plot_single_node_tab = MatplotlibWidget()
        # Ensure the plot widget expands to fill the tab
        self.plot_single_node_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Placeholder matplotlib plot
        self.update_single_node_plot()

        # Add the plot tab to the tab widget, but hide it initially
        self.show_output_tab_widget.addTab(self.plot_single_node_tab, "Plot (Time History)")
        # Make it initially hidden
        self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_single_node_tab), False)

        # Initialize modal coordinates plot
        self.plot_modal_coords_tab = PlotlyWidget()
        self.show_output_tab_widget.addTab(self.plot_modal_coords_tab, "Plot (Modal Coordinates)")
        self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_modal_coords_tab),
                                                  False)

        # Create Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("border: 1px solid #5b9bd5; padding: 10px; background-color: #ffffff;")
        self.progress_bar.setValue(0)  # Start with 0% progress
        self.progress_bar.setAlignment(Qt.AlignCenter)  # Center the progress bar text
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)

        # File selection buttons
        self.coord_file_button.clicked.connect(self.select_coord_file)
        self.stress_file_button.clicked.connect(self.select_stress_file)

        # Layouts
        main_layout = QVBoxLayout()

        # Group box for file selection
        file_group = QGroupBox("Input Files")
        file_group.setStyleSheet(group_box_style)
        file_layout = QGridLayout()

        file_layout.addWidget(self.coord_file_button, 0, 0)
        file_layout.addWidget(self.coord_file_path, 0, 1)
        file_layout.addWidget(self.stress_file_button, 1, 0)
        file_layout.addWidget(self.stress_file_path, 1, 1)
        file_layout.addWidget(self.steady_state_checkbox, 2, 0, 1, 2)
        file_layout.addWidget(self.steady_state_file_button, 3, 0)
        file_layout.addWidget(self.steady_state_file_path, 3, 1)
        file_layout.addWidget(self.deformations_checkbox, 4, 0, 1, 2)
        file_layout.addWidget(self.deformations_file_button, 5, 0)
        file_layout.addWidget(self.deformations_file_path, 5, 1)

        file_group.setLayout(file_layout)

        # Group box for outputs requested
        self.output_group = QGroupBox("Outputs")
        self.output_group.setStyleSheet(group_box_style)
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.time_history_checkbox)
        output_layout.addWidget(self.principal_stress_checkbox)
        output_layout.addWidget(self.von_mises_checkbox)
        output_layout.addWidget(self.damage_index_checkbox)
        self.output_group.setLayout(output_layout)

        # Group box for Single Node Solution (Node ID selection)
        self.single_node_group = QGroupBox("Scoping")
        self.single_node_group.setStyleSheet(group_box_style)
        self.single_node_label = QLabel("Select a node:")
        self.single_node_label.setFont(QFont('Arial', 8))
        single_node_layout = QHBoxLayout()
        single_node_layout.addWidget(self.single_node_label)
        single_node_layout.addWidget(self.node_line_edit)
        self.single_node_group.setVisible(False)
        self.single_node_group.setMaximumWidth(250)
        self.single_node_group.setLayout(single_node_layout)

        # Horizontal layout to place Outputs and Single Node Expansion side by side
        hbox = QHBoxLayout()
        hbox.addWidget(self.output_group)  # Outputs on the left
        hbox.addWidget(self.single_node_group)

        # Adding elements to main layout
        main_layout.addWidget(file_group)
        main_layout.addLayout(hbox)
        main_layout.addWidget(self.solve_button)
        main_layout.addWidget(self.show_output_tab_widget)  # Add the tab widget for the log terminal
        main_layout.addWidget(self.progress_bar)

        self.setLayout(main_layout)

        # Initially hide the "Calculate Damage Index" checkbox if "Calculate Von-Mises" is not checked
        self.toggle_damage_index_checkbox_visibility()

    def toggle_steady_state_stress_inputs(self):
        is_checked = self.steady_state_checkbox.isChecked()
        self.steady_state_file_button.setVisible(is_checked)
        self.steady_state_file_path.setVisible(is_checked)

        # Clear the file path text if the checkbox is unchecked
        if not is_checked:
            self.steady_state_file_path.clear()

    def toggle_deformations_inputs(self):
        if self.deformations_checkbox.isChecked():
            self.deformations_file_button.setVisible(True)
            self.deformations_file_path.setVisible(True)
        else:
            self.deformations_file_button.setVisible(False)
            self.deformations_file_path.setVisible(False)

    def toggle_damage_index_checkbox_visibility(self):
        if self.von_mises_checkbox.isChecked():
            self.damage_index_checkbox.setVisible(True)
        else:
            self.damage_index_checkbox.setVisible(False)

    def toggle_single_node_solution_group(self):
        try:
            if self.time_history_checkbox.isChecked():
                # Enable mutual exclusivity for stress checkboxes in Time History Mode
                self.principal_stress_checkbox.toggled.connect(self.on_principal_stress_toggled)
                self.von_mises_checkbox.toggled.connect(self.on_von_mises_toggled)

                # Show single node group and plot tab
                self.single_node_group.setVisible(True)
                self.show_output_tab_widget.setTabVisible(
                    self.show_output_tab_widget.indexOf(self.plot_single_node_tab), True)
            else:
                # Remove mutual exclusivity when Time History Mode is off
                self.principal_stress_checkbox.toggled.disconnect(self.on_principal_stress_toggled)
                self.von_mises_checkbox.toggled.disconnect(self.on_von_mises_toggled)

                # Hide single node group and plot tab
                self.single_node_group.setVisible(False)
                self.show_output_tab_widget.setTabVisible(
                    self.show_output_tab_widget.indexOf(self.plot_single_node_tab), False)
        except Exception as e:
            print(f"Error in toggling single node group visibility: {e}")

    def on_principal_stress_toggled(self):
        """Disable Von-Mises checkbox if Principal Stress checkbox is activated in Time History Mode."""
        if self.time_history_checkbox.isChecked() and self.principal_stress_checkbox.isChecked():
            self.von_mises_checkbox.setChecked(False)

    def on_von_mises_toggled(self):
        """Disable Principal Stress checkbox if Von-Mises checkbox is activated in Time History Mode."""
        if self.time_history_checkbox.isChecked() and self.von_mises_checkbox.isChecked():
            self.principal_stress_checkbox.setChecked(False)

    def update_single_node_plot(self):
        """Updates the placeholder plot inside the MatplotlibWidget."""
        x = np.linspace(0, 10, 100)
        y = np.zeros(100)
        self.plot_single_node_tab.update_plot(x, y)

    def update_single_node_plot_based_on_checkboxes(self):
        """Update the plot based on the state of the 'Principal Stress' and 'Von Mises Stress' checkboxes."""
        try:
            # Retrieve the checkbox states
            is_principal_stress = self.principal_stress_checkbox.isChecked()
            is_von_mises = self.von_mises_checkbox.isChecked()

            # Dummy data for the plot (replace this with actual data when available)
            x_data = [1, 2, 3, 4, 5]  # Time or some other X-axis data
            y_data = [0, 0, 0, 0, 0]  # Dummy Y-axis data

            # Update the plot with the current checkbox states
            self.plot_single_node_tab.update_plot(x_data, y_data, None, is_principal_stress, is_von_mises)
        except Exception as e:
            print(f"Error updating plot based on checkbox states: {e}")

    def select_coord_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Modal Coordinate File', '', 'Coordinate Files (*.mcf)')
        if file_name:
            self.coord_file_path.setText(file_name)
            self.process_modal_coordinate_file(file_name)

    def process_modal_coordinate_file(self, filename):
        try:
            # First, create an unwrapped version of the file.
            base, ext = os.path.splitext(filename)
            unwrapped_filename = base + "_unwrapped" + ext
            unwrap_mcf_file(filename, unwrapped_filename)

            # Now read the unwrapped file.
            # Find the line that contains 'Time' to know where data starts.
            with open(unwrapped_filename, 'r') as file:
                for i, line in enumerate(file):
                    if 'Time' in line:
                        start_index = i
                        break

            # Read the data starting from the identified start line.
            df = pd.read_csv(unwrapped_filename, sep='\s+', skiprows=start_index + 1, header=None)

            # Delete the unwrapped file from disk now that its data is loaded.
            os.remove(unwrapped_filename)

            # Create the column names: first column is 'Time', then Mode_1, Mode_2, etc.
            df.columns = ['Time'] + [f'Mode_{i}' for i in range(1, df.shape[1])]

            # Store the Time column separately.
            global time_values
            time_values = df['Time'].to_numpy()

            # Drop the 'Time' column and transpose the DataFrame to get modal coordinates.
            df_transposed_dropped = df.drop(columns='Time').transpose()

            # Convert the DataFrame to a global NumPy array.
            global modal_coord
            modal_coord = df_transposed_dropped.to_numpy()

            del df, df_transposed_dropped

            # Log the success and shape of the resulting array.
            self.console_textbox.append(f"Successfully processed modal coordinate input: {unwrapped_filename}")
            self.console_textbox.append(f"Modal coordinates tensor shape (m x n): {modal_coord.shape} \n")

            # Update the Plot (Modal Coordinates) tab
            self.plot_modal_coords_tab.update_plot(time_values, modal_coord)
            self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_modal_coords_tab), True)
        except Exception as e:
            self.console_textbox.append(f"Error processing modal coordinate file: {e}")
        finally:
            # After processing, update the time point controls in the DisplayTab.
            self.update_display_time_controls()

    def select_stress_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Modal Stress File', '', 'CSV Files (*.csv)')
        if file_name:
            self.stress_file_path.setText(file_name)
            self.process_modal_stress_file(file_name)

    def process_modal_stress_file(self, filename):
        try:
            # Read the modal stress CSV file into a DataFrame
            df = pd.read_csv(filename)

            # Extract 'Node ID' column and save it in another DataFrame
            global df_node_ids  # Make df_node_ids accessible globally
            df_node_ids = df[['NodeID']].to_numpy().flatten()  # Convert to a 1D array

            # If the file contains X, Y, and Z columns, store them
            if {'X', 'Y', 'Z'}.issubset(df.columns):
                global node_coords
                node_coords = df[['X', 'Y', 'Z']].to_numpy()
            else:
                node_coords = None

            # Drop the 'Node ID' column from the original DataFrame
            df = df.drop(columns=['NodeID'])

            # Extract columns related to each normal stress component
            global modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz
            modal_sx = df.filter(regex='(?i)sx_.*').to_numpy().astype(NP_DTYPE)
            modal_sy = df.filter(regex='(?i)sy_.*').to_numpy().astype(NP_DTYPE)
            modal_sz = df.filter(regex='(?i)sz_.*').to_numpy().astype(NP_DTYPE)
            modal_sxy = df.filter(regex='(?i)sxy_.*').to_numpy().astype(NP_DTYPE)
            modal_syz = df.filter(regex='(?i)syz_.*').to_numpy().astype(NP_DTYPE)
            modal_sxz = df.filter(regex='(?i)sxz_.*').to_numpy().astype(NP_DTYPE)

            # Log the success and shape of the DataFrame and arrays
            self.console_textbox.append(f"Successfully processed modal stress file: {filename}\n")
            self.console_textbox.append(f"Modal stress tensor shape (m x n): {df.shape}")
            self.console_textbox.append(f"Node IDs tensor shape: {df_node_ids.shape}\n")
            self.console_textbox.append(f"Normal stress components extracted: SX, SY, SZ, SXY, SYZ, SXZ")
            self.console_textbox.append(f"SX shape: {modal_sx.shape}, SY shape: {modal_sy.shape}, SZ shape: {modal_sz.shape}")
            self.console_textbox.append(f"SXY shape: {modal_sz.shape}, SYZ shape: {modal_syz.shape}, SXZ shape: {modal_sxz.shape}\n")
            self.console_textbox.verticalScrollBar().setValue(self.console_textbox.verticalScrollBar().maximum())
        except Exception as e:
            self.console_textbox.append(f"Error processing modal stress file: {e}")
            self.console_textbox.verticalScrollBar().setValue(self.console_textbox.verticalScrollBar().maximum())
        finally:
            # After processing, update the time point controls in the DisplayTab.
            self.update_display_time_controls()

    def select_deformations_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Modal Deformations File (.csv)', '', 'CSV Files (*.csv)')
        if file_name:
            self.deformations_file_path.setText(file_name)
            self.process_modal_deformations_file(file_name)

    def process_modal_deformations_file(self, filename):
        try:
            df = pd.read_csv(filename)
            # Assume the CSV contains a "NodeID" column and deformation columns.
            # For example, assume columns like "UX_mode1", "UY_mode1", "UZ_mode1", etc.
            global df_node_ids_deformations, modal_ux, modal_uy, modal_uz
            df_node_ids_deformations = df[['NodeID']].to_numpy().flatten()
            # Extract deformation components using a case-insensitive regex (adjust as needed)
            modal_ux = df.filter(regex='(?i)^ux_').to_numpy().astype(NP_DTYPE)
            modal_uy = df.filter(regex='(?i)^uy_').to_numpy().astype(NP_DTYPE)
            modal_uz = df.filter(regex='(?i)^uz_').to_numpy().astype(NP_DTYPE)
            self.console_textbox.append(f"Successfully processed modal deformations file: {filename}")
            self.console_textbox.append(f"Deformations array shapes: UX {modal_ux.shape}, UY {modal_uy.shape}, UZ {modal_uz.shape}")
        except Exception as e:
            self.console_textbox.append(f"Error processing modal deformations file: {e}")

    def update_display_time_controls(self):
        """
        Call the update_time_point_range() method on the DisplayTab to check if the
        modal coordinate and modal stress files are loaded. This should be called after
        processing either file.
        """
        try:
            self.window().display_tab.update_time_point_range()
        except Exception as e:
            print("Could not update display time controls:", e)

    def select_steady_state_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Steady-State Stress File (Node numbers should be included)', '', 'Stress field exported from ANSYS Mechanical (*.txt)')
        if file_name:
            self.steady_state_file_path.setText(file_name)
            self.process_steady_state_file(file_name)

    def process_steady_state_file(self, filename):
        try:
            # Read the steady-state stress file into a DataFrame
            df = pd.read_csv(filename, delimiter='\t', header=0)

            # Log the success and shape of the DataFrame
            self.console_textbox.append(f"Successfully processed steady-state stress file: {filename}\n")

            # Extract columns if they exist
            global steady_node_ids, steady_sx, steady_sy, steady_sz, steady_sxy, steady_syz, steady_sxz

            # Attempt to retrieve node IDs if the column exists
            if 'Node Number' in df.columns:
                steady_node_ids = df['Node Number'].to_numpy().reshape(-1, 1)
                self.console_textbox.append(f"Number of node IDs extracted: {len(steady_node_ids)}")
            else:
                self.console_textbox.append("Error: 'Node Number' column not found in the file.\n")

            self.console_textbox.append(f"Steady-state stress data shape (m x n): {df.shape}")

            # Extract stress components by checking for exact column names
            steady_sx = df['SX (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE) if 'SX (MPa)' in df.columns else None
            steady_sy = df['SY (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE) if 'SY (MPa)' in df.columns else None
            steady_sz = df['SZ (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE) if 'SZ (MPa)' in df.columns else None
            steady_sxy = df['SXY (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE) if 'SXY (MPa)' in df.columns else None
            steady_syz = df['SYZ (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE) if 'SYZ (MPa)' in df.columns else None
            steady_sxz = df['SXZ (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE) if 'SXZ (MPa)' in df.columns else None

            # Log extracted components and their shapes if they are present
            stress_components = {
                "SX": steady_sx, "SY": steady_sy, "SZ": steady_sz,
                "SXY": steady_sxy, "SYZ": steady_syz, "SXZ": steady_sxz
            }
            self.console_textbox.append("Extracted steady-state stress components:")
            for name, comp in stress_components.items():
                if comp is not None:
                    self.console_textbox.append(f"{name} shape: {comp.shape}")
                else:
                    self.console_textbox.append(f"{name} component not found in file.")

            # Scroll to the bottom of the console output
            self.console_textbox.verticalScrollBar().setValue(self.console_textbox.verticalScrollBar().maximum())

        except Exception as e:
            # Log error in console if the file cannot be read or processed
            self.console_textbox.append(f"Error processing steady-state stress file: {filename}")
            self.console_textbox.append(f"Error details: {e}")
            self.console_textbox.verticalScrollBar().setValue(self.console_textbox.verticalScrollBar().maximum())

    def solve(self):
        try:
            # Determine the output location
            output_directory = self.project_directory if self.project_directory else os.path.dirname(
                os.path.abspath(__file__))

            # Ensure modal data are defined before proceeding
            global modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord
            if (modal_sx is None or modal_sy is None or modal_sz is None or modal_sxy is None
                    or modal_syz is None or modal_sxz is None or modal_coord is None):
                self.console_textbox.append("Please load the modal coordinate and stress files before solving.")
                return

            # Show the progress bar at the start of the solution
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Get the current date and time
            current_time = datetime.now()

            self.console_textbox.append(f"\n******************* BEGIN SOLVE ********************\nDatetime: {current_time}\n\n")

            # Check if the checkboxes are checked
            calculate_damage = self.damage_index_checkbox.isChecked()
            calculate_von_mises = self.von_mises_checkbox.isChecked()
            calculate_principal_stress = self.principal_stress_checkbox.isChecked()

            # Check if "Time History Mode" is enabled
            is_time_history_mode = self.time_history_checkbox.isChecked()

            # Check for steady-state stress inclusion
            is_include_steady_state = self.steady_state_checkbox.isChecked()

            # # Initialize steady-state stress variables
            global steady_sx, steady_sy, steady_sz, steady_sxy, steady_syz, steady_sxz, steady_node_ids

            # If steady-state stress inclusion is requested
            if is_include_steady_state:
                if (steady_sx is None or
                    steady_sy is None or
                    steady_sz is None or
                    steady_sxy is None or
                    steady_syz is None or
                    steady_sxz is None or
                    steady_node_ids is None):
                    self.console_textbox.append("Error: Steady-state stress data is not processed yet.")
                    self.progress_bar.setVisible(False)
                    return
            else:
                # Assign steady-state stress variables as empty
                steady_sx = None
                steady_sy = None
                steady_sz = None
                steady_sxy = None
                steady_syz = None
                steady_sxz = None
                steady_node_ids = None

            # Check if modal node IDs are available
            if 'df_node_ids' not in globals() or df_node_ids is None:
                self.console_textbox.append("Error: Modal node IDs are not available.")
                self.progress_bar.setVisible(False)
                return

            if is_time_history_mode:
                # Process only the selected node for time history mode
                selected_node_id = int(self.node_line_edit.text())  # Get selected node ID as index
                selected_node_idx = get_node_index_from_id(selected_node_id, df_node_ids)

                self.console_textbox.append(f"Time History Mode enabled for Node {selected_node_id}\n")

                if 'modal_ux' in globals() and modal_ux is not None:
                    modal_deformations = (modal_ux, modal_uy, modal_uz)
                else:
                    modal_deformations = None

                # Create an instance of the solver
                self.solver = MSUPSmartSolverTransient(
                    modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                    modal_coord,
                    steady_sx=steady_sx,
                    steady_sy=steady_sy,
                    steady_sz=steady_sz,
                    steady_sxy=steady_sxy,
                    steady_syz=steady_syz,
                    steady_sxz=steady_sxz,
                    steady_node_ids=steady_node_ids,
                    modal_node_ids=df_node_ids,
                    output_directory=output_directory
                )

                # Use the new method for single node processing
                time_indices, stress_values = self.solver.process_results_for_a_single_node(
                    selected_node_idx,
                    calculate_von_mises=calculate_von_mises,
                    calculate_principal_stress=calculate_principal_stress,
                )

                if time_indices is not None and stress_values is not None:
                    # Plot the time history of the selected stress component
                    self.plot_single_node_tab.update_plot(time_values, stress_values, selected_node_id,
                                                          is_principal_stress=calculate_principal_stress,
                                                          is_von_mises=calculate_von_mises)

                self.progress_bar.setVisible(False)  # Hide progress bar for single-node operation
                return  # Exit early, no need to write files

            # Create an instance of MSUPSmartSolverTransient for batch solver
            self.solver = MSUPSmartSolverTransient(
                modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                modal_coord,
                steady_sx=steady_sx,
                steady_sy=steady_sy,
                steady_sz=steady_sz,
                steady_sxy=steady_sxy,
                steady_syz=steady_syz,
                steady_sxz=steady_sxz,
                steady_node_ids=steady_node_ids,
                modal_node_ids=df_node_ids,
                output_directory = output_directory
            )

            # Connect the solver's progress signal to the progress bar update slot
            self.solver.progress_signal.connect(self.update_progress_bar)

            # Run the process_results method
            start_time = time.time()
            self.solver.process_results(
                calculate_damage=calculate_damage,
                calculate_von_mises=calculate_von_mises,
                calculate_principal_stress=calculate_principal_stress
            )
            end_time_main_calc = time.time() - start_time

            current_time = datetime.now()

            self.console_textbox.append(f"******************** END SOLVE *********************\nDatetime: {current_time}\n\n")

            # Log the completion
            self.console_textbox.append(f"Main calculation routine completed in: {end_time_main_calc:.2f} seconds")
            self.console_textbox.moveCursor(QTextCursor.End)  # Move cursor to the end
            self.console_textbox.ensureCursorVisible()  # Ensure the cursor is visible

            # region Create maximum over time plot if solver is not run in Time History mode
            if not self.time_history_checkbox.isChecked():
                # Automatically determine which maximum arrays were computed
                vm_data = self.solver.max_over_time_svm if self.von_mises_checkbox.isChecked() else None
                principal_data = self.solver.max_over_time_s1 if self.principal_stress_checkbox.isChecked() else None

                if not hasattr(self, 'plot_max_over_time_tab'):
                    self.plot_max_over_time_tab = PlotlyMaxWidget()
                    modal_tab_index = self.show_output_tab_widget.indexOf(self.plot_modal_coords_tab)
                    self.show_output_tab_widget.insertTab(modal_tab_index + 1, self.plot_max_over_time_tab,
                                                          "Maximum Over Time")

                self.plot_max_over_time_tab.update_plot(time_values, vm_values=vm_data, principal_values=principal_data)
                self.show_output_tab_widget.setTabVisible(
                    self.show_output_tab_widget.indexOf(self.plot_max_over_time_tab), True)
            # endregion

        except Exception as e:
            self.console_textbox.append(f"Error during solving process: {e}, Datetime: {current_time}")
            self.console_textbox.moveCursor(QTextCursor.End)  # Move cursor to the end
            self.console_textbox.ensureCursorVisible()  # Ensure the cursor is visible

    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        if value >= 100:
            # Hide the progress bar once the process is finished
            self.progress_bar.setVisible(False)

    def handle_node_selection(self, node_id):
        """Handles node selection logic for both manual entry and combobox."""
        try:
            if node_id not in df_node_ids:
                QMessageBox.warning(self, "Node Not Found", f"Node ID {node_id} not found in loaded data.")
                return

            # Log the selected Node ID
            self.console_textbox.append(f"Selected Node ID: {node_id}")
            self.console_textbox.moveCursor(QTextCursor.End)
            self.console_textbox.ensureCursorVisible()

            # Dummy plot data (to be replaced with actual results if solver runs)
            x_data = [1, 2, 3, 4, 5]
            y_data = [0, 0, 0, 0, 0]

            # Check checkbox states
            is_principal_stress = self.principal_stress_checkbox.isChecked()
            is_von_mises = self.von_mises_checkbox.isChecked()

            # Update plot widget
            self.plot_single_node_tab.update_plot(
                x_data, y_data, node_id,
                is_principal_stress=is_principal_stress,
                is_von_mises=is_von_mises
            )

            # (Optional) Trigger solve immediately
            # self.solve()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while selecting node: {e}")

    def on_node_entered(self):
        """Triggered when user presses Enter after typing Node ID."""
        try:
            entered_text = self.node_line_edit.text()
            if not entered_text.isdigit():
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer Node ID.")
                return

            node_id = int(entered_text)
            self.handle_node_selection(node_id)  # Use shared method

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing entered Node ID: {e}")

    #region Handle mouse-based UI functionality
    def dragEnterEvent(self, event):
        """Accept the drag event if it contains URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle dropped files by finding the target widget under the cursor."""
        pos = event.pos()  # Position relative to the main widget
        target_widget = self.childAt(pos)  # Correct widget under cursor

        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.handle_dropped_file(target_widget, file_path)
            break  # Process only the first file

    def handle_dropped_file(self, target_widget, file_path):
        """Process the dropped file based on the target widget hierarchy."""
        # Check for stress file drop targets (button or path field)
        if self.is_target_in_widgets(target_widget, [self.stress_file_button, self.stress_file_path]):
            if file_path.endswith('.csv'):
                self.stress_file_path.setText(file_path)
                self.process_modal_stress_file(file_path)
                return

        # Check for coordinate file drop targets
        if self.is_target_in_widgets(target_widget, [self.coord_file_button, self.coord_file_path]):
            if file_path.endswith('.mcf'):
                self.coord_file_path.setText(file_path)
                self.process_modal_coordinate_file(file_path)
                return

        # Check for steady-state file drop targets
        if self.is_target_in_widgets(target_widget, [self.steady_state_file_button, self.steady_state_file_path]):
            if file_path.endswith('.txt'):
                self.steady_state_file_path.setText(file_path)
                self.process_steady_state_file(file_path)
                return

        # Unsupported file or target
        self.console_textbox.append(f"Unsupported file or drop target: {file_path}")

    def is_target_in_widgets(self, target_widget, widgets):
        """Check if the target widget is part of the allowed widgets or their children."""
        while target_widget is not None:
            if target_widget in widgets:
                return True
            target_widget = target_widget.parent()  # Move up the parent hierarchy
        return False
    #endregion

    #region Handle keyboard-based UI functionality
    def toggle_modal_coord_fullscreen_plot(self):
        # Only act if the current output tab is the Plot (Modal Coordinates) tab.
        if self.show_output_tab_widget.currentWidget() != self.plot_modal_coords_tab:
            return

        if self.modal_plot_window is None:
            # Create a new composite widget.
            composite_widget = ModalCoordCompositeWidget()
            # Use stored data from the original Plotly widget to update both plots.
            if (self.plot_modal_coords_tab.last_time_values is not None and
                    self.plot_modal_coords_tab.last_modal_coord is not None):
                composite_widget.update_time_plot(self.plot_modal_coords_tab.last_time_values,
                                                  self.plot_modal_coords_tab.last_modal_coord)
                composite_widget.update_bar_plot(self.plot_modal_coords_tab.last_modal_coord)
            # Create and show the modal window with the composite widget.
            self.modal_plot_window = ModalCoordPlotWindow(composite_widget, parent=self.window())
            self.modal_plot_window.show()
        else:
            self.modal_plot_window.close()
            self.modal_plot_window = None
    #endregion

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create the figure and the canvas for embedding the plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Set size policy to make the canvas expand and shrink with the window
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # Create the layout and add the canvas to it
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_plot(self, x, y, node_id=None, is_principal_stress=False, is_von_mises=False):
        """Update the plot with new data and dynamically adjust the Y-axis label."""

        # Clear the existing figure
        self.figure.clear()

        # Create a single plot layout
        ax = self.figure.add_subplot(1, 1, 1)

        # Determine which plot to create based on the active checkbox
        if is_principal_stress:
            ax.plot(x, y, label=r'$\sigma_1$', color='red')
            ax.set_title(f"Principal Stress (Node ID: {node_id})" if node_id else "Principal Stress", fontsize=8)
            ax.set_ylabel(r'$\sigma_1$ [MPa]', fontsize=8)
        elif is_von_mises:
            ax.plot(x, y, label=r'$\sigma_{VM}$', color='blue')
            ax.set_title(f"Von-Mises Stress (Node ID: {node_id})" if node_id else "Von-Mises Stress", fontsize=8)
            ax.set_ylabel(r'$\sigma_{VM}$ [MPa]', fontsize=8)

        # Set common properties
        ax.set_xlabel('Time [seconds]', fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Increase major ticks
        ax.grid(True, which='both', linestyle='-', linewidth=0.5)  # Enable grid
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=8)

        # Add max value text
        max_y_value = np.max(y)
        textstr = f'Max Value: {max_y_value:.4f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

        # Adjust layout and redraw canvas
        #self.figure.tight_layout()
        self.canvas.draw()

class PlotlyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.web_view = QWebEngineView(self)
        layout = QVBoxLayout()
        layout.addWidget(self.web_view)
        self.setLayout(layout)
        # Store last used data for refresh
        self.last_time_values = None
        self.last_modal_coord = None

    def update_plot(self, time_values, modal_coord):
        self.last_time_values = time_values
        self.last_modal_coord = modal_coord

        fig = go.Figure()
        num_modes = modal_coord.shape[0]
        for i in range(num_modes):
            fig.add_trace(go.Scattergl(
                x=time_values,
                y=modal_coord[i, :],
                mode='lines',  # 'markers' or 'lines+markers'
                name=f'Mode {i + 1}',
                opacity=0.7
            ))

        # Adjust layout here
        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Modal Coordinate Value",
            template="plotly_white",
            font=dict(size=7),  # global font size for labels, etc.
            margin=dict(l=40, r=40, t=10, b=0),  # figure margins
            legend=dict(
                font=dict(size=7)
            )
        )

        # Wrap the figure in a FigureResampler.
        # This enables dynamic resampling on zoom events.
        resampler_fig = FigureResampler(fig, default_n_shown_samples=1000)

        # Generate HTML and display
        html = pyo.plot(resampler_fig, include_plotlyjs='cdn', output_type='div')
        self.web_view.setHtml(html, QUrl("about:blank"))

class ModalCoordCompositeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Use a vertical splitter so the user can adjust height between plots.
        self.splitter = QSplitter(Qt.Vertical)

        # Create two PlotlyWidget instances: one for the time plot and one for the bar plot.
        self.time_plot_widget = PlotlyWidget()
        self.bar_plot_widget = PlotlyWidget()

        self.splitter.addWidget(self.time_plot_widget)
        self.splitter.addWidget(self.bar_plot_widget)
        # Set initial stretch factors (adjust as needed)
        self.splitter.setStretchFactor(0, 3)  # Top widget gets more space
        self.splitter.setStretchFactor(1, 1)  # Bottom widget gets less

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
        self.setLayout(layout)

    def update_time_plot(self, time_values, modal_coord):
        self.time_plot_widget.update_plot(time_values, modal_coord)

    def update_bar_plot(self, modal_coord):
        """
        Computes the RMS of each mode (row of modal_coord) and creates a bar plot.
        Each bar is its own trace so that a legend entry appears.
        """
        # Compute RMS values along the time axis (axis=1)
        rms = np.sqrt(np.mean(modal_coord ** 2, axis=1))
        num_modes = rms.shape[0]
        # Create mode labels
        x_labels = [f"Mode {i + 1}" for i in range(num_modes)]

        # Create a new bar plot figure.
        fig = go.Figure()
        for i in range(num_modes):
            # Each mode is a separate trace.
            fig.add_trace(go.Bar(
                x=[x_labels[i]],
                y=[rms[i]],
                name=f"Mode {i + 1}",
                opacity=0.7
            ))
        # Update layout without fixed width/height.
        fig.update_layout(
            xaxis_title="Mode",
            yaxis_title="RMS Value",
            template="plotly_white",
            font=dict(size=7),
            margin=dict(l=40, r=40, t=10, b=40),
            showlegend=True
        )
        # Generate HTML using inline Plotly JS or CDN as needed.
        html = pyo.plot(fig, include_plotlyjs='cdn', output_type='div')
        self.bar_plot_widget.web_view.setHtml(html, QUrl("about:blank"))

class ModalCoordPlotWindow(QMainWindow):
    def __init__(self, composite_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Modal Coordinates Plot")
        self.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(composite_widget)
        self.showMaximized()
        # Shortcut "M" inside the modal window to close it.
        self.shortcut_m = QShortcut(QKeySequence("M"), self)
        self.shortcut_m.activated.connect(self.close)

    def closeEvent(self, event):
        # When closed, simply accept the event.
        event.accept()

class PlotlyMaxWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.web_view = QWebEngineView(self)
        layout = QVBoxLayout()
        layout.addWidget(self.web_view)
        self.setLayout(layout)

    def update_plot(self, time_values, vm_values=None, principal_values=None):
        fig = go.Figure()

        # If both metrics are provided, add two traces.
        if vm_values is not None and principal_values is not None:
            fig.add_trace(go.Scattergl(
                x=time_values,
                y=vm_values,
                mode='lines',
                name='Von Mises',
                opacity=0.7,
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scattergl(
                x=time_values,
                y=principal_values,
                mode='lines',
                name='Max Principal',
                opacity=0.7,
                line=dict(color='red')
            ))
        # If only von Mises values are available, plot only that trace.
        elif vm_values is not None:
            fig.add_trace(go.Scattergl(
                x=time_values,
                y=vm_values,
                mode='lines',
                name='Von Mises',
                opacity=0.7,
                line=dict(color='blue')
            ))
        # If only principal stress values are available, plot that trace.
        elif principal_values is not None:
            fig.add_trace(go.Scattergl(
                x=time_values,
                y=principal_values,
                mode='lines',
                name='Max Principal',
                opacity=0.7,
                line=dict(color='red')
            ))
        else:
            # If no data is provided, display a message.
            fig.add_annotation(text="No data available", showarrow=False)

        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Stress (MPa)",
            template="plotly_white",
            font=dict(size=7),
            margin=dict(l=40, r=40, t=10, b=0),
            legend=dict(font=dict(size=7))
        )

        # Wrap the figure for dynamic resampling (improves performance on zoom)
        resampler_fig = FigureResampler(fig, default_n_shown_samples=1000)
        html = pyo.plot(resampler_fig, include_plotlyjs='cdn', output_type='div')
        self.web_view.setHtml(html, QUrl("about:blank"))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window title and dimensions
        self.setWindowTitle('MSUP Smart Solver - v0.8')
        self.setGeometry(40, 40, 800, 670)

        # Create a menu bar
        menu_bar_style = """
            QMenuBar {
                background-color: #ffffff;  /* White background */
                color: #000000;             /* Black text */
                padding: 2px;               /* Reduced padding */
                font-family: Arial;
                font-size: 12px;            /* Smaller font size */
            }
            QMenuBar::item {
                background-color: #ffffff;
                color: #000000;
                padding: 2px 5px;           /* Reduced padding */
                margin: 0px;                /* Removed margin */
            }
            QMenuBar::item:selected {
                background-color: #e0e0e0;  /* Light gray for hover effect */
                border-radius: 2px;         /* Slightly rounded corners */
            }
            QMenu {
                background-color: #ffffff;
                color: #000000;
                padding: 2px;
                border: 1px solid #d0d0d0;  /* Light gray border */
            }
            QMenu::item {
                background-color: transparent;
                padding: 2px 10px;          /* Reduced padding */
            }
            QMenu::item:selected {
                background-color: #e0e0e0;
                border-radius: 2px;
            }
        """

        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.menu_bar.setStyleSheet(menu_bar_style)

        # Add "File" menu
        file_menu = self.menu_bar.addMenu("File")

        # Add "Select Project Directory" action
        select_dir_action = QAction("Select Project Directory", self)
        select_dir_action.triggered.connect(self.select_project_directory)
        file_menu.addAction(select_dir_action)

        # Create a QTabWidget
        self.tab_widget = QTabWidget()

        tab_style = """
            QTabBar::tab {
                background-color: #e7f0fd;
                border: 2px solid #5b9bd5;
                padding: 3px;
                border-top-left-radius: 5px;  /* Upper left corner rounded */
                border-top-right-radius: 5px; /* Upper right corner rounded */
                margin: 2px;
            }
            QTabBar::tab:hover {
                background-color: #cce4ff;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;  # Active tab has a white background
                border: 2px solid #5b9bd5;
                font-weight: bold;
            }
            QTabBar::tab:!selected {
                margin-top: 3px;  # Make the unselected tabs slightly smaller
            }
        """
        self.tab_widget.setStyleSheet(tab_style)

        # Create the "Main Window" tab and add the MSUPSmartSolverGUI widget to it
        self.batch_solver_tab = MSUPSmartSolverGUI()
        self.tab_widget.addTab(self.batch_solver_tab, "Main Window")

        # Create and add Display tab
        self.display_tab = DisplayTab()
        self.display_tab.main_window = self
        self.tab_widget.addTab(self.display_tab, "Display")

        # Set the central widget of the main window to the tab widget
        self.setCentralWidget(self.tab_widget)

        # Variable to store selected project directory
        self.project_directory = None

        # Create the Navigator (File Explorer)
        self.create_navigator()

    def create_navigator(self):
        """Create a dockable navigator showing project directory contents."""
        self.navigator_dock = QDockWidget("Navigator", self)
        self.navigator_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.navigator_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)

        # Get the Desktop path dynamically
        #desktop_path = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)

        # Create file system model
        self.file_model = QFileSystemModel()
        #self.file_model.setRootPath(desktop_path)  # Initially Desktop, updates when project directory is selected
        self.file_model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)  # Show all files & folders

        # Apply file type filter
        self.file_model.setNameFilters(["*.csv", "*.mcf", "*.txt"])  # Only show CSV, MCF, TXT files
        self.file_model.setNameFilterDisables(False)  # Disable showing grayed-out files

        # Create Tree View
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        #self.tree_view.setRootIndex(self.file_model.index(desktop_path))  # Start at Desktop
        self.tree_view.setHeaderHidden(False)  # Show headers for resizing
        self.tree_view.setMinimumWidth(240)  # Set a reasonable width
        self.tree_view.setSortingEnabled(True)  # Allow sorting of files/folders

        # Hide unwanted columns: 1 = Size, 2 = Type
        self.tree_view.setColumnHidden(1, True)
        self.tree_view.setColumnHidden(2, True)

        # Adjust column width to fit content
        self.tree_view.setColumnWidth(0, 250)  # Set a reasonable default width for file names
        self.tree_view.header().setSectionResizeMode(0,
                                                     self.tree_view.header().ResizeToContents)  # Auto-resize name column

        # Apply style to match main buttons
        navigator_title_style = """
            QDockWidget::title {
                background-color: #e7f0fd;  /* Match button background */
                color: black;  /* Match button text color */
                font-weight: bold;
                font-size: 9px;
                padding-top: 2px;
                padding-bottom: 2px;
                padding-left: 8px;
                padding-right: 8px;
                border-bottom: 2px solid #5b9bd5;  /* Match button border */
            }
        """

        # Tree View Styling (for Navigator contents)
        tree_view_style = """
            QTreeView {
                font-size: 7.5pt;  /* Smaller font for tree contents */
                background-color: #ffffff;  /* Keep it clean */
                alternate-background-color: #f5f5f5;  /* Slight alternation for readability */
                border: none;
            }
            QTreeView::item:hover {
                background-color: #d0e4ff;  /* Subtle hover effect */
            }
            QTreeView::item:selected {
                background-color: #5b9bd5;  /* Active selection color */
                color: #ffffff;  /* White text when selected */
            }
            QHeaderView::section {
                background-color: #e7f0fd;  /* Match the navigator title */
                padding: 3px;
                border: none;
                font-weight: bold;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #5b9bd5;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """

        self.navigator_dock.setStyleSheet(navigator_title_style)
        self.tree_view.setStyleSheet(tree_view_style)

        # Set Tree View as the dock widget's main content
        self.navigator_dock.setWidget(self.tree_view)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.navigator_dock)  # Add it to the left

        # Add a "View" menu option to show/hide Navigator
        view_menu = self.menu_bar.addMenu("View")
        toggle_navigator_action = self.navigator_dock.toggleViewAction()
        toggle_navigator_action.setText("Navigator")  # Rename action
        view_menu.addAction(toggle_navigator_action)

        #Enable drag and drop on the TreeView
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setSelectionMode(QTreeView.SingleSelection)
        self.tree_view.setDragDropMode(QTreeView.DragDrop)

    def select_project_directory(self):
        """Open a dialog to select a project directory and update the Navigator."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.project_directory = dir_path
            print(f"Project directory selected: {self.project_directory}")

            # Update the solver GUI's project_directory
            self.batch_solver_tab.project_directory = self.project_directory  # <-- Ensures solver GUI gets updated

            # Update the navigator with the selected directory
            self.file_model.setRootPath(self.project_directory)
            self.tree_view.setRootIndex(self.file_model.index(self.project_directory))

# endregion

# region Run the main GUI
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # Use high DPI icons and images
    app = QApplication(sys.argv)

    # Create the main window and show it
    main_window = MainWindow()
    main_window.showMaximized()

    sys.exit(app.exec_())
# endregion
