# region Import libraries
print("Importing libraries...")
import math
import threading
import subprocess
import time
import sys
import os
import io

from io import StringIO
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import psutil
import gc
import vtk
import plotly.graph_objects as go
import plotly.offline as pyo
import imageio
import tempfile
import plotly.io as pio
import pyvista as pv
from pyvistaqt import QtInteractor
from plotly_resampler import FigureResampler
from scipy.signal import butter, filtfilt, detrend

from numba import njit, prange
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QMainWindow, QCheckBox, QProgressBar, QFileDialog, QGroupBox, QGridLayout, QSizePolicy,
                             QTextEdit, QTabWidget, QComboBox, QMenuBar, QAction, QDockWidget, QTreeView, QMenu,
                             QFileSystemModel, QMessageBox, QSpinBox, QDoubleSpinBox, QShortcut, QSplitter, QHeaderView,
                             QAbstractItemView, QTableView, QProgressDialog, QDialog, QDialogButtonBox, QInputDialog,
                             QWidgetAction)
from PyQt5.QtGui import (QPalette, QColor, QFont, QTextCursor, QKeySequence, QDoubleValidator, QStandardItemModel,
                         QStandardItem, QKeySequence)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QUrl, QDir, QStandardPaths, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
print("Done")

# endregion

# region Define global variables
RAM_PERCENT = 0.9 # Default RAM allocation percentage based on free RAM
DEFAULT_PRECISION = 'Double'

if DEFAULT_PRECISION == 'Single':
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32
    RESULT_DTYPE = 'float32'
elif DEFAULT_PRECISION == 'Double':
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64
    RESULT_DTYPE = 'float64'

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
        cycles.append((abs(stack[i] - stack[i + 1]), 0.5))
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

# region Define global classes
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

        # Memory details
        my_virtual_memory = psutil.virtual_memory()
        self.total_memory = my_virtual_memory.total / (1024 ** 3)
        self.available_memory = my_virtual_memory.available / (1024 ** 3)
        self.allocated_memory = my_virtual_memory.available * self.RAM_PERCENT / (1024 ** 3)
        print(f"Total system RAM: {self.total_memory:.2f} GB")
        print(f"Available system RAM: {self.available_memory:.2f} GB")
        print(f"Allocated system RAM: {self.allocated_memory:.2f} GB")

        # Store time axis once for gradient calcs
        global time_values
        self.time_values = time_values.astype(self.NP_DTYPE) if 'time_values' in globals() else None

    def estimate_chunk_size(self, num_nodes, num_time_points, calculate_von_mises, calculate_max_principal_stress,
                            calculate_damage, calculate_deformation=False,
                            calculate_velocity=False, calculate_acceleration=False):
        """Calculate the optimal chunk size for processing based on available memory."""
        available_memory = psutil.virtual_memory().available * self.RAM_PERCENT
        memory_per_node = self.get_memory_per_node(num_time_points, calculate_von_mises, calculate_max_principal_stress,
                                                   calculate_damage, calculate_deformation,
                                                   calculate_velocity, calculate_acceleration)
        max_nodes_per_iteration = available_memory // memory_per_node
        return max(1, int(max_nodes_per_iteration))  # Ensure at least one node per chunk

    def estimate_ram_required_per_iteration(self, chunk_size, memory_per_node):
        """Estimate the total RAM required per iteration to compute stresses."""
        total_memory = chunk_size * memory_per_node
        return total_memory / (1024 ** 3)  # Convert bytes to GB

    def get_memory_per_node(self, num_time_points, calculate_von_mises, calculate_max_principal_stress,
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

    def map_steady_state_stresses(self, steady_stress, steady_node_ids, modal_node_ids):
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

        return actual_sx.cpu().numpy(), actual_sy.cpu().numpy(), actual_sz.cpu().numpy(), actual_sxy.cpu().numpy(), actual_syz.cpu().numpy(), actual_sxz.cpu().numpy()

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
        if 'modal_ux' not in globals():
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
                if s1 < s2: s1, s2 = s2, s1
                if s2 < s3: s2, s3 = s3, s2
                if s1 < s2: s1, s2 = s2, s1

                # --- Step 6: Store the Final Results ---
                # Assign the sorted principal stresses to their correct place in our output arrays.
                s1_out[i, j] = s1
                s2_out[i, j] = s2
                s3_out[i, j] = s3

        # After the loops have finished, return the three complete 2D arrays of results.
        return s1_out, s2_out, s3_out

    def process_results(self,
                        calculate_damage=False,
                        calculate_von_mises=False,
                        calculate_max_principal_stress=False,
                        calculate_min_principal_stress=False,
                        calculate_deformation=False,
                        calculate_velocity=False,
                        calculate_acceleration=False):
        """Process stress results in batch to compute user requested outputs and their max/min values over time."""
        # region Initialization
        # Initialize tensor size
        num_nodes, num_modes = self.modal_sx.shape
        num_time_points = self.modal_coord.shape[1]
        # endregion

        # region Get the chunk size based on selected options
        chunk_size = self.estimate_chunk_size(
            num_nodes, num_time_points,
            calculate_von_mises, calculate_max_principal_stress, calculate_damage,
            calculate_deformation, calculate_velocity, calculate_acceleration)

        num_iterations = (num_nodes + chunk_size - 1) // chunk_size
        print(f"Estimated number of iterations to avoid memory overflow: {num_iterations}")

        memory_per_node = self.get_memory_per_node(
            num_time_points, calculate_von_mises, calculate_max_principal_stress, calculate_damage,
            calculate_deformation, calculate_velocity, calculate_acceleration)
        memory_required_per_iteration = self.estimate_ram_required_per_iteration(chunk_size, memory_per_node)
        print(f"Estimated RAM required per iteration: {memory_required_per_iteration:.2f} GB\n")
        # endregion

        # region Create temporary (memmap) files
        if calculate_max_principal_stress:
            # Initialize max over time vector
            self.max_over_time_s1 = -np.inf * np.ones(num_time_points, dtype=NP_DTYPE)

            # Create memmap files for storing the maximum principal stresses per node (s1)
            s1_max_memmap = np.memmap(os.path.join(self.output_directory, 'max_s1_stress.dat'),
                                      dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))
            s1_max_time_memmap = np.memmap(os.path.join(self.output_directory, 'time_of_max_s1_stress.dat'),
                                           dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))

        if calculate_min_principal_stress:
            self.min_over_time_s3 = np.inf * np.ones(num_time_points, dtype=NP_DTYPE)

            s3_min_memmap = np.memmap(os.path.join(self.output_directory, 'min_s3_stress.dat'),
                                      dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))
            s3_min_time_memmap = np.memmap(os.path.join(self.output_directory, 'time_of_min_s3_stress.dat'),
                                           dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))

        if calculate_von_mises:
            # Initialize max over time vector
            self.max_over_time_svm = -np.inf * np.ones(num_time_points, dtype=NP_DTYPE)

            # Create memmap files for storing the maximum von Mises stresses per node
            von_mises_max_memmap = np.memmap(os.path.join(self.output_directory, 'max_von_mises_stress.dat'),
                                             dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))
            von_mises_max_time_memmap = np.memmap(
                os.path.join(self.output_directory, 'time_of_max_von_mises_stress.dat'),
                dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))

        if calculate_deformation:
            # Initialize max over time vector
            self.max_over_time_def = -np.inf * np.ones(num_time_points, dtype=NP_DTYPE)

            # Create memmap files for storing the maximum deformation magnitudes per node
            def_max_memmap = np.memmap(os.path.join(self.output_directory, 'max_deformation.dat'),
                                       dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))
            def_time_memmap = np.memmap(os.path.join(self.output_directory, 'time_of_max_deformation.dat'),
                                        dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))

        if calculate_velocity:
            # Initialize max over time vector
            self.max_over_time_vel = -np.inf * np.ones(num_time_points, dtype=NP_DTYPE)

            # Create memmap files for storing the maximum velocity magnitudes per node
            vel_max_memmap = np.memmap(os.path.join(self.output_directory, 'max_velocity.dat'),
                                       dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))
            vel_time_memmap = np.memmap(os.path.join(self.output_directory, 'time_of_max_velocity.dat'),
                                        dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))

        if calculate_acceleration:
            # Initialize max over time vector
            self.max_over_time_acc = -np.inf * np.ones(num_time_points, dtype=NP_DTYPE)

            # Create memmap files for storing the maximum velocity magnitudes per node
            acc_max_memmap = np.memmap(os.path.join(self.output_directory, 'max_acceleration.dat'),
                                       dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))
            acc_time_memmap = np.memmap(os.path.join(self.output_directory, 'time_of_max_acceleration.dat'),
                                        dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))

        if calculate_damage:
            potential_damage_memmap = np.memmap(os.path.join(self.output_directory, 'potential_damage_results.dat'),
                                                dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))
        # endregion

        # region --- MAIN CALCULATION ROUTINE ---
        for start_idx in range(0, num_nodes, chunk_size):
            end_idx = min(start_idx + chunk_size, num_nodes)

            # region Calculate normal stresses
            start_time = time.time()
            actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
                self.compute_normal_stresses(start_idx, end_idx)
            print(f"Elapsed time for normal stresses: {(time.time() - start_time):.3f} seconds")
            # endregion

            # region Calculate the requested outputs
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

            if calculate_max_principal_stress or calculate_min_principal_stress:
                # Calculate principal stresses
                start_time = time.time()
                s1, s2, s3 = self.compute_principal_stresses(actual_sx, actual_sy, actual_sz,
                                                             actual_sxy, actual_syz, actual_sxz)
                print(f"Elapsed time for principal stresses: {(time.time() - start_time):.3f} seconds")

            if calculate_max_principal_stress:
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

            if calculate_min_principal_stress:
                # global minima over time (element-wise)
                chunk_min = np.min(s3, axis=0)
                self.min_over_time_s3 = np.minimum(self.min_over_time_s3, chunk_min)

                # per-node minimum & its time index
                min_s3_per_node = np.min(s3, axis=1)
                time_indices = np.argmin(s3, axis=1)
                time_of_min_s3_per_node = time_values[time_indices]

                s3_min_memmap[start_idx:end_idx] = min_s3_per_node
                s3_min_time_memmap[start_idx:end_idx] = time_of_min_s3_per_node

            if (calculate_velocity or calculate_acceleration or calculate_deformation) and \
                    self.modal_deformations_ux is not None:
                ux, uy, uz = self.compute_deformations(start_idx, end_idx)

                # --- ADD THIS NEW IF BLOCK ---
                if calculate_deformation:
                    start_time = time.time()
                    def_mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
                    chunk_max = np.max(def_mag, axis=0)
                    self.max_over_time_def = np.maximum(self.max_over_time_def, chunk_max)
                    max_def_per_node = np.max(def_mag, axis=1)
                    time_indices = np.argmax(def_mag, axis=1)
                    def_max_memmap[start_idx:end_idx] = max_def_per_node
                    def_time_memmap[start_idx:end_idx] = time_values[time_indices]
                    print(f"Elapsed time for deformation magnitude and time: {(time.time() - start_time):.3f} seconds")

                if calculate_velocity or calculate_acceleration:
                    start_time = time.time()
                    vel_mag, acc_mag, _, _, _, _, _, _ = self._vel_acc_from_disp(ux, uy, uz, self.time_values)
                    print(f"Elapsed time for calculation of velocity/acceleration components: {(time.time() - start_time):.3f} seconds")

                    if calculate_velocity:
                        start_time = time.time()
                        chunk_max = np.max(vel_mag, axis=0)
                        self.max_over_time_vel = np.maximum(self.max_over_time_vel, chunk_max)
                        max_vel_per_node = np.max(vel_mag, axis=1)
                        time_indices = np.argmax(vel_mag, axis=1)
                        vel_max_memmap[start_idx:end_idx]  = max_vel_per_node
                        vel_time_memmap[start_idx:end_idx] = time_values[time_indices]
                        print(f"Elapsed time for velocity magnitude and time: {(time.time() - start_time):.3f} seconds")

                    if calculate_acceleration:
                        start_time = time.time()
                        chunk_max = np.max(acc_mag, axis=0)
                        self.max_over_time_acc = np.maximum(self.max_over_time_acc, chunk_max)
                        max_acc_per_node = np.max(acc_mag, axis=1)
                        time_indices = np.argmax(acc_mag, axis=1)
                        acc_max_memmap[start_idx:end_idx]  = max_acc_per_node
                        acc_time_memmap[start_idx:end_idx] = time_values[time_indices]
                        print(f"Elapsed time for acceleration magnitude and time: {(time.time() - start_time):.3f} seconds")

            # Calculate potential damage index for all nodes in the chunk
            if calculate_damage:
                start_time = time.time()

                # Compute the signed von Mises stress using the existing von Mises results
                signed_von_mises = self.compute_signed_von_mises_stress(sigma_vm, actual_sx, actual_sy, actual_sz)

                # Use the signed von Mises stress for damage calculation
                # Instead of hardcoding, use fatigue parameters if they have been set; otherwise, fall back to defaults.
                A = self.fatigue_A if hasattr(self, 'fatigue_A') else 1
                m = self.fatigue_m if hasattr(self, 'fatigue_m') else -3
                potential_damages = compute_potential_damage_for_all_nodes(signed_von_mises, A, m)
                potential_damage_memmap[start_idx:end_idx] = potential_damages
                print(f"Elapsed time for damage index calculation: {(time.time() - start_time):.3f} seconds")
            # endregion

            # region Free up some memory
            start_time = time.time()
            del actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz

            if calculate_von_mises:
                del sigma_vm
            if calculate_max_principal_stress or calculate_min_principal_stress:
                del s1, s2, s3
            if calculate_deformation:
                del def_mag
            if calculate_velocity:
                del vel_mag
            if calculate_acceleration:
                del acc_mag

            gc.collect()
            print(f"Elapsed time for garbage collection: {(time.time() - start_time):.3f} seconds")
            current_available_memory = psutil.virtual_memory().available * self.RAM_PERCENT
            # endregion

            # region Emit progress signal as a percentage of the total iterations
            current_iteration = (start_idx // chunk_size) + 1
            progress_percentage = (current_iteration / num_iterations) * 100
            self.progress_signal.emit(int(progress_percentage))
            QApplication.processEvents()  # Keep UI responsive

            print(f"Iteration completed for nodes {start_idx} to {end_idx}. "
                  f"Allocated system RAM: {current_available_memory / (1024 ** 3):.2f} GB\n")
            # endregion
        # endregion

        # region Ensure all memmap files are flushed to disk
        if calculate_von_mises:
            von_mises_max_memmap.flush()
            von_mises_max_time_memmap.flush()
        if calculate_max_principal_stress:
            s1_max_memmap.flush()
            s1_max_time_memmap.flush()
        if calculate_min_principal_stress:
            s3_min_memmap.flush()
            s3_min_time_memmap.flush()
        if calculate_deformation:
            def_max_memmap.flush()
            def_time_memmap.flush()
        if calculate_velocity:
            vel_max_memmap.flush()
            vel_time_memmap.flush()
        if calculate_acceleration:
            acc_max_memmap.flush()
            acc_time_memmap.flush()
        if calculate_damage:
            potential_damage_memmap.flush()
        # endregion

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
        if calculate_max_principal_stress:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "max_s1_stress.dat"),
                                    os.path.join(self.output_directory, "max_s1_stress.csv"),
                                    "S1_Max")
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "time_of_max_s1_stress.dat"),
                                    os.path.join(self.output_directory, "time_of_max_s1_stress.csv"),
                                    "Time_of_S1_Max")
        if calculate_min_principal_stress:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "min_s3_stress.dat"),
                                    os.path.join(self.output_directory, "min_s3_stress.csv"),
                                    "S3_Min")

            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "time_of_min_s3_stress.dat"),
                                    os.path.join(self.output_directory, "time_of_min_s3_stress.csv"),
                                    "Time_of_S3_Min")
        if calculate_deformation:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "max_deformation.dat"),
                                    os.path.join(self.output_directory, "max_deformation.csv"),
                                    "DEF_Max")
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "time_of_max_deformation.dat"),
                                    os.path.join(self.output_directory, "time_of_max_deformation.csv"),
                                    "Time_of_DEF_Max")
        if calculate_velocity:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "max_velocity.dat"),
                                    os.path.join(self.output_directory, "max_velocity.csv"),
                                    "VEL_Max")
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "time_of_max_velocity.dat"),
                                    os.path.join(self.output_directory, "time_of_max_velocity.csv"),
                                    "Time_of_VEL_Max")
        if calculate_acceleration:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "max_acceleration.dat"),
                                    os.path.join(self.output_directory, "max_acceleration.csv"),
                                    "ACC_Max")
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "time_of_max_acceleration.dat"),
                                    os.path.join(self.output_directory, "time_of_max_acceleration.csv"),
                                    "Time_of_ACC_Max")

        if calculate_damage:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    os.path.join(self.output_directory, "potential_damage_results.dat"),
                                    os.path.join(self.output_directory, "potential_damage_results.csv"),
                                    "Potential Damage (Damage Index)")
        # endregion

    def process_results_for_a_single_node(self,
                                          selected_node_idx,
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

        # region Initialization & reassignment of variables
        selected_node_id = df_node_ids[selected_node_idx]
        # endregion

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
                s1, s2, s3 = self.compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
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
                        return np.arange(vel_mag.shape[1]), velocity_data
                    if calculate_acceleration:
                        acceleration_data = {
                            'Magnitude': acc_mag[0, :],
                            'X': acc_x[0, :],
                            'Y': acc_y[0, :],
                            'Z': acc_z[0, :]
                        }
                        return np.arange(acc_mag.shape[1]), acceleration_data

        # Return none if no output is requested
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
        self.anim_timer = None  # timer for animation
        self.time_text_actor = None
        self.current_anim_time = 0.0  # current time in the animation
        self.animation_paused = False
        self.temp_solver = None

        # Attributes for Precomputation
        self.precomputed_scalars = None  # (num_nodes, num_anim_steps)
        self.precomputed_coords = None  # (num_nodes, 3, num_anim_steps) or similar
        self.precomputed_anim_times = None  # (num_anim_steps,) - actual time values for each frame
        self.current_anim_frame_index = 0  # Index for accessing precomputed arrays
        self.data_column_name = "Stress"  # Default/placeholder name for scalars
        self.is_deformation_included_in_anim = False  # Track if deformation was computed

        self.highlight_actor = None  # This tracks the highlight sphere
        self.box_widget = None
        self.hotspot_dialog = None
        self.is_point_picking_active = False

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
        self.point_size.setValue(5)
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

        # Add Custom Context Menu
        self.plotter.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plotter.customContextMenuRequested.connect(self.show_context_menu)

        # Layout
        layout = QVBoxLayout()

        # File controls
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_path)

        # Visualization controls
        self.graphics_control_layout = QHBoxLayout()
        self.graphics_control_layout.addWidget(QLabel("Node Point Size:"))
        self.graphics_control_layout.addWidget(self.point_size)
        self.graphics_control_layout.addWidget(QLabel("Legend Range:"))
        self.graphics_control_layout.addWidget(self.scalar_min_spin)
        self.graphics_control_layout.addWidget(self.scalar_max_spin)
        self.graphics_control_layout.addStretch()

        self.graphics_control_group = QGroupBox("Visualization Controls")
        self.graphics_control_group.setStyleSheet(group_box_style)
        self.graphics_control_group.setLayout(self.graphics_control_layout)

        # Contour Time-point controls layout:
        self.selected_time_label = QLabel("Initialize / Display results for a selected time point:")
        self.selected_time_label.setStyleSheet("margin: 10px;")
        # Initially hide the checkbox, and show it once the required files are loaded.

        self.time_point_spinbox = QDoubleSpinBox()
        self.time_point_spinbox.setDecimals(5)
        self.time_point_spinbox.setPrefix("Time (seconds): ")
        # Range will be updated later from the modal coordinate file's time values.
        self.time_point_spinbox.setRange(0, 0)

        self.update_time_button = QPushButton("Update")
        self.update_time_button.clicked.connect(self.update_time_point_results)

        self.save_time_button = QPushButton("Save Time Point as CSV")
        self.save_time_button.clicked.connect(self.save_time_point_results)

        # Put the new widgets in a horizontal layout
        self.time_point_layout = QHBoxLayout()
        self.time_point_layout.addWidget(self.selected_time_label)
        self.time_point_layout.addWidget(self.time_point_spinbox)
        self.time_point_layout.addWidget(self.update_time_button)
        self.time_point_layout.addWidget(self.save_time_button)
        self.time_point_layout.addStretch()

        self.time_point_group = QGroupBox("Initialization && Time Point Controls")
        self.time_point_group.setStyleSheet(group_box_style)
        self.time_point_group.setLayout(self.time_point_layout)
        self.time_point_group.setVisible(False)

        # Animation Control Layout
        self.anim_layout = QHBoxLayout()
        # Add a spin box for animation frame interval (in milliseconds)
        self.anim_interval_spin = QSpinBox()
        self.anim_interval_spin.setRange(5, 10000)  # Allow between 5 ms and 10,000 ms delay
        self.anim_interval_spin.setValue(100)  # Default delay is 100 ms
        self.anim_interval_spin.setPrefix("Interval (ms): ")
        self.anim_layout.addWidget(self.anim_interval_spin)
        # Label and spinbox for animation start time
        self.anim_start_label = QLabel("Time Range:")
        self.anim_start_spin = QDoubleSpinBox()
        self.anim_start_spin.setPrefix("Start: ")
        self.anim_start_spin.setDecimals(5)
        self.anim_start_spin.setMinimum(0)
        self.anim_start_spin.setValue(0)
        # Label and spinbox for animation end time
        self.anim_end_spin = QDoubleSpinBox()
        self.anim_end_spin.setPrefix("End: ")
        self.anim_end_spin.setDecimals(5)
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
        self.stop_button.clicked.connect(self.stop_animation)
        # Add Time Step Mode ComboBox and Custom Step SpinBox
        self.time_step_mode_combo = QComboBox()
        self.time_step_mode_combo.addItems(["Custom Time Step", "Actual Data Time Steps"])
        self.time_step_mode_combo.setCurrentIndex(1)
        self.custom_step_spin = QDoubleSpinBox()
        self.custom_step_spin.setDecimals(5)
        self.custom_step_spin.setRange(0.000001, 10)
        self.custom_step_spin.setValue(0.01)
        self.custom_step_spin.setPrefix("Step (secs): ")

        self.actual_interval_spin = QSpinBox()
        self.actual_interval_spin.setRange(1, 1)  # Set max later after loading time_values
        self.actual_interval_spin.setValue(1)
        self.actual_interval_spin.setPrefix("Every nth: ")
        self.actual_interval_spin.setVisible(False)  # Hidden by default

        # Connect the combo box's text change signal
        self.time_step_mode_combo.currentTextChanged.connect(self.update_step_spinbox_state)
        self.update_step_spinbox_state(self.time_step_mode_combo.currentText())

        # Deformation Scale Factor
        self.deformation_scale_label = QLabel("Deformation Scale Factor:")
        self.deformation_scale_edit = QLineEdit("1")
        # Create and set a QDoubleValidator. This will allow numbers in standard or scientific notation.
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.deformation_scale_edit.setValidator(validator)
        # Connect the editingFinished signal so that pressing Enter (or losing focus) triggers validation.
        self.deformation_scale_edit.editingFinished.connect(self.validate_deformation_scale)
        # Store the last valid input – starting at 1.0.
        self.last_valid_deformation_scale = 1.0
        self.deformation_scale_label.setVisible(False)
        self.deformation_scale_edit.setVisible(False)
        self.graphics_control_layout.addWidget(self.deformation_scale_label)
        self.graphics_control_layout.addWidget(self.deformation_scale_edit)

        # Add Save Animation Button ---
        self.save_anim_button = QPushButton("Save as Video/GIF")
        self.save_anim_button.setStyleSheet(button_style)  # Apply the style
        self.save_anim_button.clicked.connect(self.save_animation)
        self.save_anim_button.setEnabled(False)  # Initially disabled
        self.save_anim_button.setToolTip(
            "Save the precomputed animation frames as MP4 or GIF.\nRequires 'imageio' and 'ffmpeg' (for MP4).")

        # Add widgets to the animation layout
        self.anim_layout.addWidget(self.time_step_mode_combo)
        self.anim_layout.addWidget(self.custom_step_spin)
        self.anim_layout.addWidget(self.actual_interval_spin)
        self.anim_layout.addWidget(self.anim_interval_spin)
        self.anim_layout.addWidget(self.anim_start_label)
        self.anim_layout.addWidget(self.anim_start_spin)
        self.anim_layout.addWidget(self.anim_end_spin)
        self.anim_layout.addWidget(self.play_button)
        self.anim_layout.addWidget(self.pause_button)
        self.anim_layout.addWidget(self.stop_button)
        self.anim_layout.addWidget(self.save_anim_button)
        # self.anim_layout.addStretch()

        self.anim_group = QGroupBox("Animation Controls")
        self.anim_group.setStyleSheet(group_box_style)
        self.anim_group.setLayout(self.anim_layout)
        self.anim_group.setVisible(False)

        layout.addLayout(file_layout)
        layout.addWidget(self.graphics_control_group)
        layout.addWidget(self.time_point_group)
        layout.addWidget(self.anim_group)
        layout.addWidget(self.plotter)
        self.setLayout(layout)

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

            # Update animation time range controls
            self.anim_start_spin.setRange(min_time, max_time)
            self.anim_end_spin.setRange(min_time, max_time)
            self.anim_start_spin.setValue(min_time)
            self.anim_end_spin.setValue(max_time)
            self.actual_interval_spin.setMaximum(len(time_values))  # max is number of time points
            self.actual_interval_spin.setValue(1)  # default to every point

            self.anim_group.setVisible(True)
            self.time_point_group.setVisible(True)
            self.deformation_scale_label.setVisible(True)
            self.deformation_scale_edit.setVisible(True)

            # Check whether modal deformations file is loaded
            has_deforms = "modal_ux" in globals()

            # Enable/disable the scale factor based on whether deformation data is loaded
            if has_deforms:
                self.deformation_scale_edit.setEnabled(True)
                self.deformation_scale_edit.setText(
                    str(self.last_valid_deformation_scale))  # Restore last valid value (e.g., "1")
            else:
                self.deformation_scale_edit.setEnabled(False)
                self.deformation_scale_edit.setText("0")  # Set to "0" when disabled

            # Initialize plotter with points
            if "node_coords" in globals() and node_coords is not None:
                mesh = pv.PolyData(node_coords)
                if "df_node_ids" in globals() and df_node_ids is not None:
                    mesh["NodeID"] = df_node_ids.astype(int)
                mesh["Index"] = np.arange(mesh.n_points)

                self.current_mesh = mesh
                self.data_column = "Index"
                self.update_visualization()
                self.plotter.reset_camera()
        else:
            # Hide the controls if the required data is not available.
            self.anim_group.setVisible(False)
            self.time_point_group.setVisible(False)
            self.deformation_scale_label.setVisible(False)
            self.deformation_scale_edit.setVisible(False)

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
        compute_max_principal = main_tab.max_principal_stress_checkbox.isChecked()
        compute_min_principal = main_tab.min_principal_stress_checkbox.isChecked()
        compute_deformation = main_tab.deformation_checkbox.isChecked()
        compute_velocity   = main_tab.velocity_checkbox.isChecked()
        compute_acceleration = main_tab.acceleration_checkbox.isChecked()

        num_outputs_selected = sum([
            compute_von,
            compute_max_principal,
            compute_min_principal,
            compute_deformation,
            compute_velocity,
            compute_acceleration,
        ])
        if num_outputs_selected > 1:
            QMessageBox.warning(self, "Multiple Outputs Selected",
                                "Please select only one output type for time point visualization.")
            return

        if not (compute_von or compute_max_principal or compute_min_principal or compute_deformation or
                compute_velocity or compute_acceleration):
            QMessageBox.warning(self, "No Selection",
                                "No valid output is selected. Please select a valid output type to compute the results.")
            return

        # Get the selected time value and find the closest index in the global time_values array.
        selected_time = self.time_point_spinbox.value()
        time_index = np.argmin(np.abs(time_values - selected_time))

        # Get the number of modes to skip from the main tab ---
        main_tab = self.main_window.batch_solver_tab
        skip_n = 0
        if main_tab.deformations_checkbox.isChecked() and main_tab.deformation_loaded:
            try:
                skip_n = int(main_tab.skip_modes_combo.currentText())
            except (ValueError, TypeError):
                skip_n = 0
        mode_slice = slice(skip_n, None)

        # Handle deformations if they are defined, ensuring they are also sliced
        modal_deformations_filtered = None
        if 'modal_ux' in globals():
            modal_deformations_filtered = (
                modal_ux[:, mode_slice],
                modal_uy[:, mode_slice],
                modal_uz[:, mode_slice])

        # Assuming modal_coord shape is [num_modes, num_time_points], we extract a column vector.
        if compute_velocity or compute_acceleration:
            WINDOW = 7  # 3 pts either side + centre
            half = WINDOW // 2
            idx0 = max(0, time_index - half)
            idx1 = min(modal_coord.shape[1], time_index + half + 1)

            # need ≥2 samples – otherwise the derivative makes no sense
            if idx1 - idx0 < 2:
                QMessageBox.warning(
                    self, "Too few samples",
                    "Velocity/acceleration need at least two time steps. "
                    "Pick another time or load more results.")
                return

            # Slice the modal coordinate matrix for the chosen time point.
            selected_modal_coord = modal_coord[mode_slice, idx0:idx1]  # windowed block
            dt_window = time_values[idx0:idx1]  # matching time vector
            centre_offset = time_index - idx0  # column we actually want
        else:
            selected_modal_coord = modal_coord[mode_slice, time_index:time_index + 1]

        # Create a temporary solver instance using the sliced modal coordinate matrix.
        include_steady = self.main_window.batch_solver_tab.steady_state_checkbox.isChecked()
        try:
            if include_steady and "steady_sx" in globals() and steady_sx is not None and "steady_node_ids" in globals():
                temp_solver = MSUPSmartSolverTransient(
                    modal_sx[:, mode_slice],
                    modal_sy[:, mode_slice],
                    modal_sz[:, mode_slice],
                    modal_sxy[:, mode_slice],
                    modal_syz[:, mode_slice],
                    modal_sxz[:, mode_slice],
                    selected_modal_coord,
                    steady_sx, steady_sy, steady_sz, steady_sxy, steady_syz, steady_sxz,
                    steady_node_ids, modal_node_ids=df_node_ids,
                    modal_deformations=modal_deformations_filtered
                )
            else:
                temp_solver = MSUPSmartSolverTransient(
                    modal_sx[:, mode_slice],
                    modal_sy[:, mode_slice],
                    modal_sz[:, mode_slice],
                    modal_sxy[:, mode_slice],
                    modal_syz[:, mode_slice],
                    modal_sxz[:, mode_slice],
                    selected_modal_coord,
                    modal_node_ids=df_node_ids,
                    modal_deformations=modal_deformations_filtered
                )
        except Exception as e:
            QMessageBox.warning(self, "Solver Error", f"Failed to create solver instance: {e}")
            return

        # Determine the number of nodes (assumed from the shape of the modal stress arrays).
        num_nodes = modal_sx.shape[0]

        # Start with the original coordinates as the default
        display_coords = node_coords

        # Check if the user wants to display the deformed shape for this time point
        if main_tab.deformations_checkbox.isChecked() and 'modal_ux' in globals():
            # Get the displacements. The temp_solver already has the correct sliced data.
            # The result will be arrays of shape (num_nodes, 1) for this single time point.
            ux_tp, uy_tp, uz_tp = temp_solver.compute_deformations(0, num_nodes)

            # Get the scale factor from the UI
            try:
                scale_factor = float(self.deformation_scale_edit.text())
            except (ValueError, TypeError):
                scale_factor = 1.0  # Default to 1 if the input is invalid

            # If velocity/acceleration was requested, the displacement arrays (ux_tp, etc.)
            # will have multiple columns. We must select only the central column of the derivation window,
            # corresponding to the user's selected time point.
            if compute_velocity or compute_acceleration:
                # Use the already calculated centre_offset to pick the correct column
                ux_tp = ux_tp[:, [centre_offset]]
                uy_tp = uy_tp[:, [centre_offset]]
                uz_tp = uz_tp[:, [centre_offset]]

            # Create a single displacement vector of shape (num_nodes, 3)
            displacement_vector = np.hstack((ux_tp, uy_tp, uz_tp))

            # Calculate the new, deformed coordinates by adding the scaled displacement
            display_coords = node_coords + (displacement_vector * scale_factor)

        # Compute normal stresses for all nodes using the temporary solver.
        actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
            temp_solver.compute_normal_stresses(0, num_nodes)

        # Based on the selected stress type, compute the scalar field.
        if compute_von:
            scalar_field = temp_solver.compute_von_mises_stress(actual_sx, actual_sy, actual_sz,
                                                                actual_sxy, actual_syz, actual_sxz)
            field_name = "SVM"
            display_name = "SVM (MPa)"

        elif compute_max_principal:
            s1, s2, s3 = temp_solver.compute_principal_stresses(actual_sx, actual_sy, actual_sz,
                                                                actual_sxy, actual_syz, actual_sxz)
            scalar_field = s1  # Choose the maximum principal stress
            field_name = "S1"
            display_name = "S1 (MPa)"

        elif compute_min_principal:
            s1, s2, s3 = temp_solver.compute_principal_stresses(actual_sx, actual_sy, actual_sz,
                                                                actual_sxy, actual_syz, actual_sxz)
            scalar_field = s3  # Choose the minimum principal stress
            field_name = "S3"
            display_name = "S3 (MPa)"

        elif compute_deformation:
            # For single time point, ux_tp has shape (num_nodes, 1)
            scalar_field = np.sqrt(ux_tp ** 2 + uy_tp ** 2 + uz_tp ** 2)
            field_name = "Deformation"
            display_name = "Deformation (mm)"

        elif compute_velocity or compute_acceleration:
            if 'modal_ux' not in globals():
                QMessageBox.warning(self, "Missing Data",
                                    "Modal deformations must be loaded for requested calculation(s).")
                return

            # Get displacements for the *windowed* modal slice
            ux_blk, uy_blk, uz_blk = temp_solver.compute_deformations(0, num_nodes)

            # Run the 6-th-order scheme on that window (use the window for 6th order algorithm, computed above)
            vel_blk, acc_blk, _, _, _, _, _, _ = temp_solver._vel_acc_from_disp(
                ux_blk, uy_blk, uz_blk, dt_window.astype(temp_solver.NP_DTYPE))

            vel_tp = vel_blk[:, centre_offset]  # pick centre column
            acc_tp = acc_blk[:, centre_offset]

            # Choose the result asked by user
            scalar_field = vel_tp if compute_velocity else acc_tp
            field_name = "Velocity" if compute_velocity else "Acceleration"
            display_name = "Velocity (mm/s)" if compute_velocity else "Acceleration (mm/s²)"
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
            mesh = pv.PolyData(display_coords)

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

            self.plotter.render()
            self.update_visualization()

            # Clear the external file path since we are generating a new view.
            self.file_path.clear()

        else:
            QMessageBox.warning(self, "Missing Data", "Node coordinates are not available.")

    def save_time_point_results(self):
        """
        Saves the currently displayed results (node coordinates and the computed scalar field)
        into a CSV file. The saved column name is aware of the currently displayed output type.
        """
        # 1. Check if there is a mesh with data to save.
        if self.current_mesh is None:
            QMessageBox.warning(self, "No Data", "No visualization data to save.")
            return

        # 2. Get the name of the currently active data array from the mesh object.
        active_scalar_name = self.current_mesh.active_scalars_name
        if not active_scalar_name:
            QMessageBox.warning(self, "No Active Data",
                                "The current mesh does not have an active scalar field to save.")
            return

        # 3. Create a smart, descriptive default filename.
        base_name = active_scalar_name.split(' ')[0]  # Extracts "SVM", "S1", "Velocity", etc.
        selected_time = self.time_point_spinbox.value()
        # Format the filename, replacing illegal characters like '.' in the time value
        default_filename = f"{base_name}_T_{selected_time:.5f}.csv".replace('.', '_')

        # 4. Open the file dialog with the suggested filename.
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Time Point Results", default_filename,
                                                   "CSV Files (*.csv)")

        if not file_name:
            return  # User cancelled the dialog

        try:
            # 5. Get all necessary data arrays from the mesh.
            coords = self.current_mesh.points
            scalar_data = self.current_mesh[active_scalar_name]

            # 6. Create the output DataFrame, starting with NodeID if available.
            df_out = pd.DataFrame()
            if 'NodeID' in self.current_mesh.array_names:
                df_out['NodeID'] = self.current_mesh['NodeID']

            # 7. Add the coordinate and scalar data with their correct headers.
            df_out['X'] = coords[:, 0]
            df_out['Y'] = coords[:, 1]
            df_out['Z'] = coords[:, 2]
            df_out[active_scalar_name] = scalar_data

            # 8. Save the complete DataFrame to the chosen CSV file.
            df_out.to_csv(file_name, index=False)
            QMessageBox.information(self, "Save Successful", f"Time point results saved successfully to:\n{file_name}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the file: {e}")

    def update_anim_range_min(self, value):
        # Ensure that the animation end spin box cannot be set to a value less than the start
        self.anim_end_spin.setMinimum(value)

    def update_anim_range_max(self, value):
        # Ensure that the animation start spin box cannot exceed the end value
        self.anim_start_spin.setMaximum(value)

    def validate_deformation_scale(self):
        """
        Validate the deformation scale factor input.
        If the input can be converted to a float, update the last valid value.
        Otherwise, revert the text to the last valid input.
        """
        text = self.deformation_scale_edit.text()
        try:
            value = float(text)
            self.last_valid_deformation_scale = value
        except ValueError:
            # Revert to the last valid input if the current text is invalid.
            self.deformation_scale_edit.setText(str(self.last_valid_deformation_scale))

    def start_animation(self):
        """Start animating through the time range by precomputing frames."""
        global time_values, modal_coord, modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz
        global node_coords, df_node_ids
        global modal_ux, modal_uy, modal_uz  # For deformations

        if self.current_mesh is None:
            QMessageBox.warning(self, "No Data",
                                "Please load or initialize the mesh before animating (e.g., by updating results for a single time point).")
            return

        # region Resume Logic
        if self.animation_paused:
            if self.precomputed_scalars is None:
                QMessageBox.warning(self, "Resume Error",
                                    "Cannot resume animation. Precomputed data is missing. Please stop and start again.")
                self.stop_animation()  # Force stop state
                return
            print("Resuming animation...")
            self.animation_paused = False
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.deformation_scale_edit.setEnabled(False)  # Keep disabled during animation
            if self.anim_timer:
                self.anim_timer.start(self.anim_interval_spin.value())
            else:  # Should not happen, but safety check
                self.anim_timer = QTimer(self)
                self.anim_timer.timeout.connect(self.animate_frame)
                self.anim_timer.start(self.anim_interval_spin.value())
            return
        # endregion

        # region Start Fresh Logic
        print("\n ---Starting animation precomputation...---")
        self.stop_animation()  # Ensure clean state (clears previous data, resets index)
        # endregion

        # region 1. Get Animation Parameters & Time Steps ---
        anim_times, anim_indices, error_msg = self._get_animation_time_steps()
        if error_msg:
            QMessageBox.warning(self, "Animation Setup Error", error_msg)
            self.stop_animation()  # Ensure UI is reset
            return
        if anim_times is None or len(anim_times) == 0:
            QMessageBox.warning(self, "Animation Setup Error", "No time steps generated for the animation.")
            self.stop_animation()
            return

        num_anim_steps = len(anim_times)
        print(f"Attempting to precompute {num_anim_steps} frames.")
        # endregion

        # region 2. Determine Required Outputs
        main_tab = self.window().batch_solver_tab  # Access main tab for settings
        compute_von = main_tab.von_mises_checkbox.isChecked()
        compute_max_principal = main_tab.max_principal_stress_checkbox.isChecked()
        compute_min_principal = main_tab.min_principal_stress_checkbox.isChecked()
        compute_deformation_anim = main_tab.deformations_checkbox.isChecked()  # This is for MOVING the nodes
        compute_deformation_contour = main_tab.deformation_checkbox.isChecked()  # This is for COLORING the nodes
        compute_velocity = main_tab.velocity_checkbox.isChecked()
        compute_acceleration = main_tab.acceleration_checkbox.isChecked()

        if not (compute_von or compute_max_principal or compute_min_principal or
                compute_deformation_contour or compute_velocity or compute_acceleration):
            QMessageBox.warning(self, "No Selection",
                                "No valid output is selected in the Main Window tab for animation.")
            self.stop_animation()
            return

        if compute_deformation_anim and 'modal_ux' not in globals():
            QMessageBox.warning(self, "Deformation Error",
                                "Deformation checkbox is checked, but modal deformation data (ux, uy, uz) is not loaded.")
            compute_deformation_anim = False  # Disable deformation if data missing
            self.is_deformation_included_in_anim = False
        elif compute_deformation_anim:
            self.is_deformation_included_in_anim = True
        else:
            self.is_deformation_included_in_anim = False
        # endregion

        # region 3. RAM Estimation and Check
        num_nodes = modal_sx.shape[0]
        estimated_gb = self._estimate_animation_ram(num_nodes, num_anim_steps, compute_deformation_anim)
        my_virtual_memory = psutil.virtual_memory()
        available_gb = my_virtual_memory.available / (1024 ** 3)
        # Use a safety factor (e.g., allow using up to 80% of available RAM)
        safe_available_gb = available_gb * RAM_PERCENT

        print(f"Estimated RAM for precomputation: {estimated_gb:.3f} GB")
        print(f"Available system RAM: {available_gb:.3f} GB (Safe threshold: {safe_available_gb:.3f} GB)")

        if estimated_gb > safe_available_gb:
            # Calculate max possible steps
            ram_per_step_gb = estimated_gb / num_anim_steps
            max_steps = int(safe_available_gb // ram_per_step_gb) if ram_per_step_gb > 0 else 0

            if max_steps > 1:
                # Try to estimate max time based on current step settings
                if self.time_step_mode_combo.currentText() == "Custom Time Step":
                    max_time_est = self.anim_start_spin.value() + max_steps * self.custom_step_spin.value()
                    suggestion = f"Try reducing the end time to around {max_time_est:.4f}s or increasing the custom step."
                else:
                    # Estimate based on average interval of actual data within the original range
                    avg_dt = np.mean(np.diff(anim_times)) if len(anim_times) > 1 else 0
                    max_time_est = self.anim_start_spin.value() + max_steps * avg_dt * self.actual_interval_spin.value() if avg_dt > 0 else self.anim_end_spin.value()
                    suggestion = f"Try reducing the end time to around {max_time_est:.4f}s or increasing the 'Every nth' value."

                QMessageBox.warning(self, "Insufficient Memory",
                                    f"Estimated RAM required ({estimated_gb:.3f} GB) exceeds the safe available RAM ({safe_available_gb:.3f} GB).\n\n"
                                    f"The current settings would require precomputing {num_anim_steps} frames.\n"
                                    f"With available memory, you can precompute approximately {max_steps} frames.\n\n"
                                    f"{suggestion}\n\n"
                                    "Please adjust the time range or step/interval and try again.")
            else:
                QMessageBox.warning(self, "Insufficient Memory",
                                    f"Estimated RAM required ({estimated_gb:.3f} GB) exceeds the safe available RAM ({safe_available_gb:.3f} GB), even for a minimal number of frames.\n\n"
                                    "Cannot start animation. Please check system resources or reduce model complexity if possible.")

            self.stop_animation()
            return
        # endregion

        # region 4. Perform Precomputation
        QApplication.setOverrideCursor(Qt.WaitCursor)  # Show busy cursor
        try:
            # Get the number of modes to skip from the main tab ---
            main_tab = self.window().batch_solver_tab
            skip_n = 0
            if main_tab.deformations_checkbox.isChecked() and main_tab.deformation_loaded:
                try:
                    skip_n = int(main_tab.skip_modes_combo.currentText())
                except (ValueError, TypeError):
                    skip_n = 0
            mode_slice = slice(skip_n, None)

            # Slice the required modal coordinates correctly
            selected_modal_coord = modal_coord[mode_slice, anim_indices]

            # Check if steady-state stress is included
            include_steady = self.main_window.batch_solver_tab.steady_state_checkbox.isChecked()
            steady_kwargs = {}
            if include_steady and "steady_sx" in globals() and steady_sx is not None and "steady_node_ids" in globals():
                steady_kwargs = {
                    'steady_sx': steady_sx, 'steady_sy': steady_sy, 'steady_sz': steady_sz,
                    'steady_sxy': steady_sxy, 'steady_syz': steady_syz, 'steady_sxz': steady_sxz,
                    'steady_node_ids': steady_node_ids
                }

            # Handle deformations, ensuring they are also sliced
            modal_deformations_arg = None
            if compute_deformation_anim:
                if 'modal_ux' in globals():
                    modal_deformations_filtered = (
                        modal_ux[:, mode_slice],
                        modal_uy[:, mode_slice],
                        modal_uz[:, mode_slice]
                    )

            # Create a temporary solver instance for the *entire* animation duration
            temp_solver = MSUPSmartSolverTransient(
                modal_sx[:, mode_slice], modal_sy[:, mode_slice], modal_sz[:, mode_slice],
                modal_sxy[:, mode_slice], modal_syz[:, mode_slice], modal_sxz[:, mode_slice],
                selected_modal_coord,  # Use the sliced coordinates
                modal_node_ids=df_node_ids,
                modal_deformations=modal_deformations_filtered,
                **steady_kwargs  # Add steady state args if needed
            )

            # Compute normal stresses for all nodes, all required time steps
            print("Computing normal stresses for animation...")
            actual_sx_anim, actual_sy_anim, actual_sz_anim, actual_sxy_anim, actual_syz_anim, actual_sxz_anim = \
                temp_solver.compute_normal_stresses(0, num_nodes)  # Shape: (num_nodes, num_anim_steps)

            # Compute the selected scalar field
            print("Computing scalar field for animation...")
            if compute_von:
                self.precomputed_scalars = temp_solver.compute_von_mises_stress(
                    actual_sx_anim, actual_sy_anim, actual_sz_anim,
                    actual_sxy_anim, actual_syz_anim, actual_sxz_anim)
                self.data_column_name = "SVM (MPa)"

            elif compute_max_principal:
                s1_anim, _, _ = temp_solver.compute_principal_stresses(
                    actual_sx_anim, actual_sy_anim, actual_sz_anim,
                    actual_sxy_anim, actual_syz_anim, actual_sxz_anim)
                self.precomputed_scalars = s1_anim
                self.data_column_name = "S1 (MPa)"

            elif compute_min_principal:
                _, _, s3_anim = temp_solver.compute_principal_stresses(
                    actual_sx_anim, actual_sy_anim, actual_sz_anim,
                    actual_sxy_anim, actual_syz_anim, actual_sxz_anim)
                self.precomputed_scalars = s3_anim
                self.data_column_name = "S3 (MPa)"

            elif compute_velocity or compute_acceleration or compute_deformation_contour:
                # Need modal deformations
                if 'modal_ux' not in globals():
                    QMessageBox.warning(self, "Deformation Error",
                                        "Deformation/Velocity/Acceleration requested but modal deformations are not loaded.")
                    self.stop_animation();
                    return

                ux_anim, uy_anim, uz_anim = temp_solver.compute_deformations(0, num_nodes)

                if compute_deformation_contour:
                    self.precomputed_scalars = np.sqrt(ux_anim ** 2 + uy_anim ** 2 + uz_anim ** 2)
                    self.data_column_name = "Deformation (mm)"

                if compute_velocity or compute_acceleration:
                    vel_mag_anim, acc_mag_anim, _, _, _, _, _, _ = temp_solver._vel_acc_from_disp(
                        ux_anim, uy_anim, uz_anim, anim_times.astype(temp_solver.NP_DTYPE))
                    if compute_velocity:
                        self.precomputed_scalars = vel_mag_anim
                        self.data_column_name = "Velocity (mm/s)"
                    else:  # acceleration
                        self.precomputed_scalars = acc_mag_anim
                        self.data_column_name = "Acceleration (mm/s²)"

            # Compute deformations if needed
            if compute_deformation_anim:
                print("Computing deformations for animation...")
                deformations_anim = temp_solver.compute_deformations(0,
                                                                     num_nodes)  # (ux, uy, uz) each (num_nodes, num_anim_steps)
                if deformations_anim is not None:
                    ux_anim, uy_anim, uz_anim = deformations_anim
                    scale_factor = float(self.deformation_scale_edit.text())
                    # Calculate absolute deformed coordinates: original + scaled displacement
                    # Reshape original coords to broadcast: (num_nodes, 3, 1)
                    original_coords_reshaped = node_coords[:, :, np.newaxis]

                    # Zero‐offset so start shape is exact
                    ux_anim = ux_anim - ux_anim[:, [0]]
                    uy_anim = uy_anim - uy_anim[:, [0]]
                    uz_anim = uz_anim - uz_anim[:, [0]]

                    # Stack displacements: (num_nodes, 3, num_anim_steps)
                    displacements_stacked = np.stack([ux_anim, uy_anim, uz_anim], axis=1)
                    # Store final coordinates
                    self.precomputed_coords = original_coords_reshaped + scale_factor * displacements_stacked
                else:
                    print("Warning: Deformation computation failed unexpectedly.")
                    self.is_deformation_included_in_anim = False

            # Store the time values corresponding to the computed frames
            self.precomputed_anim_times = anim_times

            print("Cleaning up temporary animation data...")
            del temp_solver
            del actual_sx_anim, actual_sy_anim, actual_sz_anim, actual_sxy_anim, actual_syz_anim, actual_sxz_anim
            if compute_von: pass  # Scalars already stored
            if compute_max_principal: del s1_anim
            if compute_min_principal: del s3_anim
            if compute_deformation_anim and deformations_anim is not None:
                del deformations_anim, ux_anim, uy_anim, uz_anim, displacements_stacked, original_coords_reshaped
            gc.collect()
            print(" ---Precomputation complete.---")
            self.save_anim_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Precomputation Error", f"Failed to precompute animation frames: {str(e)}")
            self.stop_animation()  # Ensure cleanup on error
            QApplication.restoreOverrideCursor()  # Restore cursor
            return
        finally:
            QApplication.restoreOverrideCursor()  # Restore cursor
        # endregion

        # --- 5. Final Setup & Start Timer ---
        self.current_anim_frame_index = 0
        self.animation_paused = False  # Ensure flag is reset

        # Update the mesh with the first frame's data before starting timer
        # We probably don't want to include this first update in the loop profile
        try:
            self.animate_frame(update_index=False)
        except Exception as e:
            QMessageBox.critical(self, "Animation Error", f"Failed initial frame render: {str(e)}")
            self.stop_animation()  # Stop if initial render fails
            return

        # Now start the timer for subsequent frames
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.animate_frame)
        self.anim_timer.start(self.anim_interval_spin.value())

        # Update UI state
        self.deformation_scale_edit.setEnabled(False)  # Disable editing during animation
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

        # Ensure save button remains enabled if precomputation succeeded
        if self.precomputed_scalars is not None:
            self.save_anim_button.setEnabled(True)  # Redundant check, but safe

        # Remove static time text if it exists from previous single time point updates
        if hasattr(self, 'time_text_actor') and self.time_text_actor is not None:
            self.plotter.remove_actor(self.time_text_actor)
            self.time_text_actor = None

        # Clear the external file path since we are generating a new view.
        self.file_path.clear()

    def pause_animation(self):
        """Pause the animation (resumes from the current frame when Play is clicked)."""
        if self.anim_timer is not None and self.anim_timer.isActive():
            self.anim_timer.stop()
            self.animation_paused = True  # Set the flag
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            print("\nAnimation paused.")
        else:
            print("\nPause command ignored: Animation timer not active.")

    def stop_animation(self):
        """Stop the animation, release precomputed data, and reset state."""
        # MODIFICATION: Check if there is an animation to stop before printing.
        is_stoppable = self.anim_timer is not None or self.precomputed_scalars is not None

        if is_stoppable:
            print("\nStopping animation and releasing resources...")

        if self.anim_timer is not None:
            self.anim_timer.stop()
            # Optional: disconnect to be sure it doesn't trigger again accidentally
            try:
                self.anim_timer.timeout.disconnect(self.animate_frame)
            except TypeError:  # Already disconnected
                pass
            self.anim_timer = None  # Allow timer to be garbage collected

        # --- Release Precomputed Data ---
        print(" ")
        if self.precomputed_scalars is not None:
            del self.precomputed_scalars
            self.precomputed_scalars = None
            print("Released precomputed scalars.")
        if self.precomputed_coords is not None:
            del self.precomputed_coords
            self.precomputed_coords = None
            print("Released precomputed coordinates.")
        if self.precomputed_anim_times is not None:
            del self.precomputed_anim_times
            self.precomputed_anim_times = None
            print("Released precomputed times.")

        # Explicitly trigger garbage collection
        gc.collect()
        # --- End Release ---

        # Reset state variables
        self.current_anim_frame_index = 0
        self.animation_paused = False
        self.is_deformation_included_in_anim = False

        # Reset UI elements
        self.deformation_scale_edit.setEnabled(True)  # Re-enable editing
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.save_anim_button.setEnabled(False)

        # Remove the animation time text actor
        if hasattr(self, 'time_text_actor') and self.time_text_actor is not None:
            self.plotter.remove_actor(self.time_text_actor)
            self.time_text_actor = None

        # Optional: Reset mesh to original state (if node_coords exist)
        if self.current_mesh and 'node_coords' in globals() and node_coords is not None:
            print("Resetting mesh to original coordinates.")
            try:
                # Check if the mesh still has points data assigned
                if self.current_mesh.points is not None:
                    # Only reset if the number of points matches
                    if self.current_mesh.n_points == node_coords.shape[0]:
                        self.current_mesh.points = node_coords.copy()  # Use copy to be safe
                        # Optionally reset scalars to 0 or initial state if desired
                        # self.current_mesh[self.data_column_name] = np.zeros(self.current_mesh.n_points)
                        self.plotter.render()  # Render the reset state
                    else:
                        print("Warning: Cannot reset mesh points, point count mismatch.")
                else:
                    print("Warning: Cannot reset mesh points, mesh points data is missing.")
            except Exception as e:
                print(f"Error resetting mesh points: {e}")

        if is_stoppable:
            print("\nAnimation stopped.")

    def animate_frame(self, update_index=True):
        """Update the display using the next precomputed animation frame."""
        # --- Use helper to update mesh ---
        # Check if data exists before calling helper
        if self.precomputed_scalars is None or self.precomputed_anim_times is None:
            print("Animation frame skipped: Precomputed data not available.")
            self.stop_animation()
            return

        if not self._update_mesh_for_frame(self.current_anim_frame_index):
            print(f"Animation frame skipped: Failed to update mesh for index {self.current_anim_frame_index}.")
            # Attempt to stop gracefully if data seems inconsistent
            self.stop_animation()
            return

        # --- Render ---
        # Render happens *after* mesh update now
        self.plotter.render()

        # --- Increment Frame Index for Next Call ---
        if update_index:
            num_frames = len(self.precomputed_anim_times) if self.precomputed_anim_times is not None else 0
            if num_frames > 0:
                self.current_anim_frame_index += 1
                if self.current_anim_frame_index >= num_frames:
                    self.current_anim_frame_index = 0  # Loop animation
            else:
                # Should not happen if called correctly, but safety check
                self.stop_animation()

    def _update_mesh_for_frame(self, frame_index):
        """Updates the mesh data (scalars and optionally points) for a given frame index."""
        if self.precomputed_scalars is None or self.precomputed_anim_times is None or self.current_mesh is None:
            print("Error: Cannot update mesh - precomputed data or mesh missing.")
            return False

        num_frames = len(self.precomputed_anim_times)
        if frame_index < 0 or frame_index >= num_frames:
            print(f"Error: Invalid frame index {frame_index} for {num_frames} frames.")
            return False  # Invalid index

        try:
            current_scalars = self.precomputed_scalars[:, frame_index]
            current_time = self.precomputed_anim_times[frame_index]

            # Get deformed coordinates if available and enabled
            if self.is_deformation_included_in_anim and self.precomputed_coords is not None:
                if frame_index >= self.precomputed_coords.shape[2]:
                    print(f"Error: Frame index {frame_index} out of bounds for precomputed coordinates.")
                    return False
                current_coords = self.precomputed_coords[:, :, frame_index]
                # Ensure mesh object still exists
                if self.current_mesh is not None:
                    self.current_mesh.points = current_coords  # Update node positions
                else:
                    print("Error: Current mesh is None, cannot update points.")
                    return False

            # Update scalars on the mesh
            if self.current_mesh is not None:
                self.current_mesh[self.data_column_name] = current_scalars
                # Ensure the active scalar is set correctly
                if self.current_mesh.active_scalars_name != self.data_column_name:
                    self.current_mesh.set_active_scalars(self.data_column_name)
            else:
                print("Error: Current mesh is None, cannot update scalars.")
                return False

            # Update the scalar bar range if necessary (using fixed range from UI)
            fixed_min = self.scalar_min_spin.value()
            fixed_max = self.scalar_max_spin.value()
            if self.current_actor and hasattr(self.current_actor, 'mapper') and self.current_actor.mapper:
                # Check if range needs setting
                current_range = self.current_actor.mapper.GetScalarRange()
                # Use a tolerance for float comparison
                if abs(current_range[0] - fixed_min) > 1e-6 or abs(current_range[1] - fixed_max) > 1e-6:
                    self.current_actor.mapper.SetScalarRange(fixed_min, fixed_max)
                    # Update scalar bar title if needed
                    if hasattr(self.plotter, 'scalar_bar') and self.plotter.scalar_bar:
                        self.plotter.scalar_bar.SetTitle(self.data_column_name)

            # --- Update Time Text ---
            time_text = f"Time: {current_time:.5f} s"
            if hasattr(self, 'time_text_actor') and self.time_text_actor is not None:
                # Check if actor still exists in renderer before trying to set input
                try:
                    # VTKPythonCore could throw errors if the underlying VTK object is deleted
                    if self.time_text_actor.GetViewProps() is not None:
                        self.time_text_actor.SetInput(time_text)
                    else:  # Underlying object gone, recreate
                        # Safer to attempt removal by ref first
                        self.plotter.remove_actor(self.time_text_actor, render=False)
                        self.time_text_actor = self.plotter.add_text(time_text, position=(0.8, 0.9), viewport=False,
                                                                     font_size=10)
                except (
                        AttributeError,
                        ReferenceError):  # Actor might have been garbage collected or VTK object deleted
                    # Attempt removal if reference still exists
                    try:
                        self.plotter.remove_actor(self.time_text_actor, render=False)
                    except:
                        pass
                    self.time_text_actor = self.plotter.add_text(time_text, position=(0.8, 0.9), viewport=False,
                                                                 font_size=10)
            else:
                self.time_text_actor = self.plotter.add_text(time_text, position=(0.8, 0.9), viewport=False,
                                                             font_size=10)

            return True

        except IndexError as e:
            print(f"Error: Index {frame_index} out of bounds during mesh update. {e}")
            return False
        except Exception as e:
            # Log the error type and message
            print(f"Error updating mesh for frame {frame_index}: {type(e).__name__}: {e}")
            # Optionally show a QMessageBox here for critical errors
            # QMessageBox.critical(self, "Animation Error", f"Failed to update mesh for frame {frame_index}: {str(e)}")
            return False

    def _capture_animation_frame(self, frame_index):
        """Updates the plotter for the given frame index and returns a screenshot (NumPy array)."""
        # Update mesh data, scalar bar, time text for the target frame
        if not self._update_mesh_for_frame(frame_index):
            print(f"Warning: Failed to update mesh for frame {frame_index} before capture.")
            return None  # Indicate failure

        # Render the scene *after* updating
        self.plotter.render()
        # Allow the event loop to process the render command, might help prevent blank frames
        QApplication.processEvents()
        time.sleep(0.01)  # Small delay, sometimes helps ensure rendering completes before screenshot

        # Capture screenshot
        try:
            # Use window_size=None to capture the current interactive window size
            # Ensure plotter and renderer are valid
            if self.plotter and self.plotter.renderer:
                frame_image = self.plotter.screenshot(transparent_background=False, return_img=True, window_size=None)
                if frame_image is None:
                    print(f"Warning: Screenshot returned None for frame {frame_index}.")
                    return None
                return frame_image
            else:
                print(f"Error: Plotter or renderer invalid for frame {frame_index}.")
                return None
        except Exception as e:
            print(f"Error taking screenshot for frame {frame_index}: {type(e).__name__}: {e}")
            return None

    def save_animation(self):
        """Saves the precomputed animation frames as a video (MP4) or GIF."""
        if self.precomputed_scalars is None or self.precomputed_anim_times is None:
            QMessageBox.warning(self, "Cannot Save", "Animation data must be precomputed first (click Play).")
            return

        num_frames = len(self.precomputed_anim_times)
        if num_frames == 0:
            QMessageBox.warning(self, "Cannot Save", "No frames were precomputed.")
            return

        # --- File Dialog ---
        options = QFileDialog.Options()
        # Use project directory if available in the main window
        default_dir = ""
        if hasattr(self.window(), 'project_directory') and self.window().project_directory:
            default_dir = self.window().project_directory
        elif self.file_path.text():  # Fallback to directory of loaded viz file
            default_dir = os.path.dirname(self.file_path.text())

        fileName, selectedFilter = QFileDialog.getSaveFileName(self,
                                                               "Save Animation", default_dir,
                                                               "MP4 Video (*.mp4);;Animated GIF (*.gif)",
                                                               "MP4 Video (*.mp4)",  # Default filter
                                                               options=options)
        if not fileName:
            return  # User cancelled

        # Determine format and ensure correct extension
        file_format = ""
        if selectedFilter == "MP4 Video (*.mp4)":
            file_format = "MP4"
            if not fileName.lower().endswith(".mp4"):
                fileName += ".mp4"
        elif selectedFilter == "Animated GIF (*.gif)":
            file_format = "GIF"
            if not fileName.lower().endswith(".gif"):
                fileName += ".gif"
        else:  # If filter somehow is unexpected, try deriving from extension
            if fileName.lower().endswith(".mp4"):
                file_format = "MP4"
            elif fileName.lower().endswith(".gif"):
                file_format = "GIF"
            else:  # Add default extension if none provided and filter unknown
                QMessageBox.warning(self, "Cannot Determine Format",
                                    "Could not determine file format. Please use .mp4 or .gif extension.")
                # Defaulting to MP4, force extension
                # file_format = "MP4"
                # if not fileName.lower().endswith(".mp4"): fileName += ".mp4"
                return

        fps = 1000.0 / self.anim_interval_spin.value()
        print(f"---Saving animation...---")
        print(f"Attempting to save {num_frames} frames to '{fileName}' as {file_format} at {fps:.2f} FPS.")

        # --- Progress Dialog ---
        progress = QProgressDialog("Saving animation frames...", "Cancel", 0, num_frames,
                                   self.window())  # Parent to main window
        progress.setWindowModality(Qt.WindowModal)  # Block interaction with main window
        progress.setWindowTitle("Saving Animation")
        progress.setMinimumDuration(1000)  # Show only if saving takes > 1 second
        progress.setValue(0)
        # Don't call progress.show() yet, wait until first frame attempt

        # --- Store original state to restore later ---
        original_frame_index = self.current_anim_frame_index
        original_is_paused = self.animation_paused
        was_timer_active = self.anim_timer is not None and self.anim_timer.isActive()

        # Pause the live animation timer if it's running
        if was_timer_active:
            self.anim_timer.stop()
            print("Live animation timer paused for saving.")

        # --- Saving Loop (using imageio writer for memory efficiency) ---
        cancelled = False
        writer = None  # Initialize writer to None
        try:
            # Prepare writer arguments
            writer_kwargs = {'fps': fps}
            if file_format == 'MP4':
                # pixelformat is crucial for MP4 compatibility (like QuickTime)
                # quality can be set (0-10, 10 is highest, default is often 5)
                writer_kwargs.update({'macro_block_size': None, 'pixelformat': 'yuv420p', 'quality': 7})
            elif file_format == 'GIF':
                # loop=0 means infinite loop
                # palettesize can affect colors/size
                # subrectangles=True might optimize for smaller GIFs if only parts change (unlikely here)
                writer_kwargs.update({'macro_block_size': None, 'loop': 0, 'palettesize': 256})

            # --- Preserve Camera State ---
            # GetState() returns a dictionary or tuple, depending on VTK version
            initial_camera_state = None
            if self.plotter and self.plotter.camera:
                # Let's try the dictionary method first, common in newer PyVista/VTK
                try:
                    initial_camera_state = self.plotter.camera.GetState()
                    print("Saved camera state (dict/tuple).")
                except AttributeError:  # Fallback for older versions possibly returning tuple directly from position etc.
                    initial_camera_state = (
                        self.plotter.camera.position,
                        self.plotter.camera.focal_point,
                        self.plotter.camera.up
                    )
                    print("Saved camera state (pos/focal/up).")

            # Start the writer *before* the loop
            writer = imageio.get_writer(fileName, format=file_format, mode='I',
                                        **writer_kwargs)  # mode='I' for multiple images
            progress.show()  # Show progress dialog now writer is ready

            for i in range(num_frames):
                if progress.wasCanceled():
                    cancelled = True
                    print("Save cancelled by user.")
                    break

                # --- Restore Camera State before capturing each frame ---
                if initial_camera_state is not None and self.plotter and self.plotter.camera:
                    try:
                        if isinstance(initial_camera_state, dict):
                            self.plotter.camera.SetState(initial_camera_state)
                        elif isinstance(initial_camera_state, tuple) and len(
                                initial_camera_state) == 3:  # pos/focal/up tuple
                            self.plotter.camera.position = initial_camera_state[0]
                            self.plotter.camera.focal_point = initial_camera_state[1]
                            self.plotter.camera.up = initial_camera_state[2]
                        # No else needed, if it's not recognized, we just don't restore
                    except Exception as cam_err:
                        print(f"Warning: Could not restore camera state for frame {i}: {cam_err}")

                # Capture the frame using the helper function
                frame_image = self._capture_animation_frame(i)

                if frame_image is None:  # Handle potential errors during capture
                    # Option 1: Skip frame (video might look glitchy)
                    print(f"Warning: Skipping frame {i} due to capture failure.")
                    # Option 2: Abort saving
                    # raise RuntimeError(f"Failed to capture frame {i}")
                    # Let's skip for now, user can retry if it looks bad
                    progress.setValue(i + 1)  # Still update progress
                    QApplication.processEvents()
                    continue  # Go to next frame

                writer.append_data(frame_image)  # Append frame to file
                progress.setValue(i + 1)
                QApplication.processEvents()  # Keep UI responsive, update progress

        except ImportError as e:
            # Specific error for missing backend
            error_msg = f"ImportError: {e}. Cannot save animation.\n\n"
            if file_format == 'MP4':
                error_msg += "Saving MP4 requires 'ffmpeg'. Please install it.\nTry: pip install imageio[ffmpeg]"
            else:
                error_msg += "Ensure 'imageio' is installed correctly."
            QMessageBox.critical(self, "Missing Dependency", error_msg)
            print(error_msg)
            cancelled = True  # Treat as cancellation
        except Exception as e:
            error_msg = f"Failed to save animation:\n{type(e).__name__}: {e}\n\n"
            error_msg += "Check write permissions for the directory.\n"
            if file_format == 'MP4':
                error_msg += "Ensure 'ffmpeg' is installed and accessible.\n"
            error_msg += "Check console output for more details."
            QMessageBox.critical(self, "Save Error", error_msg)
            print(f"Imageio save error: {type(e).__name__}: {e}")  # Log detailed error
            cancelled = True  # Treat error as cancellation for cleanup logic
        finally:
            # --- Cleanup ---
            if writer is not None:
                try:
                    writer.close()  # Ensure writer is closed
                    print("Imageio writer closed.")
                except Exception as close_err:
                    print(f"Error closing imageio writer: {close_err}")

            progress.close()  # Ensure progress dialog is closed

            # --- Restore original animation state ---
            print("Restoring plotter state...")
            # Restore mesh/plotter to the state it was in before saving started
            # Use a try-except block for robustness
            try:
                self._update_mesh_for_frame(original_frame_index)
                self.plotter.render()  # Render the restored state
                print(f"Restored view to frame {original_frame_index}.")
            except Exception as restore_err:
                print(f"Warning: Could not fully restore plotter state: {restore_err}")

            # Restore live animation timer if it was running *and* wasn't paused originally
            if was_timer_active and not original_is_paused:
                # Check if timer still exists (might be None if stop_animation was called)
                if self.anim_timer:
                    self.anim_timer.start(self.anim_interval_spin.value())
                    print("Live animation timer restarted.")
                else:  # Recreate timer if needed (edge case)
                    print("Recreating live animation timer.")
                    self.anim_timer = QTimer(self)
                    self.anim_timer.timeout.connect(self.animate_frame)
                    self.anim_timer.start(self.anim_interval_spin.value())
            elif was_timer_active and original_is_paused:
                print("Leaving live animation timer paused (was paused before saving).")

            # Ensure paused state is correct
            self.animation_paused = original_is_paused

            # --- Clean up potentially incomplete/cancelled file ---
            if cancelled and os.path.exists(fileName):
                try:
                    print(f"Attempting to remove cancelled/incomplete file: {fileName}")
                    os.remove(fileName)
                    print("File removed.")
                except OSError as remove_error:
                    print(f"Could not remove cancelled/incomplete file: {remove_error}")

        # --- Final Feedback ---
        if not cancelled:
            QMessageBox.information(self, "Save Successful", f"Animation successfully saved to:\n{fileName}")
            print("---Animation saving process finished.---\n")
        else:
            # Message box already shown for error, only show warning for user cancellation
            if not progress.wasCanceled():  # i.e., cancelled due to an error
                pass  # Error message already shown
            else:  # Cancelled by user clicking button
                QMessageBox.warning(self, "Save Cancelled", "Animation saving was cancelled by user.")
            print("Animation saving process aborted.")

    def _get_animation_time_steps(self):
        """
        Determines the time values and corresponding indices from global time_values
        needed for the animation based on user settings.

        Returns:
            tuple: (
                anim_times: np.array or None - The actual time values for each animation frame.
                anim_indices: np.array or None - The indices in global time_values corresponding to anim_times.
                error_message: str or None - An error message if inputs are invalid or no steps generated, otherwise None.
            )
        """
        global time_values
        # --- Input Validation ---
        if time_values is None or len(time_values) == 0:
            return None, None, "Global time_values not loaded or empty."

        start_time = self.anim_start_spin.value()
        end_time = self.anim_end_spin.value()

        if start_time >= end_time:
            return None, None, "Animation start time must be less than end time."

        # Initialize as standard Python lists
        anim_times_list = []
        anim_indices_list = []

        # --- Logic based on Time Step Mode ---
        if self.time_step_mode_combo.currentText() == "Custom Time Step":
            step = self.custom_step_spin.value()
            if step <= 0:
                return None, None, "Custom time step must be positive."

            current_t = start_time
            last_added_idx = -1  # Keep track of the last index added

            # Loop through custom time steps
            while current_t <= end_time:
                # Find the index of the closest actual time point in the data
                idx = np.argmin(np.abs(time_values - current_t))

                # Ensure the found index corresponds to a time within the overall bounds
                # (Handles cases where closest time might be outside start/end due to large steps)
                # And ensure we don't add duplicate indices consecutively
                if time_values[idx] >= start_time and time_values[idx] <= end_time and idx != last_added_idx:
                    anim_indices_list.append(idx)
                    anim_times_list.append(time_values[idx])  # Use the actual data time point
                    last_added_idx = idx  # Update last added index

                # Prevent infinite loops for very small steps, break if time doesn't advance significantly
                if current_t + step <= current_t:
                    print("Warning: Custom time step is too small, breaking loop.")
                    break
                current_t += step

            # Ensure the time point closest to the requested end_time is included, if not already the last one
            end_idx = np.argmin(np.abs(time_values - end_time))
            if time_values[end_idx] >= start_time and time_values[end_idx] <= end_time:
                if not anim_indices_list or end_idx != anim_indices_list[-1]:
                    anim_indices_list.append(end_idx)
                    anim_times_list.append(time_values[end_idx])

        else:  # "Actual Data Time Steps"
            nth = self.actual_interval_spin.value()
            if nth <= 0:
                return None, None, "Actual data step interval (Every nth) must be positive."

            # Find indices of actual data points within the requested time range
            valid_indices = np.where((time_values >= start_time) & (time_values <= end_time))[0]

            if len(valid_indices) == 0:
                return None, None, "No actual data time steps found within the specified range."

            # Select every nth index from the valid ones
            selected_indices_np = valid_indices[::nth]

            # Convert to list for easier manipulation and checking
            selected_indices_list = selected_indices_np.tolist()

            # Ensure the very first point in range is included if skipped by nth
            first_valid_idx = valid_indices[0]
            if first_valid_idx not in selected_indices_list:
                selected_indices_list.insert(0, first_valid_idx)

            # Ensure the very last point in range is included if skipped by nth
            last_valid_idx = valid_indices[-1]
            if last_valid_idx not in selected_indices_list:
                # Check if the list is empty before trying to access last element
                if not selected_indices_list or last_valid_idx != selected_indices_list[-1]:
                    selected_indices_list.append(last_valid_idx)

            # Use the final list of indices
            anim_indices_list = selected_indices_list
            # Get the corresponding time values from the global array
            anim_times_list = time_values[anim_indices_list].tolist()  # Convert result to list

        # --- Final Check and Return ---
        # Perform the emptiness check ON THE LISTS before returning/converting
        if not anim_times_list:
            return None, None, "No animation frames generated for the selected time range and step."

        # If the lists are not empty, THEN convert to NumPy arrays and return
        # Use np.unique to remove potential duplicates introduced by start/end point logic, preserving order
        unique_indices, order_indices = np.unique(anim_indices_list, return_index=True)
        final_indices = unique_indices[np.argsort(order_indices)]
        final_times = time_values[final_indices]

        return np.array(final_times), np.array(final_indices, dtype=int), None

    def _estimate_animation_ram(self, num_nodes, num_anim_steps, include_deformation):
        """
        Estimates the peak RAM needed in GB for precomputing animation data.
        This revised version considers the intermediate arrays needed during calculation.
        """
        element_size = np.dtype(NP_DTYPE).itemsize  # Size of NP_DTYPE selected

        # RAM for the 6 intermediate normal/shear stress arrays (sx, sy, sz, sxy, syz, sxz)
        # These are needed to compute the final scalars (Von Mises or S1)
        # Shape: (num_nodes, num_anim_steps) for each of the 6 components.
        normal_stress_ram = num_nodes * 6 * num_anim_steps * element_size

        # RAM for the final stored scalar array (Von Mises or S1)
        # Shape: (num_nodes, num_anim_steps)
        scalar_ram = num_nodes * 1 * num_anim_steps * element_size

        # RAM for deformation calculation and storage (if requested)
        intermediate_displacement_ram = 0
        final_coordinate_ram = 0
        if include_deformation:
            # RAM for intermediate displacement arrays (ux_anim, uy_anim, uz_anim)
            # Shape: (num_nodes, num_anim_steps) for each of the 3 components.
            intermediate_displacement_ram = num_nodes * 3 * num_anim_steps * element_size

            # RAM for the final stored deformed coordinate array (precomputed_coords)
            # Stores X, Y, Z coordinates for each node at each step.
            # Shape: (num_nodes, 3, num_anim_steps) -> num_nodes * 3 * num_anim_steps elements.
            final_coordinate_ram = num_nodes * 3 * num_anim_steps * element_size

        # Total estimated peak RAM is the sum of intermediate stresses, final scalars,
        # and potentially intermediate displacements and final coordinates.
        # We assume the peak occurs when most of these arrays exist simultaneously
        # before intermediate ones are deleted by garbage collection.
        total_ram_bytes = (normal_stress_ram +
                           scalar_ram +
                           intermediate_displacement_ram +
                           final_coordinate_ram)

        # Add a safety buffer (e.g., 20%) for Python overhead, temporary copies, etc.
        # Increased buffer slightly as the calculation involves several large steps.
        total_ram_bytes *= 1.20

        # Convert bytes to Gigabytes (GB)
        return total_ram_bytes / (1024 ** 3)

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
        compute_max_principal = main_tab.max_principal_stress_checkbox.isChecked()
        # If neither is selected, return zeros (or you could raise an error).
        if not (compute_von or compute_max_principal):
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

        elif compute_max_principal:
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
        """Handle data visualization with robust data cleaning, surface reconstruction, and interpolation."""
        try:
            # 1. Load the data and clean it as before
            df = pd.read_csv(filename)
            if df.empty:
                raise ValueError("The selected CSV file is empty.")

            df.columns = [col.strip() for col in df.columns]
            x_col = next((c for c in df.columns if c.upper() == 'X'), None)
            y_col = next((c for c in df.columns if c.upper() == 'Y'), None)
            z_col = next((c for c in df.columns if c.upper() == 'Z'), None)
            nodeid_col = next((c for c in df.columns if c.upper() == 'NODEID'), None)

            if not all([x_col, y_col, z_col]):
                raise ValueError("CSV file must contain X, Y, and Z columns.")

            df_clean = df.dropna(subset=[x_col, y_col, z_col])
            coords = df_clean[[x_col, y_col, z_col]].values

            potential_data_cols = [c for c in df_clean.columns if c.upper() not in ['NODEID', 'X', 'Y', 'Z']]
            if not potential_data_cols:
                raise ValueError("No data column found in the CSV file.")
            self.data_column = potential_data_cols[0]

            scalar_values = df_clean[self.data_column].fillna(0).values

            # 2. Create the ORIGINAL point cloud
            point_cloud = pv.PolyData(coords)

            # 3. Add the scalar data to the ORIGINAL point cloud FIRST
            point_cloud[self.data_column] = scalar_values

            # 4. Reconstruct the surface (this will create the new points)
            print("Reconstructing surface... This may create new points.")
            reconstructed_mesh = point_cloud.reconstruct_surface()
            print(f"Original points: {point_cloud.n_points}, New surface points: {reconstructed_mesh.n_points}")

            # 5. CRITICAL FIX: Interpolate the data from the original cloud onto the new surface
            # This intelligently assigns values to the new points based on their neighbors.
            print("Interpolating data onto new surface...")
            # A radius of 1.0 is a generic starting point; you might need to adjust this
            # if your model is very large or very small.
            final_mesh = reconstructed_mesh.interpolate(point_cloud, radius=1.0)

            # 6. Assign the interpolated mesh as the current mesh
            self.current_mesh = final_mesh
            self.current_mesh.set_active_scalars(self.data_column)

            # We don't add NodeID here, as the new points don't have original IDs.
            # The hover-over will show the interpolated data value.

            # 7. Update UI controls
            data_min, data_max = self.current_mesh.get_data_range(self.data_column)
            self.scalar_min_spin.blockSignals(True)
            self.scalar_max_spin.blockSignals(True)
            self.scalar_min_spin.setRange(data_min, data_max)
            self.scalar_min_spin.setValue(data_min)
            self.scalar_max_spin.setRange(data_min, 1e30)
            self.scalar_max_spin.setValue(data_max)
            self.scalar_min_spin.blockSignals(False)
            self.scalar_max_spin.blockSignals(False)

            # 8. Finalize visualization
            if not self.camera_widget:
                self.camera_widget = self.plotter.add_camera_orientation_widget()
                self.camera_widget.EnabledOn()

            self.update_visualization()
            self.plotter.reset_camera()
            self.plotter.camera.zoom(1)

        except Exception as e:
            self.clear_visualization()
            QMessageBox.critical(self, "Visualization Error", f"Failed to visualize data:\n\n{str(e)}")

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
            self.custom_step_spin.setVisible(False)
            self.actual_interval_spin.setVisible(True)
        else:
            self.custom_step_spin.setVisible(True)
            self.actual_interval_spin.setVisible(False)

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
                'title_font_size': 14,
                'label_font_size': 12,
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

        # Ensure the camera widget is re-enabled if it was removed.
        if not self.camera_widget:
            self.camera_widget = self.plotter.add_camera_orientation_widget()
            self.camera_widget.EnabledOn()

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
                self.hover_annotation.SetText(2, f"Node ID: {node_id}\n{self.data_column}: {value:.5f}")
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
        """
        Handles dynamic point size updates efficiently by modifying the actor directly
        while also correctly re-initializing hover annotations. This avoids clearing
        the entire scene.
        """
        # We need a mesh and an actor to be present to do anything
        if self.current_mesh and self.current_actor:
            # 1. Clear the old hover annotations and their observers
            self.clear_hover_elements()

            # 2. Directly modify the properties of the existing actor
            new_size = self.point_size.value()
            self.current_actor.prop.point_size = new_size
            self.current_actor.prop.render_points_as_spheres = True

            # 3. Re-create the hover annotations for the updated plot
            self.setup_hover_annotation()

            # 4. Render the changes to the screen.
            self.plotter.render()

    def clear_visualization(self):
        """Properly clear existing visualization"""
        self.stop_animation()
        self.clear_hover_elements()

        # Manually disable and remove the box widget if it exists
        if self.box_widget:
            self.box_widget.Off()
            self.box_widget = None

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

        self.file_path.clear()

    def show_context_menu(self, position):
        """Creates and displays the right-click context menu."""
        # Do nothing if the scene is empty – prevents right-click menu entirely
        if self.current_mesh is None:
            return

        context_menu = QMenu(self)

        context_menu.setStyleSheet("""
            QMenu {
                background-color: #e7f0fd;      /* Main background - matches buttons */
                color: black;                   /* Text color */
                border: 1px solid #5b9bd5;      /* Border color - matches group boxes */
                border-radius: 5px;             /* Rounded corners */
                padding: 5px;                   /* Padding around the whole menu */
            }
            QMenu::item {
                background-color: transparent;  /* Make items transparent by default */
                padding: 5px 25px 5px 20px;     /* Set padding for each item */
                margin: 2px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #cce4ff;      /* Highlight color on hover - matches button hover */
                color: black;
            }
            QMenu::item:disabled {
                color: #808080;                 /* Gray text color for disabled items */
                background-color: transparent;  /* Ensure it has no background */
            }
            QMenu::separator {
                height: 1px;
                background-color: #5b9bd5;      /* Color of the separator line */
                margin: 5px 0px;                /* Space above and below the line */
            }
        """)

        title_style = """
            font-weight: bold; color: #333; 
            text-decoration: underline; padding: 4px 0px 6px 7px;
        """

        # Title for the Selection Tools (such as Box) Group
        box_title_label = QLabel("Selection Tools")
        box_title_label.setStyleSheet(title_style)
        box_title_action = QWidgetAction(context_menu)
        box_title_action.setDefaultWidget(box_title_label)
        context_menu.addAction(box_title_action)

        # Add/Remove Box
        if self.box_widget is None:
            box_action_text = "Add Selection Box"
        else:
            box_action_text = "Remove Selection Box"
        toggle_box_action = QAction(box_action_text, self)
        toggle_box_action.triggered.connect(self.toggle_selection_box)
        context_menu.addAction(toggle_box_action)

        # Pick Center
        pick_action = QAction("Pick Box Center", self)
        pick_action.setCheckable(True)
        pick_action.setChecked(self.is_point_picking_active)
        pick_action.setEnabled(self.current_mesh is not None)
        pick_action.triggered.connect(self.toggle_point_picking_mode)
        context_menu.addAction(pick_action)

        context_menu.addSeparator()

        # Title for Hotspot Analysis
        hotspot_title_label = QLabel("Hotspot Analysis")
        hotspot_title_label.setStyleSheet(title_style)
        hotspot_title_action = QWidgetAction(context_menu)
        hotspot_title_action.setDefaultWidget(hotspot_title_label)
        context_menu.addAction(hotspot_title_action)

        # Action for finding hotspots on the whole view
        hotspot_action = QAction("Find Hotspots (on current view)", self)
        hotspot_action.setEnabled(self.current_mesh and self.current_mesh.active_scalars is not None)
        hotspot_action.triggered.connect(self.find_hotspots_on_view)
        context_menu.addAction(hotspot_action)

        # Find in Box
        find_in_box_action = QAction("Find Hotspots in Selection", self)
        find_in_box_action.setEnabled(self.box_widget is not None)
        find_in_box_action.triggered.connect(self.find_hotspots_in_box)
        context_menu.addAction(find_in_box_action)

        context_menu.addSeparator()

        # Title for View Control
        view_title_label = QLabel("View Control")
        view_title_label.setStyleSheet(title_style)
        view_title_action = QWidgetAction(context_menu)
        view_title_action.setDefaultWidget(view_title_label)
        context_menu.addAction(view_title_action)

        # Reset Camera action
        reset_camera_action = QAction("Reset Camera", self)
        reset_camera_action.triggered.connect(self.plotter.reset_camera)
        context_menu.addAction(reset_camera_action)

        context_menu.exec_(self.plotter.mapToGlobal(position))

    def _find_and_show_hotspots(self, mesh_to_analyze):
        """A helper function to run hotspot analysis on a given mesh."""
        if not mesh_to_analyze or mesh_to_analyze.n_points == 0:
            QMessageBox.information(self, "No Nodes Found", "No nodes were found in the selected area.")
            return

        # Ask user for Top N
        num_hotspots, ok = QInputDialog.getInt(self, "Number of Hotspots", "How many top nodes to find?", 10, 1, 1000)
        if not ok:
            return

        # Get data from the provided mesh
        try:
            node_ids = mesh_to_analyze['NodeID']
            scalar_values = mesh_to_analyze.active_scalars
            scalar_name = mesh_to_analyze.active_scalars_name
            if scalar_name is None:
                scalar_name = "Result"

            df = pd.DataFrame({'NodeID': node_ids, scalar_name: scalar_values})
            df_hotspots = df.sort_values(by=scalar_name, ascending=False).head(num_hotspots).copy()
            df_hotspots.insert(0, 'Rank', range(1, 1 + len(df_hotspots)))
            df_hotspots.reset_index(drop=True, inplace=True)

            # If a dialog is already open, close it before creating a new one
            if self.hotspot_dialog is not None:
                self.hotspot_dialog.close()

            # Create and launch the dialog
            dialog = HotspotDialog(df_hotspots, self)
            dialog.node_selected.connect(self.highlight_and_focus_on_node)
            dialog.finished.connect(self._cleanup_hotspot_analysis)  # Clean up when closed

            if self.box_widget is not None:
                self.box_widget.Off() # Disable the widget to lock its position and size

            self.hotspot_dialog = dialog
            self.hotspot_dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to find hotspots: {e}")

    def find_hotspots_on_view(self):
        """Finds hotspots only on the points currently visible to the camera."""
        if not self.current_mesh:
            QMessageBox.warning(self, "No Data", "There is no mesh loaded to find hotspots on.")
            return

        # 1. Create the VTK filter for selecting visible points
        selector = vtk.vtkSelectVisiblePoints()

        # 2. Configure the selector with the input mesh and the plotter's renderer
        selector.SetInputData(self.current_mesh)
        selector.SetRenderer(self.plotter.renderer)
        selector.Update()  # Execute the filter

        # 3. Get the result and wrap it as a PyVista PolyData object
        visible_mesh = pv.wrap(selector.GetOutput())

        # Check if any points were actually visible
        if visible_mesh.n_points == 0:
            QMessageBox.information(self, "No Visible Points", "No points are visible in the current camera view.")
            return

        # Pass the new, filtered mesh to your existing analysis function
        self._find_and_show_hotspots(visible_mesh)

    def highlight_and_focus_on_node(self, node_id):
        if self.current_mesh is None:
            QMessageBox.warning(self, "No Mesh", "Cannot highlight node because no mesh is loaded.")
            return

        # --- THIS IS THE FIX ---
        # 1. If a highlight actor from a previous selection exists, remove it
        #    directly from the VTK renderer.
        if self.highlight_actor:
            self.plotter.renderer.RemoveActor(self.highlight_actor)
            self.highlight_actor = None
        # --- END OF FIX ---

        try:
            # 1. Find the node's index and coordinates (this part is the same)
            node_indices = np.where(self.current_mesh['NodeID'] == node_id)[0]
            if len(node_indices) == 0:
                print(f"Node ID {node_id} not found in the current mesh.")
                return

            point_index = node_indices[0]
            point_coords = self.current_mesh.points[point_index]

            # 2. Create the label text
            label_text = f"Node {node_id}"

            # 3. Add the point label actor instead of a sphere
            #    This creates a visible point and text label at the coordinates.
            self.highlight_actor = self.plotter.add_point_labels(
                point_coords, [label_text],
                name="hotspot_label",
                font_size=16,
                point_color='red',
                point_size=15,
                text_color='red',
                always_visible=True # Ensures the label is not hidden by the mesh
            )

            # 4. Move the camera to focus on the point
            self.plotter.fly_to(point_coords)

        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Could not highlight node {node_id}: {e}")

    def toggle_selection_box(self):
        """Adds or removes the box widget from the plotter."""
        if self.box_widget is None:
            # Add the widget and store a reference to it
            self.box_widget = self.plotter.add_box_widget(callback=self._dummy_callback)

            # Set the initial size of the box to be 75% of the dataset's bounds.
            # A smaller box will have smaller handles.
            self.box_widget.SetPlaceFactor(0.75)

            # Get the property for the handles and change their color
            handle_property = self.box_widget.GetHandleProperty()
            handle_property.SetColor(0.8, 0.4, 0.2) # Set to a less obtrusive orange color
            handle_property.SetPointSize(1)

            # Get the property for the currently selected handle
            selected_handle_property = self.box_widget.GetSelectedHandleProperty()
            selected_handle_property.SetColor(1.0, 0.5, 0.0) # Set to a bright orange when selected

        else:
            # Use the recommended way to remove all widgets of this type
            self.plotter.clear_box_widgets()
            self.box_widget = None
        # We need to render to see the change
        self.plotter.render()

    def find_hotspots_in_box(self):
        """Clips the mesh to the box bounds and runs the hotspot analysis."""
        if self.box_widget is None:
            return  # Should not happen if the menu is disabled, but a good safety check

        # Create a vtk.vtkPolyData object to store the box's geometry
        box_geometry = vtk.vtkPolyData()
        # Ask the widget to populate our object with its current geometry
        self.box_widget.GetPolyData(box_geometry)
        # Now, get the bounds from the geometry object, which has the GetBounds() method
        bounds = box_geometry.GetBounds()

        # Clip the main mesh using these bounds
        clipped_mesh = self.current_mesh.clip_box(bounds, invert=False)

        # Call the existing helper function with the clipped mesh
        self._find_and_show_hotspots(clipped_mesh)

    def _dummy_callback(self, *args):
        """A do-nothing callback function to satisfy the widget's requirement."""
        pass

    def toggle_point_picking_mode(self, checked):
        """Toggles the point picking mode on the plotter."""
        self.is_point_picking_active = checked
        if checked:
            # Disables other interactions and sets up our callback
            self.plotter.enable_point_picking(
                callback=self.on_point_picked,
                show_message=False,  # Don't show the default PyVista message box
                use_picker=True,  # Ensures we pick a point on the mesh
                left_clicking = True
            )
            self.plotter.setCursor(Qt.CrossCursor)  # Give user visual feedback
        else:
            self.plotter.disable_picking()
            self.plotter.setCursor(Qt.ArrowCursor)

    def on_point_picked(self, *args):
        """Callback executed when a point is picked on the mesh."""
        # Use *args to robustly handle different PyVista versions
        # Check if args is empty or if the coordinate array has a size of 0
        if not args or args[0].size == 0:
            return

        center = args[0]

        # If the box widget doesn't exist yet, create it now
        if self.box_widget is None:
            self.box_widget = self.plotter.add_box_widget(callback=self._dummy_callback)
            # Apply our custom properties
            self.box_widget.GetHandleProperty().SetColor(0.8, 0.4, 0.2)
            self.box_widget.GetSelectedHandleProperty().SetColor(1.0, 0.5, 0.0)
            self.box_widget.GetHandleProperty().SetPointSize(10)
            self.box_widget.GetSelectedHandleProperty().SetPointSize(15)

            # Define a default size for the new box
            size = self.current_mesh.length * 0.1
            bounds = [
                center[0] - size / 2.0, center[0] + size / 2.0,
                center[1] - size / 2.0, center[1] + size / 2.0,
                center[2] - size / 2.0, center[2] + size / 2.0,
            ]
        else:
            # If the box already exists, get its current size
            box_geometry = vtk.vtkPolyData()
            self.box_widget.GetPolyData(box_geometry)
            current_bounds = box_geometry.GetBounds()

            x_size = current_bounds[1] - current_bounds[0]
            y_size = current_bounds[3] - current_bounds[2]
            z_size = current_bounds[5] - current_bounds[4]
            # Calculate new bounds centered on the picked point
            bounds = [
                center[0] - x_size / 2.0, center[0] + x_size / 2.0,
                center[1] - y_size / 2.0, center[1] + y_size / 2.0,
                center[2] - z_size / 2.0, center[2] + z_size / 2.0,
            ]

        # Move the box widget to the new bounds directly
        self.box_widget.PlaceWidget(bounds)
        self.plotter.render()

        # Turn off picking mode after one use
        self.toggle_point_picking_mode(False)

    def _cleanup_hotspot_analysis(self):
        """Removes all highlight labels and re-enables the box widget."""
        # Remove the text label actor
        if hasattr(self, 'highlight_actor') and self.highlight_actor:
            self.plotter.remove_actor("hotspot_label", reset_camera=False)
            self.highlight_actor = None

        # Re-enable the box widget if it still exists
        if self.box_widget:
            self.box_widget.On()

        # Clear the reference to the now-closed dialog
        self.hotspot_dialog = None

        self.plotter.render()

    def __del__(self):
        """Ensure proper cleanup"""
        self.clear_visualization()


class HotspotDialog(QDialog):
    # Signal to be emitted when a node is selected from the table
    # It will carry the integer Node ID.
    node_selected = pyqtSignal(int)

    def __init__(self, hotspot_df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hotspot Analysis Results")
        self.setMinimumSize(300, 300)

        self.table_view = QTableView()
        self.model = QStandardItemModel(self)
        self.table_view.setModel(self.model)

        # Make the table non-editable and select whole rows at a time
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)

        # Populate the table with the data
        self.populate_table(hotspot_df)

        # When a row is clicked, trigger our handler
        self.table_view.clicked.connect(self.on_row_clicked)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Click a row to navigate to the node in the Display tab."))
        layout.addWidget(self.table_view)
        self.setLayout(layout)

    def populate_table(self, df):
        """Populates the table, formatting floats to 4 decimal places."""
        self.model.setHorizontalHeaderLabels(df.columns)

        for index, row in df.iterrows():
            items = []
            for col_name, val in row.items():
                # Keep Rank and NodeID as integers
                if col_name in ['Rank', 'NodeID']:
                    items.append(QStandardItem(str(int(float(val)))))
                # Format all other columns as floats with 4 decimal places
                else:
                    items.append(QStandardItem(f"{val:.4f}"))
            self.model.appendRow(items)

        self.table_view.resizeColumnsToContents()

    def on_row_clicked(self, index):
        # Get the row of the clicked cell
        row = index.row()
        # Assume 'NodeID' is the second column (index 1)
        node_id_item = self.model.item(row, 1)
        if node_id_item:
            node_id = int(float(node_id_item.text()))
            # Emit the signal with the node ID
            self.node_selected.emit(node_id)


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Attributes for interactivity
        self.ax = None
        self.annot = None
        self.plotted_lines = []

        # Matplotlib canvas on the left
        self.figure = plt.Figure(tight_layout=True) #tight layout for better spacing
        self.canvas = FigureCanvas(self.figure)
        # make it expand/shrink with the window
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # Add the Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Data table on the right
        self.table = QTableView(self)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.model = QStandardItemModel(self)
        self.model.setHorizontalHeaderLabels(["Time [s]", "Value"])
        self.table.setModel(self.model)

        # Ctrl+C to copy the selected block
        copy_sc = QShortcut(QKeySequence.Copy, self.table)
        copy_sc.activated.connect(self.copy_selection)

        # Split view
        self.splitter = QSplitter(Qt.Horizontal, self)

        # Create a container for plot and toolbar
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        self.splitter.addWidget(plot_container)
        self.splitter.addWidget(self.table)

        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)
        self.setLayout(layout)

        # Connect a hover event
        self.canvas.mpl_connect("motion_notify_event", self.hover)

    def showEvent(self, event):
        """
        This event is called when the widget is shown.
        """
        # First, run the default event processing
        super().showEvent(event)

        # Schedule the splitter adjustment to run after this event is processed.
        # This ensures the widget has its final geometry.
        QTimer.singleShot(50, self.adjust_splitter_size)

    def adjust_splitter_size(self):
        """
        Calculates the ideal width for the table, including the vertical scrollbar,
        and resizes the splitter.
        """
        header = self.table.horizontalHeader()

        # Temporarily set the resize mode to calculate the ideal content width
        # Note: For PyQt6/PySide6, use QHeaderView.ResizeMode.ResizeToContents
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        #
        # --- THE KEY FIX IS HERE ---
        #
        # 1. Check if a vertical scrollbar is visible and get its width.
        v_scrollbar = self.table.verticalScrollBar()
        scrollbar_width = v_scrollbar.width() if v_scrollbar.isVisible() else 0

        # 2. Calculate the required width, now including the scrollbar.
        required_width = (header.length() +
                          self.table.verticalHeader().width() +
                          self.table.frameWidth() * 2 +
                          scrollbar_width)

        # Restore interactive resizing so the user can adjust columns manually
        # Note: For PyQt6/PySide6, use QHeaderView.ResizeMode.Interactive
        header.setSectionResizeMode(QHeaderView.Interactive)

        # Now, adjust the splitter with the corrected width
        total_width = self.splitter.width()
        plot_width = total_width - required_width

        # Enforce a minimum width for the plot for usability
        if plot_width < 450:
            plot_width = 450

        # Recalculate table width in case the plot width was clipped
        new_table_width = total_width - plot_width

        self.splitter.setSizes([int(plot_width), int(new_table_width)])

    def hover(self, event):
        """Show an annotation when hovering over a data point."""
        # Check if the event is valid and within the axes
        if not event.inaxes or self.ax is None or self.annot is None:
            return

        visible = self.annot.get_visible()

        # Check all plotted lines
        for line in self.plotted_lines:
            cont, ind = line.contains(event)
            if cont:
                # Get the data coordinates of the hovered point
                pos = line.get_xydata()[ind["ind"][0]]
                x_coord, y_coord = pos[0], pos[1]

                # Update annotation text and position
                self.annot.xy = (x_coord, y_coord)
                self.annot.set_text(f"Time: {x_coord:.4f}\nValue: {y_coord:.4f}")

                # Set annotation visibility and draw
                if not visible:
                    self.annot.set_visible(True)
                    self.canvas.draw_idle()
                return  # Stop after finding the first point

        # If the mouse is not over any point, hide the annotation
        if visible:
            self.annot.set_visible(False)
            self.canvas.draw_idle()

    def update_plot(self, x, y, node_id=None,
                    is_max_principal_stress=False,
                    is_min_principal_stress=False,
                    is_von_mises=False,
                    is_deformation=False,
                    is_velocity=False,
                    is_acceleration=False):
        # Clear the figure
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)

        # --- Define plot styles ---
        styles = {
            'Magnitude': {'color': 'black', 'linestyle': '-', 'linewidth': 2},
            'X': {'color': 'red', 'linestyle': '--', 'linewidth': 1},
            'Y': {'color': 'green', 'linestyle': '--', 'linewidth': 1},
            'Z': {'color': 'blue', 'linestyle': '--', 'linewidth': 1},
        }

        self.model.clear()
        textstr = ""

        # Check if y is a dictionary for multi-component data
        if isinstance(y, dict):
            # This is multi-component data (Deformation, Velocity, etc.)
            if is_velocity:
                prefix, units = "Velocity", "(mm/s)"
            elif is_acceleration:
                prefix, units = "Acceleration", "(mm/s²)"
            else:  # is_deformation
                prefix, units = "Deformation", "(mm)"

            ax.set_title(f"{prefix} (Node ID: {node_id})", fontsize=8)
            ax.set_ylabel(f"{prefix} {units}", fontsize=8)

            self.model.setHorizontalHeaderLabels(
                ["Time [s]", f"Mag {units}", f"X {units}", f"Y {units}", f"Z {units}"])

            # This loop is now safe because we've confirmed y is a dictionary
            for component, data in y.items():
                style = styles.get(component, {})
                ax.plot(x, data, label=f'{prefix} ({component})', **style)

            for i in range(len(x)):
                items = [
                    QStandardItem(f"{x[i]:.5f}"),
                    QStandardItem(f"{y['Magnitude'][i]:.5f}"),
                    QStandardItem(f"{y['X'][i]:.5f}"),
                    QStandardItem(f"{y['Y'][i]:.5f}"),
                    QStandardItem(f"{y['Z'][i]:.5f}")
                ]
                self.model.appendRow(items)

            max_y_value = np.max(y['Magnitude'])
            time_of_max = x[np.argmax(y['Magnitude'])]
            textstr = f'Max Magnitude: {max_y_value:.4f}\nTime of Max: {time_of_max:.5f} s'

        else:
            # This is single-component data (Stress or a placeholder)
            self.model.setHorizontalHeaderLabels(["Time [s]", "Value"])
            for xi, yi in zip(x, y):
                self.model.appendRow([QStandardItem(f"{xi:.5f}"), QStandardItem(f"{yi:.5f}")])

            if is_min_principal_stress:
                ax.plot(x, y, label=r'$\sigma_3$', color='green')
                ax.set_title(f"Min Principal Stress (Node ID: {node_id})" if node_id else "Min Principal Stress",
                             fontsize=8)
                ax.set_ylabel(r'$\sigma_3$ [MPa]', fontsize=8)
                min_y_value = np.min(y)
                time_of_min = x[np.argmin(y)]
                textstr = f'Min Magnitude: {min_y_value:.4f}\nTime of Min: {time_of_min:.5f} s'
            else:
                title = "Stress"
                label = "Value"
                color = 'blue'
                if is_max_principal_stress:
                    title, label, color = "Max Principal Stress", r'$\sigma_1$', 'red'
                elif is_von_mises:
                    title, label, color = "Von Mises Stress", r'$\sigma_{VM}$', 'blue'

                ax.plot(x, y, label=label, color=color)
                ax.set_title(f"{title} (Node ID: {node_id})" if node_id else title, fontsize=8)
                ax.set_ylabel(f'{label} [MPa]', fontsize=8)

                if len(y) > 0 and np.any(y):  # Check if y is not empty or all zeros
                    max_y_value = np.max(y)
                    time_of_max = x[np.argmax(y)]
                    textstr = f'Max Magnitude: {max_y_value:.4f}\nTime of Max: {time_of_max:.5f} s'

        # Common plot styling
        ax.set_xlabel('Time [seconds]', fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=8)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7)

        if textstr:
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

        # Resize table columns to fit the new content
        self.table.resizeColumnsToContents()
        self.canvas.draw()

        QTimer.singleShot(0, self.adjust_splitter_size)

    def copy_selection(self):
        """Copy the selected rectangular block of cells as TSV to the clipboard."""
        sel = self.table.selectedIndexes()
        if not sel:
            return

        rows = sorted(idx.row() for idx in sel)
        cols = sorted(idx.column() for idx in sel)
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]

        lines = []

        # 1) Header labels
        headers = [
            self.model.headerData(c, Qt.Horizontal)
            for c in range(c0, c1 + 1)
        ]
        lines = ['\t'.join(headers)]

        for r in range(r0, r1 + 1):
            row_data = []
            for c in range(c0, c1 + 1):
                text = self.model.index(r, c).data() or ""
                row_data.append(text)
            lines.append('\t'.join(row_data))

        QApplication.clipboard().setText('\n'.join(lines))

    def clear_plot(self):
        """Clears the plot and the data table, and draws an empty placeholder plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.set_title("Time History (No Data)", fontsize=8)
        ax.set_xlabel('Time [seconds]', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=8)
        self.canvas.draw()

        self.model.removeRows(0, self.model.rowCount())
        self.model.setHorizontalHeaderLabels(["Time [s]", "Value"])

        self.table.resizeColumnsToContents()
        QTimer.singleShot(0, self.adjust_splitter_size)


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
        main_win = self.window()
        main_win.load_fig_to_webview(resampler_fig, self.web_view)

    def clear_plot(self):
        """Clears the plot and resets stored data."""
        self.web_view.setHtml("")  # Clear the web view content
        self.last_time_values = None
        self.last_modal_coord = None


class PlotlyMaxWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- left: Plotly web view ---
        self.web_view = QWebEngineView(self)

        # --- right: Data table ---
        self.table = QTableView(self)
        # Allow rectangular selection, multi‑select with Shift/Ctrl
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Model with 3 columns
        self.model = QStandardItemModel(self)
        self.model.setHorizontalHeaderLabels(["Time [s]", "Data Value"])
        self.table.setModel(self.model)

        # Ctrl+C shortcut bound to copy_selection()
        copy_sc = QShortcut(QKeySequence.Copy, self.table)
        copy_sc.activated.connect(self.copy_selection)

        # Splitter to hold plot + table
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self.web_view)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 90)  # plot ~90%
        splitter.setStretchFactor(1, 10)  # table ~10%

        lay = QVBoxLayout(self)
        lay.addWidget(splitter)
        self.setLayout(lay)

    def update_plot(self, time_values, traces=None):
        """
        Dynamically plots multiple data traces and populates a table.
        - traces: A list of dictionaries, e.g., [{'name': 'Von Mises (MPa)', 'data': np.array([...])}]
        """
        if traces is None:
            traces = []

        # 1) Build figure by iterating through the provided traces
        fig = go.Figure()
        for trace_info in traces:
            fig.add_trace(go.Scattergl(x=time_values, y=trace_info['data'], mode='lines', name=trace_info['name']))

        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Value",  # Generic Y-axis title
            template="plotly_white",
            font=dict(size=7),
            margin=dict(l=40, r=40, t=10, b=0),
            legend=dict(font=dict(size=7))
        )

        # 2) Wrap in resampler
        resfig = FigureResampler(fig, default_n_shown_samples=50000)

        # Show the plot
        main_win = self.window()
        main_win.load_fig_to_webview(resfig, self.web_view)

        # 3) Dynamically populate the table
        headers = ["Time [s]"] + [trace['name'] for trace in traces]
        self.model.setHorizontalHeaderLabels(headers)
        self.model.removeRows(0, self.model.rowCount())

        for i, t in enumerate(time_values):
            # Start each row with the time value
            row_items = [QStandardItem(f"{t:.5f}")]
            # Add the data from each trace for the current time step
            for trace in traces:
                row_items.append(QStandardItem(f"{trace['data'][i]:.6f}"))
            self.model.appendRow(row_items)

    def copy_selection(self):
        """Copy the currently selected block of cells to the clipboard as TSV."""
        sel = self.table.selectedIndexes()
        if not sel:
            return
        # determine the extents of the selection
        rows = sorted(idx.row() for idx in sel)
        cols = sorted(idx.column() for idx in sel)
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]

        lines = []

        # 1) Header labels
        headers = [
            self.model.headerData(c, Qt.Horizontal)
            for c in range(c0, c1 + 1)
        ]
        lines = ['\t'.join(headers)]

        for r in range(r0, r1 + 1):
            row_data = []
            for c in range(c0, c1 + 1):
                idx = self.model.index(r, c)
                text = idx.data() or ""
                row_data.append(text)
            lines.append('\t'.join(row_data))
        QApplication.clipboard().setText('\n'.join(lines))

    def clear_plot(self):
        """Clears the plot and the data table."""
        self.web_view.setHtml("")
        self.model.removeRows(0, self.model.rowCount())
        # Also reset the headers to a default state
        self.model.setHorizontalHeaderLabels(["Time [s]", "Data Value"])


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

        # Set up a single logger instance
        self.logger = Logger(self.console_textbox)
        sys.stdout = self.logger  # Redirect stdout to the logger

        # Enable drag-and-drop for file selection buttons and text fields
        self.setAcceptDrops(True)

        # Flags to check whether primary inputs are loaded
        self.coord_loaded = False
        self.deformation_loaded = False
        self.stress_loaded      = False

        self._update_solve_button_state()

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
        self.coord_file_path.setStyleSheet(
            "background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;")

        # Modal Stress File Section
        self.stress_file_button = QPushButton('Read Modal Stress File (.csv)')
        self.stress_file_button.setStyleSheet(button_style)
        self.stress_file_button.setFont(QFont('Arial', 8))
        self.stress_file_path = QLineEdit()
        self.stress_file_path.setReadOnly(True)
        self.stress_file_path.setStyleSheet(
            "background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;")

        # Checkbox for "Include Steady-State Stress Field"
        self.steady_state_checkbox = QCheckBox("Include Steady-State Stress Field (Optional)")
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
        self.deformations_checkbox = QCheckBox("Include Deformations (Optional)")
        self.deformations_checkbox.setStyleSheet("margin: 10px 0;")
        self.deformations_checkbox.toggled.connect(self.toggle_deformations_inputs)

        self.deformations_file_button = QPushButton('Read Modal Deformations File (.csv)')
        self.deformations_file_button.setStyleSheet("/* use your button style */")
        self.deformations_file_button.setFont(QFont('Arial', 8))
        self.deformations_file_path = QLineEdit()
        self.deformations_file_button.setStyleSheet(button_style)
        self.deformations_file_path.setReadOnly(True)
        self.deformations_file_path.setStyleSheet(
            "background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;")
        self.deformations_file_button.clicked.connect(self.select_deformations_file)

        # Initially hide the deformations file controls until the checkbox is checked.
        self.deformations_file_button.setVisible(False)
        self.deformations_file_path.setVisible(False)

        # Create label and combobox for skipping first n modes
        self.skip_modes_label = QLabel("Skip first n modes:")
        self.skip_modes_label.setVisible(False)  # hidden until file is loaded

        self.skip_modes_combo = QComboBox()
        self.skip_modes_combo.setFixedWidth(80)  # compact width
        self.skip_modes_combo.setVisible(False)
        self.skip_modes_combo.currentTextChanged.connect(self.on_skip_modes_changed)

        # Checkbox for Time History Mode (Single Node)
        self.time_history_checkbox = QCheckBox('Time History Mode (Single Node)')
        self.time_history_checkbox.setStyleSheet("margin: 10px 0;")
        self.time_history_checkbox.toggled.connect(self.toggle_single_node_solution_group)
        self.time_history_checkbox.toggled.connect(self._on_time_history_toggled)

        # Checkbox for Calculate Principal Stress
        self.max_principal_stress_checkbox = QCheckBox('Max Principal Stress')
        self.max_principal_stress_checkbox.setStyleSheet("margin: 10px 0;")
        self.max_principal_stress_checkbox.toggled.connect(self.update_single_node_plot_based_on_checkboxes)
        self.min_principal_stress_checkbox = QCheckBox("Min Principal Stress")
        self.min_principal_stress_checkbox.setStyleSheet("margin: 10px 0;")
        self.min_principal_stress_checkbox.toggled.connect(self.update_single_node_plot_based_on_checkboxes)

        # Checkbox for Calculate Von-Mises Stress
        self.von_mises_checkbox = QCheckBox('Von-Mises Stress')
        self.von_mises_checkbox.setStyleSheet("margin: 10px 0;")
        self.von_mises_checkbox.toggled.connect(self.update_single_node_plot_based_on_checkboxes)

        # Checkbox for Calculating Velocity
        self.velocity_checkbox = QCheckBox('Velocity')
        self.velocity_checkbox.setStyleSheet("margin: 10px 0;")

        # Checkbox for Calculating Acceleration
        self.acceleration_checkbox = QCheckBox('Acceleration')
        self.acceleration_checkbox.setStyleSheet("margin: 10px 0;")

        # Checkbox for Calculating Deformation
        self.deformation_checkbox = QCheckBox('Deformation')
        self.deformation_checkbox.setStyleSheet("margin: 10px 0;")

        # Checkbox for Calculate Damage Index
        self.damage_index_checkbox = QCheckBox('Damage Index / Potential Damage')
        self.damage_index_checkbox.setStyleSheet("margin: 10px 0;")
        self.damage_index_checkbox.toggled.connect(self.toggle_fatigue_params_visibility)

        # Connect checkbox signal to the method for controlling the visibility of the damage index checkbox
        self.von_mises_checkbox.toggled.connect(self.toggle_damage_index_checkbox_visibility)

        # Create Fatigue Parameters group box (initially hidden)
        self.fatigue_params_group = QGroupBox("Fatigue Parameters")
        self.fatigue_params_group.setStyleSheet(group_box_style)
        fatigue_group_main_layout = QHBoxLayout()
        fatigue_inputs_layout = QVBoxLayout()
        self.A_line_edit = QLineEdit()
        self.A_line_edit.setPlaceholderText("Enter Fatigue Strength Coefficient [MPa]")
        self.A_line_edit.setValidator(QDoubleValidator())
        # Signal will be emitted when the user hits Enter or when the field loses focus:
        self.A_line_edit.editingFinished.connect(lambda: print("A value changed:", self.A_line_edit.text()))
        self.m_line_edit = QLineEdit()
        self.m_line_edit.setPlaceholderText("Enter Fatigue Strength Exponent")
        self.m_line_edit.setValidator(QDoubleValidator())
        self.m_line_edit.editingFinished.connect(lambda: print("m value changed:", self.m_line_edit.text()))

        # Add labels and line edits to the layout
        fatigue_inputs_layout.addWidget(QLabel("σ’f"))
        fatigue_inputs_layout.addWidget(self.A_line_edit)
        fatigue_inputs_layout.addWidget(QLabel("b:"))
        fatigue_inputs_layout.addWidget(self.m_line_edit)
        self.fatigue_params_group.setLayout(fatigue_inputs_layout)
        self.fatigue_params_group.setVisible(False)  # hide by default

        # LineEdit for Node ID input
        self.node_line_edit = QLineEdit()
        self.node_line_edit.setPlaceholderText("Enter Node ID")
        self.node_line_edit.setStyleSheet(button_style)
        self.node_line_edit.setMaximumWidth(150)
        self.node_line_edit.setMinimumWidth(100)
        self.node_line_edit.returnPressed.connect(self.on_node_entered)

        # Solve Button
        self.solve_button = QPushButton('SOLVE')
        self.solve_button.setStyleSheet(button_style)
        self.solve_button.setFont(QFont('Arial', 9, QFont.Bold))
        self.solve_button.clicked.connect(self.solve)

        # Read-only Output Console
        self.console_textbox = QTextEdit()
        self.console_textbox.setReadOnly(True)
        self.console_textbox.setStyleSheet("background-color: #ffffff; border: 1px solid #5b9bd5")
        self.console_textbox.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.console_textbox.setText('Console Output:\n')

        # Set monospaced font for log terminal
        terminal_font = QFont("Consolas", 8)
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
        file_layout.addWidget(self.skip_modes_label, 5, 2)
        file_layout.addWidget(self.skip_modes_combo, 5, 3)

        file_group.setLayout(file_layout)

        # Group box for outputs requested
        self.output_group = QGroupBox("Outputs")
        self.output_group.setStyleSheet(group_box_style)
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.time_history_checkbox)
        output_layout.addWidget(self.max_principal_stress_checkbox)
        output_layout.addWidget(self.min_principal_stress_checkbox)
        output_layout.addWidget(self.von_mises_checkbox)
        output_layout.addWidget(self.deformation_checkbox)
        output_layout.addWidget(self.velocity_checkbox)
        output_layout.addWidget(self.acceleration_checkbox)
        output_layout.addWidget(self.damage_index_checkbox)
        self.output_group.setLayout(output_layout)

        # outputs that require ONLY the deformation file
        self._deformation_outputs = [self.deformation_checkbox,
                                     self.velocity_checkbox,
                                     self.acceleration_checkbox]

        # outputs that need both the modal coordinate and the modal stress file
        self._coord_stress_outputs = [self.time_history_checkbox,
                                      self.max_principal_stress_checkbox,
                                      self.min_principal_stress_checkbox,
                                      self.von_mises_checkbox,
                                      self.damage_index_checkbox]

        # Keep output checkboxes disabled until relevant input files are loaded
        for cb in (self._deformation_outputs + self._coord_stress_outputs):
            cb.setEnabled(False)

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
        hbox_user_inputs = QHBoxLayout()
        hbox_user_inputs.addWidget(self.output_group)
        hbox_user_inputs.addWidget(self.fatigue_params_group)
        hbox_user_inputs.addWidget(self.single_node_group)

        # Adding elements to main layout
        main_layout.addWidget(file_group)
        main_layout.addLayout(hbox_user_inputs)
        main_layout.addWidget(self.solve_button)
        main_layout.addWidget(self.show_output_tab_widget)  # Add the tab widget for the log terminal
        main_layout.addWidget(self.progress_bar)

        self.setLayout(main_layout)

        # Initially hide the "Calculate Damage Index" checkbox if "Calculate Von-Mises" is not checked
        self.toggle_damage_index_checkbox_visibility()

        # A master list for all outputs that are mutually exclusive in Time History mode
        self.time_history_exclusive_outputs = [
            self.max_principal_stress_checkbox,
            self.min_principal_stress_checkbox,
            self.von_mises_checkbox,
            self.deformation_checkbox,
            self.velocity_checkbox,
            self.acceleration_checkbox
        ]

    def update_output_checkboxes_state(self):
        """
        Enables or disables output checkboxes based on which primary data files are loaded.
        """
        # Stress-related outputs require both primary files
        stress_enabled = self.coord_loaded and self.stress_loaded
        for cb in self._coord_stress_outputs:
            cb.setEnabled(stress_enabled)
            if not stress_enabled:
                cb.setChecked(False)

        # Deformation-related outputs require the coordinate file and the optional deformation file
        deformations_enabled = (self.coord_loaded and
                                self.deformations_checkbox.isChecked() and
                                self.deformation_loaded)
        for cb in self._deformation_outputs:
            cb.setEnabled(deformations_enabled)
            if not deformations_enabled:
                cb.setChecked(False)

    def toggle_steady_state_stress_inputs(self):
        is_checked = self.steady_state_checkbox.isChecked()
        self.steady_state_file_button.setVisible(is_checked)
        self.steady_state_file_path.setVisible(is_checked)

        # Clear the file path text if the checkbox is unchecked
        if not is_checked:
            self.steady_state_file_path.clear()

    def toggle_deformations_inputs(self):
        """Shows or hides the UI controls for loading a modal deformations file."""
        is_checked = self.deformations_checkbox.isChecked()

        # Control the visibility of the file input widgets and the "skip modes" combo box.
        self.deformations_file_button.setVisible(is_checked)
        self.deformations_file_path.setVisible(is_checked)

        are_details_enabled = is_checked and self.deformation_loaded
        self.skip_modes_label.setVisible(are_details_enabled)
        self.skip_modes_combo.setVisible(are_details_enabled)

        # Call helper method to ensure exclusivity due to availability of input files
        self.update_output_checkboxes_state()

        if not is_checked:
            self.deformations_file_path.clear()
            self.deformation_loaded = False

    def toggle_damage_index_checkbox_visibility(self):
        if self.von_mises_checkbox.isChecked():
            self.damage_index_checkbox.setVisible(True)
        else:
            self.damage_index_checkbox.setVisible(False)

    def toggle_fatigue_params_visibility(self, checked):
        self.fatigue_params_group.setVisible(checked)

    def toggle_single_node_solution_group(self):
        try:
            if self.time_history_checkbox.isChecked():
                # Connect all exclusive checkboxes to the single, unified handler
                for cb in self.time_history_exclusive_outputs:
                    # Use a lambda to pass a reference to the checkbox that was clicked
                    cb.toggled.connect(
                        lambda checked, a_checkbox=cb: self.on_exclusive_output_toggled(checked, a_checkbox)
                    )

                # Show single node group and plot tab
                self.single_node_group.setVisible(True)
                self.show_output_tab_widget.setTabVisible(
                    self.show_output_tab_widget.indexOf(self.plot_single_node_tab), True)
            else:
                # Disconnect only the specific handler for mutual exclusivity leaving the original connections (like for the damage index) intact.
                for checkbox in self.time_history_exclusive_outputs:
                    try:
                        # This specifically targets the function we added earlier.
                        checkbox.toggled.disconnect(self.on_exclusive_output_toggled)
                    except TypeError:
                        # This error can occur if the slot was already disconnected.
                        # It's safe to ignore it in this context.
                        pass

                # Hide single node group and plot tab
                self.single_node_group.setVisible(False)
                self.show_output_tab_widget.setTabVisible(
                    self.show_output_tab_widget.indexOf(self.plot_single_node_tab), False)
        except Exception as e:
            print(f"Error in toggling single node group visibility: {e}")

    def _on_time_history_toggled(self, is_checked):
        """When Time History Mode is checked, uncheck all other output options."""
        if is_checked:
            # Combine all output checkboxes into one list for easy iteration
            all_output_checkboxes = self._coord_stress_outputs + self._deformation_outputs

            for checkbox in all_output_checkboxes:
                # Don't uncheck the Time History checkbox itself
                if checkbox is self.time_history_checkbox:
                    continue

                # Temporarily block signals to prevent other logic from firing, then uncheck
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)

    def on_exclusive_output_toggled(self, is_checked, sender_checkbox):
        """
        Ensures that only one of the exclusive output checkboxes is selected at a time
        when in Time History Mode.
        """
        # We only act if a box was just checked, not unchecked
        if self.time_history_checkbox.isChecked() and is_checked:
            for checkbox in self.time_history_exclusive_outputs:
                # If the checkbox is not the one that triggered the signal...
                if checkbox is not sender_checkbox:
                    # Temporarily block its signals to prevent a chain reaction of toggled events
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)

    def update_single_node_plot(self):
        """Updates the placeholder plot inside the MatplotlibWidget."""
        x = np.linspace(0, 10, 100)
        y = np.zeros(100)
        self.plot_single_node_tab.update_plot(x, y)

    def update_single_node_plot_based_on_checkboxes(self):
        """Update the plot based on the state of the 'Principal Stress' and 'Von Mises Stress' checkboxes."""
        try:
            # Retrieve the checkbox states
            is_max_principal_stress = self.max_principal_stress_checkbox.isChecked()
            is_min_principal_stress = self.min_principal_stress_checkbox.isChecked()
            is_von_mises = self.von_mises_checkbox.isChecked()
            is_deformation = self.deformation_checkbox.isChecked()
            is_velocity = self.velocity_checkbox.isChecked()
            is_acceleration = self.acceleration_checkbox.isChecked()

            # Dummy data for the plot (replace this with actual data when available)
            x_data = [1, 2, 3, 4, 5]  # Time or some other X-axis data
            y_data = [0, 0, 0, 0, 0]  # Dummy Y-axis data

            # Update the plot with the current checkbox states
            self.plot_single_node_tab.update_plot(x_data, y_data, None,
                                                  is_max_principal_stress= is_max_principal_stress,
                                                  is_min_principal_stress= is_min_principal_stress,
                                                  is_von_mises=is_von_mises, is_deformation=is_deformation,
                                                  is_velocity=is_velocity, is_acceleration=is_acceleration)
        except Exception as e:
            print(f"Error updating plot based on checkbox states: {e}")

    def select_coord_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Modal Coordinate File', '', 'Coordinate Files (*.mcf)')
        if file_name:
            self.process_modal_coordinate_file(file_name)

    def process_modal_coordinate_file(self, filename):
        """Validates the MCF file, and if valid, processes it and updates the GUI."""
        try:
            # --- 1. Validation Step ---
            base, ext = os.path.splitext(filename)
            unwrapped_filename = base + "_unwrapped" + ext
            unwrap_mcf_file(filename, unwrapped_filename)
            with open(unwrapped_filename, 'r') as file:
                start_index = next(i for i, line in enumerate(file) if 'Time' in line)
            df_val = pd.read_csv(unwrapped_filename, sep='\\s+', skiprows=start_index + 1, header=None)
            os.remove(unwrapped_filename)

            if df_val.empty or df_val.shape[1] < 2:
                raise ValueError("File appears to be empty or has no mode columns.")
            if not all(pd.api.types.is_numeric_dtype(df_val[c]) for c in df_val.columns):
                raise ValueError("File contains non-numeric data where modal coordinates are expected.")

        except Exception as e:
            QMessageBox.warning(self, "Invalid File", f"The selected Modal Coordinate File is not valid.\n\nError: {e}")
            return

        # --- 2. If Validation Passes, Clear OLD Coordinate/Time Data & Plots ---
        global modal_coord, time_values
        modal_coord, time_values = None, None
        self.plot_modal_coords_tab.clear_plot()
        self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_modal_coords_tab), False)
        if hasattr(self, 'plot_max_over_time_tab'):
            self.plot_max_over_time_tab.clear_plot()
            self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_max_over_time_tab), False)
        if hasattr(self, 'plot_min_over_time_tab'):
            self.plot_min_over_time_tab.clear_plot()
            self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_min_over_time_tab), False)

        # --- 3. Load NEW Data and Update UI ---
        self.coord_file_path.setText(filename)
        time_values = df_val.iloc[:, 0].to_numpy()
        modal_coord = df_val.drop(columns=df_val.columns[0]).transpose().to_numpy()
        del df_val

        self.console_textbox.append(f"Successfully validated and loaded modal coordinate file: {os.path.basename(filename)}\n")
        self.console_textbox.append(f"Modal coordinates tensor shape (m x n): {modal_coord.shape} \n")
        self.plot_modal_coords_tab.update_plot(time_values, modal_coord)
        self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_modal_coords_tab), True)

        self.coord_loaded = True
        self.update_output_checkboxes_state()
        self.update_time_and_animation_ui()
        self._update_solve_button_state()

    def select_stress_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Modal Stress File', '', 'CSV Files (*.csv)')
        if file_name:
            self.process_modal_stress_file(file_name)

    def process_modal_stress_file(self, filename):
        """Validates the modal stress CSV file, and if valid, processes it."""
        try:
            # --- 1. Validation Step ---
            df_val = pd.read_csv(filename)
            if 'NodeID' not in df_val.columns:
                raise ValueError("Required 'NodeID' column not found.")
            stress_components = ['sx_', 'sy_', 'sz_', 'sxy_', 'syz_', 'sxz_']
            for comp in stress_components:
                if df_val.filter(regex=f'(?i){comp}').empty:
                    raise ValueError(f"Required stress component columns matching '{comp}*' not found.")

        except Exception as e:
            QMessageBox.warning(self, "Invalid File", f"The selected Modal Stress File is not valid.\n\nError: {e}")
            return

        # --- 2. If Validation Passes, Clear OLD Stress/Node Data & Plots ---
        global df_node_ids, node_coords, modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz
        df_node_ids, node_coords = None, None
        modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz = (None,) * 6
        self.window().display_tab.clear_visualization()
        self.plot_single_node_tab.clear_plot()

        # --- 3. Load NEW Data and Update UI ---
        self.stress_file_path.setText(filename)
        df_node_ids = df_val['NodeID'].to_numpy().flatten()
        if {'X', 'Y', 'Z'}.issubset(df_val.columns):
            node_coords = df_val[['X', 'Y', 'Z']].to_numpy()

        modal_sx = df_val.filter(regex='(?i)sx_.*').to_numpy().astype(NP_DTYPE)
        modal_sy = df_val.filter(regex='(?i)sy_.*').to_numpy().astype(NP_DTYPE)
        modal_sz = df_val.filter(regex='(?i)sz_.*').to_numpy().astype(NP_DTYPE)
        modal_sxy = df_val.filter(regex='(?i)sxy_.*').to_numpy().astype(NP_DTYPE)
        modal_syz = df_val.filter(regex='(?i)syz_.*').to_numpy().astype(NP_DTYPE)
        modal_sxz = df_val.filter(regex='(?i)sxz_.*').to_numpy().astype(NP_DTYPE)
        del df_val

        self.console_textbox.append(f"Successfully validated and loaded modal stress file: {os.path.basename(filename)}\n")
        self.console_textbox.append(f"Node IDs tensor shape: {df_node_ids.shape}\n")
        self.console_textbox.append(f"Normal stress components extracted: SX, SY, SZ, SXY, SYZ, SXZ")
        self.console_textbox.append(
            f"SX shape: {modal_sx.shape}, SY shape: {modal_sy.shape}, SZ shape: {modal_sz.shape}")
        self.console_textbox.append(
            f"SXY shape: {modal_sz.shape}, SYZ shape: {modal_syz.shape}, SXZ shape: {modal_sxz.shape}\n")
        self.console_textbox.verticalScrollBar().setValue(self.console_textbox.verticalScrollBar().maximum())

        self.stress_loaded = True
        self.update_output_checkboxes_state()
        self.update_time_and_animation_ui()
        self._update_solve_button_state()

    def select_deformations_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Modal Deformations File', '', 'CSV Files (*.csv)')
        if file_name:
            self.process_modal_deformations_file(file_name)

    def process_modal_deformations_file(self, filename):
        """Validates the modal deformations CSV file, and if valid, processes it."""
        try:
            # --- 1. Validation Step ---
            df_val = pd.read_csv(filename)

            if 'NodeID' not in df_val.columns:
                raise ValueError("Required 'NodeID' column not found.")

            deform_components = ['ux_', 'uy_', 'uz_']
            for comp in deform_components:
                if df_val.filter(regex=f'(?i){comp}').empty:
                    raise ValueError(f"Required deformation columns matching '{comp}*' not found.")

        except Exception as e:
            # If validation fails, ensure deformation features are turned off
            self.deformation_loaded = False
            self.deformations_file_path.clear()
            self.toggle_deformations_inputs()
            QMessageBox.warning(self, "Invalid File",
                                f"The selected Modal Deformations File is not valid.\n\nError: {e}")
            return

        # --- 2. If Validation Passes, Clear OLD Deformation Data ---
        global df_node_ids_deformations, modal_ux, modal_uy, modal_uz
        df_node_ids_deformations, modal_ux, modal_uy, modal_uz = None, None, None, None

        # Also clear UI elements that depend on it
        self.skip_modes_combo.clear()

        # --- 3. Load NEW Data and Update UI ---
        self.deformations_file_path.setText(filename)

        df_node_ids_deformations = df_val['NodeID'].to_numpy().flatten()
        modal_ux = df_val.filter(regex='(?i)^ux_').to_numpy().astype(NP_DTYPE)
        modal_uy = df_val.filter(regex='(?i)^uy_').to_numpy().astype(NP_DTYPE)
        modal_uz = df_val.filter(regex='(?i)^uz_').to_numpy().astype(NP_DTYPE)
        del df_val

        # Repopulate the skip modes combo box with new data
        num_modes = modal_ux.shape[1]
        self.skip_modes_combo.addItems([str(i) for i in range(num_modes + 1)])

        self.deformation_loaded = True
        self.console_textbox.append(f"Successfully validated and loaded modal deformations file: {os.path.basename(filename)}\n")
        self.console_textbox.append(
            f"Deformations array shapes: UX {modal_ux.shape}, UY {modal_uy.shape}, UZ {modal_uz.shape}")

        # Refresh the state of all related UI controls
        self.toggle_deformations_inputs()
        self.update_output_checkboxes_state()
        self.update_time_and_animation_ui()
        sys.stdout.flush()

    def on_skip_modes_changed(self, text):
        """
        Notifies the user in the console when the number of skipped modes changes.
        """
        try:
            if not text or not text.isdigit(): return
            num_skipped = int(text)
            message = (f"\n[INFO] Skip Modes option is set to {num_skipped}. "
                       f"The first {num_skipped} modes will be excluded from the next calculation.\n")
            if 'modal_sx' in globals() and modal_sx is not None:
                total_modes = modal_sx.shape[1]
                modes_used = total_modes - num_skipped
                message += f"       - Modes to be used: {modes_used} (from mode {num_skipped + 1} to {total_modes})\n"
            self.console_textbox.append(message)
            self.console_textbox.verticalScrollBar().setValue(self.console_textbox.verticalScrollBar().maximum())
        except (ValueError, TypeError) as e:
            self.console_textbox.append(f"\n[DEBUG] Could not parse skip modes value: {text}. Error: {e}")

    def update_time_and_animation_ui(self):
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
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Steady-State Stress File', '', 'Text Files (*.txt)')
        if file_name:
            self.process_steady_state_file(file_name)

    def process_steady_state_file(self, filename):
        """Validates the steady-state stress TXT file, and if valid, processes it."""
        try:
            # --- 1. Validation Step ---
            df_val = pd.read_csv(filename, delimiter='\t', header=0)

            # Define and check for all required columns
            required_cols = [
                'Node Number', 'SX (MPa)', 'SY (MPa)', 'SZ (MPa)',
                'SXY (MPa)', 'SYZ (MPa)', 'SXZ (MPa)'
            ]
            for col in required_cols:
                if col not in df_val.columns:
                    raise ValueError(f"Required column '{col}' not found.")

        except Exception as e:
            QMessageBox.warning(self, "Invalid File", f"The selected Steady-State Stress File is not valid.\n\nError: {e}")
            return # Stop if validation fails

        # --- 2. If Validation Passes, Clear OLD Steady-State Data ---
        global steady_node_ids, steady_sx, steady_sy, steady_sz, steady_sxy, steady_syz, steady_sxz
        steady_node_ids, steady_sx, steady_sy, steady_sz, steady_sxy, steady_syz, steady_sxz = (None,) * 7

        # --- 3. Load NEW Data and Update UI ---
        self.steady_state_file_path.setText(filename)

        steady_node_ids = df_val['Node Number'].to_numpy().reshape(-1, 1)
        steady_sx = df_val['SX (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
        steady_sy = df_val['SY (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
        steady_sz = df_val['SZ (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
        steady_sxy = df_val['SXY (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
        steady_syz = df_val['SYZ (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
        steady_sxz = df_val['SXZ (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
        del df_val

        self.console_textbox.append(f"Successfully validated and loaded steady-state stress file: {os.path.basename(filename)}\n")
        self.console_textbox.append(f"Steady-state stress data shape (m x n): {df.shape}")
        sys.stdout.flush()

    def solve(self):
        try:
            # Check if "Time History Mode" is enabled
            is_time_history_mode = self.time_history_checkbox.isChecked()

            # Check if the checkboxes are checked
            calculate_damage = self.damage_index_checkbox.isChecked()
            calculate_von_mises = self.von_mises_checkbox.isChecked()
            calculate_max_principal_stress = self.max_principal_stress_checkbox.isChecked()
            calculate_min_principal_stress = self.min_principal_stress_checkbox.isChecked()
            calculate_deformation = self.deformation_checkbox.isChecked()
            calculate_velocity      = self.velocity_checkbox.isChecked()
            calculate_acceleration  = self.acceleration_checkbox.isChecked()

            # Validation for Time History mode
            if is_time_history_mode:
                node_id_text = self.node_line_edit.text()
                if not node_id_text:
                    QMessageBox.warning(self, "Missing Input", "Please enter a Node ID for Time History mode.")
                    return

                try:
                    selected_node_id = int(node_id_text)
                    # This check requires that the modal stress file has been loaded first
                    if selected_node_id not in df_node_ids:
                         QMessageBox.warning(self, "Invalid Node ID", f"Node ID {selected_node_id} was not found in the loaded modal stress file.")
                         return
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "The entered Node ID is not a valid integer.")
                    return
                except NameError:
                     QMessageBox.warning(self, "Missing Data", "Cannot validate Node ID because the modal stress file has not been loaded.")
                     return

                # Validate that at least one output is selected
                time_history_outputs_selected = [
                    calculate_von_mises, calculate_max_principal_stress, calculate_min_principal_stress,
                    calculate_deformation, calculate_velocity, calculate_acceleration
                ]
                if not any(time_history_outputs_selected):
                    QMessageBox.warning(self, "No Output Selected", "Please select an output to plot for the time history analysis.")
                    return

            # Determine the output location
            output_directory = self.project_directory if self.project_directory else os.path.dirname(
                os.path.abspath(__file__))

            # Ensure modal data are defined before proceeding
            global modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord
            if (modal_sx is None or modal_sy is None or modal_sz is None or modal_sxy is None
                    or modal_syz is None or modal_sxz is None or modal_coord is None):
                self.console_textbox.append("Please load the modal coordinate and stress files before solving.")
                return

            if 'modal_ux' in globals() and modal_ux is not None:
                modal_deformations = (modal_ux, modal_uy, modal_uz)
            else:
                modal_deformations = None

            # region Apply Mode Skipping
            # Get the number of modes to be skipped, if defined by user
            skip_n = 0
            if hasattr(self, 'skip_modes_combo') and self.skip_modes_combo.isVisible():
                try:
                    skip_n = int(self.skip_modes_combo.currentText())
                except (ValueError, TypeError):
                    skip_n = 0

            # This slice will be applied to all modal arrays without modifying them globally
            mode_slice = slice(skip_n, None)

            # Ensure the number of modes to skip is valid
            if skip_n >= modal_sx.shape[1]:
                QMessageBox.critical(self, "Calculation Error",
                                     f"Cannot skip {skip_n} modes as only {modal_sx.shape[1]} are available.")
                self.progress_bar.setVisible(False)
                return

            # Slice the deformation tuple separately for clarity
            modal_deformations_filtered = None
            if modal_deformations is not None:
                modal_deformations_filtered = (
                    modal_deformations[0][:, mode_slice],
                    modal_deformations[1][:, mode_slice],
                    modal_deformations[2][:, mode_slice]
                )
            # endregion

            # Show the progress bar at the start of the solution
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Get the current date and time
            current_time = datetime.now()

            self.console_textbox.append(
                f"\n******************* BEGIN SOLVE ********************\nDatetime: {current_time}\n\n")

            # Check for steady-state stress inclusion
            is_include_steady_state = self.steady_state_checkbox.isChecked()

            # Check if Damage Index / Potential Damage is requested
            if self.damage_index_checkbox.isChecked():
                try:
                    fatigue_A = float(self.A_line_edit.text())
                    fatigue_m = float(self.m_line_edit.text())
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input",
                                        "Please enter valid numbers for fatigue parameters A and m.")
                    return
                # Save these values to the solver instance
                self.solver.fatigue_A = fatigue_A
                self.solver.fatigue_m = fatigue_m

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

                # Create an instance of the solver
                self.solver = MSUPSmartSolverTransient(
                    modal_sx[:, mode_slice],
                    modal_sy[:, mode_slice],
                    modal_sz[:, mode_slice],
                    modal_sxy[:, mode_slice],
                    modal_syz[:, mode_slice],
                    modal_sxz[:, mode_slice],
                    modal_coord[mode_slice, :],
                    steady_sx=steady_sx,
                    steady_sy=steady_sy,
                    steady_sz=steady_sz,
                    steady_sxy=steady_sxy,
                    steady_syz=steady_syz,
                    steady_sxz=steady_sxz,
                    steady_node_ids=steady_node_ids,
                    modal_node_ids=df_node_ids,
                    output_directory=output_directory,
                    modal_deformations=modal_deformations_filtered
                )

                # Use the new method for single node processing
                time_indices, y_data = self.solver.process_results_for_a_single_node(
                    selected_node_idx,
                    calculate_von_mises = calculate_von_mises,
                    calculate_max_principal_stress = calculate_max_principal_stress,
                    calculate_min_principal_stress = calculate_min_principal_stress,
                    calculate_deformation          = calculate_deformation,
                    calculate_velocity             = calculate_velocity,
                    calculate_acceleration         = calculate_acceleration
                )

                if time_indices is not None and y_data is not None:
                    # Plot the time history of the selected stress component
                    self.plot_single_node_tab.update_plot(time_values, y_data, selected_node_id,
                                                          is_max_principal_stress=calculate_max_principal_stress,
                                                          is_min_principal_stress=calculate_min_principal_stress,
                                                          is_von_mises=calculate_von_mises,
                                                          is_deformation=calculate_deformation,
                                                          is_velocity=calculate_velocity,
                                                          is_acceleration=calculate_acceleration)

                self.progress_bar.setVisible(False)  # Hide progress bar for single-node operation
                return  # Exit early, no need to write files

            # Create an instance of MSUPSmartSolverTransient for batch solver
            self.solver = MSUPSmartSolverTransient(
                modal_sx[:, mode_slice],
                modal_sy[:, mode_slice],
                modal_sz[:, mode_slice],
                modal_sxy[:, mode_slice],
                modal_syz[:, mode_slice],
                modal_sxz[:, mode_slice],
                modal_coord[mode_slice, :],
                steady_sx=steady_sx,
                steady_sy=steady_sy,
                steady_sz=steady_sz,
                steady_sxy=steady_sxy,
                steady_syz=steady_syz,
                steady_sxz=steady_sxz,
                steady_node_ids=steady_node_ids,
                modal_node_ids=df_node_ids,
                output_directory=output_directory,
                modal_deformations = modal_deformations_filtered
            )

            # Connect the solver's progress signal to the progress bar update slot
            self.solver.progress_signal.connect(self.update_progress_bar)

            # Run the process_results method
            start_time = time.time()
            self.solver.process_results(
                calculate_damage=calculate_damage,
                calculate_von_mises=calculate_von_mises,
                calculate_max_principal_stress=calculate_max_principal_stress,
                calculate_min_principal_stress=calculate_min_principal_stress,
                calculate_deformation=calculate_deformation,
                calculate_velocity=calculate_velocity,
                calculate_acceleration=calculate_acceleration
            )
            end_time_main_calc = time.time() - start_time

            current_time = datetime.now()

            self.console_textbox.append(
                f"******************** END SOLVE *********************\nDatetime: {current_time}\n\n")

            # Log the completion
            self.console_textbox.append(f"Main calculation routine completed in: {end_time_main_calc:.2f} seconds\n")
            self.console_textbox.moveCursor(QTextCursor.End)  # Move cursor to the end
            self.console_textbox.ensureCursorVisible()  # Ensure the cursor is visible

            def update_plot(self, time_values, traces=None):
                """
                Dynamically plots multiple data traces and populates a table.
                - traces: A list of dictionaries, e.g., [{'name': 'Von Mises (MPa)', 'data': np.array([...])}]
                """
                if traces is None:
                    traces = []

                # 1) Build figure by iterating through the provided traces
                fig = go.Figure()
                for trace_info in traces:
                    fig.add_trace(
                        go.Scattergl(x=time_values, y=trace_info['data'], mode='lines', name=trace_info['name']))

                fig.update_layout(
                    xaxis_title="Time [s]",
                    yaxis_title="Value",  # Generic Y-axis title
                    template="plotly_white",
                    font=dict(size=7),
                    margin=dict(l=40, r=40, t=10, b=0),
                    legend=dict(font=dict(size=7))
                )

                # 2) Wrap in resampler
                resfig = FigureResampler(fig, default_n_shown_samples=50000)

                # Show the plot
                main_win = self.window()
                main_win.load_fig_to_webview(resfig, self.web_view)

                # 3) Dynamically populate the table
                headers = ["Time [s]"] + [trace['name'] for trace in traces]
                self.model.setHorizontalHeaderLabels(headers)
                self.model.removeRows(0, self.model.rowCount())

                for i, t in enumerate(time_values):
                    # Start each row with the time value
                    row_items = [QStandardItem(f"{t:.5f}")]
                    # Add the data from each trace for the current time step
                    for trace in traces:
                        row_items.append(QStandardItem(f"{trace['data'][i]:.5f}"))
                    self.model.appendRow(items)

            # region Create maximum over time plot if solver is not run in Time History mode
            if not self.time_history_checkbox.isChecked():
                # --- Maximum Over Time Plot ---
                max_traces = []
                min_traces = []
                if self.von_mises_checkbox.isChecked():
                    max_traces.append({'name': 'Von Mises (MPa)', 'data': self.solver.max_over_time_svm})
                    von_mises_data_max_over_time = self.solver.max_over_time_svm
                else:
                    von_mises_data_max_over_time = None

                if self.max_principal_stress_checkbox.isChecked():
                    max_traces.append({'name': 'S1 (MPa)', 'data': self.solver.max_over_time_s1})
                    max_principal_data_max_over_time = self.solver.max_over_time_s1
                else:
                    max_principal_data_max_over_time = None

                if self.deformation_checkbox.isChecked():
                    max_traces.append({'name': 'Deformation (mm)', 'data': self.solver.max_over_time_def})
                    deformation_data_max_over_time = self.solver.max_over_time_def
                else:
                    deformation_data_max_over_time = None

                if self.velocity_checkbox.isChecked():
                    max_traces.append({'name': 'Velocity (mm/s)', 'data': self.solver.max_over_time_vel})
                    velocity_data_max_over_time = self.solver.max_over_time_vel
                else:
                    velocity_data_max_over_time = None

                if self.acceleration_checkbox.isChecked():
                    max_traces.append({'name': 'Acceleration (mm/s²)', 'data': self.solver.max_over_time_acc})
                    acceleration_data_max_over_time = self.solver.max_over_time_acc
                else:
                    acceleration_data_max_over_time = None

                if max_traces:  # Only create and show the tab if there is data
                    if not hasattr(self, 'plot_max_over_time_tab'):
                        self.plot_max_over_time_tab = PlotlyMaxWidget()
                        modal_tab_index = self.show_output_tab_widget.indexOf(self.plot_modal_coords_tab)
                        self.show_output_tab_widget.insertTab(modal_tab_index + 1, self.plot_max_over_time_tab,
                                                              "Maximum Over Time")

                    self.plot_max_over_time_tab.update_plot(time_values, traces=max_traces)
                    self.show_output_tab_widget.setTabVisible(
                        self.show_output_tab_widget.indexOf(self.plot_max_over_time_tab), True)

                # --- Minimum Over Time Plot ---
                if calculate_min_principal_stress:
                    min_traces = [{'name': 'S3 (MPa)', 'data': self.solver.min_over_time_s3}]

                    if not hasattr(self, 'plot_min_over_time_tab'):
                        self.plot_min_over_time_tab = PlotlyMaxWidget()
                        idx = self.show_output_tab_widget.indexOf(self.plot_max_over_time_tab)
                        self.show_output_tab_widget.insertTab(idx + 1, self.plot_min_over_time_tab, "Minimum Over Time")

                    self.plot_min_over_time_tab.update_plot(time_values, traces=min_traces)
                    self.show_output_tab_widget.setTabVisible(
                        self.show_output_tab_widget.indexOf(self.plot_min_over_time_tab), True)

                # region Update the scalar range spinboxes in the Display tab using the calculated min and max values
                if von_mises_data_max_over_time is not None:
                    scalar_min = np.min(von_mises_data_max_over_time)
                    scalar_max = np.max(von_mises_data_max_over_time)
                elif max_principal_data_max_over_time is not None:
                    scalar_min = np.min(max_principal_data_max_over_time)
                    scalar_max = np.max(max_principal_data_max_over_time)
                elif deformation_data_max_over_time is not None:
                    scalar_min = np.min(deformation_data_max_over_time)
                    scalar_max = np.max(deformation_data_max_over_time)
                elif velocity_data_max_over_time is not None:
                    scalar_min = np.min(velocity_data_max_over_time)
                    scalar_max = np.max(velocity_data_max_over_time)
                elif acceleration_data_max_over_time is not None:
                    scalar_min = np.min(acceleration_data_max_over_time)
                    scalar_max = np.max(acceleration_data_max_over_time)
                else:
                    scalar_min = None
                    scalar_max = None

                if scalar_min is not None and scalar_max is not None:
                    # Retrieve the DisplayTab instance
                    display_tab = self.window().display_tab
                    display_tab.scalar_min_spin.blockSignals(True)
                    display_tab.scalar_max_spin.blockSignals(True)
                    display_tab.scalar_min_spin.setRange(scalar_min, scalar_max)
                    # We use 1e30 as an arbitrary high upper bound
                    display_tab.scalar_max_spin.setRange(scalar_min, 1e30)
                    display_tab.scalar_min_spin.setValue(scalar_min)
                    display_tab.scalar_max_spin.setValue(scalar_max)
                    display_tab.scalar_min_spin.blockSignals(False)
                    display_tab.scalar_max_spin.blockSignals(False)
                # endregion
            # endregion

        except Exception as e:
            self.console_textbox.append(f"Error during solving process: {e}, Datetime: {current_time}")
            self.console_textbox.moveCursor(QTextCursor.End)  # Move cursor to the end
            self.console_textbox.ensureCursorVisible()  # Ensure the cursor is visible

    def _update_solve_button_state(self):
        """Enables the SOLVE button only if all required primary inputs are loaded."""
        if self.coord_loaded and self.stress_loaded:
            self.solve_button.setEnabled(True)
        else:
            self.solve_button.setEnabled(False)

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
            is_max_principal_stress = self.max_principal_stress_checkbox.isChecked()
            is_von_mises = self.von_mises_checkbox.isChecked()

            # Update plot widget
            self.plot_single_node_tab.update_plot(x_data, y_data, node_id,
                                                  is_max_principal_stress=is_max_principal_stress,
                                                  is_von_mises=is_von_mises)

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

    # region Handle mouse-based UI functionality
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

    # endregion

    # region Handle keyboard-based UI functionality
    # Nothing here yet
    # endregion


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.temp_files = []  # List to track temp files

        # Window title and dimensions
        self.setWindowTitle('MSUP Smart Solver - v0.96.2')
        self.setGeometry(40, 40, 600, 800)

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

        # Create the Navigator (File Explorer)
        self.create_navigator()

        # Add "File" menu
        file_menu = self.menu_bar.addMenu("File")

        # Add "Select Project Directory" action
        select_dir_action = QAction("Select Project Directory", self)
        select_dir_action.triggered.connect(self.select_project_directory)
        file_menu.addAction(select_dir_action)

        # Add a "View" menu option to show/hide Navigator
        view_menu = self.menu_bar.addMenu("View")
        toggle_navigator_action = self.navigator_dock.toggleViewAction()
        toggle_navigator_action.setText("Navigator")
        view_menu.addAction(toggle_navigator_action)

        # Add Settings menu
        settings_menu = self.menu_bar.addMenu("Settings")

        # Add "Advanced" action under Settings menu
        advanced_settings_action = QAction("Advanced", self)
        advanced_settings_action.triggered.connect(self.open_advanced_settings)
        settings_menu.addAction(advanced_settings_action)

        # Create a QTabWidget
        self.tab_widget = QTabWidget()

        tab_style = """
        QTabBar::tab {
            background-color: #d6e4f5;     /* Paler blue for inactive tabs */
            color: #666666;                /* Dimmed text for inactive tabs */
            border: 1px solid #5b9bd5;
            padding: 3px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            margin: 2px;
            font-size: 8pt;
            min-width: 100px;
        }

        QTabBar::tab:selected {
            background-color: #e7f0fd;     /* Active tab: your current blue theme */
            color: #000000;                /* Bold text */
            border: 3px solid #5b9bd5;
        }

        QTabBar::tab:hover {
            background-color: #cce4ff;     /* Lighter blue on hover */
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

    def create_navigator(self):
        """Create a dockable navigator showing project directory contents."""
        self.navigator_dock = QDockWidget("Navigator", self)
        self.navigator_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.navigator_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)

        # Get the Desktop path dynamically
        # desktop_path = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)

        # Create file system model
        self.file_model = QFileSystemModel()
        # self.file_model.setRootPath(desktop_path)  # Initially Desktop, updates when project directory is selected
        self.file_model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)  # Show all files & folders

        # Apply file type filter
        self.file_model.setNameFilters(["*.csv", "*.mcf", "*.txt"])  # Only show CSV, MCF, TXT files
        self.file_model.setNameFilterDisables(False)  # Disable showing grayed-out files

        # Create Tree View
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        # self.tree_view.setRootIndex(self.file_model.index(desktop_path))  # Start at Desktop
        self.tree_view.doubleClicked.connect(self.open_navigator_file)
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

        # Enable drag and drop on the TreeView
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

    def open_navigator_file(self, index):
        """
        Opens the double-clicked file from the navigator in the default system
        application, attempting to maximize it on Windows.
        """
        if self.file_model.isDir(index):
            return  # Do nothing for directories

        file_path = self.file_model.filePath(index)

        try:
            subprocess.run(['cmd', '/c', 'start', '/max', '', file_path], shell=True)

        except Exception as e:
            print(f"Error opening file '{file_path}': {e}")

    def load_fig_to_webview(self, fig, web_view):
        """Generates full HTML with embedded JS, saves to temp file, and loads."""
        try:
            # Handle FigureResampler object if passed
            plotly_fig = fig.figure if hasattr(fig, 'figure') else fig

            html_content = pio.to_html(plotly_fig,
                                       full_html=True,
                                       include_plotlyjs=True,  # Embed JS
                                       config={'responsive': True})

            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
                tmp_file.write(html_content)
                file_path = tmp_file.name
                self.temp_files.append(file_path)

            web_view.setUrl(QUrl.fromLocalFile(file_path))
            web_view.show()
        except Exception as e:
            print(f"Error loading figure to webview: {e}")
            traceback.print_exc()
            # Display error in webview
            error_html = f"<html><body><h1>Error loading plot</h1><pre>{e}</pre><pre>{traceback.format_exc()}</pre></body></html>"
            try:
                web_view.setHtml(error_html)
            except Exception:
                pass  # Ignore errors setting error html

    def closeEvent(self, event):
        """Clean up temporary files on application close."""
        self.clear_plot_cache(show_message=False)
        event.accept()

    def open_advanced_settings(self):
        """Opens the advanced settings dialog and applies changes if accepted."""
        dialog = AdvancedSettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_settings()
            self.apply_advanced_settings(settings)
            QMessageBox.information(self, "Settings Applied",
                                    "New advanced settings have been applied.\n"
                                    "They will be used for the next solve operation.")

    def apply_advanced_settings(self, settings):
        """Updates the global variables based on the dialog's settings."""
        global RAM_PERCENT, DEFAULT_PRECISION, IS_GPU_ACCELERATION_ENABLED
        global NP_DTYPE, TORCH_DTYPE, RESULT_DTYPE

        # Update the primary global variables
        RAM_PERCENT = settings["ram_percent"]
        DEFAULT_PRECISION = settings["precision"]
        IS_GPU_ACCELERATION_ENABLED = settings["gpu_acceleration"]

        # Update the derived precision-related variables
        if DEFAULT_PRECISION == 'Single':
            NP_DTYPE = np.float32
            TORCH_DTYPE = torch.float32
            RESULT_DTYPE = 'float32'
        elif DEFAULT_PRECISION == 'Double':
            NP_DTYPE = np.float64
            TORCH_DTYPE = torch.float64
            RESULT_DTYPE = 'float64'

        print("\n--- Advanced settings updated ---")
        print(f"  RAM Allocation: {RAM_PERCENT * 100:.0f}%")
        print(f"  Solver Precision: {DEFAULT_PRECISION}")
        print(f"  GPU Acceleration: {'Enabled' if IS_GPU_ACCELERATION_ENABLED else 'Disabled'}")
        print("---------------------------------")


class AdvancedSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setMinimumWidth(400)

        # --- Define fonts for different elements ---
        main_font = QFont()
        main_font.setPointSize(10)  # Main font for labels and controls

        group_title_font = QFont()
        group_title_font.setPointSize(10)

        # --- Current values for reference ---
        global_settings_text = (
            f"Current settings:\n"
            f"- Precision: {DEFAULT_PRECISION}\n"
            f"- RAM Limit: {RAM_PERCENT * 100:.0f}%\n"
            f"- GPU Acceleration: {'Enabled' if IS_GPU_ACCELERATION_ENABLED else 'Disabled'}"
        )
        self.current_settings_label = QLabel(global_settings_text)
        # Style this label specifically for a 'console' look
        self.current_settings_label.setStyleSheet("""
            background-color: #f0f0f0; 
            border: 1px solid #dcdcdc; 
            padding: 8px; 
            border-radius: 3px;
            font-family: Consolas, Courier New, monospace;
            font-size: 9pt;
        """)

        # --- Create widgets for modification ---
        self.ram_label = QLabel("Set RAM Allocation (%):")
        self.ram_spinbox = QSpinBox()
        self.ram_spinbox.setRange(10, 95)
        self.ram_spinbox.setValue(int(RAM_PERCENT * 100))
        self.ram_spinbox.setToolTip("Set the maximum percentage of available RAM the solver can use. It will based on allowable free memory.")

        self.precision_label = QLabel("Set Solver Precision:")
        self.precision_combobox = QComboBox()
        self.precision_combobox.addItems(["Single", "Double"])
        self.precision_combobox.setCurrentText(DEFAULT_PRECISION)
        self.precision_combobox.setToolTip(
            "Single precision is faster and uses less memory.\nDouble precision is more accurate but slower.")

        self.gpu_checkbox = QCheckBox("Enable GPU Acceleration (Only works if NVIDIA CUDA is installed in PC)")
        self.gpu_checkbox.setChecked(IS_GPU_ACCELERATION_ENABLED)
        self.gpu_checkbox.setToolTip("Uses the GPU for matrix multiplication if a compatible NVIDIA GPU is found and CUDA is installed in the system.")

        # --- Apply font to the widgets ---
        self.ram_label.setFont(main_font)
        self.ram_spinbox.setFont(main_font)
        self.precision_label.setFont(main_font)
        self.precision_combobox.setFont(main_font)
        self.gpu_checkbox.setFont(main_font)

        # --- Layout ---
        layout = QGridLayout()
        layout.setSpacing(15)
        layout.addWidget(self.ram_label, 0, 0)
        layout.addWidget(self.ram_spinbox, 0, 1)
        layout.addWidget(self.precision_label, 1, 0)
        layout.addWidget(self.precision_combobox, 1, 1)
        layout.addWidget(self.gpu_checkbox, 2, 0, 1, 2)

        # --- GroupBox to hold the settings ---
        settings_group = QGroupBox("Modify Global Parameters")
        settings_group.setFont(group_title_font)
        settings_group.setLayout(layout)

        # --- OK and Cancel buttons ---
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.setFont(main_font)  # Apply font to buttons

        # --- Main layout for the dialog ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.current_settings_label)
        main_layout.addWidget(settings_group)
        main_layout.addStretch()  # Add a spacer
        main_layout.addWidget(self.buttons)
        self.setLayout(main_layout)

    def get_settings(self):
        """Returns the selected settings from the dialog widgets."""
        return {
            "ram_percent": self.ram_spinbox.value() / 100.0,
            "precision": self.precision_combobox.currentText(),
            "gpu_acceleration": self.gpu_checkbox.isChecked(),
        }
# endregion

# region Run the main GUI
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # Use high DPI icons and images
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox, QTextEdit, QLineEdit {
            font-size: 8pt;
        }
    """)

    # Create the main window and show it
    main_window = MainWindow()
    main_window.showMaximized()

    sys.exit(app.exec_())
# endregion
# region Run the main GUI
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # Use high DPI icons and images
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox, QTextEdit, QLineEdit {
            font-size: 8pt;
        }
    """)

    # Create the main window and show it
    main_window = MainWindow()
    main_window.showMaximized()

    sys.exit(app.exec_())
# endregion
