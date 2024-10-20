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
                             QTextEdit, QTabWidget, QComboBox)
from PyQt5.QtGui import QPalette, QColor, QFont, QTextCursor
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QUrl
import sys
from io import StringIO
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
# endregion

# region Define global variables
NP_DTYPE = np.float64
TORCH_DTYPE = torch.float64
RESULT_DTYPE = 'float64'

IS_GPU_ACCELERATION_ENABLED = False

IS_WRITE_TO_DISK_STRESS_S1_MAX_AT_ALL_TIME_POINTS = False
IS_WRITE_TO_DISK_TIMES_OF_MAX_STRESS_S1 = False

IS_WRITE_TO_DISK_VON_MISES_MAX_AT_ALL_TIME_POINTS = False
IS_WRITE_TO_DISK_TIMES_OF_MAX_VON_STRESS_AT_EACH_NODE = False

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
# endregion

# region Define global class & functions
class MSUPSmartSolverTransient(QObject):
    progress_signal = pyqtSignal(int)

    def __init__(self, modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord):
        super().__init__()

        # Global settings
        self.NP_DTYPE = NP_DTYPE
        self.TORCH_DTYPE = TORCH_DTYPE
        self.RESULT_DTYPE = RESULT_DTYPE
        self.ELEMENT_SIZE = np.dtype(self.NP_DTYPE).itemsize
        self.RAM_PERCENT = 0.1

        # Initialize modal inputs
        self.device = torch.device("cuda" if IS_GPU_ACCELERATION_ENABLED and torch.cuda.is_available() else "cpu")
        self.modal_sx = torch.tensor(modal_sx, dtype=TORCH_DTYPE).to(self.device)
        self.modal_sy = torch.tensor(modal_sy, dtype=TORCH_DTYPE).to(self.device)
        self.modal_sz = torch.tensor(modal_sz, dtype=TORCH_DTYPE).to(self.device)
        self.modal_sxy = torch.tensor(modal_sxy, dtype=TORCH_DTYPE).to(self.device)
        self.modal_syz = torch.tensor(modal_syz, dtype=TORCH_DTYPE).to(self.device)
        self.modal_sxz = torch.tensor(modal_sxz, dtype=TORCH_DTYPE).to(self.device)
        self.modal_coord = torch.tensor(modal_coord, dtype=TORCH_DTYPE).to(self.device)

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
    def compute_normal_stresses(modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord, start_idx,
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
    def compute_normal_stresses_for_a_single_node(modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                                                       modal_coord, selected_node_idx):
        """Compute actual stresses using matrix multiplication."""
        actual_sx = torch.matmul(modal_sx[selected_node_idx: selected_node_idx +1, :], modal_coord)
        actual_sy = torch.matmul(modal_sy[selected_node_idx: selected_node_idx +1, :], modal_coord)
        actual_sz = torch.matmul(modal_sz[selected_node_idx: selected_node_idx +1, :], modal_coord)
        actual_sxy = torch.matmul(modal_sxy[selected_node_idx: selected_node_idx +1, :], modal_coord)
        actual_syz = torch.matmul(modal_syz[selected_node_idx: selected_node_idx +1, :], modal_coord)
        actual_sxz = torch.matmul(modal_sxz[selected_node_idx: selected_node_idx +1, :], modal_coord)

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
    def compute_principal_stresses_yedek(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
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
    def compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
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

        chunk_size = self.get_chunk_size(num_nodes, num_time_points)
        num_iterations = (num_nodes + chunk_size - 1) // chunk_size
        print(f"Estimated number of iterations to avoid memory overflow: {num_iterations}")

        memory_required_per_iteration = self.estimate_ram_required_per_iteration(chunk_size, num_time_points,
                                                                                 self.ELEMENT_SIZE)
        print(f"Estimated RAM required per iteration: {memory_required_per_iteration:.2f} GB\n")

        if calculate_principal_stress:
            # Create memmap files for storing the maximum principal stresses per node (s1)
            s1_max_memmap = np.memmap('max_s1_stress.dat', dtype=RESULT_DTYPE, mode='w+', shape=(num_nodes,))
            s1_max_time_memmap = np.memmap('time_of_max_s1_stress.dat', dtype=RESULT_DTYPE, mode='w+',
                                           shape=(num_nodes,))

        if calculate_von_mises:
            # Create memmap files for storing the maximum von Mises stresses per node
            von_mises_max_memmap = np.memmap('max_von_mises_stress.dat', dtype=RESULT_DTYPE, mode='w+',
                                             shape=(num_nodes,))
            von_mises_max_time_memmap = np.memmap('time_of_max_von_mises_stress.dat', dtype=RESULT_DTYPE, mode='w+',
                                                  shape=(num_nodes,))

        if calculate_damage:
            potential_damage_memmap = np.memmap('potential_damage_results.dat', dtype=RESULT_DTYPE,
                                                mode='w+', shape=(num_nodes,))

        for start_idx in range(0, num_nodes, chunk_size):
            end_idx = min(start_idx + chunk_size, num_nodes)

            # Calculate normal stresses
            start_time = time.time()
            actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
                self.compute_normal_stresses(self.modal_sx, self.modal_sy, self.modal_sz, self.modal_sxy,
                                             self.modal_syz, self.modal_sxz, self.modal_coord, start_idx, end_idx)
            print(f"Elapsed time for normal stresses: {(time.time() - start_time):.3f} seconds")

            if calculate_von_mises:
                # Calculate von Mises stresses
                start_time = time.time()
                sigma_vm = self.compute_von_mises_stress(actual_sx, actual_sy, actual_sz,
                                                         actual_sxy, actual_syz, actual_sxz)
                print(f"Elapsed time for von Mises stresses: {(time.time() - start_time):.3f} seconds")

                # Calculate the maximum von Mises stress and its time index for each node
                start_time = time.time()
                max_von_mises_stress_per_node = np.max(sigma_vm, axis=1)
                time_of_max_von_mises_stress_per_node = np.argmax(sigma_vm, axis=1)
                von_mises_max_memmap[start_idx:end_idx] = max_von_mises_stress_per_node
                von_mises_max_time_memmap[start_idx:end_idx] = time_of_max_von_mises_stress_per_node
                print(f"Elapsed time for max von Mises stress and time: {(time.time() - start_time):.3f} seconds")

            if calculate_principal_stress:
                # Calculate principal stresses
                start_time = time.time()
                s1, s2, s3 = self.compute_principal_stresses(actual_sx, actual_sy, actual_sz,
                                                             actual_sxy, actual_syz, actual_sxz)
                print(f"Elapsed time for principal stresses: {(time.time() - start_time):.3f} seconds")

                # Calculate the maximum principal stress (s1) and its time index for each node
                start_time = time.time()
                max_s1_per_node = np.max(s1, axis=1)
                time_of_max_s1_per_node = np.argmax(s1, axis=1)
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
                  f"Available RAM: {current_available_memory / (1024 ** 3):.2f} GB\n")

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
                                    "max_von_mises_stress.dat",
                                    "max_von_mises_stress.csv",
                                    "SVM_Max")
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    "time_of_max_von_mises_stress.dat",
                                    "time_of_max_von_mises_stress.csv",
                                    "Time_of_SVM_Max")
        if calculate_principal_stress:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    "max_s1_stress.dat",
                                    "max_s1_stress.csv",
                                    "S1_Max")
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    "time_of_max_s1_stress.dat",
                                    "time_of_max_s1_stress.csv",
                                    "Time_of_S1_Max")
        if calculate_damage:
            self.convert_dat_to_csv(df_node_ids, num_nodes,
                                    "potential_damage_results.dat",
                                    "potential_damage_results.csv",
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
            self.compute_normal_stresses_for_a_single_node(self.modal_sx, self.modal_sy, self.modal_sz, self.modal_sxy,
                                         self.modal_syz, self.modal_sxz, self.modal_coord, selected_node_idx)

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
        """Converts a .dat file to a .csv file with NodeID as the first column and an appropriate header for the results."""
        try:
            # Read the memmap file as a NumPy array
            data = np.memmap(dat_filename, dtype=RESULT_DTYPE, mode='r', shape=(num_nodes,))

            # Combine NodeID and data into a single DataFrame
            df = pd.DataFrame({
                'NodeID': node_ids,  # First column is NodeID
                header: data  # Second column is the result with appropriate header
            })

            # Save to CSV
            df.to_csv(csv_filename, index=False)
            print(f"Successfully converted {dat_filename} to {csv_filename}.")
        except Exception as e:
            print(f"Error converting {dat_filename} to {csv_filename}: {e}")

class Logger:
    def __init__(self, text_edit):
        self.text_edit = text_edit
        self.terminal = sys.stdout
        self.log_stream = StringIO()

    def write(self, message):
        self.terminal.write(message)  # Keep writing to the original terminal
        self.log_stream.write(message)  # Save to internal buffer
        self.text_edit.moveCursor(QTextCursor.End)
        self.text_edit.insertPlainText(message)  # Add text to QTextEdit
        self.text_edit.moveCursor(QTextCursor.End)
        self.text_edit.ensureCursorVisible()  # Ensure the latest message is visible
        QApplication.processEvents()  # Process UI updates to refresh the log in real time

    def flush(self):
        pass

class MSUPSmartSolverGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

        # Initialize solver attribute
        self.solver = None

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

        # Checkbox for Time History Mode (Single Node)
        self.time_history_checkbox = QCheckBox('Time History Mode (Single Node)')
        self.time_history_checkbox.setStyleSheet("margin: 10px 0;")
        self.time_history_checkbox.toggled.connect(self.toggle_single_node_solution_group)

        # Checkbox for Calculate Principal Stress
        self.principal_stress_checkbox = QCheckBox('Principal Stress')
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

        # ComboBox for Node ID selection
        self.node_combo_box = QComboBox()
        self.node_combo_box.setStyleSheet(button_style)
        self.node_combo_box.currentIndexChanged.connect(self.on_node_selected)

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

        # Create the QWebEngineView for the Plotly Plot
        self.plot_single_node_tab = MatplotlibWidget()
        # Ensure the plot widget expands to fill the tab
        self.plot_single_node_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Placeholder Plotly plot
        self.update_single_node_plot()

        # Add the plot tab to the tab widget, but hide it initially
        self.show_output_tab_widget.addTab(self.plot_single_node_tab, "Plot (Time History)")
        # Make it initially hidden
        self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_single_node_tab), False)

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
        self.single_node_group = QGroupBox("Single Node Expansion")
        self.single_node_group.setStyleSheet(group_box_style)
        self.single_node_label = QLabel("Select a node:")
        self.single_node_label.setFont(QFont('Arial', 8))
        single_node_layout = QHBoxLayout()
        single_node_layout.addWidget(self.single_node_label)
        single_node_layout.addWidget(self.node_combo_box)
        self.single_node_group.setVisible(False)
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

    def toggle_damage_index_checkbox_visibility(self):
        if self.von_mises_checkbox.isChecked():
            self.damage_index_checkbox.setVisible(True)
        else:
            self.damage_index_checkbox.setVisible(False)

    def toggle_single_node_solution_group(self):
        try:
            if self.time_history_checkbox.isChecked():
                self.single_node_group.setVisible(True)
                self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_single_node_tab), True)
            else:
                self.single_node_group.setVisible(False)
                self.show_output_tab_widget.setTabVisible(self.show_output_tab_widget.indexOf(self.plot_single_node_tab), False)
        except Exception as e:
            print(f"Error in toggling single node group visibility: {e}")

    def update_single_node_plot(self):
        """Updates the placeholder plot inside the MatplotlibWidget."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
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
            # Read the file into a DataFrame, skipping the header information
            with open(filename, 'r') as file:
                for i, line in enumerate(file):
                    if 'Time' in line:
                        start_index = i
                        break

            # Read the data starting from the identified start line
            df = pd.read_csv(filename, sep='\s+', skiprows=start_index + 1, header=None)

            # Create the column names
            df.columns = ['Time'] + [f'Mode_{i}' for i in range(1, df.shape[1])]

            # Store the Time column separately
            global time_values
            time_values = df['Time'].to_numpy()

            # Drop the 'Time' column and transpose the DataFrame
            df_transposed_dropped = df.drop(columns='Time').transpose()

            # Convert the DataFrame to a global NumPy array
            global modal_coord
            modal_coord = df_transposed_dropped.to_numpy()

            del df, df_transposed_dropped

            # Log the success and shape of the resulting array
            self.console_textbox.append(f"Successfully processed modal coordinate input: {filename}")
            self.console_textbox.append(f"Modal coordinates tensor shape (m x n): {modal_coord.shape} \n")
        except Exception as e:
            self.console_textbox.append(f"Error processing modal coordinate file: {e}")

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

            # Populate the node combo box with actual node IDs from the modal stress file
            self.node_combo_box.clear()  # Clear any existing items
            self.node_combo_box.addItems([str(node_id) for node_id in df_node_ids])  # Add new node IDs

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

    def solve(self):
        try:
            # Ensure modal data are defined before proceeding
            global modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord
            if modal_sx is None or modal_sy is None or modal_sz is None or modal_sxy is None or modal_syz is None or modal_sxz is None or modal_coord is None:
                self.console_textbox.append("Please load the modal coordinate and stress files before solving.")
                return

            # Show the progress bar at the start of the solution
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Get the current date and time
            current_time = datetime.now()

            self.console_textbox.append(f"\n******************* BEGIN SOLVE ********************\nDatetime: {current_time}\n\n")

            # Set up the logger to redirect print statements to the log terminal
            logger = Logger(self.console_textbox)
            sys.stdout = logger

            # Check if the checkboxes are checked
            calculate_damage = self.damage_index_checkbox.isChecked()
            calculate_von_mises = self.von_mises_checkbox.isChecked()
            calculate_principal_stress = self.principal_stress_checkbox.isChecked()

            # Check if "Time History Mode" is enabled
            is_time_history_mode = self.time_history_checkbox.isChecked()

            if is_time_history_mode:
                # Process only the selected node for time history mode
                selected_node_id = int(self.node_combo_box.currentText())  # Get selected node ID as index
                selected_node_idx = get_node_index_from_id(selected_node_id, df_node_ids)

                self.console_textbox.append(f"Time History Mode enabled for Node {selected_node_id}\n")

                # Create an instance of the solver
                self.solver = MSUPSmartSolverTransient(modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                                                       modal_coord)

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
                return  # Exit early, no need to write files or calculate damage

            # Create an instance of MSUPSmartSolverTransient for batch solver
            self.solver = MSUPSmartSolverTransient(modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                                                   modal_coord)

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

            # Reset stdout to default
            sys.stdout = sys.__stdout__
        except Exception as e:
            self.console_textbox.append(f"Error during solving process: {e}, Datetime: {current_time}")
            self.console_textbox.moveCursor(QTextCursor.End)  # Move cursor to the end
            self.console_textbox.ensureCursorVisible()  # Ensure the cursor is visible
            sys.stdout = sys.__stdout__

    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        if value >= 100:
            # Hide the progress bar once the process is finished
            self.progress_bar.setVisible(False)

    def on_node_selected(self):
        """Update the plot when a new node is selected."""
        try:
            selected_node_id = self.node_combo_box.currentText()

            # Dummy data for the plot (replace with actual data)
            x_data = [1, 2, 3, 4, 5]
            y_data = [0, 0, 0, 0, 0]

            # Check if "Principal Stress" or "Von Mises" is selected
            is_principal_stress = self.principal_stress_checkbox.isChecked()
            is_von_mises = self.von_mises_checkbox.isChecked()

            # Update the plot with the selected node ID and the checkbox statuses
            self.plot_single_node_tab.update_plot(x_data, y_data, selected_node_id, is_principal_stress, is_von_mises)

            # Log the selected Node ID in the log terminal
            self.console_textbox.append(f"Selected Node ID: {selected_node_id}")
            self.console_textbox.moveCursor(QTextCursor.End)
            self.console_textbox.ensureCursorVisible()  # Ensure the log scrolls to the bottom
        except Exception as e:
            print(f"Error updating plot with selected node: {e}")

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
        self.ax.clear()  # Clear the existing plot
        self.ax.plot(x, y)

        # Turn on grid
        # Enable major and minor grids
        self.ax.grid(True, which='major', linestyle='-', linewidth=0.5)  # Major grid
        self.ax.grid(True, which='minor', linestyle=':', linewidth=0.25, color='gray')  # Minor grid

        # Turn on the minor ticks
        self.ax.minorticks_on()

        # Set the title with a smaller font size
        if node_id:
            self.ax.set_title(f"Node ID: {node_id}", fontsize=8)  # Decreased title font size
        else:
            self.ax.set_title("Plot", fontsize=8)  # Default title with smaller font size

        # Set smaller font sizes for axis labels
        self.ax.set_xlabel('Time [seconds]', fontsize=8)

        # Dynamically adjust the Y-axis label based on the checkbox selection
        if is_principal_stress:
            self.ax.set_ylabel(r'$\sigma_1$ [MPa]', fontsize=8)  # Principal Stress with LaTeX-style subscript
        elif is_von_mises:
            self.ax.set_ylabel(r'$\sigma_{VM}$ [MPa]', fontsize=8)  # Von Mises Stress with LaTeX-style subscript
        else:
            self.ax.set_ylabel('Y-Axis', fontsize=8)  # Default Y-Axis label

        # Add a textbox to display the maximum Y value
        max_y_value = np.max(y)
        textstr = f'Max Value: {max_y_value:.4f}'
        self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes,
                     fontsize=8, verticalalignment='top', horizontalalignment='left',
                     bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

        # Adjust tick label font sizes
        self.ax.tick_params(axis='both', which='major', labelsize=8)  # Decrease tick label size

        # Manually adjust the margins to provide space for x-axis label
        self.figure.subplots_adjust(bottom=0.2)  # Set bottom margin
        self.canvas.draw()  # Redraw the canvas

    def resizeEvent(self, event):
        """Handle the resize event to make sure the plot fits the window."""
        self.canvas.resize(event.size())  # Resize the canvas along with the widget
        self.canvas.draw()  # Redraw the plot on resizing

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window title and dimensions
        self.setWindowTitle('MSUP Smart Solver - v0.1')
        self.setGeometry(100, 100, 600, 620)

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

        # Create the "Batch Solver" tab and add the MSUPSmartSolverGUI widget to it
        self.batch_solver_tab = MSUPSmartSolverGUI()
        self.tab_widget.addTab(self.batch_solver_tab, "Batch Solver")

        # Set the central widget of the main window to the tab widget
        self.setCentralWidget(self.tab_widget)
# endregion


# region Run the main GUI
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # Use high DPI icons and images
    app = QApplication(sys.argv)

    # Create the main window and show it
    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())
# endregion
