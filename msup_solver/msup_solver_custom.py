# region Import libraries
import numpy as np
import pandas as pd
import torch
import psutil
import gc
from numba import njit, prange
import time
from datetime import datetime

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, QProgressBar, QFileDialog, QGroupBox, QGridLayout, QTextEdit
from PyQt5.QtGui import QPalette, QColor, QFont, QTextCursor
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot
import sys
from io import StringIO
# endregion

# region Define global variables
IS_GPU_ACCELERATION_ENABLED = False

IS_WRITE_TO_DISK_STRESS_S1_MAX_AT_ALL_TIME_POINTS = False
IS_WRITE_TO_DISK_TIMES_OF_MAX_STRESS_S1 = False

IS_WRITE_TO_DISK_VON_MISES_MAX_AT_ALL_TIME_POINTS = False
IS_WRITE_TO_DISK_TIMES_OF_MAX_VON_STRESS_AT_EACH_NODE = False

IS_WRITE_TO_DISK_POTENTIAL_DAMAGE = True
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
    damages = np.zeros(num_nodes, dtype=np.float32)
    for i in prange(num_nodes):
        series = sigma_vm[i, :]
        ranges, counts = rainflow_counter(series)
        # Compute damage
        damage = np.sum(counts / (A / ((ranges + 1e-10) ** m)))
        damages[i] = damage
    return damages
# endregion

# region Define global class & functions
class PotentialDamageSolverByMSUP(QObject):
    progress_signal = pyqtSignal(int)

    def __init__(self, modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord):
        super().__init__()

        # Global settings
        self.write_to_disk_potential_damage = IS_WRITE_TO_DISK_POTENTIAL_DAMAGE
        self.DTYPE = np.float32
        self.ELEMENT_SIZE = np.dtype(self.DTYPE).itemsize
        self.RAM_PERCENT = 0.6

        # Initialize modal inputs
        self.device = torch.device("cuda" if IS_GPU_ACCELERATION_ENABLED and torch.cuda.is_available() else "cpu")
        self.modal_sx = torch.tensor(modal_sx, dtype=torch.float32).to(self.device)
        self.modal_sy = torch.tensor(modal_sy, dtype=torch.float32).to(self.device)
        self.modal_sz = torch.tensor(modal_sz, dtype=torch.float32).to(self.device)
        self.modal_sxy = torch.tensor(modal_sxy, dtype=torch.float32).to(self.device)
        self.modal_syz = torch.tensor(modal_syz, dtype=torch.float32).to(self.device)
        self.modal_sxz = torch.tensor(modal_sxz, dtype=torch.float32).to(self.device)
        self.modal_coord = torch.tensor(modal_coord, dtype=torch.float32).to(self.device)

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

    def process_results(self):
        """Process stress results to compute potential (relative) fatigue damage."""
        num_nodes, num_modes = self.modal_sx.shape
        num_time_points = self.modal_coord.shape[1]

        chunk_size = self.get_chunk_size(num_nodes, num_time_points)
        num_iterations = (num_nodes + chunk_size - 1) // chunk_size
        print(f"Estimated number of iterations to avoid memory overflow: {num_iterations}")

        memory_required_per_iteration = self.estimate_ram_required_per_iteration(chunk_size, num_time_points,
                                                                                 self.ELEMENT_SIZE)
        print(f"Estimated RAM required per iteration: {memory_required_per_iteration:.2f} GB\n")

        if self.write_to_disk_potential_damage:
            potential_damage_memmap = np.memmap('potential_damage_results.dat', dtype='float32',
                                                mode='w+', shape=(num_nodes,))

        # Create memmap file for storing the maximum von Mises stresses per node
        von_mises_max_memmap = np.memmap('max_von_mises_stress.dat', dtype='float32', mode='w+',
                                         shape=(num_nodes,))

        # Create memmap file for storing the time points of max von Mises stress per node
        von_mises_max_time_memmap = np.memmap('time_of_max_von_mises_stress.dat', dtype='float32', mode='w+',
                                              shape=(num_nodes,))

        for start_idx in range(0, num_nodes, chunk_size):
            end_idx = min(start_idx + chunk_size, num_nodes)

            # region Calculate principal stresses
            start_time = time.time()
            actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
                self.compute_principal_stresses(self.modal_sx, self.modal_sy, self.modal_sz, self.modal_sxy,
                                                self.modal_syz, self.modal_sxz, self.modal_coord, start_idx, end_idx)
            print(f"Elapsed time for principal stresses: {(time.time() - start_time):.3f} seconds")
            # endregion

            # region Calculate von-Mises stresses
            start_time = time.time()
            sigma_vm = self.compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz)
            print(f"Elapsed time for Von-Mises stresses: {(time.time() - start_time):.3f} seconds")
            # endregion

            # Calculate the maximum von Mises stress for each node
            start_time = time.time()
            max_von_mises_stress_per_node = np.max(sigma_vm, axis=1)
            time_of_max_von_mises_stress_per_node = np.argmax(sigma_vm, axis=1)

            von_mises_max_memmap[
            start_idx:end_idx] = max_von_mises_stress_per_node  # Write max von Mises stress to disk

            von_mises_max_time_memmap[
            start_idx:end_idx] = time_of_max_von_mises_stress_per_node  # Write time points of max stress
            print(f"Elapsed time for calculating max von Mises stress and time: {time.time() - start_time:.3f} seconds")

            # region Free up some memory
            start_time = time.time()
            del actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz
            print(f"Elapsed time for garbage collection: {time.time() - start_time:.3f} seconds")
            # endregion

            # Calculate potential damage index for all nodes in the chunk
            start_time = time.time()
            A = 1
            m = -3
            potential_damages = compute_potential_damage_for_all_nodes(sigma_vm, A, m)
            print(f"Elapsed time for rainflow counting and damage index calculation: {time.time() - start_time:.3f} seconds")

            if self.write_to_disk_potential_damage:
                potential_damage_memmap[start_idx:end_idx] = potential_damages  # Write damage results to disk

            current_available_memory = psutil.virtual_memory().available * self.RAM_PERCENT

            # Emit progress signal as a percentage of the total iterations
            current_iteration = (start_idx // chunk_size) + 1
            progress_percentage = (current_iteration / num_iterations) * 100  # Calculate the percentage
            self.progress_signal.emit(int(progress_percentage))  # Emit the progress percentage
            QApplication.processEvents()  # Keep UI responsive

            print(
                f"Iteration completed for nodes {start_idx} to {end_idx}. Available RAM: {current_available_memory / (1024 ** 3):.2f} GB \n")

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
        # Window title and dimensions
        self.setWindowTitle('MSUP Smart Solver - v0.1')
        self.setGeometry(100, 100, 700, 500)

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

        # Calculate Damage Index Checkbox
        self.damage_index_checkbox = QCheckBox('Calculate Damage Index')
        self.damage_index_checkbox.setStyleSheet("margin: 10px 0;")

        # Solve Button
        self.solve_button = QPushButton('Solve')
        self.solve_button.setStyleSheet(button_style)
        self.solve_button.setFont(QFont('Arial', 8, QFont.Bold))
        self.solve_button.clicked.connect(self.solve)

        # Read-only Log Terminal
        self.log_terminal = QTextEdit()
        self.log_terminal.setReadOnly(True)
        self.log_terminal.setStyleSheet("background-color: #ffffff; border: 1px solid #5b9bd5")
        self.log_terminal.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.log_terminal.setText('LOG TERMINAL:\n')

        # Set monospaced font for log terminal
        terminal_font = QFont("Consolas", 7)
        terminal_font.setStyleHint(QFont.Monospace)  # For a more console-like textbox
        self.log_terminal.setFont(terminal_font)

        # Create Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("border: 1px solid #5b9bd5; padding: 10px; background-color: #ffffff;")
        self.progress_bar.setValue(0)  # Start with 0% progress
        self.progress_bar.setAlignment(Qt.AlignCenter)  # Center the progress bar text
        self.progress_bar.setTextVisible(True)

        # File selection buttons
        self.coord_file_button.clicked.connect(self.select_coord_file)
        self.stress_file_button.clicked.connect(self.select_stress_file)

        # Layouts
        main_layout = QVBoxLayout()

        # Group box for file selection
        file_group = QGroupBox()
        file_group.setStyleSheet("font-weight: bold; border: 1px solid #5b9bd5; padding: 10px;")
        file_layout = QGridLayout()

        file_layout.addWidget(self.coord_file_button, 0, 0)
        file_layout.addWidget(self.coord_file_path, 0, 1)
        file_layout.addWidget(self.stress_file_button, 1, 0)
        file_layout.addWidget(self.stress_file_path, 1, 1)

        file_group.setLayout(file_layout)

        # Adding elements to main layout
        main_layout.addWidget(file_group)
        main_layout.addWidget(self.damage_index_checkbox)
        main_layout.addWidget(self.solve_button)
        main_layout.addWidget(self.log_terminal)
        main_layout.addWidget(self.progress_bar)

        self.setLayout(main_layout)

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

            # Drop the 'Time' column and transpose the DataFrame
            df_transposed_dropped = df.drop(columns='Time').transpose()

            # Convert the DataFrame to a global NumPy array
            global modal_coord
            modal_coord = df_transposed_dropped.to_numpy()

            del df, df_transposed_dropped

            # Log the success and shape of the resulting array
            self.log_terminal.append(f"Successfully processed modal coordinate input: {filename}")
            self.log_terminal.append(f"Modal coordinates tensor shape (m x n): {modal_coord.shape} \n")
        except Exception as e:
            self.log_terminal.append(f"Error processing modal coordinate file: {e}")

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
            df_node_ids = df[['NodeID']]

            # Drop the 'Node ID' column from the original DataFrame
            df = df.drop(columns=['NodeID'])

            # Extract columns related to each principal stress component
            global modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz
            modal_sx = df.filter(regex='(?i)sx_.*').to_numpy().astype(np.float32)
            modal_sy = df.filter(regex='(?i)sy_.*').to_numpy().astype(np.float32)
            modal_sz = df.filter(regex='(?i)sz_.*').to_numpy().astype(np.float32)
            modal_sxy = df.filter(regex='(?i)sxy_.*').to_numpy().astype(np.float32)
            modal_syz = df.filter(regex='(?i)syz_.*').to_numpy().astype(np.float32)
            modal_sxz = df.filter(regex='(?i)sxz_.*').to_numpy().astype(np.float32)

            # Log the success and shape of the DataFrame and arrays
            self.log_terminal.append(f"Successfully processed modal stress file: {filename}\n")
            self.log_terminal.append(f"Modal stress tensor shape (m x n): {df.shape}")
            self.log_terminal.append(f"Node IDs tensor shape: {df_node_ids.shape}\n")
            self.log_terminal.append(f"Principal stress components extracted: SX, SY, SZ, SXY, SYZ, SXZ")
            self.log_terminal.append(f"SX shape: {modal_sx.shape}, SY shape: {modal_sy.shape}, SZ shape: {modal_sz.shape}")
            self.log_terminal.append(f"SXY shape: {modal_sz.shape}, SYZ shape: {modal_syz.shape}, SXZ shape: {modal_sxz.shape}\n")
            self.log_terminal.verticalScrollBar().setValue(self.log_terminal.verticalScrollBar().maximum())
        except Exception as e:
            self.log_terminal.append(f"Error processing modal stress file: {e}")
            self.log_terminal.verticalScrollBar().setValue(self.log_terminal.verticalScrollBar().maximum())

    def solve(self):
        try:
            # Ensure modal data are defined before proceeding
            global modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord
            if modal_sx is None or modal_sy is None or modal_sz is None or modal_sxy is None or modal_syz is None or modal_sxz is None or modal_coord is None:
                self.log_terminal.append("Please load the modal coordinate and stress files before solving.")
                return

            # Get the current date and time
            current_time = datetime.now()

            self.log_terminal.append(f"********** BEGIN SOLVE *********** , Datetime: {current_time}\n\n")

            # Set up the logger to redirect print statements to the log terminal
            logger = Logger(self.log_terminal)
            sys.stdout = logger

            # Create an instance of PotentialDamageSolverByMSUP
            self.solver = PotentialDamageSolverByMSUP(modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz,
                                                      modal_coord)

            # Connect the solver's progress signal to the progress bar update slot
            self.solver.progress_signal.connect(self.update_progress_bar)

            # Run the process_results method
            start_time = time.time()
            self.solver.process_results()
            end_time_main_calc = time.time() - start_time

            current_time = datetime.now()

            self.log_terminal.append(f"********** END SOLVE *********** , Datetime: {current_time}\n\n")

            # Log the completion
            self.log_terminal.append(f"Main calculation routine completed in: {end_time_main_calc:.2f} seconds")

            # Reset stdout to default
            sys.stdout = sys.__stdout__
        except Exception as e:
            self.log_terminal.append(f"Error during solving process: {e}, Datetime: {current_time}")
            sys.stdout = sys.__stdout__

    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
# endregion


# region Run the main GUI
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # Use high DPI icons and images
    app = QApplication(sys.argv)
    solverGUI = MSUPSmartSolverGUI()
    solverGUI.show()
    sys.exit(app.exec_())
# endregion
