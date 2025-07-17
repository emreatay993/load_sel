"""
MSUP Smart Solver for Transient Analysis.

This application provides a graphical user interface (GUI) for performing
transient structural analysis using the Mode Superposition (MSUP) method.
It allows users to load modal data, compute stress and deformation results
over time, and visualize the outputs in various ways.
"""
# region Import libraries
print("Importing libraries...")

# ---- Standard Library Imports ----
import gc
import io
import math
import os
import subprocess
import sys
import tempfile
import time
import threading
from datetime import datetime
from io import StringIO

# ---- Third-Party Imports ----
# ImageIO and Psutil
import imageio
import psutil

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# Numba
from numba import njit, prange

# NumPy
import numpy as np

# Pandas
import pandas as pd

# Plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
from plotly_resampler import FigureResampler

# PySide/PyQt
from PyQt5.QtCore import (QDir, QObject, QStandardPaths, Qt, QTimer, QUrl,
                          pyqtSignal, pyqtSlot)
from PyQt5.QtGui import (QColor, QDoubleValidator, QFont, QKeySequence,
                         QPalette, QStandardItem, QStandardItemModel,
                         QTextCursor)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (QAbstractItemView, QAction, QApplication, QDialog, QVBoxLayout,
                             QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDockWidget, QDoubleSpinBox, QFileDialog,
                             QFileSystemModel, QGridLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                             QLineEdit, QMainWindow, QMenu, QMenuBar,
                             QMessageBox, QProgressBar, QProgressDialog,
                             QPushButton, QShortcut, QSizePolicy, QSpinBox,
                             QSplitter, QTabWidget, QTableView, QTextEdit,
                             QTreeView, QVBoxLayout, QWidget, QWidgetAction)

# PyTorch
import torch

# PyVista
import pyvista as pv
from pyvistaqt import QtInteractor

# SciPy
from scipy.signal import butter, detrend, filtfilt

# VTK
import vtk

# Back-end Modules
from solver_engine import MSUPSmartSolverTransient
import solver_engine
from display_tab import DisplayTab

print("Done.")

# endregion

# region Define global variables

# --- Solver Configuration ---
# These constants control the core behavior and precision of the solver.

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
# Set OpenBLAS to use all available CPU cores for NumPy operations.
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

# endregion

# region Define global functions
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


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Attributes for interactivity
        self.ax = None
        self.annot = None
        self.plotted_lines = []
        self.legend_map = {} # Used to map legend items to plot lines

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
        # Connect a legend pick event
        self.canvas.mpl_connect('pick_event', self.on_legend_pick)

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

    def on_legend_pick(self, event):
        """
        Handles clicks on legend items to toggle plot line visibility.
        """
        # The artist that was picked (e.g., a legend line or text)
        artist = event.artist

        # Check if the picked artist is part of our interactive legend
        if artist in self.legend_map:
            # Find the original plot line this legend item corresponds to
            original_line = self.legend_map[artist]

            # Toggle the visibility of the plot line
            is_visible = not original_line.get_visible()
            original_line.set_visible(is_visible)

            # Visually update all parts of the corresponding legend entry (line and text)
            for leg_artist, line in self.legend_map.items():
                if line == original_line:
                    leg_artist.set_alpha(1.0 if is_visible else 0.2)  # Dim if not visible

            # Redraw the canvas to apply the changes
            self.canvas.draw()

    def update_plot(self, x, y, node_id=None,
                    is_max_principal_stress=False,
                    is_min_principal_stress=False,
                    is_von_mises=False,
                    is_deformation=False,
                    is_velocity=False,
                    is_acceleration=False):
        # Clear the figure
        self.figure.clear()
        self.plotted_lines.clear()
        self.legend_map.clear()

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
                line, = ax.plot(x, data, label=f'{prefix} ({component})', **style)
                self.plotted_lines.append(line)

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
        ax.set_xlim(np.min(x), np.max(x))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=8)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Create the legend
            leg = ax.legend(handles, labels, fontsize=7)

            # Map legend artists to original plot lines and make them pickable
            for legline, legtext, origline in zip(leg.get_lines(), leg.get_texts(), self.plotted_lines):
                legline.set_picker(True)  # Enable picking on the legend line
                legline.set_pickradius(5)  # Set a click-able area
                self.legend_map[legline] = origline  # Map legend line to plot line

                legtext.set_picker(True)
                self.legend_map[legtext] = origline  # Map legend text to the same plot line

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
    initial_data_loaded = pyqtSignal(object)
    time_point_result_ready = pyqtSignal(object, str, float, float)
    animation_data_ready = pyqtSignal(object)

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

        # Flag for right-click plot dialog
        self.plot_dialog = None

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
        self.time_history_checkbox.toggled.connect(self._update_damage_index_state)
        self.von_mises_checkbox.toggled.connect(self._update_damage_index_state)

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
        self.solve_button.clicked.connect(lambda: self.solve())

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

    def _check_and_emit_initial_data(self):
        """Checks if all necessary data is loaded and emits a signal."""
        if self.coord_loaded and self.stress_loaded:
            # Package all necessary data into a tuple and emit it (for example to display tab)
            initial_data = (time_values, node_coords, df_node_ids, self.deformation_loaded)
            self.initial_data_loaded.emit(initial_data)

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

    def _update_damage_index_state(self):
        """Disables the Damage Index checkbox if Time History is active or Von Mises is inactive."""
        is_time_history_checked = self.time_history_checkbox.isChecked()
        is_von_mises_checked = self.von_mises_checkbox.isChecked()

        # Damage index is only available when Von Mises is checked AND Time History is OFF.
        is_enabled = is_von_mises_checked and not is_time_history_checked
        self.damage_index_checkbox.setEnabled(is_enabled)

        # If the checkbox becomes disabled, also uncheck and hide it for clarity.
        if not is_enabled:
            self.damage_index_checkbox.setChecked(False)
            self.damage_index_checkbox.setVisible(False)

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
        self._check_and_emit_initial_data()
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
        self._check_and_emit_initial_data()
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
        self._check_and_emit_initial_data()
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

    def solve(self, force_time_history_for_node_id=None):
        try:
            # Check if we are in time history mode either by the checkbox OR the forced argument
            is_time_history_mode = self.time_history_checkbox.isChecked() or \
                                   (force_time_history_for_node_id is not None)

            # Check if the checkboxes are checked
            calculate_damage = self.damage_index_checkbox.isChecked()
            calculate_von_mises = self.von_mises_checkbox.isChecked()
            calculate_max_principal_stress = self.max_principal_stress_checkbox.isChecked()
            calculate_min_principal_stress = self.min_principal_stress_checkbox.isChecked()
            calculate_deformation = self.deformation_checkbox.isChecked()
            calculate_velocity      = self.velocity_checkbox.isChecked()
            calculate_acceleration  = self.acceleration_checkbox.isChecked()

            selected_node_id = None  # Initialize to None

            # Validation for Time History mode
            if is_time_history_mode:
                # If we are forcing this mode, use the provided node ID. Otherwise, get it from the UI.
                if force_time_history_for_node_id is not None:
                    selected_node_id = force_time_history_for_node_id
                    # Basic validation for the passed node_id
                    if selected_node_id not in df_node_ids:
                        QMessageBox.warning(self, "Invalid Node ID", f"Node ID {selected_node_id} was not found...")
                        return

                else:
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
                selected_node_idx = get_node_index_from_id(selected_node_id, df_node_ids)

                # Check if the node index was found before proceeding
                if selected_node_idx is None:
                    # The get_node_index_from_id function already prints an error,
                    # so we can just return.
                    return None

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
                    time_values,
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
                    selected_node_id,
                    df_node_ids,
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

                    # Return plot arguments for later use
                    plot_args = (time_values, y_data, selected_node_id,
                                 calculate_max_principal_stress, calculate_min_principal_stress,
                                 calculate_von_mises, calculate_deformation,
                                 calculate_velocity, calculate_acceleration)
                    return plot_args

                self.progress_bar.setVisible(False)  # Hide progress bar for single-node operation
                return None  # Exit early, no need to write files

            # Show the progress bar at the start of the solution
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Create an instance of MSUPSmartSolverTransient for batch solver
            self.solver = MSUPSmartSolverTransient(
                modal_sx[:, mode_slice],
                modal_sy[:, mode_slice],
                modal_sz[:, mode_slice],
                modal_sxy[:, mode_slice],
                modal_syz[:, mode_slice],
                modal_sxz[:, mode_slice],
                modal_coord[mode_slice, :],
                time_values,
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
                time_values,
                df_node_ids,
                node_coords,
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

        return None

    def _update_solve_button_state(self):
        """Enables the SOLVE button only if all required primary inputs are loaded."""
        if self.coord_loaded and self.stress_loaded:
            self.solve_button.setEnabled(True)
        else:
            self.solve_button.setEnabled(False)

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

    def show_plot_in_new_dialog(self, time_values, y_data, node_id, is_max_s1, is_min_s3, is_svm, is_def, is_vel,
                                is_acc):
        """Creates and shows a non-modal dialog with a new MatplotlibWidget."""
        # If another plot dialog is already open, close it first.
        if self.plot_dialog:
            self.plot_dialog.close()

        # Create the dialog and the new plot widget
        self.plot_dialog = QDialog(self)
        plot_widget = MatplotlibWidget()

        # Update the plot in the new widget with the data
        plot_widget.update_plot(time_values, y_data, node_id,
                                is_max_principal_stress=is_max_s1,
                                is_min_principal_stress=is_min_s3,
                                is_von_mises=is_svm,
                                is_deformation=is_def,
                                is_velocity=is_vel,
                                is_acceleration=is_acc)

        # Set up the dialog's layout and appearance
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(plot_widget)
        self.plot_dialog.setLayout(dialog_layout)
        self.plot_dialog.setWindowTitle(f"Time History for Node {node_id}")
        self.plot_dialog.setMinimumSize(800, 600)

        # Get the current window flags
        flags = self.plot_dialog.windowFlags()
        # Add the minimize and maximize buttons
        flags |= Qt.WindowMinimizeButtonHint
        flags |= Qt.WindowMaximizeButtonHint
        # Set the new flags
        self.plot_dialog.setWindowFlags(flags)

        # Show the dialog without blocking the main window
        self.plot_dialog.show()

    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        if value >= 100:
            # Hide the progress bar once the process is finished
            self.progress_bar.setVisible(False)

    @pyqtSlot(int)
    def plot_history_for_node(self, node_id):
        """
        This slot receives a node_id, validates that a single output is selected,
        and calls the solver to generate a plot in a new dialog.
        """
        print(f"Slot received Node ID: {node_id}")

        # 1. Check if primary files are loaded
        if not (self.coord_loaded and self.stress_loaded):
            QMessageBox.warning(self, "Data Not Loaded",
                                "Please load Modal Coordinate and Modal Stress files before plotting a time history.")
            return

        # 2. Find which output is checked and ensure it's only one
        outputs_checked = []
        # Map allows for specific error messages
        output_map = {
            'Von-Mises Stress': self.von_mises_checkbox,
            'Max Principal Stress': self.max_principal_stress_checkbox,
            'Min Principal Stress': self.min_principal_stress_checkbox,
            'Deformation': self.deformation_checkbox,
            'Velocity': self.velocity_checkbox,
            'Acceleration': self.acceleration_checkbox,
        }
        for name, cb in output_map.items():
            if cb.isChecked():
                outputs_checked.append(name)

        # Give specific feedback to the user based on their selection
        if len(outputs_checked) == 0:
            QMessageBox.warning(self, "No Output Selected",
                                "Please go to the 'Main Window' tab and check an output box (e.g., 'Von-Mises Stress') to plot its time history.")
            return

        if len(outputs_checked) > 1:
            QMessageBox.warning(self, "Multiple Outputs Selected",
                                f"Multiple outputs are checked: {', '.join(outputs_checked)}.\n\nPlease go to the 'Main Window' tab and select only one output to plot.")
            return

        # 3. Specifically check if deformation data is missing for relevant outputs
        selected_output_name = outputs_checked[0]
        if selected_output_name in ['Deformation', 'Velocity', 'Acceleration'] and not self.deformation_loaded:
            QMessageBox.warning(self, "Missing Deformation Data",
                                f"Plotting '{selected_output_name}' requires the optional Modal Deformations file, which has not been loaded.\n\nPlease check the 'Include Deformations' box and load the file first.")
            return

        # 4. If all checks pass, call the solver
        plot_args = self.solve(force_time_history_for_node_id=node_id)

        # 5. Show the plot dialog or provide a fallback error message
        if plot_args:
            self.show_plot_in_new_dialog(*plot_args)
        else:
            # This message will now appear if something unexpected happens inside the solver
            QMessageBox.warning(self, "Calculation Failed",
                                "The time history calculation did not produce a result. Please check the console output for more details.")

    @pyqtSlot(float, dict)
    def perform_time_point_calculation(self, selected_time, options):
        """Receives a request, performs a single time-point calc, and emits the result."""
        print("Control Panel: Received request to perform calculation.")

        try:
            # --- This is the calculation logic moved from DisplayTab ---
            if not (self.coord_loaded and self.stress_loaded):
                QMessageBox.warning(self, "Missing Data", "Core data files are not loaded.")
                return

            num_outputs_selected = sum([
                options['compute_von_mises'], options['compute_max_principal'],
                options['compute_min_principal'], options['compute_deformation_contour'],
                options['compute_velocity'], options['compute_acceleration']
            ])
            if num_outputs_selected > 1:
                QMessageBox.warning(self, "Multiple Outputs Selected",
                                    "Please select only one output type for time point visualization.")
                return
            if num_outputs_selected == 0:
                QMessageBox.warning(self, "No Selection",
                                    "No valid output is selected. Please select a valid output type.")
                return

            time_index = np.argmin(np.abs(time_values - selected_time))
            mode_slice = slice(options['skip_n_modes'], None)

            modal_deformations_filtered = None
            if options['display_deformed_shape'] and self.deformation_loaded:
                modal_deformations_filtered = (
                modal_ux[:, mode_slice], modal_uy[:, mode_slice], modal_uz[:, mode_slice])

            is_vel_or_accel = options['compute_velocity'] or options['compute_acceleration']
            if is_vel_or_accel:
                WINDOW, half = 7, 3
                idx0, idx1 = max(0, time_index - half), min(modal_coord.shape[1], time_index + half + 1)
                if idx1 - idx0 < 2:
                    QMessageBox.warning(self, "Too few samples", "Velocity/acceleration need at least two time steps.")
                    return
                selected_modal_coord = modal_coord[mode_slice, idx0:idx1]
                dt_window = time_values[idx0:idx1]
                centre_offset = time_index - idx0
            else:
                selected_modal_coord = modal_coord[mode_slice, time_index:time_index + 1]
                dt_window = time_values[time_index:time_index + 1]

            steady_kwargs = {}
            if options['include_steady'] and steady_sx is not None:
                steady_kwargs = {'steady_sx': steady_sx, 'steady_sy': steady_sy, 'steady_sz': steady_sz,
                                 'steady_sxy': steady_sxy, 'steady_syz': steady_syz, 'steady_sxz': steady_sxz,
                                 'steady_node_ids': steady_node_ids}

            temp_solver = MSUPSmartSolverTransient(
                modal_sx[:, mode_slice], modal_sy[:, mode_slice], modal_sz[:, mode_slice],
                modal_sxy[:, mode_slice], modal_syz[:, mode_slice], modal_sxz[:, mode_slice],
                selected_modal_coord, dt_window, modal_node_ids=df_node_ids,
                modal_deformations=modal_deformations_filtered, **steady_kwargs
            )

            num_nodes = modal_sx.shape[0]
            display_coords = node_coords
            ux_tp, uy_tp, uz_tp = None, None, None

            if options['display_deformed_shape'] and self.deformation_loaded:
                ux_tp, uy_tp, uz_tp = temp_solver.compute_deformations(0, num_nodes)
                if is_vel_or_accel:
                    ux_tp, uy_tp, uz_tp = ux_tp[:, [centre_offset]], uy_tp[:, [centre_offset]], uz_tp[:,
                                                                                                [centre_offset]]
                displacement_vector = np.hstack((ux_tp, uy_tp, uz_tp))
                display_coords = node_coords + (displacement_vector * options['scale_factor'])

            actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = temp_solver.compute_normal_stresses(0,
                                                                                                                      num_nodes)

            scalar_field, display_name = None, "Result"
            if options['compute_von_mises']:
                scalar_field = temp_solver.compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy,
                                                                    actual_syz, actual_sxz)
                display_name = "SVM (MPa)"
            elif options['compute_max_principal']:
                s1, _, _ = temp_solver.compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy,
                                                                  actual_syz, actual_sxz)
                scalar_field = s1
                display_name = "S1 (MPa)"
            elif options['compute_min_principal']:
                _, _, s3 = temp_solver.compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy,
                                                                  actual_syz, actual_sxz)
                scalar_field = s3
                display_name = "S3 (MPa)"
            elif options['compute_deformation_contour']:
                if not self.deformation_loaded:
                    QMessageBox.warning(self, "Missing Data", "Modal deformations must be loaded for this calculation.")
                    return
                if ux_tp is None:  # Recalculate if not already done for deformed shape
                    ux_tp, uy_tp, uz_tp = temp_solver.compute_deformations(0, num_nodes)
                scalar_field = np.sqrt(ux_tp ** 2 + uy_tp ** 2 + uz_tp ** 2)
                display_name = "Deformation (mm)"
            elif is_vel_or_accel:
                if not self.deformation_loaded:
                    QMessageBox.warning(self, "Missing Data", "Modal deformations must be loaded for this calculation.")
                    return
                ux_blk, uy_blk, uz_blk = temp_solver.compute_deformations(0, num_nodes)
                vel_blk, acc_blk, _, _, _, _, _, _ = temp_solver._vel_acc_from_disp(ux_blk, uy_blk, uz_blk,
                                                                                    dt_window.astype(
                                                                                        temp_solver.NP_DTYPE))

                if options['compute_velocity']:
                    scalar_field = vel_blk[:, [centre_offset]]
                    display_name = "Velocity (mm/s)"
                else:  # Acceleration
                    scalar_field = acc_blk[:, [centre_offset]]
                    display_name = "Acceleration (mm/s²)"

            if scalar_field is None:
                print("No valid output was calculated.")
                return

            # --- Finalize and Emit ---
            mesh = pv.PolyData(display_coords)
            if df_node_ids is not None:
                mesh["NodeID"] = df_node_ids.astype(int)
            mesh[display_name] = scalar_field
            mesh.set_active_scalars(display_name)

            data_min, data_max = np.min(scalar_field), np.max(scalar_field)

            self.time_point_result_ready.emit(mesh, display_name, data_min, data_max)

        except Exception as e:
            print(f"ERROR during time point calculation: {e}")
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()

    @pyqtSlot(dict)
    def perform_animation_precomputation(self, params):
        """
        Receives animation parameters from the DisplayTab, runs the heavy precomputation,
        and emits the resulting data arrays back. This will freeze the GUI while running.
        """
        try:
            # region 1. Get Animation Time Steps
            # We must call the helper methods that belong to the DisplayTab
            display_tab = self.window().display_tab
            anim_times, anim_indices, error_msg = display_tab._get_animation_time_steps()

            if error_msg:
                QMessageBox.warning(self, "Animation Setup Error", error_msg)
                raise ValueError(error_msg)
            if anim_times is None or len(anim_times) == 0:
                QMessageBox.warning(self, "Animation Setup Error", "No time steps generated.")
                raise ValueError("No animation time steps generated.")

            num_anim_steps = len(anim_times)
            print(f"Attempting to precompute {num_anim_steps} frames.")
            # endregion

            # region 2. Determine Required Outputs
            # Use the flags passed in the params dictionary
            if not any([params['compute_von_mises'], params['compute_max_principal'], params['compute_min_principal'],
                        params['compute_deformation_contour'], params['compute_velocity'],
                        params['compute_acceleration']]):
                QMessageBox.warning(self, "No Selection", "No valid output is selected for animation.")
                raise ValueError("No output selected for animation.")

            # Determine if deformed shape animation is possible and requested
            compute_deformation_anim = params['compute_deformation_anim']
            if compute_deformation_anim and not self.deformation_loaded:
                QMessageBox.warning(self, "Deformation Error",
                                    "Deformation is checked, but deformation data is not loaded.")
                compute_deformation_anim = False

            is_deformation_included_in_anim = compute_deformation_anim
            # endregion

            # region 3. RAM Estimation and Check
            num_nodes = modal_sx.shape[0]
            estimated_gb = display_tab._estimate_animation_ram(num_nodes, num_anim_steps, compute_deformation_anim)
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            safe_available_gb = available_gb * solver_engine.RAM_PERCENT

            print(f"Estimated RAM for precomputation: {estimated_gb:.3f} GB")
            print(f"Available system RAM: {available_gb:.3f} GB (Safe threshold: {safe_available_gb:.3f} GB)")

            if estimated_gb > safe_available_gb:
                # The detailed suggestion message is complex and UI-related, a simpler one is fine here.
                QMessageBox.warning(self, "Insufficient Memory",
                                    f"Estimated RAM required ({estimated_gb:.3f} GB) exceeds safe limit ({safe_available_gb:.3f} GB). Adjust time range or step.")
                raise ValueError("Insufficient RAM for animation.")
            # endregion

            # region 4. Perform Precomputation
            # Initialize local variables to hold the results
            precomputed_scalars = None
            precomputed_coords = None
            data_column_name = "Result"

            mode_slice = slice(params['skip_n_modes'], None)
            selected_modal_coord = modal_coord[mode_slice, anim_indices]

            steady_kwargs = {}
            if params['include_steady'] and 'steady_sx' in globals() and steady_sx is not None:
                steady_kwargs = {
                    'steady_sx': steady_sx, 'steady_sy': steady_sy, 'steady_sz': steady_sz,
                    'steady_sxy': steady_sxy, 'steady_syz': steady_syz, 'steady_sxz': steady_sxz,
                    'steady_node_ids': steady_node_ids
                }

            modal_deformations_filtered = None
            if compute_deformation_anim:
                modal_deformations_filtered = (
                modal_ux[:, mode_slice], modal_uy[:, mode_slice], modal_uz[:, mode_slice])

            temp_solver = MSUPSmartSolverTransient(
                modal_sx[:, mode_slice], modal_sy[:, mode_slice], modal_sz[:, mode_slice],
                modal_sxy[:, mode_slice], modal_syz[:, mode_slice], modal_sxz[:, mode_slice],
                selected_modal_coord, anim_times, modal_node_ids=df_node_ids,
                modal_deformations=modal_deformations_filtered, **steady_kwargs
            )

            print("Computing normal stresses for animation...")
            actual_sx_anim, actual_sy_anim, actual_sz_anim, actual_sxy_anim, actual_syz_anim, actual_sxz_anim = \
                temp_solver.compute_normal_stresses(0, num_nodes)

            print("Computing scalar field for animation...")
            if params['compute_von_mises']:
                precomputed_scalars = temp_solver.compute_von_mises_stress(actual_sx_anim, actual_sy_anim,
                                                                           actual_sz_anim, actual_sxy_anim,
                                                                           actual_syz_anim, actual_sxz_anim)
                data_column_name = "SVM (MPa)"
            elif params['compute_max_principal']:
                s1_anim, _, _ = temp_solver.compute_principal_stresses(actual_sx_anim, actual_sy_anim, actual_sz_anim,
                                                                       actual_sxy_anim, actual_syz_anim,
                                                                       actual_sxz_anim)
                precomputed_scalars = s1_anim
                data_column_name = "S1 (MPa)"
            # ... (add elif for min_principal, etc. following the same pattern) ...
            elif params['compute_velocity'] or params['compute_acceleration'] or params['compute_deformation_contour']:
                if not self.deformation_loaded:
                    raise ValueError("Deformation data not loaded for requested calculation.")

                ux_anim, uy_anim, uz_anim = temp_solver.compute_deformations(0, num_nodes)
                if params['compute_deformation_contour']:
                    precomputed_scalars = np.sqrt(ux_anim ** 2 + uy_anim ** 2 + uz_anim ** 2)
                    data_column_name = "Deformation (mm)"
                if params['compute_velocity'] or params['compute_acceleration']:
                    vel_mag_anim, acc_mag_anim, _, _, _, _, _, _ = temp_solver._vel_acc_from_disp(ux_anim, uy_anim,
                                                                                                  uz_anim,
                                                                                                  anim_times.astype(
                                                                                                      temp_solver.NP_DTYPE))
                    if params['compute_velocity']:
                        precomputed_scalars = vel_mag_anim
                        data_column_name = "Velocity (mm/s)"
                    else:
                        precomputed_scalars = acc_mag_anim
                        data_column_name = "Acceleration (mm/s²)"

            if compute_deformation_anim:
                print("Computing deformations for animation...")
                deformations_anim = temp_solver.compute_deformations(0, num_nodes)
                if deformations_anim is not None:
                    ux_anim, uy_anim, uz_anim = deformations_anim
                    scale_factor = params['scale_factor']
                    original_coords_reshaped = node_coords[:, :, np.newaxis]
                    ux_anim -= ux_anim[:, [0]]
                    uy_anim -= uy_anim[:, [0]]
                    uz_anim -= uz_anim[:, [0]]
                    displacements_stacked = np.stack([ux_anim, uy_anim, uz_anim], axis=1)
                    precomputed_coords = original_coords_reshaped + scale_factor * displacements_stacked

            print("Cleaning up temporary animation data...")
            del temp_solver, actual_sx_anim, actual_sy_anim, actual_sz_anim, actual_sxy_anim, actual_syz_anim, actual_sxz_anim
            gc.collect()
            print("---Precomputation complete.---")
            # endregion

            # Package all results into a tuple
            results = (
            precomputed_scalars, precomputed_coords, anim_times, data_column_name, is_deformation_included_in_anim)

            # Emit the finished data
            self.animation_data_ready.emit(results)

        except Exception as e:
            # On any failure, restore the cursor and emit None to signal the DisplayTab
            print(f"ERROR during animation precomputation: {e}")
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            self.animation_data_ready.emit(None)

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
        self.setWindowTitle('MSUP Smart Solver - v0.97.0')
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

        # Connect signals
        self.batch_solver_tab.initial_data_loaded.connect(self.display_tab.setup_initial_view)
        self.display_tab.node_picked_signal.connect(self.batch_solver_tab.plot_history_for_node)
        self.display_tab.time_point_update_requested.connect(self.batch_solver_tab.perform_time_point_calculation)
        self.batch_solver_tab.time_point_result_ready.connect(self.display_tab.update_view_with_results)
        self.display_tab.animation_precomputation_requested.connect(self.batch_solver_tab.perform_animation_precomputation)
        self.batch_solver_tab.animation_data_ready.connect(self.display_tab.on_animation_data_ready)

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

        # Update the primary global variables
        solver_engine.RAM_PERCENT = settings["ram_percent"]
        solver_engine.DEFAULT_PRECISION = settings["precision"]
        solver_engine.IS_GPU_ACCELERATION_ENABLED = settings["gpu_acceleration"]

        # Update the derived precision-related variables
        if solver_engine.DEFAULT_PRECISION == 'Single':
            solver_engine.NP_DTYPE = np.float32
            solver_engine.TORCH_DTYPE = torch.float32
            solver_engine.RESULT_DTYPE = 'float32'
        elif solver_engine.DEFAULT_PRECISION == 'Double':
            solver_engine.NP_DTYPE = np.float64
            solver_engine.TORCH_DTYPE = torch.float64
            solver_engine.RESULT_DTYPE = 'float64'

        print("\n--- Advanced settings updated ---")
        print(f"  RAM Allocation: {solver_engine.RAM_PERCENT * 100:.0f}%")
        print(f"  Solver Precision: {solver_engine.DEFAULT_PRECISION}")
        print(f"  GPU Acceleration: {'Enabled' if solver_engine.IS_GPU_ACCELERATION_ENABLED else 'Disabled'}")
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
            f"- Precision: {solver_engine.DEFAULT_PRECISION}\n"
            f"- RAM Limit: {solver_engine.RAM_PERCENT * 100:.0f}%\n"
            f"- GPU Acceleration: {'Enabled' if solver_engine.IS_GPU_ACCELERATION_ENABLED else 'Disabled'}"
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