'''
Author: Kamil Emre Atay (k5483)
Version: 0.97.1
Script Name: we_mechload_viewer.py

Tested in Python version "3.10" and "3.12"
Please use "requirements.txt file" supplied with this code to be able to compile it in the future by using pyinstaller.
'''

# region Import libraries
import os
os.system('color F0')
print("Importing libraries...")
import sys
import csv
import math
import pandas as pd
import re
from time import sleep
from collections import OrderedDict
from natsort import natsorted
from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QLineEdit, QSpinBox,
                             QSplitter, QComboBox, QLabel, QSizePolicy, QPushButton, QCheckBox, QGroupBox,
                             QDialog, QListWidget, QListWidgetItem, QMenuBar, QMenu, QAction, QDoubleSpinBox,
                             QDockWidget, QTreeView, QFileSystemModel, QMainWindow)
from PyQt5.QtGui import QFont, QIcon
import plotly.graph_objects as go
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
import traceback
import shutil
from endaq.plot import rolling_min_max_envelope
from endaq.plot import spectrum_over_time
from endaq.calc.fft import rolling_fft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tempfile
import plotly.io as pio
from scipy.signal import butter, filtfilt
from scipy.signal.windows import tukey

print("Done.")
# endregion

# region Define global variables / functions
DATA_DOMAIN = None
# endregion

# region Read input file
#######################################
# region Set up functions to be used for reading raw input
def select_directory(title):
    folder = QFileDialog.getExistingDirectory(None, title)
    if not folder:
        QMessageBox.critical(None, 'Error', f"No folder is selected for input files\n(*.full.pld and *.log files)\n\nExiting the program.")
        sys.exit(1)
        return None
    return folder

def get_file_path(folder, file_suffix):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(file_suffix)]

def read_pld_log_file(file_path):
    df = pd.read_csv(file_path, delimiter='|', skipinitialspace=True, skip_blank_lines=True)
    df = df.iloc[:,1].dropna().str.strip().to_frame() # set df.iloc[:,1]... for max.pld type header files and df.iloc[:,5] for log type files
    return df.T

def insert_phase_columns(df):
    # Extract the Interface Label row
    interface_labels = df.iloc[0, :].copy()

    # Create a new list for the transformed labels
    transformed_labels = []

    for label in interface_labels:
        # Add the original label
        transformed_labels.append(label)

        # Add the corresponding phase label
        phase_label = f"Phase_{label}"
        transformed_labels.append(phase_label)

    # Replace the original Interface Label row with the transformed labels
    df = pd.DataFrame([transformed_labels], index=["Interface Label"])
    return df

def read_pld_file(file_path):
    df = pd.read_csv(file_path, delimiter='|', skipinitialspace=True, skip_blank_lines=True, comment='_', low_memory=False)
    df = df.apply(pd.to_numeric)
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df.columns = df.columns.str.strip()
    df.reset_index(drop=True, inplace=True)
    return df
# endregion

# region Run the input file reader and parse the data
#######################
# region Define the main routine of the input file reader
def main():
    try:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        folder_selected = select_directory('Please select a directory for raw data and headers')

        if not folder_selected:
            sys.exit()

        file_path_full_data = get_file_path(folder_selected, 'full.pld')
        file_path_headers_data = get_file_path(folder_selected, 'max.pld')

        if not file_path_full_data or not file_path_headers_data:
            QMessageBox.critical(None, 'Error', "No required files found! Exiting.")
            sys.exit()

        dfs = [read_pld_file(file_path) for file_path in file_path_full_data]
        df = pd.concat(dfs, ignore_index=True)

        df_intf_before = read_pld_log_file(file_path_headers_data[0])
        if 'FREQ' in df.columns:
            df_intf = insert_phase_columns(df_intf_before)
            df_intf_labels = df_intf.iloc[0]
            new_columns = ['NO'] + ['FREQ'] + df_intf_labels.tolist()
            DATA_DOMAIN = 'FREQ'
        elif 'TIME' in df.columns:
            df_intf = df_intf_before
            df_intf_labels = df_intf.iloc[0]
            new_columns = ['NO'] + ['TIME'] + df_intf_labels.tolist()
            DATA_DOMAIN = 'TIME'
        additional_columns_needed = len(df.columns) - len(new_columns)
        if additional_columns_needed > 0:
            extended_new_columns = new_columns + [f"Extra_Column_{i}" for i in range(1, additional_columns_needed + 1)]
            df.columns = extended_new_columns
        else:
            df.columns = new_columns[:len(df.columns)]

        df.columns = new_columns[:len(df.columns)]

        print(df)

        return df, DATA_DOMAIN, folder_selected
    except Exception as e:
        QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")
        sys.exit()
# endregion

# region Run the file reader
if __name__ == "__main__":
    data, DATA_DOMAIN, selected_folder = main()

    # Write the raw data with correct headers inside the solution directory
    data.to_csv("full_data.csv", index=False)
# endregion
#######################
# endregion
#######################################
# endregion

# region Set up the main GUI and its functions
class WE_load_plotter(QMainWindow):
    # region Initialize main window & tabs
    def __init__(self, parent=None, raw_data_folder=None):
        super(WE_load_plotter, self).__init__(parent)
        self.raw_data_folder = raw_data_folder  # Store the raw data folder path
        self.temp_files = []  # List to track temporary files
        self.init_variables()
        self.init_ui()

    def create_directory_tree_dock(self):
        # Create a dock widget titled "Data Folders"
        dock = QDockWidget("Data Folders", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self.tree_view = QTreeView(dock)
        self.file_model = QFileSystemModel()
        self.file_model.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs)

        parent_folder = os.path.dirname(self.raw_data_folder) if self.raw_data_folder else os.getcwd()
        self.file_model.setRootPath(parent_folder)
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(parent_folder))

        # Hide extra columns so that only the folder names show.
        self.tree_view.setColumnHidden(1, True)
        self.tree_view.setColumnHidden(2, True)
        self.tree_view.setColumnHidden(3, True)

        # Optionally, hide the header for a minimalist look.
        self.tree_view.header().setVisible(False)

        # Apply a custom stylesheet for a modern look.
        self.tree_view.setStyleSheet("""
                QTreeView {
                    background-color: #f7f7f7;
                    border: none;
                }
                QTreeView::item {
                    padding: 5px;
                }
                QTreeView::item:selected {
                    background-color: #00838f;
                    color: white;
                }
            """)

        # Enable multi-selection using Ctrl/Shift.
        self.tree_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        dock.setWidget(self.tree_view)
        self.tree_view.selectionModel().selectionChanged.connect(self.on_directory_selected)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        # Store the dock widget for later use (like toggle actions)
        self.tree_dock = dock

    def init_variables(self):

        self.legend_visible = True
        self.legend_positions = ['default', 'top left', 'top right', 'bottom right', 'bottom left']
        self.current_legend_position = 1
        self.df = pd.read_csv('full_data.csv')
        self.df_compare = None
        self.df_extracted_loads_for_ansys = None
        self.side_filter_selector_for_compare = QComboBox()
        self.current_plot_data = {}
        self.app_ansys = None
        self.mechanical = None

        # Calculate sample rate
        if 'TIME' in self.df.columns:
            time_diffs = self.df['TIME'].diff().dropna()
            self.sample_rate = 1 / time_diffs.mean()

        self.default_font_size = 12
        self.legend_font_size = 10
        self.hover_font_size = 15
        self.hover_mode = 'closest'

        # DataFrames for plots
        self.original_df_tab1 = None
        self.working_df_tab1 = None
        self.original_df_phase_tab1 = None
        self.working_df_phase_tab1 = None
        self.original_df_t_series_tab2 = None
        self.working_df_t_series_tab2 = None
        self.original_df_r_series_tab2 = None
        self.working_df_r_series_tab2 = None
        self.original_df_t_series_tab3 = None
        self.working_df_t_series_tab3 = None
        self.original_df_r_series_tab3 = None
        self.working_df_r_series_tab3 = None
        self.original_df_time_domain_tab4 = None
        self.working_df_time_domain_tab4 = None
        self.original_df_compare_tab = None
        self.working_df_compare_tab = None
        self.original_df_absolute_diff_tab = None
        self.working_df_absolute_diff_tab = None
        self.original_df_relative_diff_tab = None
        self.working_df_relative_diff_tab = None
        self.original_df_compare_t_series_tab = None
        self.working_df_compare_t_series_tab = None
        self.original_df_compare_r_series_tab = None
        self.working_df_compare_r_series_tab = None

        self.number_limit_of_data_points_shown_for_each_trace = 100000

    def init_ui(self):
        # Create a menu bar and add the File menu with an Open action.
        menu_bar = QMenuBar(self)
        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_new_data)
        file_menu.addAction(open_action)

        # Clear Plot Cache Action ---
        file_menu.addSeparator() # Optional separator
        clear_cache_action = QAction("Clear Plot Cache", self)
        clear_cache_action.triggered.connect(self.clear_plot_cache)
        file_menu.addAction(clear_cache_action)

        # Create and add the directory tree dock.
        self.create_directory_tree_dock()

        # Create a View menu to toggle the directory dock.
        view_menu = QMenu("View", self)
        # Add the toggle action for the dock widget.
        view_menu.addAction(self.tree_dock.toggleViewAction())
        menu_bar.addMenu(view_menu)

        # Set the menu bar for the QMainWindow.
        self.setMenuBar(menu_bar)

        # Create a central widget with a vertical layout.
        central_widget = QWidget(self)
        central_layout = QVBoxLayout(central_widget)

        # Create the tab widget and add your tabs.
        self.tab_widget = QTabWidget(central_widget)
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                background: #00838f;
                color: white;
                min-width: 120px; /* Adjust this value as needed */
                padding: 5px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 5px;
            }
            QTabBar::tab:selected {
                background: #00acc1;
                font-weight: normal; /* Remove bold */
            }
            QTabBar::tab:disabled {
                background: #cccccc;
                color: #777777;
            }
            QTabWidget::pane {
                border-top: 2px solid #ccc;
                border-left: 1px solid #ccc;
                border-right: 1px solid #ccc;
                border-bottom: 1px solid #ccc;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        self.tab1 = self.create_tab("Single Data", self.setupTab1)
        self.tab2 = self.create_tab("Interface Data", self.setupTab2)
        self.tab3 = self.create_tab("Part Loads", self.setupTab3)
        if 'FREQ' in self.df.columns:
            self.tab4 = self.create_tab("Time Domain Representation", self.setupTab4)
        compare_tab = self.create_tab("Compare Data", self.setupCompareTab)
        compare_part_loads_tab = self.create_tab("Compare Data (Part Loads)", self.setupComparePartLoadsTab)
        self.settings_tab = self.create_tab("Settings", self.setupSettingsTab)

        self.tab_widget.addTab(self.tab1, "Single Data")
        self.tab_widget.addTab(self.tab2, "Interface Data")
        self.tab_widget.addTab(self.tab3, "Part Loads")
        if 'FREQ' in self.df.columns:
            self.tab_widget.addTab(self.tab4, "Time Domain Representation")
        self.tab_widget.addTab(compare_tab, "Compare Data")
        self.tab_widget.addTab(compare_part_loads_tab, "Compare Data (Part Loads)")
        self.tab_widget.addTab(self.settings_tab, "Settings")

        # Add the tab widget to the central layout.
        central_layout.addWidget(self.tab_widget)
        # Set the central widget of the QMainWindow.
        self.setCentralWidget(central_widget)

        # Update the window title and icon.
        folder_name = os.path.basename(self.raw_data_folder) if self.raw_data_folder else ""
        parent_folder = os.path.basename(os.path.dirname(self.raw_data_folder)) if self.raw_data_folder else ""
        self.setWindowTitle(f"WE MechLoad Viewer - v0.97.1    |    (Directory Folder: {parent_folder})")
        icon_path = self.get_resource_path("icon.ico")
        self.setWindowIcon(QIcon(icon_path))
        self.showMaximized()

    def get_resource_path(self, relative_path):
        """Get the absolute path to a resource."""
        # Check if running in a PyInstaller bundle
        if hasattr(sys, "_MEIPASS"):
            # Running in a bundled environment
            return os.path.join(sys._MEIPASS, relative_path)
        else:
            # Running in a development environment
            return os.path.join(os.path.dirname(__file__), relative_path)

    def create_tab(self, name, setup_method):
        tab = QWidget()
        setup_method(tab)
        return tab
    # endregion

    # region Initialize widgets & layouts inside each tab of the main window
    # "Single Data" tab
    def setupTab1(self, tab):
        self.splitter_tab1 = QSplitter(QtCore.Qt.Vertical)
        self.regular_plot = QtWebEngineWidgets.QWebEngineView()
        self.splitter_tab1.addWidget(self.regular_plot)
        if 'FREQ' in self.df.columns:
            self.phase_plot = QtWebEngineWidgets.QWebEngineView()
            self.splitter_tab1.addWidget(self.phase_plot)
            self.splitter_tab1.setSizes([self.height() // 2, self.height() // 2])

        self.column_selector = QComboBox()
        self.column_selector.setEditable(False)
        regular_columns = [col for col in self.df.columns if
                           'Phase_' not in col and col != 'FREQ' and col != 'TIME' and col != 'NO']
        self.column_selector.addItems(regular_columns)
        self.column_selector.currentIndexChanged.connect(self.update_plots_tab1)

        if 'TIME' in self.df.columns:
            self.column_selector.currentIndexChanged.connect(self.update_spectrum_plot)

        self.spectrum_checkbox = QCheckBox("Show Spectrum Plot")
        self.spectrum_checkbox.stateChanged.connect(self.toggle_spectrum_plot)
        self.spectrum_checkbox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        self.plot_type_selector = QComboBox()
        self.plot_type_selector.setEditable(False)
        self.plot_type_selector.addItems(['Heatmap', 'Surface', 'Waterfall', 'Animation', 'Peak', 'Lines'])
        self.plot_type_selector.setVisible(False)
        self.plot_type_selector.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.plot_type_selector.currentIndexChanged.connect(self.update_spectrum_plot)

        self.specgram_checkbox = QCheckBox("Show Spectrum in Full Scale Plot")
        self.specgram_checkbox.stateChanged.connect(self.toggle_spectrum_plot)

        if 'TIME' in self.df.columns:
            self.filter_checkbox = QCheckBox("Apply Low-Pass Filter")
            self.filter_checkbox.stateChanged.connect(self.toggle_filter_options)

            self.cutoff_frequency_label = QLabel("Cutoff Freq [Hz]:")
            self.cutoff_frequency_input = QLineEdit()  # Added line
            self.cutoff_frequency_input.setPlaceholderText("Cutoff Frequency [Hz]")
            self.cutoff_frequency_label.setVisible(False)
            self.cutoff_frequency_input.setVisible(False)
            self.cutoff_frequency_input.textChanged.connect(self.update_plots_tab1)

            self.filter_order_label = QLabel("Order:")
            self.filter_order_input = QSpinBox()
            self.filter_order_input.setRange(1, 10)
            self.filter_order_input.setValue(2)
            self.filter_order_label.setVisible(False)
            self.filter_order_input.setVisible(False)
            self.filter_order_input.valueChanged.connect(self.update_plots_tab1)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(self.column_selector)

        if 'TIME' in self.df.columns:
            selector_layout.addWidget(self.spectrum_checkbox)
            selector_layout.addWidget(self.plot_type_selector)
            selector_layout.addWidget(self.specgram_checkbox)

            selector_layout.addWidget(self.filter_checkbox)
            selector_layout.addWidget(self.cutoff_frequency_label)
            selector_layout.addWidget(self.cutoff_frequency_input)
            selector_layout.addWidget(self.filter_order_label)
            selector_layout.addWidget(self.filter_order_input)

        layout = QVBoxLayout(tab)
        layout.addLayout(selector_layout)
        layout.addWidget(self.splitter_tab1)

        self.spectrum_plot = QtWebEngineWidgets.QWebEngineView()
        self.splitter_tab1.addWidget(self.spectrum_plot)
        self.spectrum_plot.setVisible(False)
        self.spectrum_plot.setVisible(False)

    # Show or hide low-pass filter options and labels based on the state of low-pass filter checkbox
    def toggle_filter_options(self):
        filter_enabled = self.filter_checkbox.isChecked()
        self.cutoff_frequency_input.setVisible(filter_enabled)
        self.cutoff_frequency_label.setVisible(filter_enabled)
        self.filter_order_input.setVisible(filter_enabled)
        self.filter_order_label.setVisible(filter_enabled)
        self.update_plots_tab1()

    # "Interface Data" tab

    def setupTab2(self, tab):
        layout = QVBoxLayout(tab)
        self.setupInterfaceSelector(layout)
        self.setupSideSelectionTab2(layout)
        self.setupPlots(layout)

    # "Part Loads" tab
    def setupTab3(self, tab):
        layout = QVBoxLayout(tab)
        upper_layout = QHBoxLayout()

        self.setupSideFilterSelector(upper_layout)

        self.exclude_checkbox = QCheckBox(r"Filter out T2, T3, R2, and R3 from graphs")
        self.exclude_checkbox.stateChanged.connect(self.update_plots_tab3)
        upper_layout.addWidget(self.exclude_checkbox)

        self.tukey_checkbox_tab3 = QCheckBox("Apply Tukey Window")
        self.tukey_checkbox_tab3.stateChanged.connect(self.update_plots_tab3)
        upper_layout.addWidget(self.tukey_checkbox_tab3)

        self.tukey_alpha_spin = QDoubleSpinBox()
        self.tukey_alpha_spin.setRange(0.1, 0.5)
        self.tukey_alpha_spin.setSingleStep(0.05)
        self.tukey_alpha_spin.setValue(0.1)
        self.tukey_alpha_spin.setVisible(False)
        self.tukey_alpha_spin.valueChanged.connect(self.update_plots_tab3)
        upper_layout.addWidget(self.tukey_alpha_spin)

        self.tukey_checkbox_tab3.stateChanged.connect(
            lambda s: self.tukey_alpha_spin.setVisible(s == QtCore.Qt.Checked)
        )

        self.section_checkbox = QCheckBox("Section Data")
        #self.section_checkbox.stateChanged.connect(self.update_plots_tab3)
        upper_layout.addWidget(self.section_checkbox)

        self.section_min_label = QLabel("Min Time [sec]")
        self.section_min_input = QLineEdit()
        self.section_min_label.setVisible(False)
        self.section_min_input.setVisible(False)
        self.section_min_input.returnPressed.connect(self.validate_section_min)

        self.section_max_label = QLabel("Max Time [sec]")
        self.section_max_input = QLineEdit()
        self.section_max_label.setVisible(False)
        self.section_max_input.setVisible(False)
        self.section_max_input.returnPressed.connect(self.validate_section_max)

        self.section_checkbox.stateChanged.connect(
            lambda s: [w.setVisible(s == QtCore.Qt.Checked) for w in
                       (self.section_min_label, self.section_min_input,
                        self.section_max_label, self.section_max_input)]
        )

        upper_layout.addWidget(self.section_min_label)
        upper_layout.addWidget(self.section_min_input)
        upper_layout.addWidget(self.section_max_label)
        upper_layout.addWidget(self.section_max_input)

        layout.addLayout(upper_layout)

        splitter = QSplitter(QtCore.Qt.Vertical)
        self.t_series_plot_tab3 = QtWebEngineWidgets.QWebEngineView()
        self.r_series_plot_tab3 = QtWebEngineWidgets.QWebEngineView()
        splitter.addWidget(self.t_series_plot_tab3)
        splitter.addWidget(self.r_series_plot_tab3)
        splitter.setSizes([self.height() // 2, self.height() // 2])
        layout.addWidget(splitter)

        self.data_point_selector_tab3 = QComboBox()
        self.data_point_selector_tab3.setEditable(True)

        # Change the content and label of the combobox based on whether the input data is in time or frequency domain
        if 'FREQ' in self.df.columns:
            self.data_point_selector_tab3.addItem("Select a frequency [Hz] to extract the raw data")
            self.data_point_selector_tab3.addItems([str(freq_point) for freq_point in sorted(self.df['FREQ'].unique())])
        if 'TIME' in self.df.columns:
            self.data_point_selector_tab3.addItem("Select a time point [sec] from the list to extract the raw data in the frequency domain")
            self.data_point_selector_tab3.addItems([str(time_point) for time_point in sorted(self.df['TIME'].unique())])

        # Make the placeholder text in the combobox non-selectable
        index = self.data_point_selector_tab3.count() - 1
        self.data_point_selector_tab3.setItemData(index, 0, QtCore.Qt.UserRole - 1)
        data_point_selector_tab3_layout = QHBoxLayout()
        data_point_selector_tab3_layout.addWidget(self.data_point_selector_tab3)

        # Display the buttons designed specifically for time or frequency domain based data
        if 'FREQ' in self.df.columns:
            self.extract_data_button = QPushButton("Extract Data for Selected Frequency")
            self.extract_all_data_button = QPushButton("Extract Part Loads as FEA Input (ANSYS)")
            self.extract_data_button.clicked.connect(self.extract_single_frequency_data_point)
            data_point_selector_tab3_layout.addWidget(self.extract_data_button)
            self.extract_all_data_button.clicked.connect(self.extract_all_data_points_in_frequency_domain)
            data_point_selector_tab3_layout.addWidget(self.extract_all_data_button)
        if 'TIME' in self.df.columns:
            self.extract_data_button = QPushButton("Extract Data for Selected Time (Not Active)")
            self.extract_all_data_button = QPushButton("Extract Part Loads as FEA Input (ANSYS)")
            self.extract_all_data_button.clicked.connect(self.extract_all_data_points_in_time_domain)
            self.extract_data_button.setStyleSheet("color: gray;")
            data_point_selector_tab3_layout.addWidget(self.extract_data_button)
            data_point_selector_tab3_layout.addWidget(self.extract_all_data_button)

        layout.addLayout(data_point_selector_tab3_layout)

    # "Time Domain Representation" tab (active only if input data is in frequency domain)
    def setupTab4(self, tab):
        layout = QVBoxLayout(tab)

        self.data_point_selector = QComboBox()
        self.data_point_selector.setEditable(True)
        self.data_point_selector.addItem("Select a frequency [Hz] to plot the time domain data")
        self.data_point_selector.addItems([str(freq) for freq in sorted(self.df['FREQ'].unique())])
        self.data_point_selector.currentIndexChanged.connect(self.update_time_domain_plot)
        data_point_selector_layout = QHBoxLayout()
        data_point_selector_label = QLabel("Select a Frequency")
        data_point_selector_layout.addWidget(data_point_selector_label)
        data_point_selector_layout.addWidget(self.data_point_selector)
        layout.addLayout(data_point_selector_layout)

        self.time_domain_plot = QtWebEngineWidgets.QWebEngineView()
        layout.addWidget(self.time_domain_plot)

        self.interval_selector = QComboBox()
        self.interval_selector.setEditable(True)
        self.interval_selector.addItem("Select an Interval")
        for i in range(1, 361):
            if 360 % i == 0:
                self.interval_selector.addItem(str(i))
        interval_selector_layout = QHBoxLayout()
        interval_selector_label = QLabel("Select an Interval")
        interval_selector_layout.addWidget(interval_selector_label)
        interval_selector_layout.addWidget(self.interval_selector)

        self.extract_button = QPushButton("Extract Data at Each Interval as CSV file")
        self.extract_button.clicked.connect(self.extract_time_representation_data)
        interval_selector_layout.addWidget(self.extract_button)

        layout.addLayout(interval_selector_layout)
        tab.setLayout(layout)

    # Add controls common to all tabs

    def add_common_controls(self, layout):
        # Checkbox for Rolling Min-Max Envelope
        self.rolling_min_max_checkbox = QCheckBox("Show Plots as Rolling Min-Max Envelope")
        self.rolling_min_max_checkbox.stateChanged.connect(self.update_all_transient_data_plots)

        # Checkbox for Plot as Bars
        self.plot_as_bars_checkbox = QCheckBox("Plot as Bars")
        self.plot_as_bars_checkbox.stateChanged.connect(self.update_all_transient_data_plots)

        # Input box for Desired Num Points
        self.desired_num_points_label = QLabel("Number of Points Shown:")
        self.desired_num_points_input = QLineEdit()
        self.desired_num_points_input.setPlaceholderText("Enter an Integer Number")
        self.desired_num_points_input.setFixedWidth(150)
        self.desired_num_points_input.setText("50000")  # Set initial value
        self.desired_num_points_input.textChanged.connect(self.update_all_transient_data_plots)

        # Enable the checkbox by default if the x data length is more than a certain number of data points
        x_data_length = len(self.df['TIME']) if 'TIME' in self.df.columns else len(self.df['FREQ'])
        if x_data_length > self.number_limit_of_data_points_shown_for_each_trace:
            self.rolling_min_max_checkbox.setChecked(True)
            self.plot_as_bars_checkbox.setChecked(True)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.rolling_min_max_checkbox)
        control_layout.addWidget(self.plot_as_bars_checkbox)
        control_layout.addWidget(self.desired_num_points_label)
        control_layout.addWidget(self.desired_num_points_input)

        layout.addLayout(control_layout)

    # Set the visibility of settings based on input data type (time or frequency domain)
    def update_control_visibility(self):
        is_time_data = 'TIME' in self.df.columns
        is_rolling_min_max_checked = self.rolling_min_max_checkbox.isChecked()

        self.rolling_min_max_checkbox.setVisible(is_time_data)
        self.plot_as_bars_checkbox.setVisible(is_time_data and is_rolling_min_max_checked)
        self.desired_num_points_label.setVisible(is_time_data and is_rolling_min_max_checked)
        self.desired_num_points_input.setVisible(is_time_data and is_rolling_min_max_checked)

    def toggle_spectrum_plot(self, state):
        try:
            show_spectrum = self.spectrum_checkbox.isChecked()
            show_specgram = self.specgram_checkbox.isChecked()

            if show_spectrum:
                self.specgram_checkbox.setChecked(False)
                self.specgram_checkbox.setDisabled(True)
                self.plot_type_selector.setVisible(True)  # Show the plot type selector
            else:
                self.specgram_checkbox.setDisabled(False)
                self.plot_type_selector.setVisible(False)  # Hide the plot type selector

            if show_specgram:
                self.spectrum_checkbox.setChecked(False)
                self.spectrum_checkbox.setDisabled(True)
            else:
                self.spectrum_checkbox.setDisabled(False)

            if show_spectrum or show_specgram:
                self.spectrum_plot.setVisible(True)
                if 'FREQ' in self.df.columns:
                    self.splitter_tab1.setSizes([self.height() // 3, self.height() // 3, self.height() // 3])
                else:
                    self.splitter_tab1.setSizes([self.height() // 2, self.height() // 2])
                self.spectrum_plot.setHtml("<html><body></body></html>")
                self.update_spectrum_plot()
            else:
                self.spectrum_plot.setVisible(False)
                if 'FREQ' in self.df.columns:
                    self.splitter_tab1.setSizes([self.height() // 2, self.height() // 2])
                else:
                    self.splitter_tab1.setSizes([self.height()])  # Only one plot visible

        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")

    def setupSettingsTab(self, tab):
        layout = QVBoxLayout(tab)

        # Data Processing Tools Group
        data_processing_group = QGroupBox("Data Processing Tools")
        data_processing_layout = QVBoxLayout()

        self.add_common_controls(data_processing_layout)

        data_processing_group.setLayout(data_processing_layout)
        layout.addWidget(data_processing_group)

        # Graphical Settings Group
        graphical_settings_group = QGroupBox("Graphical Settings")
        graphical_settings_layout = QVBoxLayout()

        # Legend Font Size
        legend_font_size_layout = QHBoxLayout()
        legend_font_size_label = QLabel("Legend Font Size")
        self.legend_font_size_selector = QComboBox()
        self.legend_font_size_selector.addItems([str(size) for size in range(4, 30)])
        self.legend_font_size_selector.setCurrentText(str(self.legend_font_size))
        self.legend_font_size_selector.currentIndexChanged.connect(self.update_font_settings)
        legend_font_size_layout.addWidget(legend_font_size_label)
        legend_font_size_layout.addWidget(self.legend_font_size_selector)
        graphical_settings_layout.addLayout(legend_font_size_layout)

        # Default Font Size
        default_font_size_layout = QHBoxLayout()
        default_font_size_label = QLabel("Default Font Size")
        self.default_font_size_selector = QComboBox()
        self.default_font_size_selector.addItems([str(size) for size in range(8, 30)])
        self.default_font_size_selector.setCurrentText(str(self.default_font_size))
        self.default_font_size_selector.currentIndexChanged.connect(self.update_font_settings)
        default_font_size_layout.addWidget(default_font_size_label)
        default_font_size_layout.addWidget(self.default_font_size_selector)
        graphical_settings_layout.addLayout(default_font_size_layout)

        # Hover Font Size
        hover_font_size_layout = QHBoxLayout()
        hover_font_size_label = QLabel("Hover Font Size")
        self.hover_font_size_selector = QComboBox()
        self.hover_font_size_selector.addItems([str(size) for size in range(4, 21)])
        self.hover_font_size_selector.setCurrentText(str(self.hover_font_size))
        self.hover_font_size_selector.currentIndexChanged.connect(self.update_font_settings)
        hover_font_size_layout.addWidget(hover_font_size_label)
        hover_font_size_layout.addWidget(self.hover_font_size_selector)
        graphical_settings_layout.addLayout(hover_font_size_layout)

        # Hover Mode
        hover_mode_layout = QHBoxLayout()
        hover_mode_label = QLabel("Hover Mode")
        #hover_mode_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.hover_mode_selector = QComboBox()
        self.hover_mode_selector.addItems(['closest', 'x', 'y', 'x unified', 'y unified'])
        self.hover_mode_selector.setCurrentText(self.hover_mode)
        self.hover_mode_selector.currentIndexChanged.connect(self.update_font_settings)
        hover_mode_layout.addWidget(hover_mode_label)
        hover_mode_layout.addWidget(self.hover_mode_selector)
        graphical_settings_layout.addLayout(hover_mode_layout)

        graphical_settings_group.setLayout(graphical_settings_layout)
        layout.addWidget(graphical_settings_group)

        # Add a contact label at the bottom
        contact_label = QLabel(
            "Please reach K. Emre Atay (Compressor Module: Stationary Parts Team) for bug reports or feature requests.")
        layout.addWidget(contact_label, alignment=QtCore.Qt.AlignBottom)

        # Apply stylesheet to group boxes
        data_processing_group.setStyleSheet("""
            QGroupBox {
                color: #00838f;
                background-color: #f0f0f0;
                border: 1px solid lightgray;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
        """)
        graphical_settings_group.setStyleSheet("""
            QGroupBox {
                color: #00838f;
                background-color: #f0f0f0;
                border: 1px solid lightgray;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
        """)

        tab.setLayout(layout)

    def setupCompareTab(self, tab):
        splitter_main = QSplitter(QtCore.Qt.Vertical)
        splitter_upper = QSplitter(QtCore.Qt.Vertical)
        splitter_lower = QSplitter(QtCore.Qt.Vertical)

        self.compare_regular_plot = QtWebEngineWidgets.QWebEngineView()
        # self.compare_phase_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_absolute_diff_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_percent_diff_plot = QtWebEngineWidgets.QWebEngineView()

        splitter_upper.addWidget(self.compare_regular_plot)
        # splitter_upper.addWidget(self.compare_phase_plot)
        splitter_upper.setSizes([self.height() // 4, self.height() // 4])

        splitter_lower.addWidget(self.compare_absolute_diff_plot)
        splitter_lower.addWidget(self.compare_percent_diff_plot)
        splitter_lower.setSizes([self.height() // 4, self.height() // 4])

        splitter_main.addWidget(splitter_upper)
        splitter_main.addWidget(splitter_lower)
        splitter_main.setSizes([self.height() // 2, self.height() // 2])

        self.compare_column_selector = QComboBox()
        self.compare_column_selector.setEditable(False)
        layout = QVBoxLayout(tab)
        layout.addWidget(self.compare_column_selector)
        layout.addWidget(splitter_main)

        self.compare_button = QPushButton("Select Data for Comparison")
        self.compare_button.clicked.connect(self.select_compare_data)
        self.compare_button.setStyleSheet("""
            QPushButton {
                background-color: #00838f;
                color: white;
                border: 2px solid #006064;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #00acc1;
                border-color: #006064;
            }
            QPushButton:pressed {
                background-color: #006064;
                border-color: #004d40;
            }
        """)
        layout.addWidget(self.compare_button)

        tab.setLayout(layout)

    def setupComparePartLoadsTab(self, tab):
        layout = QVBoxLayout(tab)
        upper_layout = QHBoxLayout()

        self.setupSideFilterSelectorForCompare(upper_layout)

        self.exclude_checkbox_compare = QCheckBox("Filter out T2, T3, R2, and R3 from graphs")
        self.exclude_checkbox_compare.stateChanged.connect(self.update_compare_part_loads_plots)
        upper_layout.addWidget(self.exclude_checkbox_compare)

        layout.addLayout(upper_layout)
        self.setupComparePartLoadsPlots(layout)

    def setupComparePartLoadsPlots(self, layout):
        splitter = QSplitter(QtCore.Qt.Vertical)
        self.compare_t_series_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_r_series_plot = QtWebEngineWidgets.QWebEngineView()
        splitter.addWidget(self.compare_t_series_plot)
        splitter.addWidget(self.compare_r_series_plot)
        splitter.setSizes([self.height() // 2, self.height() // 2])
        layout.addWidget(splitter)

    # Setting up the canvas of each plot area of each tab
    def setupPlots(self, layout):
        splitter = QSplitter(QtCore.Qt.Vertical)
        self.t_series_plot = QtWebEngineWidgets.QWebEngineView()
        self.r_series_plot = QtWebEngineWidgets.QWebEngineView()
        splitter.addWidget(self.t_series_plot)
        splitter.addWidget(self.r_series_plot)
        splitter.setSizes([self.height() // 2, self.height() // 2])
        layout.addWidget(splitter)
    # endregion

    # region Define the logics to be executed behind each button
    def select_compare_data(self):
        try:
            # Ask for one folder containing both comparison full.pld and max.pld files
            folder_selected = select_directory('Please select a directory for raw data and headers (Comparison)')
            if not folder_selected:
                return

            file_path_full_data = get_file_path(folder_selected, 'full.pld')
            file_path_headers_data = get_file_path(folder_selected, 'max.pld')

            if not file_path_full_data or not file_path_headers_data:
                QMessageBox.critical(None, 'Error', "No required files found! Exiting.")
                return

            # Load the 'full.pld' data
            dfs = [read_pld_file(file_path) for file_path in file_path_full_data]
            df_compare = pd.concat(dfs, ignore_index=True)

            # Check if the TIME or FREQ column of the compared data matches the original data
            original_first_col = self.df.columns[1]
            compared_first_col = df_compare.columns[1]
            if original_first_col != compared_first_col:
                QMessageBox.critical(
                    None,
                    'Error',
                    f"The X data columns of original and compared data are different:\n"
                    f"Original Data: {original_first_col}\nCompared Data: {compared_first_col}"
                )
                return

            # Load headers (max.pld)
            df_intf_before = read_pld_log_file(file_path_headers_data[0])

            # Figure out the domain from the current data (self.df)
            # and build new_columns accordingly
            if 'FREQ' in self.df.columns:
                df_intf = insert_phase_columns(df_intf_before)
                df_intf_labels = pd.DataFrame(df_intf.iloc[0]).T
                new_columns = ['NO'] + ['FREQ'] + df_intf_labels.iloc[0].tolist()
            elif 'TIME' in self.df.columns:
                df_intf = df_intf_before
                df_intf_labels = pd.DataFrame(df_intf.iloc[0]).T
                new_columns = ['NO'] + ['TIME'] + df_intf_labels.iloc[0].tolist()

            additional_columns_needed = len(df_compare.columns) - len(new_columns)
            if (additional_columns_needed > 0):
                extended_new_columns = new_columns + [
                    f"Extra_Column_{i}" for i in range(1, additional_columns_needed + 1)
                ]
                df_compare.columns = extended_new_columns
            else:
                df_compare.columns = new_columns[:len(df_compare.columns)]

            df_compare.columns = new_columns[:len(df_compare.columns)]

            # Ensure unique column labels
            df_compare = df_compare.loc[:, ~df_compare.columns.duplicated()]

            # Align columns with the main dataframe
            self.df_compare = df_compare.reindex(columns=self.df.columns, fill_value=np.nan)

            # Also store it in df_compare in case you need it raw
            self.df_compare = df_compare

            # Update compare_column_selector with columns from the new data
            self.compare_column_selector.clear()
            compare_columns = [
                col for col in self.df_compare.columns
                if 'Phase_' not in col and col not in ['FREQ', 'TIME', 'NO']
            ]
            self.compare_column_selector.addItems(compare_columns)
            self.compare_column_selector.currentIndexChanged.connect(self.update_compare_plots)

        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")

    def extract_time_representation_data(self):
        try:
            interval = int(self.interval_selector.currentText())
            num_points = 360 // interval + 1
            theta_points = np.linspace(0, 360, num_points, endpoint=True)

            if 360 % interval == 0 and len(theta_points) > 1 and theta_points[-2] == 360:
                theta_points = theta_points[:-1]

            data_dict = {'Theta': theta_points}

            for col, data in self.current_plot_data.items():
                full_theta = np.degrees(data['theta'])
                full_y_data = data['y_data']

                indices = np.searchsorted(full_theta, theta_points, side='left')
                indices = np.clip(indices, 0, len(full_y_data) - 1)
                sampled_y_data = full_y_data[indices]

                data_dict[col] = sampled_y_data

            extracted_data = pd.DataFrame(data_dict)
            extracted_data.to_csv("extracted_time_data_values.csv", index=False)
            self.convert_to_Nmm_units(extracted_data)
            QMessageBox.information(None, "Extraction Complete",
                                    "Data has been extracted and saved to:\n\n extracted_time_data_values.csv and extracted_time_data_values_in_Nmm_units.csv.")
        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")

    def convert_to_Nmm_units(self, data):
        nmm_data = data.copy()
        for col in nmm_data.columns:
            if col != 'Theta' or col != 'TIME':
                nmm_data[col] = nmm_data[col].astype(float) * 1000
        nmm_data.to_csv("extracted_time_data_values_in_Nmm_units.csv", index=False)

    def get_selected_sides(self):
        # Get the currently selected side from the side_filter_selector
        current_side = self.side_filter_selector.currentText()

        # Get all available sides
        all_sides = [self.side_filter_selector.itemText(i) for i in range(self.side_filter_selector.count())]

        # Create and configure the dialog
        dialog = QDialog(self)  # Create a QDialog
        dialog.setWindowTitle("Select Parts to Extract Interface Loads")  # Set dialog title
        list_widget = QListWidget()  # Create the QListWidget
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)  # Allow multiple selections
        for side in all_sides:
            item = QListWidgetItem(side)
            list_widget.addItem(item)
            if side == current_side:
                item.setSelected(True)  # Select the currently active side by default

        # Add the list widget and buttons to the dialog
        # Add the list widget and buttons to the dialog
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        help_button = QPushButton("Help")

        # Add help functionality
        help_button.clicked.connect(lambda: QMessageBox.information(
            dialog,
            "Help",
            (
                "<p style=\"text-align: justify;\">"
                "- Only the T1, T2, T3 (force) and R1, R2, R3 (moment) components are extracted. Resultant load vectors "
                "such as T2/T3 or R2/R3 are not included in the export.</p>"
                "<p style=\"text-align: justify;\">"
                "- All the components are multiplied by 1000 to convert them to SI units (N and N.mm), assuming they are "
                "given in kN (for force components) and kN.mm (for moment components), respectively. Please check whether "
                "these units are correct.</p>"
                "<p style=\"text-align: justify;\">"
                "- All extracted part loads are treated as either force or moment types. After generating the FEA template, "
                "please manually review and remove any loads that you don't need or that have incorrect types.</p>"
            )
        ))

        button_layout = QHBoxLayout()
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(help_button)
        layout = QVBoxLayout()
        layout.addWidget(list_widget)
        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Show the dialog and get the results
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_sides = [list_widget.item(i).text() for i in range(list_widget.count()) if
                              list_widget.item(i).isSelected()]
            return selected_sides
        else:
            return None  # User canceled the dialog

    def extract_single_frequency_data_point(self):
        selected_frequency_tab3 = self.data_point_selector_tab3.currentText()
        selected_side = self.side_filter_selector.currentText()

        if selected_frequency_tab3 == "Select a Frequency":
            QMessageBox.information(self, "Selection Required", "Please select a valid frequency.")
            return

        try:
            freq = float(selected_frequency_tab3)
            filtered_df = self.df[self.df['FREQ'] == freq]

            side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')
            columns = ['FREQ']
            columns += [col for col in filtered_df.columns
                        if side_pattern.search(col)
                        and col != 'FREQ'
                        and "T2/T3" not in col
                        and "R2/R3" not in col]

            result_df = filtered_df[columns]

            original_file_path = f"extracted_data_for_{selected_side}_at_{selected_frequency_tab3}_Hz.csv"
            result_df.to_csv(original_file_path, index=False)

            for col in result_df.columns:
                if not col.startswith('Phase_') and col != 'FREQ':
                    result_df[col] = result_df[col] * 1000

            converted_file_path = f"extracted_data_for_{selected_side}_at_{selected_frequency_tab3}_Hz_multiplied_by_1000.csv"
            result_df.to_csv(converted_file_path, index=False)

            QMessageBox.information(self, "Extraction Complete",
                                    f"Data has been extracted and converted. \n\nOriginal data is saved to: \n{original_file_path}. \n\nConverted data saved to: \n{converted_file_path}.")
        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")

    def extract_all_data_points_in_frequency_domain(self):
        selected_sides = self.get_selected_sides()
        if not selected_sides:
            QMessageBox.information(self, "Selection Required", "Please select valid sides.")
            return

        try:
            df_selected_parts_combined = pd.DataFrame()

            for selected_side in selected_sides:
                side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')
                columns = ['FREQ'] + [col for col in self.df.columns if side_pattern.search(
                    col) and col != 'FREQ' and 'T2/T3' not in col and 'R2/R3' not in col]

                df_selected_part = self.df[columns]

                # Ensure no duplicate FREQ column in the combined DataFrame
                if 'FREQ' in df_selected_parts_combined.columns:
                    df_selected_part = df_selected_part.drop(columns=['FREQ'], errors='ignore')

                df_selected_parts_combined = pd.concat([df_selected_parts_combined, df_selected_part], axis=1)

                original_file_path = f"extracted_data_for_{selected_side}_in_original_units.csv"
                df_selected_part.to_csv(original_file_path, index=False)

                # Perform unit conversion
                df_selected_part_converted = df_selected_part.copy()
                for col in df_selected_part_converted.columns:
                    if col != 'FREQ' and not col.startswith('Phase_'):
                        df_selected_part_converted[col] *= 1000
                # Perform unit conversion for combined data
                df_selected_parts_combined_converted = df_selected_parts_combined.copy()
                for col in df_selected_parts_combined_converted.columns:
                    if col != 'FREQ' and not col.startswith('Phase_'):
                        df_selected_parts_combined_converted[col] *= 1000

                file_path_selected_part_converted = f"extracted_data_for_{selected_side}_multiplied_by_1000.csv"
                df_selected_part_converted.to_csv(file_path_selected_part_converted, index=False)

                QMessageBox.information(self, "Extraction Complete",
                                        f"Data for {selected_side} has been extracted and converted.\n\nOriginal data saved to: {original_file_path}\nConverted data saved to: {file_path_selected_part_converted}.")

            combined_file_path = "extracted_loads_of_all_selected_parts_in_converted_units.csv"
            df_selected_parts_combined_converted.to_csv(combined_file_path, index=False)
            QMessageBox.information(self, "Extraction Complete",
                                    f"Combined data for all sides has been saved to {combined_file_path}.")

            self.df_extracted_loads_for_ansys = df_selected_parts_combined
            self.create_ansys_mechanical_input_template_harmonic()

        except Exception as e:
            traceback_info = traceback.format_exc()
            print(traceback_info)
            QMessageBox.critical(None, 'Error', f"An error occurred during the execution of this button: {str(e)}")

    def extract_all_data_points_in_time_domain(self):
        selected_sides = self.get_selected_sides()
        if not selected_sides:
            QMessageBox.information(self, "Selection Required", "Please select valid sides.")
            return

        try:
            df_selected_parts_combined = pd.DataFrame()

            for selected_side in selected_sides:
                side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')
                columns = ['TIME'] + [col for col in self.df.columns if side_pattern.search(
                    col) and col != 'TIME' and 'T2/T3' not in col and 'R2/R3' not in col]

                df_selected_part = self.df[columns]

                # Section Data filter
                if getattr(self, 'section_checkbox', None) and self.section_checkbox.isChecked():
                    t_min = self.section_min_spin.value()
                    t_max = self.section_max_spin.value()
                    df_selected_part = df_selected_part[
                        (df_selected_part['TIME'] >= t_min) &
                        (df_selected_part['TIME'] <= t_max)
                        ]

                # Apply Tukey window
                if getattr(self, 'tukey_checkbox_tab3', None) and self.tukey_checkbox_tab3.isChecked():
                    window = tukey(len(df_selected_part), self.tukey_alpha_spin.value())
                    load_cols = [c for c in df_selected_part.columns if c != 'TIME']
                    df_selected_part[load_cols] = df_selected_part[load_cols].multiply(window, axis=0)

                # Ensure no duplicate TIME column in the combined DataFrame
                if 'TIME' in df_selected_parts_combined.columns:
                    df_selected_part = df_selected_part.drop(columns=['TIME'], errors='ignore')

                df_selected_parts_combined = pd.concat([df_selected_parts_combined, df_selected_part], axis=1)

                original_file_path = f"extracted_data_for_{selected_side}_in_original_units.csv"
                df_selected_part.to_csv(original_file_path, index=False)

                # Perform unit conversion
                df_selected_part_converted = df_selected_part.copy()
                for col in df_selected_part_converted.columns:
                    if col != 'TIME' and not col.startswith('Phase_'):
                        df_selected_part_converted[col] *= 1000
                # Perform unit conversion for combined data
                df_selected_parts_combined_converted = df_selected_parts_combined.copy()
                for col in df_selected_parts_combined_converted.columns:
                    if col != 'TIME':
                        df_selected_parts_combined_converted[col] *= 1000

                file_path_selected_part_converted = f"extracted_{selected_side}_loads_multiplied_by_1000.csv"
                df_selected_part_converted.to_csv(file_path_selected_part_converted, index=False)

                QMessageBox.information(self, "Extraction Complete",
                                        f"Data for {selected_side} has been extracted and converted.\n\nOriginal data saved to: {original_file_path}\nConverted data saved to: {file_path_selected_part_converted}.")

            file_path_selected_parts_combined = "extracted_loads_of_all_selected_parts_in_converted_units.csv"
            df_selected_parts_combined_converted.to_csv(file_path_selected_parts_combined, index=False)
            QMessageBox.information(self, "Extraction Complete",
                                    f"Combined data for all sides has been saved to {file_path_selected_parts_combined}.")

            self.df_extracted_loads_for_ansys = df_selected_parts_combined
            self.create_ansys_mechanical_input_template_transient()

        except Exception as e:
            traceback_info = traceback.format_exc()
            print(traceback_info)
            QMessageBox.critical(None, 'Error', f"An error occurred during the execution of this button: {str(e)}")

    def create_APDL_table(self, result_df, table_name="my_table"):
        global DATA_DOMAIN  # Access the data type from global variables
        col_index_name = result_df.columns.name
        num_rows, num_cols = result_df.shape
        apdl_lines = []

        # Convert the DataFrame to a NumPy array for faster access
        values = result_df.values
        row_indices = result_df.index
        col_indices = result_df.columns

        # DIM command
        apdl_lines.append("\n\n\n" + f"! Create load table {table_name}\n")
        apdl_lines.append(f"*DIM,{table_name},TABLE,{num_rows},{num_cols},1,{DATA_DOMAIN}\n\n")

        # Add row index values (vectorized)
        apdl_lines.append(f"! {table_name}, {DATA_DOMAIN} Values\n")
        apdl_lines.extend([f"*SET,{table_name}({i+1},0,1),{row_index}\n" for i, row_index in enumerate(row_indices)])
        apdl_lines.append("\n")

        # Add table values (vectorized)
        apdl_lines.append(f"! {table_name} Data Values\n")
        for i in range(num_rows):
            for j in range(num_cols):
                data_value = values[i, j]
                if not pd.isna(data_value):  # Only write non-NaN values
                    apdl_lines.append(f"*SET,{table_name}({i+1},{j+1},1),{data_value}\n")

        return apdl_lines

    def create_ansys_mechanical_input_template_harmonic(self):
        ############################################################################
        # region Collect interface names and metadata to be used as input loads for the selected part

        # List of all possible keys for load components and phase angles
        all_keys = ["T1", "T2", "T3", "R1", "R2", "R3", "Phase_T1", "Phase_T2", "Phase_T3", "Phase_R1", "Phase_R2",
                    "Phase_R3"]

        # Function to extract the interface name including phase angles
        def get_full_interface_name(column_name):
            column_name = re.sub(r'\s+[TR][1-3]$', '', column_name)
            column_name = re.sub(r'^Phase_', '', column_name)
            return column_name

        # Identify unique interfaces and their corresponding columns, including phase angles
        full_interfaces = OrderedDict()
        for col in self.df_extracted_loads_for_ansys.columns:
            if col != "FREQ":
                interface_name = get_full_interface_name(col)
                if interface_name not in full_interfaces:
                    full_interfaces[interface_name] = []
                full_interfaces[interface_name].append(col)

        # Function to create interface dictionary with all keys populated, missing ones with zeroes
        def create_full_interface_dict(df, columns):
            # Initialize the dictionary with all keys set to lists of zeroes
            interface_dict = {key: [0] * len(df) for key in all_keys}
            for col in columns:
                key = col.split()[-1]  # Get the last part which is T1, T2, etc.
                if col.startswith("Phase_"):
                    phase_key = "Phase_" + key
                    interface_dict[phase_key] = df[col].tolist()
                else:
                    interface_dict[key] = df[col].tolist()
            return interface_dict

        # Create full dictionaries for each unique interface
        interface_dicts_full = {interface: create_full_interface_dict(self.df_extracted_loads_for_ansys, cols) for
                                interface, cols in
                                full_interfaces.items()}

        # List of interfaces for the selected part
        list_of_part_interface_names = list(interface_dicts_full.keys())
        # endregion

        # region Import libraries
        print("Importing ansys-mechanical-core library...")
        import ansys.api.mechanical
        import ansys.mechanical.core
        from ansys.mechanical.core import global_variables
        from ansys.mechanical.core import App
        print("Imported.")
        # endregion

        # region Initialization of ANSYS Mechanical App
        print("Starting Ansys Mechanical...")
        app_ansys = App(private_appdata=False)
        print("Running the scripts...")
        globals().update(global_variables(app_ansys))
        print("Updating the global variables...")
        # endregion

        # region Set Mechanical Units as Standard MKS
        ExtAPI.Application.ActiveUnitSystem = Ansys.ACT.Interfaces.Common.MechanicalUnitSystem.StandardMKS
        scale_factor = 1000
        # endregion

        # region Initialize the list of frequency points extracted
        # Create a separate list for FREQ
        list_of_all_frequencies = self.df_extracted_loads_for_ansys["FREQ"].tolist()

        # Convert list of frequencies to a list of frequencies as quantities (in Hz)
        list_of_all_frequencies_as_quantity = []
        for frequency_value in list_of_all_frequencies:
            list_of_all_frequencies_as_quantity.append(Quantity(frequency_value, "Hz"))
        # endregion

        # region Initialize geometry import
        geometry_source = Model.AddGeometryImportGroup()
        geometry_source.AddGeometryImport()
        # endregion

        # region Create a static analysis environment template for pre-stressed solution
        print("Creating the static analysis environment...")
        analysis_static = Model.AddStaticStructuralAnalysis()
        analysis_settings_static = analysis_static.AnalysisSettings
        analysis_settings_static.PropertyByName("SolverUnitsControl").InternalValue = 1  # Manual
        analysis_settings_static.PropertyByName("SelectedSolverUnitSystem").InternalValue  # nmm unit system
        # endregion

        # region Create a modal analysis environment template for MSUP based solution
        print("Creating the modal analysis environment...")
        analysis_modal = Model.AddModalAnalysis()
        analysis_settings_modal = analysis_modal.AnalysisSettings
        analysis_settings_modal_IC = DataModel.GetObjectsByName("Pre-Stress (None)")[0]
        analysis_settings_modal_IC.PreStressICEnvironment = analysis_static
        analysis_settings_modal.PropertyByName("NumModesToFind").InternalValue = 100
        analysis_settings_modal.PropertyByName("RangeSearch").InternalValue = 1  # Limit Search to Range: On
        analysis_settings_modal.PropertyByName("MaxFrequency").InternalValue = \
            self.df_extracted_loads_for_ansys['FREQ'].max()*1.5  # Max Freq. to find is 1.5x the max freq. of interest
        analysis_settings_modal.PropertyByName("MSUPSkipExpansion").InternalValue = 1  # On-Demand Expansion Option: Yes
        analysis_settings_modal.PropertyByName("SolverUnitsControl").InternalValue = 1  # Manual
        analysis_settings_modal.PropertyByName("SelectedSolverUnitSystem").InternalValue  # nmm unit system
        # endregion

        # region Create tree objects in Mechanical for harmonic analysis and initialize analysis settings
        print("Creating the harmonic analysis environment...")
        analysis_HR = Model.AddHarmonicResponseAnalysis()
        analysis_settings_HR = analysis_HR.AnalysisSettings
        analysis_settings_HR.PropertyByName("HarmonicForcingFrequencyMax").InternalValue = \
            self.df_extracted_loads_for_ansys['FREQ'].max()
        analysis_settings_HR_IC = DataModel.GetObjectsByName("Pre-Stress/Modal (None)")[0]
        analysis_settings_HR_IC.PreStressICEnvironment = analysis_modal
        analysis_settings_HR.PropertyByName("MSUPSkipExpansion").InternalValue = 1  # On-Demand Expansion Option: Yes
        analysis_settings_HR.PropertyByName("CombineDistResultFile").InternalValue = 1
        analysis_settings_HR.PropertyByName("ExpandResultFrom").InternalValue = 1  # Expand from Modal Solution
        analysis_settings_HR.PropertyByName("HarmonicForcingFrequencyIntervals").InternalValue = 1
        analysis_settings_HR.PropertyByName("HarmonicSolutionMethod").InternalValue = 1
        analysis_settings_HR.PropertyByName("ConstantDampingValue").InternalValue = 0.02  # 2% damping by default

        # endregion

        # region Create load objects and their dependent objects for each interface
        interface_index_no = 1
        for interface_name in list_of_part_interface_names:

            # region Create dependent objects (coordinate systems and remote points)
            # Add a reference coordinate system for each interface
            CS_interface = Model.CoordinateSystems.AddCoordinateSystem()
            CS_interface.Name = "CS_" + interface_name

            # Create remote points for each interface
            RP_interface = Model.AddRemotePoint()
            RP_interface.Name = "RP_" + interface_name
            RP_interface.CoordinateSystem = CS_interface
            RP_interface.PilotNodeAPDLName = "RP_" + str(interface_index_no)
            # endregion

            print(f"Creating force & moment objects for {interface_name}...")

            # region Create remote force objects at each interface
            force_HR = analysis_HR.AddRemoteForce()
            force_HR.DefineBy = Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
            force_HR.Name = "RF_" + interface_name
            force_HR.PropertyByName("GeometryDefineBy").InternalValue = 2  # Scoped to a remote point
            force_HR.Location = RP_interface
            force_HR.Suppressed = False
            force_HR_index_name = "RF_" + str(interface_index_no)

            # Create moments at each interface
            moment_HR = analysis_HR.AddMoment()
            moment_HR.DefineBy = Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
            moment_HR.Name = "RM_" + interface_name + "_(For Visualization Purposes Only, Delete it or Keep it Suppressed)"
            moment_HR.PropertyByName("GeometryDefineBy").InternalValue = 2  # Scoped to remote point
            moment_HR.Location = RP_interface
            moment_HR.Suppressed = True
            moment_HR_index_name = "RM_" + str(interface_index_no)
            # endregion

            # region Define the numerical values of loads and their phase angles
            # region Initialize the lists of values
            list_of_fx_values = []
            list_of_fy_values = []
            list_of_fz_values = []
            list_of_angle_fx_values = []
            list_of_angle_fy_values = []
            list_of_angle_fz_values = []

            list_of_mx_values = []
            list_of_my_values = []
            list_of_mz_values = []
            list_of_angle_mx_values = []
            list_of_angle_my_values = []
            list_of_angle_mz_values = []
            # endregion

            # region Create lists of quantities (for T1, T2, T3)
            for fx, fy, fz, angle_fx, angle_fy, angle_fz in zip(interface_dicts_full[interface_name]["T1"],
                                                                interface_dicts_full[interface_name]["T2"],
                                                                interface_dicts_full[interface_name]["T3"],
                                                                interface_dicts_full[interface_name]["Phase_T1"],
                                                                interface_dicts_full[interface_name]["Phase_T2"],
                                                                interface_dicts_full[interface_name]["Phase_T3"]):
                list_of_fx_values.append(Quantity(fx, "kN"))
                list_of_fy_values.append(Quantity(fy, "kN"))
                list_of_fz_values.append(Quantity(fz, "kN"))
                list_of_angle_fx_values.append(Quantity(angle_fx, "deg"))
                list_of_angle_fy_values.append(Quantity(angle_fy, "deg"))
                list_of_angle_fz_values.append(Quantity(angle_fz, "deg"))
            # endregion

            # region Create harmonic force dataframes (which will serve as sources when creating APDL tables)
            # region Fx Dataframe (Real and Imag)
            df_load_table_fx = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'T1': interface_dicts_full[interface_name]["T1"]})
            df_load_table_fx.set_index('FREQ', inplace=True)

            df_load_table_phase_fx = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'Phase_T1': interface_dicts_full[interface_name]["Phase_T1"]})
            df_load_table_phase_fx.set_index('FREQ', inplace=True)

            df_load_table_fx_real = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_fx_real.set_index('FREQ', inplace=True)
            df_load_table_fx_real['Real_T1'] = df_load_table_fx['T1'] * np.cos(df_load_table_phase_fx['Phase_T1'])

            df_load_table_fx_imag = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_fx_imag.set_index('FREQ', inplace=True)
            df_load_table_fx_imag['Real_T1'] = df_load_table_fx['T1'] * np.sin(df_load_table_phase_fx['Phase_T1'])
            # endregion

            # region Fy Dataframe (Real and Imag)
            df_load_table_fy = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'T2': interface_dicts_full[interface_name]["T2"]})
            df_load_table_fy.set_index('FREQ', inplace=True)

            df_load_table_phase_fy = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'Phase_T2': interface_dicts_full[interface_name]["Phase_T2"]})
            df_load_table_phase_fy.set_index('FREQ', inplace=True)

            df_load_table_fy_real = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_fy_real.set_index('FREQ', inplace=True)
            df_load_table_fy_real['Real_T2'] = df_load_table_fy['T2'] * np.cos(df_load_table_phase_fy['Phase_T2'])

            df_load_table_fy_imag = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_fy_imag.set_index('FREQ', inplace=True)
            df_load_table_fy_imag['Real_T2'] = df_load_table_fy['T2'] * np.sin(df_load_table_phase_fy['Phase_T2'])
            # endregion

            # region Fz Dataframe (Real and Imag)
            df_load_table_fz = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'T3': interface_dicts_full[interface_name]["T3"]})
            df_load_table_fz.set_index('FREQ', inplace=True)

            df_load_table_phase_fz = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'Phase_T3': interface_dicts_full[interface_name]["Phase_T3"]})
            df_load_table_phase_fz.set_index('FREQ', inplace=True)

            df_load_table_fz_real = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_fz_real.set_index('FREQ', inplace=True)
            df_load_table_fz_real['Real_T3'] = df_load_table_fz['T3'] * np.cos(df_load_table_phase_fz['Phase_T3'])

            df_load_table_fz_imag = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_fz_imag.set_index('FREQ', inplace=True)
            df_load_table_fz_imag['Real_T3'] = df_load_table_fz['T3'] * np.sin(df_load_table_phase_fz['Phase_T3'])
            # endregion
            # endregion

            # region Create lists of quantities (for R1, R2, R3)
            for mx, my, mz, angle_mx, angle_my, angle_mz in zip(interface_dicts_full[interface_name]["R1"],
                                                                interface_dicts_full[interface_name]["R2"],
                                                                interface_dicts_full[interface_name]["R3"],
                                                                interface_dicts_full[interface_name]["Phase_R1"],
                                                                interface_dicts_full[interface_name]["Phase_R2"],
                                                                interface_dicts_full[interface_name]["Phase_R3"]):
                list_of_mx_values.append(Quantity(mx, "kN m"))
                list_of_my_values.append(Quantity(my, "kN m"))
                list_of_mz_values.append(Quantity(mz, "kN m"))
                list_of_angle_mx_values.append(Quantity(angle_mx, "deg"))
                list_of_angle_my_values.append(Quantity(angle_my, "deg"))
                list_of_angle_mz_values.append(Quantity(angle_mz, "deg"))
            # endregion

            # region Create harmonic moment dataframes (which will serve as sources when creating APDL tables)
            # region Mx Dataframe (Real and Imag)
            df_load_table_mx = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'R1': interface_dicts_full[interface_name]["R1"]})
            df_load_table_mx.set_index('FREQ', inplace=True)

            df_load_table_phase_mx = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'Phase_R1': interface_dicts_full[interface_name]["Phase_R1"]})
            df_load_table_phase_mx.set_index('FREQ', inplace=True)

            df_load_table_mx_real = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_mx_real.set_index('FREQ', inplace=True)
            df_load_table_mx_real['Real_R1'] = df_load_table_mx['R1'] * np.cos(df_load_table_phase_mx['Phase_R1'])

            df_load_table_mx_imag = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_mx_imag.set_index('FREQ', inplace=True)
            df_load_table_mx_imag['Real_R1'] = df_load_table_mx['R1'] * np.sin(df_load_table_phase_mx['Phase_R1'])
            # endregion

            # region My Dataframe (Real and Imag)
            df_load_table_my = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'R2': interface_dicts_full[interface_name]["R2"]})
            df_load_table_my.set_index('FREQ', inplace=True)

            df_load_table_phase_my = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'Phase_R2': interface_dicts_full[interface_name]["Phase_R2"]})
            df_load_table_phase_my.set_index('FREQ', inplace=True)

            df_load_table_my_real = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_my_real.set_index('FREQ', inplace=True)
            df_load_table_my_real['Real_R2'] = df_load_table_my['R2'] * np.cos(df_load_table_phase_my['Phase_R2'])

            df_load_table_my_imag = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_my_imag.set_index('FREQ', inplace=True)
            df_load_table_my_imag['Real_R2'] = df_load_table_my['R2'] * np.sin(df_load_table_phase_my['Phase_R2'])
            # endregion

            # region Mz Dataframe (Real and Imag)
            df_load_table_mz = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'R3': interface_dicts_full[interface_name]["R3"]})
            df_load_table_mz.set_index('FREQ', inplace=True)

            df_load_table_phase_mz = pd.DataFrame({
                'FREQ': list_of_all_frequencies,
                'Phase_R3': interface_dicts_full[interface_name]["Phase_R3"]})
            df_load_table_phase_mz.set_index('FREQ', inplace=True)

            df_load_table_mz_real = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_mz_real.set_index('FREQ', inplace=True)
            df_load_table_mz_real['Real_R3'] = df_load_table_mz['R3'] * np.cos(df_load_table_phase_mz['Phase_R3'])

            df_load_table_mz_imag = pd.DataFrame({'FREQ': list_of_all_frequencies})
            df_load_table_mz_imag.set_index('FREQ', inplace=True)
            df_load_table_mz_imag['Real_R3'] = df_load_table_mz['R3'] * np.sin(df_load_table_phase_mz['Phase_R3'])
            # endregion
            # endregion
            # endregion

            # region Populate load objects with numerical values obtained from each relevant interface
            print(f"Populating the input tabular data for {interface_name}...")
            # Define remote force frequencies
            force_HR.XComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            force_HR.YComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            force_HR.ZComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            force_HR.XPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            force_HR.YPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            force_HR.ZPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity

            # Define remote force forces and angles
            force_HR.XComponent.Output.DiscreteValues = list_of_fx_values
            force_HR.YComponent.Output.DiscreteValues = list_of_fy_values
            force_HR.ZComponent.Output.DiscreteValues = list_of_fz_values
            force_HR.XPhaseAngle.Output.DiscreteValues = list_of_angle_fx_values
            force_HR.YPhaseAngle.Output.DiscreteValues = list_of_angle_fy_values
            force_HR.ZPhaseAngle.Output.DiscreteValues = list_of_angle_fz_values

            # Define moment frequencies
            moment_HR.XComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment_HR.YComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment_HR.ZComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment_HR.XPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment_HR.YPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment_HR.ZPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity

            # Define moments and angles
            moment_HR.XComponent.Output.DiscreteValues = list_of_mx_values
            moment_HR.YComponent.Output.DiscreteValues = list_of_my_values
            moment_HR.ZComponent.Output.DiscreteValues = list_of_mz_values
            moment_HR.XPhaseAngle.Output.DiscreteValues = list_of_angle_mx_values
            moment_HR.YPhaseAngle.Output.DiscreteValues = list_of_angle_my_values
            moment_HR.ZPhaseAngle.Output.DiscreteValues = list_of_angle_mz_values
            # endregion

            # region Define T1,T2,T3 and R1, R2, R3 loads via Command Objects
            apdl_lines_RFx = self.create_APDL_table(df_load_table_fx_real * scale_factor,"table_X_" + force_HR_index_name)
            apdl_lines_RFy = self.create_APDL_table(df_load_table_fy_real * scale_factor, "table_Y_" + force_HR_index_name)
            apdl_lines_RFz = self.create_APDL_table(df_load_table_fz_real * scale_factor, "table_Z_" + force_HR_index_name)
            apdl_lines_RFxi = self.create_APDL_table(df_load_table_fx_imag * scale_factor,"table_Xi_" + force_HR_index_name)
            apdl_lines_RFyi = self.create_APDL_table(df_load_table_fy_imag * scale_factor, "table_Yi_" + force_HR_index_name)
            apdl_lines_RFzi = self.create_APDL_table(df_load_table_fz_imag * scale_factor, "table_Zi_" + force_HR_index_name)

            apdl_lines_RMx = self.create_APDL_table(df_load_table_mx_real * scale_factor,"table_X_" + moment_HR_index_name)
            apdl_lines_RMy = self.create_APDL_table(df_load_table_my_real * scale_factor, "table_Y_" + moment_HR_index_name)
            apdl_lines_RMz = self.create_APDL_table(df_load_table_mz_real * scale_factor, "table_Z_" + moment_HR_index_name)
            apdl_lines_RMxi = self.create_APDL_table(df_load_table_mx_imag * scale_factor,"table_Xi_" + moment_HR_index_name)
            apdl_lines_RMyi = self.create_APDL_table(df_load_table_my_imag * scale_factor, "table_Yi_" + moment_HR_index_name)
            apdl_lines_RMzi = self.create_APDL_table(df_load_table_mz_imag * scale_factor, "table_Zi_" + moment_HR_index_name)

            # region Create APDL command snippet for force loads (only intended for APDL users)
            command_snippet_RF = analysis_HR.AddCommandSnippet()
            command_snippet_RF.Name = "Commands_RF_" + interface_name

            command_snippet_RF.AppendText(''.join(apdl_lines_RFx))
            command_snippet_RF.AppendText(''.join(apdl_lines_RFy))
            command_snippet_RF.AppendText(''.join(apdl_lines_RFz))
            command_snippet_RF.AppendText(''.join(apdl_lines_RFxi))
            command_snippet_RF.AppendText(''.join(apdl_lines_RFyi))
            command_snippet_RF.AppendText(''.join(apdl_lines_RFzi))

            # region Apply load as a nodal force directly from the APDL table defined for RF
            command_snippet_RF.AppendText("\n\n"+ f"! Apply the load on the remote point specified for the interface\n")
            command_snippet_RF.AppendText("nsel,s,node,," + "RP_" + str(interface_index_no) + "\n")
            command_snippet_RF.AppendText(
                f"f, all, fx, %{'table_X_' + force_HR_index_name}%, %{'table_Xi_' + force_HR_index_name}%\n")
            command_snippet_RF.AppendText(
                f"f, all, fy, %{'table_Y_' + force_HR_index_name}%, %{'table_Yi_' + force_HR_index_name}%\n")
            command_snippet_RF.AppendText(
                f"f, all, fz, %{'table_Z_' + force_HR_index_name}%, %{'table_Zi_' + force_HR_index_name}%\n")
            command_snippet_RF.AppendText("nsel,all\n")

            command_snippet_RF.AppendText(
                "\n\n"
                + (
                    "! The above APDL code is intended for ANSYS Classic / APDL users only.\n "
                    "! Please delete it or keep this command snippet suppressed if you "
                    "intend to only use native force and moment objects from ANSYS Mechanical\n"
                ))
            # endregion

            # region Create APDL command snippet for moment loads
            command_snippet_RM = analysis_HR.AddCommandSnippet()
            command_snippet_RM.Name = "Commands_RM_" + interface_name

            command_snippet_RM.AppendText(''.join(apdl_lines_RMx))
            command_snippet_RM.AppendText(''.join(apdl_lines_RMy))
            command_snippet_RM.AppendText(''.join(apdl_lines_RMz))
            command_snippet_RM.AppendText(''.join(apdl_lines_RMxi))
            command_snippet_RM.AppendText(''.join(apdl_lines_RMyi))
            command_snippet_RM.AppendText(''.join(apdl_lines_RMzi))
            command_snippet_RM.AppendText("\n\n"+ f"! Apply the load on the remote point specified for the interface\n")

            # Apply load as a nodal moment directly from the APDL table defined for RM
            command_snippet_RM.AppendText("nsel,s,node,," + "RP_" + str(interface_index_no) + "\n")
            command_snippet_RM.AppendText(
                f"f, all, mx, %{'table_X_' + moment_HR_index_name}%, %{'table_Xi_' + moment_HR_index_name}%\n")
            command_snippet_RM.AppendText(
                f"f, all, my, %{'table_Y_' + moment_HR_index_name}%, %{'table_Yi_' + moment_HR_index_name}%\n")
            command_snippet_RM.AppendText(
                f"f, all, mz, %{'table_Z_' + moment_HR_index_name}%, %{'table_Zi_' + moment_HR_index_name}%\n")
            command_snippet_RM.AppendText("nsel,all\n")

            command_snippet_RM.AppendText(
                "\n\n"
                + (
                    "! The above APDL code is intended to be used for both ANSYS Mechanical & Classical Users.\n"
                    "! DO NOT DELETE IT!\n"
                ))
            # endregion
            # endregion

            # region Make the command snippet objects for loads to be initially suppressed in the tree if they are not used
            command_snippet_RF.Suppressed = True
            #command_snippet_RM.Suppressed = True
            # endregion

            # region Suppress moment objects as they are only used for debugging/visualization purposes
            moment_HR.Suppressed = True
            # endregion

            # region Delete force or moment object if T1,T2,T3 or R1,R2,R3 components are all zero, making obj undefined
            # Check whether input lists are all zero for a set of lists
            def are_all_zeroes(*lists):
                return all(all(x == 0 for x in lst) for lst in lists)

            if are_all_zeroes(interface_dicts_full[interface_name]["R1"],
                              interface_dicts_full[interface_name]["R2"],
                              interface_dicts_full[interface_name]["R3"]):
                moment_HR.Delete()
                command_snippet_RM.Delete()

            if are_all_zeroes(interface_dicts_full[interface_name]["T1"],
                              interface_dicts_full[interface_name]["T2"],
                              interface_dicts_full[interface_name]["T3"]):
                force_HR.Delete()
                command_snippet_RF.Delete()
            # endregion

            # region Delete command snippet for force (obsolete code)
            command_snippet_RF.Delete()
            # endregion

            # Increase the interface index counter by 1
            interface_index_no += 1

        # region Save the analysis template in the solution directory
        app_ansys.save(os.path.join(os.getcwd(), "WE_Loading_Template.mechdat"))

        dir_path = 'WE_Loading_Template_Mech_Files'
        try:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                print(f'WE_Loading_Template.mechdat file is created successfully in: \n\n "{os.getcwd()}"')
            else:
                print(f"The directory {dir_path} does not exist.")
        except Exception as e:
            print(f"An error occurred while trying to delete the directory {dir_path}: {e}")

        app_ansys.print_tree(DataModel.Project.Model)

        QMessageBox.information(self, "Extraction Complete",
                                f'WE_Loading_Template.mechdat file is created successfully in: \n\n "{os.getcwd()}." \n\n '
                                f'Please restart the whole program first if you need to re-run this button.')

        # Open up the folder of the output file
        os.startfile(os.getcwd())
        # endregion

        app_ansys.close()
        # endregion
        ############################################################################

    def create_ansys_mechanical_input_template_transient(self):
        ############################################################################
        # region Collect interface names and metadata to be used as input loads for the selected part
        # List of all possible keys for load components and phase angles
        all_keys = ["T1", "T2", "T3", "R1", "R2", "R3"]

        # Function to extract the interface name
        def get_full_interface_name(column_name):
            column_name = re.sub(r'\s+[TR][1-3]$', '', column_name)
            return column_name

        # Identify unique interfaces and their corresponding columns
        full_interfaces = OrderedDict()
        for col in self.df_extracted_loads_for_ansys.columns:
            if col != "TIME":
                interface_name = get_full_interface_name(col)
                if interface_name not in full_interfaces:
                    full_interfaces[interface_name] = []
                full_interfaces[interface_name].append(col)

        # Function to create interface dictionary with all keys populated, missing ones with zeroes
        def create_full_interface_dict(df, columns):
            # Initialize the dictionary with all keys set to lists of zeroes
            interface_dict = {key: [0] * len(df) for key in all_keys}
            for col in columns:
                key = col.split()[-1]  # Get the last part which is T1, T2, etc.
                interface_dict[key] = df[col].tolist()
            return interface_dict

        # Create full dictionaries for each unique interface
        interface_dicts_full = {interface: create_full_interface_dict(self.df_extracted_loads_for_ansys, cols) for
                                interface, cols in
                                full_interfaces.items()}

        # List of interfaces for the selected part
        list_of_part_interface_names = list(interface_dicts_full.keys())
        # endregion

        # region Import libraries
        print("Importing ansys-mechanical-core library...")
        import ansys.api.mechanical
        import ansys.mechanical.core
        from ansys.mechanical.core import global_variables
        from ansys.mechanical.core import App
        print("Imported.")
        # endregion

        # region Initialization of ANSYS Mechanical App
        print("Starting Ansys Mechanical...")
        app_ansys = App(private_appdata=False)
        print("Running the scripts...")
        globals().update(global_variables(app_ansys))
        print("Updating the global variables...")
        # endregion

        # region Set Mechanical Units as Standard MKS
        ExtAPI.Application.ActiveUnitSystem = Ansys.ACT.Interfaces.Common.MechanicalUnitSystem.StandardMKS
        scale_factor = 1000
        # endregion

        # region Helper function to partition long dataframes of time vs. force / moment tabular data
        @staticmethod
        def partition_dataframe_for_load_input(df, partition_size):
            partitions = []
            prev_last_row = None  # To store the last row of the previous partition

            # Get the column names (assuming 'TIME' is the first column and the data column is the second)
            time_column = df.columns[0]
            data_column = df.columns[1]

            for i in range(0, len(df), partition_size):
                partition = df[i:i + partition_size]

                # Create a new row with 0 for both time and the corresponding data column
                zero_row = pd.DataFrame({time_column: [0], data_column: [0]})

                # If this is not the first partition, add the last row of the previous partition with the data set to 0
                if prev_last_row is not None:
                    prev_last_row_zero = prev_last_row.copy()
                    prev_last_row_zero[data_column] = 0  # Set the data column to zero
                    partition_with_last = pd.concat([zero_row, prev_last_row_zero, partition]).reset_index(drop=True)
                else:
                    # For the first partition, just add the zero row
                    partition_with_last = pd.concat([zero_row, partition]).reset_index(drop=True)

                # Store the last row of the current partition for the next iteration
                prev_last_row = partition.tail(1)

                partitions.append(partition_with_last)

            return partitions
        # endregion

        # region Initialize the list of time points extracted
        # Create a separate list for FREQ
        list_of_all_time_points = self.df_extracted_loads_for_ansys["TIME"].tolist()

        # Convert list of time points to a list of time points as quantities (in seconds)
        list_of_all_time_points_as_quantity = []
        for time_point in list_of_all_time_points:
            list_of_all_time_points_as_quantity.append(Quantity(time_point, "sec"))
        # endregion

        # region Initialize geometry import
        geometry_source = Model.AddGeometryImportGroup()
        geometry_source.AddGeometryImport()
        # endregion

        # region Create a static analysis environment template for pre-stressed solution
        print("Creating the static analysis environment...")

        analysis_static = Model.AddStaticStructuralAnalysis()
        analysis_settings_static = analysis_static.AnalysisSettings
        analysis_settings_static.PropertyByName("SolverUnitsControl").InternalValue = 1  # Manual
        analysis_settings_static.PropertyByName("SelectedSolverUnitSystem").InternalValue  # nmm unit system
        # endregion

        # region Create a modal analysis environment template for MSUP based solution
        print("Creating the modal analysis environment...")
        analysis_modal = Model.AddModalAnalysis()
        analysis_settings_modal = analysis_modal.AnalysisSettings
        analysis_settings_modal_IC = DataModel.GetObjectsByName("Pre-Stress (None)")[0]
        analysis_settings_modal_IC.PreStressICEnvironment = analysis_static
        analysis_settings_modal.PropertyByName("NumModesToFind").InternalValue = 100
        analysis_settings_modal.PropertyByName("RangeSearch").InternalValue = 1  # Limit Search to Range: On
        analysis_settings_modal.PropertyByName("MaxFrequency").InternalValue = \
            self.sample_rate  # Max Freq. to find is 2x the freq of the time points, calculated from average sampling rate (since nyquist frequency is about 2x or more)
        #analysis_settings_modal.PropertyByName("MSUPSkipExpansion").InternalValue = 1  # On-Demand Expansion Option: Yes
        analysis_settings_modal.PropertyByName("SolverUnitsControl").InternalValue = 1  # Manual
        analysis_settings_modal.PropertyByName("SelectedSolverUnitSystem").InternalValue  # nmm unit system
        # endregion

        # region Create tree objects in Mechanical for transient analysis and initialize analysis settings
        print("Creating the transient analysis environment...")
        analysis_TR = Model.AddTransientStructuralAnalysis()
        analysis_settings_TR = analysis_TR.AnalysisSettings
        analysis_settings_TR_IC = DataModel.GetObjectsByName("Modal (None)")[0]
        analysis_settings_TR_IC.ModalICEnvironment = analysis_modal
        analysis_settings_TR.PropertyByName("TimeStepDefineby").InternalValue = 0  # by Substeps
        analysis_settings_TR.PropertyByName("NumberOfSubSteps").InternalValue = len(list_of_all_time_points)
        analysis_settings_TR.PropertyByName("EndTime").InternalValue = max(list_of_all_time_points)
        analysis_settings_TR.PropertyByName("MSUPSkipExpansion").InternalValue = 1  # Yes
        analysis_settings_TR.PropertyByName("CombineDistResultFile").InternalValue = 1  # Yes
        analysis_settings_TR.PropertyByName("ExpandResultFrom").InternalValue = 1  # Expand from Modal Solution
        analysis_settings_TR.PropertyByName("ConstantDampingValue").InternalValue = 0.02  # 2% damping by default
        analysis_settings_TR.PropertyByName("SolverUnitsControl").InternalValue = 1  # Manual
        analysis_settings_TR.PropertyByName("SelectedSolverUnitSystem").InternalValue  # nmm unit system
        analysis_settings_TR.IncludeResidualVector = True
        # endregion

        # region Add a python object for clearing-up unnecessary files in solution folder after extracting the MCF file
        python_code_for_cleanup = analysis_TR.Solution.AddPythonCodeEventBased()
        python_code_for_cleanup.TargetCallback = Ansys.Mechanical.DataModel.Enums.PythonCodeTargetCallback.OnAfterPost
        python_code_for_cleanup.Text ="""
import os

def delete_files_except_mcf(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path) and not file_name.endswith(".mcf"):
            os.remove(file_path)

def after_post(this, solution):  # Do not edit this line
    # Get the working directory
    solution_directory_path = solution.WorkingDir

    if solution_directory_path:
        delete_files_except_mcf(solution_directory_path)
    else:
        print("Error: Working directory not found.")
"""
        # endregion

        # region Determine the number of partitions required
        number_of_rows_in_mechanical = 50000
        number_of_partitions = math.ceil(len(list_of_all_time_points) / number_of_rows_in_mechanical)
        # endregion

        # region Create load objects and their dependent objects for each interface
        interface_index_no = 1
        for interface_name in list_of_part_interface_names:

            # region Create dependent objects

            # region Add a named selection for each interface face (or element face)
            NS_interface = Model.AddNamedSelection()
            NS_interface.Name = "NS_" + interface_name
            # endregion

            # region Create force and/or moment objects at each interface
            # Add a reference coordinate system for each interface
            CS_interface = Model.CoordinateSystems.AddCoordinateSystem()
            CS_interface.Name = "CS_" + interface_name

            # Create remote points for each interface
            RP_interface = Model.AddRemotePoint()
            RP_interface.Name = "RP_" + interface_name
            RP_interface.CoordinateSystem = CS_interface
            RP_interface.PilotNodeAPDLName = "RP_" + str(interface_index_no)
            # endregion

            force_TR_index_name = "RF_" + str(interface_index_no)
            moment_TR_index_name = "RM_" + str(interface_index_no)
            # endregion

            # region Define numerical values of loads
            # region Create force dataframes (which will serve as sources when creating APDL tables)
            # region Fx Dataframe
            df_load_table_fx = pd.DataFrame({
                'TIME': list_of_all_time_points,
                'T1': interface_dicts_full[interface_name]["T1"]})
            #df_load_table_fx.set_index('TIME', inplace=True)
            # endregion

            # region Fy Dataframe
            df_load_table_fy = pd.DataFrame({
                'TIME': list_of_all_time_points,
                'T2': interface_dicts_full[interface_name]["T2"]})
            #df_load_table_fy.set_index('TIME', inplace=True)
            # endregion

            # region Fz Dataframe
            df_load_table_fz = pd.DataFrame({
                'TIME': list_of_all_time_points,
                'T3': interface_dicts_full[interface_name]["T3"]})
            #df_load_table_fz.set_index('TIME', inplace=True)
            # endregion
            # endregion

            # region Create moment dataframes (which will serve as sources when creating APDL tables)
            # region Mx Dataframe
            df_load_table_mx = pd.DataFrame({
                'TIME': list_of_all_time_points,
                'R1': interface_dicts_full[interface_name]["R1"]})
            #df_load_table_mx.set_index('TIME', inplace=True)
            # endregion

            # region My Dataframe
            df_load_table_my = pd.DataFrame({
                'TIME': list_of_all_time_points,
                'R2': interface_dicts_full[interface_name]["R2"]})
            #df_load_table_my.set_index('TIME', inplace=True)
            # endregion

            # region Mz Dataframe
            df_load_table_mz = pd.DataFrame({
                'TIME': list_of_all_time_points,
                'R3': interface_dicts_full[interface_name]["R3"]})
            #df_load_table_mz.set_index('TIME', inplace=True)
            # endregion
            # endregion

            # region Partition force dataframes
            print(f"Partitioning force data for {interface_name}...")

            partitioned_df_load_table_fx = partition_dataframe_for_load_input(
                df_load_table_fx, partition_size=50000)
            partitioned_df_load_table_fy = partition_dataframe_for_load_input(
                df_load_table_fy, partition_size=50000)
            partitioned_df_load_table_fz = partition_dataframe_for_load_input(
                df_load_table_fz, partition_size=50000)
            # endregion

            # region Partition moment dataFrames
            print(f"Partitioning moment data for {interface_name}...")

            partitioned_df_load_table_mx = partition_dataframe_for_load_input(
                df_load_table_mx, partition_size=50000)
            partitioned_df_load_table_my = partition_dataframe_for_load_input(
                df_load_table_my, partition_size=50000)
            partitioned_df_load_table_mz = partition_dataframe_for_load_input(
                df_load_table_mz, partition_size=50000)
            # endregion
            # endregion

            # region Define numerical values of loads as Quantity objects
            def convert_partition_to_quantity(partitioned_dfs, key, unit):
                partitioned_dfs_quantity = []
                for df in partitioned_dfs:
                    df_quantity = df.copy()
                    # Convert the values from the column corresponding to the key (e.g., 'T1', 'R1')
                    df_quantity[key] = [Quantity(value, unit) for value in df[key]]
                    df_quantity['TIME'] = [Quantity(value, "sec") for value in df['TIME']]
                    partitioned_dfs_quantity.append(df_quantity)
                return partitioned_dfs_quantity

            # Convert the partitioned DataFrames to quantity-based DataFrames
            print('Converting partitioned force data into ANSYS Quantity objects...')
            partitioned_df_load_table_fx_quantity = convert_partition_to_quantity(partitioned_df_load_table_fx, 'T1', "kN")
            partitioned_df_load_table_fy_quantity = convert_partition_to_quantity(partitioned_df_load_table_fy, 'T2', "kN")
            partitioned_df_load_table_fz_quantity = convert_partition_to_quantity(partitioned_df_load_table_fz, 'T3', "kN")
            print('Converting partitioned moment data into ANSYS Quantity objects...')
            partitioned_df_load_table_mx_quantity = convert_partition_to_quantity(partitioned_df_load_table_mx, 'R1', "kN m")
            partitioned_df_load_table_my_quantity = convert_partition_to_quantity(partitioned_df_load_table_my, 'R2', "kN m")
            partitioned_df_load_table_mz_quantity = convert_partition_to_quantity(partitioned_df_load_table_mz, 'R3', "kN m")

            # endregion

            # region Define T1,T2,T3 and R1, R2, R3 loads via Command Objects
            print(f"Populating the input tabular data for {interface_name} as APDL tables...")
            command_snippet_RF = analysis_TR.AddCommandSnippet()
            command_snippet_RM = analysis_TR.AddCommandSnippet()
            command_snippet_RF.Name = "Commands_RF_" + interface_name
            command_snippet_RM.Name = "Commands_RM_" + interface_name

            apdl_lines_RFx = self.create_APDL_table(df_load_table_fx * scale_factor,"table_X_" + force_TR_index_name)
            apdl_lines_RFy = self.create_APDL_table(df_load_table_fy * scale_factor, "table_Y_" + force_TR_index_name)
            apdl_lines_RFz = self.create_APDL_table(df_load_table_fz * scale_factor, "table_Z_" + force_TR_index_name)

            apdl_lines_RMx = self.create_APDL_table(df_load_table_mx * scale_factor,"table_X_" + moment_TR_index_name)
            apdl_lines_RMy = self.create_APDL_table(df_load_table_my * scale_factor, "table_Y_" + moment_TR_index_name)
            apdl_lines_RMz = self.create_APDL_table(df_load_table_mz * scale_factor, "table_Z_" + moment_TR_index_name)

            command_snippet_RF.AppendText(''.join(apdl_lines_RFx))
            command_snippet_RF.AppendText(''.join(apdl_lines_RFy))
            command_snippet_RF.AppendText(''.join(apdl_lines_RFz))
            command_snippet_RF.AppendText("\n\n"+ f"! Apply the load on the remote point specified for the interface\n")
            command_snippet_RF.AppendText("nsel,s,node,," + "RP_" + str(interface_index_no) + "\n")

            command_snippet_RF.AppendText(
                f"f, all, fx, %{'table_X_' + force_TR_index_name}%\n")
            command_snippet_RF.AppendText(
                f"f, all, fy, %{'table_Y_' + force_TR_index_name}%\n")
            command_snippet_RF.AppendText(
                f"f, all, fz, %{'table_Z_' + force_TR_index_name}%\n")
            command_snippet_RF.AppendText("nsel,all\n")

            command_snippet_RF.AppendText(
                "\n\n"
                + (
                    "! The above APDL code is intended for ANSYS Classic / APDL users only.\n "
                    "! Please delete it or keep this command snippet suppressed if you "
                    "intend to only use native force and moment objects from ANSYS Mechanical\n"
                ))

            command_snippet_RM.AppendText(''.join(apdl_lines_RMx))
            command_snippet_RM.AppendText(''.join(apdl_lines_RMy))
            command_snippet_RM.AppendText(''.join(apdl_lines_RMz))
            command_snippet_RM.AppendText("\n\n"+ f"! Apply the load on the remote point specified for the interface\n")
            command_snippet_RM.AppendText("nsel,s,node,," + "RP_" + str(interface_index_no) + "\n")

            command_snippet_RM.AppendText(
                f"f, all, mx, %{'table_X_' + moment_TR_index_name}%\n")
            command_snippet_RM.AppendText(
                f"f, all, my, %{'table_Y_' + moment_TR_index_name}%\n")
            command_snippet_RM.AppendText(
                f"f, all, mz, %{'table_Z_' + moment_TR_index_name}%\n")
            command_snippet_RM.AppendText("nsel,all\n")

            command_snippet_RF.AppendText(
                "\n\n"
                + (
                    "! The above APDL code is intended for ANSYS Classic / APDL users only.\n "
                    "! Please delete it or keep this command snippet suppressed if you "
                    "intend to only use native force and moment objects from ANSYS Mechanical\n"
                ))
            # endregion

            # region Define T1,T2,T3 and R1,R2,R3 loads as force and moment objects (partitioned)
            print(f"Creating force & moment objects for {interface_name}...")

            force_TR_index_name = "RF_" + str(interface_index_no) # Reserved for future implementation in APDL(not used)
            moment_TR_index_name = "RM_" + str(interface_index_no)# Reserved for future implementation in APDL(not used)

            force_TR_list_of_all_partitions = []  # List to collect all force_TR objects
            moment_TR_list_of_all_partitions = []  # List to collect all moment_TR objects

            for i in range(number_of_partitions):
                print(f"Applying the partitioned load: {i+1}/{number_of_partitions}")
                # region Create remote force objects at each interface
                force_TR = analysis_TR.AddRemoteForce()
                force_TR.DefineBy = Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
                force_TR.Name = "RF_" + interface_name + "_Part_" + str(i+1)
                force_TR.PropertyByName("GeometryDefineBy").InternalValue = 1  # Scoped to named selection
                force_TR.Location = NS_interface
                force_TR.CoordinateSystem = CS_interface
                force_TR_list_of_all_partitions.append(force_TR)
                # endregion

                # region Create moments at each interface
                moment_TR = analysis_TR.AddMoment()
                moment_TR.DefineBy = Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
                moment_TR.Name = "RM_" + interface_name + "_Part_" + str(i+1)
                moment_TR.PropertyByName("GeometryDefineBy").InternalValue = 2  # Scoped to remote_point
                moment_TR.Location = RP_interface
                moment_TR_list_of_all_partitions.append(moment_TR)
                # endregion

                # region Populate load objects with numerical values obtained from each relevant interface
                print(f"Populating the input tabular data for {interface_name}...")
                # Define force time points for each partition
                force_TR.XComponent.Inputs[0].DiscreteValues = partitioned_df_load_table_fx_quantity[i]['TIME'].tolist()
                force_TR.YComponent.Inputs[0].DiscreteValues = partitioned_df_load_table_fy_quantity[i]['TIME'].tolist()
                force_TR.ZComponent.Inputs[0].DiscreteValues = partitioned_df_load_table_fz_quantity[i]['TIME'].tolist()

                # Assign force values (fx, fy, fz) for each partition
                force_TR.XComponent.Output.DiscreteValues = partitioned_df_load_table_fx_quantity[i]['T1'].tolist()
                force_TR.YComponent.Output.DiscreteValues = partitioned_df_load_table_fy_quantity[i]['T2'].tolist()
                force_TR.ZComponent.Output.DiscreteValues = partitioned_df_load_table_fz_quantity[i]['T3'].tolist()

                # Define moment time points for each partition
                moment_TR.XComponent.Inputs[0].DiscreteValues = partitioned_df_load_table_mx_quantity[i]['TIME'].tolist()
                moment_TR.YComponent.Inputs[0].DiscreteValues = partitioned_df_load_table_my_quantity[i]['TIME'].tolist()
                moment_TR.ZComponent.Inputs[0].DiscreteValues = partitioned_df_load_table_mz_quantity[i]['TIME'].tolist()

                # Assign moment values (mx, my, mz) for each partition
                moment_TR.XComponent.Output.DiscreteValues = partitioned_df_load_table_mx_quantity[i]['R1'].tolist()
                moment_TR.YComponent.Output.DiscreteValues = partitioned_df_load_table_my_quantity[i]['R2'].tolist()
                moment_TR.ZComponent.Output.DiscreteValues = partitioned_df_load_table_mz_quantity[i]['R3'].tolist()
                # endregion
                # endregion
            # endregion

            # region Suppress command snippets for forces and moments as they are not used in transient analysis
            command_snippet_RF.Suppressed = True
            command_snippet_RM.Suppressed = True
            # endregion

            # region Delete force or moment object if T1,T2,T3 or R1,R2,R3 components are all zero, making obj undefined
            # Check whether input lists are all zero for a set of lists
            def are_all_zeroes(*lists):
                return all(all(x == 0 for x in lst) for lst in lists)

            if are_all_zeroes(interface_dicts_full[interface_name]["R1"],
                              interface_dicts_full[interface_name]["R2"],
                              interface_dicts_full[interface_name]["R3"]):
                [moment_TR.Delete() for moment_TR in moment_TR_list_of_all_partitions]
                command_snippet_RM.Delete()

            if are_all_zeroes(interface_dicts_full[interface_name]["T1"],
                              interface_dicts_full[interface_name]["T2"],
                              interface_dicts_full[interface_name]["T3"]):
                [force_TR.Delete() for force_TR in force_TR_list_of_all_partitions]
                command_snippet_RF.Delete()
            # endregion

            # Increase the interface index counter by 1
            interface_index_no += 1

        # region Save the analysis template in the solution directory
        app_ansys.save(os.path.join(os.getcwd(), "WE_Loading_Template.mechdat"))

        dir_path = 'WE_Loading_Template_Mech_Files'
        try:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                print(f'WE_Loading_Template.mechdat file is created successfully in: \n\n "{os.getcwd()}"')
            else:
                print(f"The directory {dir_path} does not exist.")
        except Exception as e:
            print(f"An error occurred while trying to delete the directory {dir_path}: {e}")

        app_ansys.print_tree(DataModel.Project.Model)

        QMessageBox.information(self, "Extraction Complete",
                                f'WE_Loading_Template.mechdat file is created successfully in: \n\n "{os.getcwd()}." \n\n '
                                f'Please restart the whole program first if you need to re-run this button.')

        # Open up the folder of the output file
        os.startfile(os.getcwd())
        # endregion

        app_ansys.close()
        # endregion
        ############################################################################

    def open_new_data(self):
        try:
            # Ask for one folder containing both raw data and header files
            folder_selected = select_directory('Please select a directory for raw data and headers')
            if not folder_selected:
                return

            file_path_full_data = get_file_path(folder_selected, 'full.pld')
            file_path_headers_data = get_file_path(folder_selected, 'max.pld')

            if not file_path_full_data or not file_path_headers_data:
                QMessageBox.critical(self, 'Error', "No required files found!")
                return

            # Read and combine all full.pld files in that folder
            dfs = [read_pld_file(file_path) for file_path in file_path_full_data]
            new_df = pd.concat(dfs, ignore_index=True)

            # Determine the new data domain
            if 'FREQ' in new_df.columns:
                new_domain = 'FREQ'
            elif 'TIME' in new_df.columns:
                new_domain = 'TIME'
            else:
                QMessageBox.critical(self, "Error", "Data does not contain a recognized domain ('FREQ' or 'TIME').")
                return

            # Compare with the current domain
            global DATA_DOMAIN
            if new_domain != DATA_DOMAIN:
                QMessageBox.warning(
                    self,
                    "Domain Mismatch",
                    f"Cannot load new data. Current domain is '{DATA_DOMAIN}' while new data domain is '{new_domain}'."
                )
                return

            # Continue as before
            df_intf_before = read_pld_log_file(file_path_headers_data[0])
            if new_domain == 'FREQ':
                df_intf = insert_phase_columns(df_intf_before)
                df_intf_labels = df_intf.iloc[0]
                new_columns = ['NO', 'FREQ'] + df_intf_labels.tolist()
            elif new_domain == 'TIME':
                df_intf = df_intf_before
                df_intf_labels = df_intf.iloc[0]
                new_columns = ['NO', 'TIME'] + df_intf_labels.tolist()

            additional_columns_needed = len(new_df.columns) - len(new_columns)
            if additional_columns_needed > 0:
                extended_new_columns = new_columns + [f"Extra_Column_{i}" for i in range(1, additional_columns_needed + 1)]
                new_df.columns = extended_new_columns
            else:
                new_df.columns = new_columns[:len(new_df.columns)]

            self.df = new_df
            self.df.to_csv("full_data.csv", index=False)

            # Optionally update UI elements (e.g. repopulate selectors) here...
            # ---

            self.raw_data_folder = folder_selected  # store the newly selected folder
            folder_name = os.path.basename(self.raw_data_folder) if self.raw_data_folder else ""
            parent_folder = os.path.basename(os.path.dirname(self.raw_data_folder)) if self.raw_data_folder else ""
            # Rename the windows title so that directory is updated
            self.setWindowTitle(f"WE MechLoad Viewer - v0.97.1    |    (Directory Folder: {parent_folder})")

            self.refresh_directory_tree()

            QMessageBox.information(self, "File Opened", "New data loaded successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open new data: {str(e)}")

    def refresh_directory_tree(self, new_folder=None):
        """
        Recompute and set the root path for self.file_model and update
        the tree view's root index. If new_folder is given, use that.
        Otherwise, default to self.raw_data_folder.
        """
        if not new_folder:
            new_folder = self.raw_data_folder

        if not new_folder:
            return

        # If you want to show the parent folder in the tree:
        parent_folder = os.path.dirname(new_folder)
        # Or, if you prefer the folder itself to be the root, you can do:
        # parent_folder = new_folder

        # Update the file model's root path
        self.file_model.setRootPath(parent_folder)

        # Retrieve the matching index in the model
        new_index = self.file_model.index(parent_folder)
        if new_index.isValid():
            self.tree_view.setRootIndex(new_index)


    def on_directory_selected(self, selected, deselected):
        indexes = self.tree_view.selectionModel().selectedRows()
        if not indexes:
            return

        combined_dfs = []
        valid_indexes = []  # To store indexes that pass validation

        for index in indexes:
            folder = self.file_model.filePath(index)
            if not os.path.isdir(folder):
                continue
            files = os.listdir(folder)
            has_full = any(f.endswith("full.pld") for f in files)
            has_max = any(f.endswith("max.pld") for f in files)
            if not (has_full and has_max):
                QMessageBox.warning(
                    self,
                    "Invalid Folder",
                    f"Folder '{os.path.basename(folder)}' does not contain the required input files.\n\nSkipping this folder."
                )
                # Remove navigation highlight for skipped / invalid folder
                with QtCore.QSignalBlocker(self.tree_view.selectionModel()):
                        self.tree_view.selectionModel().select(index, QtCore.QItemSelectionModel.Deselect)
                continue

            try:
                dfs = [read_pld_file(os.path.join(folder, f)) for f in files if f.endswith("full.pld")]
                if not dfs:
                    continue
                df_temp = pd.concat(dfs, ignore_index=True)
                if 'FREQ' in df_temp.columns:
                    folder_data_type = 'FREQ'
                elif 'TIME' in df_temp.columns:
                    folder_data_type = 'TIME'
                else:
                    continue

                global DATA_DOMAIN
                if folder_data_type != DATA_DOMAIN:
                    QMessageBox.warning(
                        self,
                        "Domain Mismatch",
                        f"Folder '{os.path.basename(folder)}' has data type '{folder_data_type}', but the program is initialized as '{DATA_DOMAIN}'.\nSkipping this folder."
                    )
                    # deselect mismatching-domain folder
                    with QtCore.QSignalBlocker(self.tree_view.selectionModel()):
                        self.tree_view.selectionModel().select(index, QtCore.QItemSelectionModel.Deselect)
                    continue

                header_files = [os.path.join(folder, f) for f in files if f.endswith("max.pld")]
                if not header_files:
                    continue
                df_intf_before = read_pld_log_file(header_files[0])
                if 'FREQ' in df_temp.columns:
                    df_intf = insert_phase_columns(df_intf_before)
                    df_intf_labels = df_intf.iloc[0]
                    new_columns = ['NO', 'FREQ'] + df_intf_labels.tolist()
                else:
                    df_intf = df_intf_before
                    df_intf_labels = df_intf.iloc[0]
                    new_columns = ['NO', 'TIME'] + df_intf_labels.tolist()

                additional_columns_needed = len(df_temp.columns) - len(new_columns)
                if additional_columns_needed > 0:
                    extended_new_columns = new_columns + [f"Extra_Column_{i}" for i in
                                                          range(1, additional_columns_needed + 1)]
                    df_temp.columns = extended_new_columns
                else:
                    df_temp.columns = new_columns[:len(df_temp.columns)]

                # Tag the DataFrame with the folder name.
                df_temp['DataFolder'] = os.path.basename(folder)
                combined_dfs.append(df_temp)
                valid_indexes.append(index)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data from {folder}: {str(e)}")
                return

        if not combined_dfs:
            print("No valid data found in selected folders.")
            return

        # Concatenate valid Dataframes
        new_df = pd.concat(combined_dfs, ignore_index=True)
        self.df = new_df
        self.df.to_csv("full_data.csv", index=False)

        # Instead of clearing and resetting the selection, let the valid selections remain.
        # If needed, you might selectively remove invalid selections without overriding the user’s multi‑selection.

        QMessageBox.information(self, "Data Loaded", "Data from selected folders loaded successfully!")

        # Decide which tabs to enable
        num_loaded_folders = len(combined_dfs)  # 1 → single, >1 → multiple
        total_tabs = self.tab_widget.count()

        # Disable all tabs except Single Data (index 0) and Settings (last index) in multi-folder mode
        for idx in range(total_tabs):
            if num_loaded_folders > 1:  # multi-folder mode
                enable = (idx == 0) or (idx == total_tabs - 1)
            else:  # single-folder mode
                enable = True
            self.tab_widget.setTabEnabled(idx, enable)

        self.update_all_transient_data_plots()

        num_valid_folders = len(valid_indexes)
        print(f"Loaded {num_valid_folders} valid folder(s).")
    # endregion

    # region Define the logic for each combobox

    def update_font_settings(self):
        self.legend_font_size = int(self.legend_font_size_selector.currentText())
        self.default_font_size = int(self.default_font_size_selector.currentText())
        self.hover_font_size = int(self.hover_font_size_selector.currentText())
        self.hover_mode = self.hover_mode_selector.currentText()
        self.update_plots_tab1()
        self.update_plots_tab2()
        self.update_plots_tab3()
        if 'FREQ' in self.df.columns:
            self.update_time_domain_plot()
        self.update_compare_plots()
        self.update_compare_part_loads_plots()

    def setupInterfaceSelector(self, layout):
        self.interface_selector = QComboBox()
        self.interface_selector.setEditable(True)
        interface_pattern = re.compile(r'I\d{1,6}[A-Za-z]?')
        interfaces = sorted(set(filter(None, [interface_pattern.match(col.split(' ')[0]).group()
                                              if interface_pattern.match(col.split(' ')[0]) else None for col in
                                              self.df.columns])))
        interfaces = natsorted(interfaces)
        self.interface_selector.addItems(interfaces)
        self.interface_selector.currentIndexChanged.connect(self.update_plots_tab2)
        layout.addWidget(self.interface_selector)

    def setupSideSelectionTab2(self, layout):
        side_selection_layout = QHBoxLayout()
        side_selection_label = QLabel("Part Side Filter")
        side_selection_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.side_selector = QComboBox()
        self.side_selector.setEditable(True)
        self.side_selector.currentIndexChanged.connect(self.update_side_selection_tab_2)
        side_selection_layout.addWidget(side_selection_label)
        side_selection_layout.addWidget(self.side_selector)
        layout.addLayout(side_selection_layout)

    def setupSideFilterSelector(self, layout, for_compare=False):
        self.side_filter_selector = QComboBox()
        self.side_filter_selector.setEditable(True)
        self.populate_side_filter_selector()
        if for_compare:
            self.side_filter_selector.currentIndexChanged.connect(self.update_compare_part_loads_plots)
        else:
            self.side_filter_selector.currentIndexChanged.connect(self.update_plots_tab3)
        layout.addWidget(self.side_filter_selector)

    def setupSideFilterSelectorForCompare(self, layout):
        self.side_filter_selector_for_compare = QComboBox()
        self.side_filter_selector_for_compare.setEditable(True)
        self.side_filter_selector_for_compare.currentIndexChanged.connect(self.update_compare_part_loads_plots)
        layout.addWidget(self.side_filter_selector_for_compare)

    def populate_side_filter_selector(self):
        pattern = re.compile(r'(?<=\s-)(.*?)(?=\s*\()')
        sides = set()
        for col in self.df.columns:
            # Skip columns starting with 'Phase_'
            if col.startswith('Phase_'):
                continue
            match = pattern.search(col)
            if match:
                sides.add(match.group(1).strip())

        self.side_filter_selector.addItems(sorted(sides))
        self.side_filter_selector_for_compare.addItems(sorted(sides))

    def populate_side_selector_tab_2(self, interface):
        pattern = re.compile(r'I\d+[a-zA-Z]?\s*-\s*(.*?)(?=\s*\()')
        relevant_columns = [col for col in self.df.columns if re.match(rf"^{re.escape(interface)}(?=\D)", col)]
        sides = sorted(set(pattern.search(col).group(1).strip() for col in relevant_columns if pattern.search(col)))

        self.side_selector.clear()
        if sides:
            self.side_selector.addItems(sides)
    # endregion

    # region Define custom keyboard events and their logic
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_L:
            self.legend_visible = not self.legend_visible
            self.update_plots_tab1()
            self.update_plots_tab2()
            self.update_plots_tab3()
            if 'FREQ' in self.df.columns:
                self.update_time_domain_plot()
            self.update_compare_plots()
            self.update_compare_part_loads_plots()
        elif event.key() == QtCore.Qt.Key_K:
            self.current_legend_position = (self.current_legend_position + 1) % len(self.legend_positions)
            self.update_plots_tab1()
            self.update_plots_tab2()
            self.update_plots_tab3()
            if 'FREQ' in self.df.columns:
                self.update_time_domain_plot()
            self.update_compare_plots()
            self.update_compare_part_loads_plots()

    def get_legend_position(self):
        positions = {
            'default': {'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
            'top left': {'x': 0, 'y': 1, 'xanchor': 'auto', 'yanchor': 'auto'},
            'top right': {'x': 1, 'y': 1, 'xanchor': 'auto', 'yanchor': 'auto'},
            'bottom right': {'x': 1, 'y': 0, 'xanchor': 'auto', 'yanchor': 'auto'},
            'bottom left': {'x': 0, 'y': 0, 'xanchor': 'auto', 'yanchor': 'auto'}
        }
        return positions.get(self.legend_positions[self.current_legend_position],
                             {'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'})
    # endregion

    # region Helper methods for updating the plots
    def get_x_axis_data_and_label(self):
        if 'TIME' in self.df.columns:
            return self.df['TIME'], 'Time [s]'
        elif 'FREQ' in self.df.columns:
            return self.df['FREQ'], 'Freq [Hz]'
        else:
            raise ValueError("Neither 'TIME' nor 'FREQ' columns found in the DataFrame.")

    def update_T_and_R_plots(self, t_plot, r_plot, interface=None, side=None, exclude_t2_t3_r2_r3=False):
        if not side:
            return

        side_pattern = re.compile(rf'\b{re.escape(side)}\b')

        # Filtering function to whether exclude T2/T3 or R2/R3 data from plots
        def should_exclude_resultant_components(col):
            if re.search(r'\bT2\b', col) and not re.search(r'T2/T3', col):
                print(f"Excluding column {col} due to T2 exclusion rule.")
                return True
            if re.search(r'\bT3\b', col) and not re.search(r'T2/T3', col):
                print(f"Excluding column {col} due to T3 exclusion rule.")
                return True
            if re.search(r'\bR2\b', col) and not re.search(r'R2/R3', col):
                print(f"Excluding column {col} due to R2 exclusion rule.")
                return True
            if re.search(r'\bR3\b', col) and not re.search(r'R2/R3', col):
                print(f"Excluding column {col} due to R3 exclusion rule.")
                return True
            return False

        # Lists to hold filtered columns
        t_series_columns = []
        r_series_columns = []

        # Loop through columns
        for col in self.df.columns:

            # Check individual conditions
            check_1 = not interface or re.match(r'^' + re.escape(interface) + r'([-\s]|$)', col)
            r"""Find whether column name starts with the specified interface (e.g. "I1"), 
            followed by either a hyphen (-), a whitespace (\s), or the end of the string ($)"""

            check_2_force = any(sub in col for sub in ["T1", "T2", "T3", "T2/T3"])
            r"""If any of the substrings ("T1", "T2", "T3", "T2/T3") is found in the column name (col), 
            the expression will evaluate to True."""

            check_2_moment = any(sub in col for sub in ["R1", "R2", "R3", "R2/R3"])
            r"""If any of the substrings ("R1", "R2", "R3", "R2/R3") is found in the column name (col), 
            the expression will evaluate to True."""

            check_3 = not col.startswith('Phase_')
            "Filters out columns that starts with 'Phase_1' prefix"

            check_4 = side_pattern.search(col) is not None
            """Filter columns relevant to a selected part side"""

            check_5 = not (exclude_t2_t3_r2_r3 and should_exclude_resultant_components(col))
            """If exclude_t2_t3_r2_r3 flag is raised, 
            resultant components are filtered out based using a search pattern for exclusion"""

            # Append to the T-series or R-series columns based on checks
            if check_1 and check_2_force and check_3 and check_4 and check_5:
                t_series_columns.append(col)

            if check_1 and check_2_moment and check_3 and check_4 and check_5:
                r_series_columns.append(col)

        # Debugging: Print filtered columns
        print(f"Filtered T series columns: {t_series_columns}")
        print(f"Filtered R series columns: {r_series_columns}")

        # Use helper function to get x_data
        x_data, x_label = self.get_x_axis_data_and_label()

        # Replace the block that builds/plots the data to avoid overwriting it before filtering
        df_for_plot = self.df.copy()

        # Section-based trimming
        if ('TIME' in df_for_plot.columns
            and getattr(self, 'section_checkbox', None)
            and self.section_checkbox.isChecked()
            and hasattr(self, 'section_min_value')
            and hasattr(self, 'section_max_value')):

            t_min = self.section_min_value
            t_max = self.section_max_value
            df_for_plot = df_for_plot[(df_for_plot['TIME'] >= t_min) & (df_for_plot['TIME'] <= t_max)]

        # Apply Tukey window
        if 'TIME' in df_for_plot.columns and getattr(self, 'tukey_checkbox_tab3', None) \
                and self.tukey_checkbox_tab3.isChecked():
            w = tukey(len(df_for_plot), self.tukey_alpha_spin.value())
            cols = [c for c in df_for_plot.columns if c not in ('TIME', 'FREQ', 'NO')]
            df_for_plot[cols] = df_for_plot[cols].multiply(w, axis=0)

        # x-axis for the figure
        x_data = df_for_plot['TIME'] if 'TIME' in df_for_plot.columns else df_for_plot['FREQ']

        # Create actual plots
        self.create_and_style_figure(t_plot, df_for_plot[t_series_columns], x_data, 'Translational Components')
        self.create_and_style_figure(r_plot, df_for_plot[r_series_columns], x_data, 'Rotational Components')

    def calculate_differences(self, df, df_compare, columns, is_freq_data):
        results = []
        missing_keys = []
        for col in columns:
            try:
                magnitude1 = df[col]
                magnitude2 = df_compare[col]
                phase_col = f'Phase_{col}'

                if is_freq_data and all(col in df.columns for col in [col, phase_col]) and all(
                        col in df_compare.columns for col in [col, phase_col]):
                    phase1 = df[phase_col]
                    phase2 = df_compare[phase_col]

                    complex1 = magnitude1 * np.exp(1j * phase1)
                    complex2 = magnitude2 * np.exp(1j * phase2)

                    complex_diff = complex1 - complex2
                    magnitude_diff = np.abs(complex_diff)
                else:
                    magnitude_diff = magnitude1 - magnitude2

                results.append((col, magnitude_diff))
            except KeyError as e:
                missing_keys.append(str(e))

        if missing_keys:
            QMessageBox.critical(
                None,
                'Key Error',
                f"The following keys were not found in at least one of the datasets: {', '.join(missing_keys)}. "
                f"The program will show the rest of the common traces."
            )

        return results

    def create_and_style_figure(self, web_view, df, x_data, title):
        try:
            if df is None:
                web_view.setHtml("<html><body><p>No data available to plot here.</p></body></html>")
                return

            # Set the x_data as the index of the DataFrame
            df.index = x_data

            legend_position = self.get_legend_position()

            # Check if rolling min-max envelope should be used
            if self.rolling_min_max_checkbox.isChecked():
                self.plot_as_bars_checkbox.setVisible(True)
                self.desired_num_points_label.setVisible(True)
                self.desired_num_points_input.setVisible(True)
                try:
                    desired_num_points = int(self.desired_num_points_input.text())
                except ValueError:
                    print("Invalid input. Please enter a valid integer number.")
                    desired_num_points = 2000

                is_plot_as_bars = self.plot_as_bars_checkbox.isChecked()

                fig = rolling_min_max_envelope(
                    df,
                    opacity=0.6,
                    desired_num_points=desired_num_points,
                    plot_as_bars=is_plot_as_bars,
                    plot_title=f'Rolling Min-Max Envelope for {title}'
                )

                fig.update_layout(
                    margin=dict(l=20, r=20, t=35, b=35),
                    legend=dict(
                        font=dict(family='Open Sans', size=self.legend_font_size, color='black'),
                        x=legend_position['x'],
                        y=legend_position['y'],
                        xanchor=legend_position.get('xanchor', 'auto'),
                        yanchor=legend_position.get('yanchor', 'top'),
                        bgcolor='rgba(255, 255, 255, 0.5)'
                    ),
                    hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=self.hover_font_size),
                    hovermode=self.hover_mode,
                    font=dict(family='Open Sans', size=self.default_font_size, color='black'),
                    showlegend=self.legend_visible,
                    yaxis_title="Data"
                )

            else:
                self.plot_as_bars_checkbox.setVisible(False)
                self.desired_num_points_label.setVisible(False)
                self.desired_num_points_input.setVisible(False)

                y_data_list = [df[col] for col in df.columns]
                column_names = df.columns
                custom_hover = ('%{fullData.name}<br>' + (
                    'Hz: ' if 'FREQ' in self.df.columns else 'Time: ') + '%{x}<br>Value: %{y:.3f}<extra></extra>')

                fig = go.Figure()
                for y_data, name in zip(y_data_list, column_names):
                    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=name, hovertemplate=custom_hover))

                fig.update_layout(
                    title=title,
                    margin=dict(l=20, r=20, t=35, b=35),
                    legend=dict(
                        font=dict(family='Open Sans', size=self.legend_font_size, color='black'),
                        x=legend_position['x'],
                        y=legend_position['y'],
                        xanchor=legend_position.get('xanchor', 'auto'),
                        yanchor=legend_position.get('yanchor', 'top'),
                        bgcolor='rgba(255, 255, 255, 0.5)'
                    ),
                    hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=self.hover_font_size),
                    hovermode=self.hover_mode,
                    font=dict(family='Open Sans', size=self.default_font_size, color='black'),
                    showlegend=self.legend_visible,
                    yaxis_title="Data"
                )

            self.load_fig_to_webview(fig, web_view)

        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred while creating and styling figures: {str(e)}")

    def populate_web_view_with_plot(self, web_view, columns, title):
        if not columns:
            web_view.setHtml("<html><body><p>No data available to plot here.</p></body></html>")
            return

        if 'FREQ' in self.df.columns:
            x_data = self.df['FREQ']
            custom_hover = ('%{fullData.name}<br>Hz: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>')
        elif 'TIME' in self.df.columns:
            x_data = self.df['TIME']
            custom_hover = ('%{fullData.name}<br>Time: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>')

        y_data_dict = {col: self.df[col] for col in columns}
        df = pd.DataFrame(y_data_dict)

        self.create_and_style_figure(web_view, df, x_data, title)

    def load_fig_to_webview(self, fig, web_view):
        """Generates full HTML with embedded JS, saves to temp file, and loads."""
        try:
            # Generate full HTML with embedded Plotly.js
            html_content = pio.to_html(fig,
                                       full_html=True,         # Generate <html>, <head>, <body>
                                       include_plotlyjs=True,  # Embed the JS library
                                       config={'responsive': True}) # Make plot responsive

            # Create a temporary file (delete=False is important here)
            # Suffix helps identify the file type, encoding ensures compatibility
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
                 tmp_file.write(html_content)
                 file_path = tmp_file.name
                 self.temp_files.append(file_path) # Keep track for potential cleanup

            # Load the local file URL
            web_view.setUrl(QtCore.QUrl.fromLocalFile(file_path))
            web_view.show() # Ensure it's visible

        except Exception as e:
            print(f"Error loading figure to webview: {e}")
            traceback.print_exc()
            try:
                # Fallback: Display error message in the webview
                error_html = f"<html><body><h1>Error loading plot</h1><pre>{e}</pre><pre>{traceback.format_exc()}</pre></body></html>"
                web_view.setHtml(error_html)
            except Exception as fallback_e:
                 print(f"Error displaying fallback error message: {fallback_e}")

    def plot_with_rolling_min_max_envelope(self, df, x_data, columns, web_view, plot_title, x_label):
        if 'TIME' in self.df.columns and self.rolling_min_max_checkbox.isChecked():
            try:
                desired_num_points = int(self.desired_num_points_input.text())
            except ValueError:
                print("Invalid input. Please enter a valid integer number.")
                desired_num_points = 2000

            is_plot_as_bars = self.plot_as_bars_checkbox.isChecked()

            fig = rolling_min_max_envelope(
                df[columns],
                opacity=0.6,
                desired_num_points=desired_num_points,
                plot_as_bars=is_plot_as_bars,
                plot_title=f'Rolling Min-Max Envelope for {plot_title}'
            )

            legend_position = self.get_legend_position()
            fig.update_layout(
                margin=dict(l=20, r=20, t=35, b=35),
                legend=dict(
                    font=dict(family='Open Sans', size=self.legend_font_size, color='black'),
                    x=legend_position['x'],
                    y=legend_position['y'],
                    xanchor=legend_position.get('xanchor', 'auto'),
                    yanchor=legend_position.get('yanchor', 'top'),
                    bgcolor='rgba(255, 255, 255, 0.5)'
                ),
                hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=self.hover_font_size),
                hovermode=self.hover_mode,
                font=dict(family='Open Sans', size=self.default_font_size, color='black'),
                showlegend=self.legend_visible,
                yaxis_title="Data"
            )

            self.load_fig_to_webview(fig, web_view)

        else:
            self.create_and_style_figure(
                web_view,
                df[columns],
                x_data,
                plot_title
            )

    def update_all_transient_data_plots(self):
        self.update_plots_tab1()
        self.update_plots_tab2()
        self.update_plots_tab3()
        self.update_compare_plots()
        self.update_compare_part_loads_plots()

    def update_spectrum_plot(self):
        if self.working_df_tab1 is not None:
            value_col = self.column_selector.currentText()
            plot_type = self.plot_type_selector.currentText()
            if value_col:
                try:
                    # Remove the previous canvas if it exists
                    if hasattr(self, 'spectrum_canvas') and self.spectrum_canvas is not None:
                        self.spectrum_plot.layout().removeWidget(self.spectrum_canvas)
                        self.spectrum_canvas.deleteLater()
                        self.spectrum_canvas = None

                    y_data = self.working_df_tab1[value_col].values

                    # Handle non-uniform time data by interpolating to a uniform grid
                    if isinstance(self.working_df_tab1.index, pd.DatetimeIndex) or isinstance(
                            self.working_df_tab1.index, pd.Index):
                        time_data = self.working_df_tab1.index.values
                        time_diffs = np.diff(time_data)
                        delta_t_min = np.min(time_diffs)
                        uniform_time = np.arange(time_data.min(), time_data.max(), delta_t_min)
                        y_data = np.interp(uniform_time, time_data, y_data)
                        fs = 1 / delta_t_min  # Recalculate sample rate for the uniform grid
                    else:
                        fs = self.sample_rate

                    # Dynamically set nperseg as a fraction of the data length
                    nperseg = max(len(y_data) // 8, 128)  # Use at least 128 data points per segment
                    noverlap = min(nperseg // 2, len(y_data) // 4)  # Ensure noverlap is less than nperseg

                    if self.specgram_checkbox.isChecked():
                        # Create a Matplotlib figure and axis
                        fig, ax = plt.subplots()

                        # Generate the spectrogram
                        Pxx, freqs, bins, im = ax.specgram(y_data, NFFT=nperseg, Fs=fs, noverlap=noverlap,
                                                           cmap='viridis')

                        # Set plot titles and labels
                        ax.set_title(f'Spectrogram of {value_col}')
                        ax.set_xlabel('Time [s]')
                        ax.set_ylabel('Frequency [Hz]')

                        # Synchronize width
                        plotly_graph_width = self.regular_plot.size().width()
                        fig.set_size_inches(1.05 * plotly_graph_width / fig.dpi, fig.get_figheight())
                        fig.subplots_adjust(left=0.055, right=0.975, top=0.9, bottom=0.1)

                        # Create a FigureCanvas and add it to the spectrum_plot layout
                        self.spectrum_canvas = FigureCanvas(fig)
                        layout = self.spectrum_plot.layout()
                        if layout is None:
                            layout = QVBoxLayout(self.spectrum_plot)
                        layout.addWidget(self.spectrum_canvas)

                        # Draw the canvas
                        self.spectrum_canvas.draw()

                        plt.close(fig)  # Close the figure after drawing to save memory
                    elif self.spectrum_checkbox.isChecked():
                        # Apply FFT on the uniform time grid data
                        fft_df = rolling_fft(self.working_df_tab1[[value_col]], num_slices=400, add_resultant=True)
                        fig_spectrum = spectrum_over_time(fft_df, plot_type=plot_type, freq_max=None,
                                                     var_to_process=value_col)

                        self.load_fig_to_webview(fig_spectrum, self.spectrum_plot)
                except Exception as e:
                    QMessageBox.critical(None, 'Error', f"An error occurred while creating the spectrum plot: {str(e)}")

    def apply_butterworth_filter(self, data):
        try:
            cutoff = float(self.cutoff_frequency_input.text())
            order = self.filter_order_input.value()
            fs = self.sample_rate  # Sampling frequency

            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist

            # Get the filter coefficients
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data)
            return y
        except ValueError:
            return data  # Return unfiltered data if there's an issue
    # endregion

    # region Helper methods for creating and modifying the dataframes for plots
    def create_dataframes(self, x_data, x_label, y_data_dict):
        try:
            # Ensure x_data and y_data_dict have the same length and make if y_data_dict is not empty
            if y_data_dict and len(x_data) == len(list(y_data_dict.values())[0]):
                y_data_dict[x_label] = x_data
                original_df = pd.DataFrame(y_data_dict)
                original_df.set_index(x_label, inplace=True)
                working_df = original_df.copy()
                return original_df, working_df
            else:
                print("Warning: Length mismatch between X and Y data values. This is likely due to initialization.")
                return None, None
        except Exception as e:
            print(f"Error in create_dataframes: {str(e)}")
            return None, None
    # endregion

    # region Main methods for updating the plots after a selection is made

    def update_plots_tab1_Yedek(self):
        selected_column = self.column_selector.currentText()
        if selected_column:
            x_data = self.df['FREQ'] if 'FREQ' in self.df.columns else self.df['TIME']
            x_label = 'Freq [Hz]' if 'FREQ' in self.df.columns else 'Time [s]'
            custom_hover = ('%{fullData.name}<br>' + (
                'Hz: ' if 'FREQ' in self.df.columns else 'Time: ') + '%{x}<br>Value: %{y:.3f}<extra></extra>')

            # Creating dataframe container for tab1 (magnitude)
            y_data_dict = {selected_column: self.df[selected_column]}
            self.original_df_tab1, self.working_df_tab1 = self.create_dataframes(x_data, x_label, y_data_dict)

            # Apply Butterworth filter if enabled and TIME is in the columns
            if 'TIME' in self.df.columns and self.filter_checkbox.isChecked():
                filtered_y_data = self.apply_butterworth_filter(self.working_df_tab1[selected_column])
                self.working_df_tab1[selected_column] = filtered_y_data

            # Plot with rolling min-max envelope
            self.plot_with_rolling_min_max_envelope(
                self.working_df_tab1,
                x_data,
                [selected_column],
                self.regular_plot,
                f'{selected_column} Plot',
                x_label
            )

            # Handle phase plot if 'FREQ' is in the columns
            if 'FREQ' in self.df.columns:
                phase_column = 'Phase_' + selected_column
                y_data_dict = {phase_column: self.df[phase_column]}
                self.original_df_phase_tab1, self.working_df_phase_tab1 = self.create_dataframes(x_data, x_label,
                                                                                                 y_data_dict)

                self.create_and_style_figure(
                    self.phase_plot,
                    self.working_df_phase_tab1,
                    x_data,
                    f'Phase {selected_column} Plot'
                )

                # Creating dataframe container for tab1 (phase)
                y_data_dict = {phase_column: self.df[phase_column]}
                self.original_df_phase_tab1, self.working_df_phase_tab1 = self.create_dataframes(x_data, x_label,
                                                                                                 y_data_dict)

    def update_plots_tab1(self):
        selected_column = self.column_selector.currentText()
        if not selected_column:
            # Optionally clear the views if no column is selected
            if hasattr(self, 'regular_plot'): self.regular_plot.setHtml("")
            if hasattr(self, 'phase_plot'): self.phase_plot.setHtml("")
            return

        # Add error handling around data access
        try:
            # Determine x-axis data and label based on domain.
            if 'FREQ' in self.df.columns:
                x_data = self.df['FREQ']
                x_label = 'Freq [Hz]'
            elif 'TIME' in self.df.columns:
                x_data = self.df['TIME']
                x_label = 'Time [s]'
            else:
                # Handle case where neither column exists
                raise ValueError("DataFrame must contain 'FREQ' or 'TIME' column.")

            # Create a copy with x_data as index to preserve proper x-axis values.
            # Ensure column exists before copying
            if selected_column not in self.df.columns:
                 raise KeyError(f"Selected column '{selected_column}' not found in DataFrame.")
            if 'Phase_' + selected_column not in self.df.columns and 'FREQ' in self.df.columns:
                 print(f"Warning: Phase column for '{selected_column}' not found.")
                 # Decide how to handle this - perhaps disable phase plot? For now, it will error later.

            df_plot = self.df.copy()
            df_plot.index = x_data
            df_plot.index.name = x_label # Set index name for axis labels

            # Create working DataFrame for the selected column.
            # Make sure data exists before creating dataframe
            y_data_dict = {}
            if selected_column in df_plot.columns:
                 y_data_dict[selected_column] = df_plot[selected_column]

            # create_dataframes now takes index directly
            self.original_df_tab1, self.working_df_tab1 = self.create_dataframes(df_plot.index, x_label, y_data_dict)

            # Apply Butterworth filter if enabled (for TIME domain). Check if working_df is valid.
            if not self.working_df_tab1.empty and 'TIME' in self.df.columns and self.filter_checkbox.isChecked():
                try:
                    # Check if column exists after dataframe creation (should normally)
                    if selected_column in self.working_df_tab1.columns:
                         filtered_y_data = self.apply_butterworth_filter(self.working_df_tab1[selected_column])
                         self.working_df_tab1[selected_column] = filtered_y_data
                    else:
                         print(f"Warning: Column '{selected_column}' missing in working_df_tab1 for filtering.")
                except Exception as e:
                    print(f"Error applying filter: {e}")

            # Define custom hover template and legend settings.
            custom_hover = (
                    '%{fullData.name}<br>' +
                    ('Hz: ' if 'FREQ' in self.df.columns else 'Time: ') +
                    '%{x}<br>Value: %{y:.3f}<extra></extra>'
            )
            legend_position = self.get_legend_position()

            # Check if rolling min-max envelope option is enabled. Check if checkbox exists.
            rolling_checked = False
            if hasattr(self, 'rolling_min_max_checkbox'):
                 rolling_checked = self.rolling_min_max_checkbox.isChecked()

            # --- Magnitude Plot Figure Creation ---
            fig = None # Initialize fig
            plot_title = f'{selected_column} Plot' # Base title

            if rolling_checked and 'TIME' in self.df.columns:
                # --- Rolling Envelope Figure ---
                plot_title += ' (Envelope)'
                # Checkbox visibility managed here
                if hasattr(self, 'plot_as_bars_checkbox'): self.plot_as_bars_checkbox.setVisible(True)
                if hasattr(self, 'desired_num_points_label'): self.desired_num_points_label.setVisible(True)
                if hasattr(self, 'desired_num_points_input'): self.desired_num_points_input.setVisible(True)

                desired_num_points = 50000 # Default
                is_plot_as_bars = False # Default
                try:
                    if hasattr(self, 'desired_num_points_input'):
                         desired_num_points = int(self.desired_num_points_input.text())
                except ValueError: pass # Keep default
                if hasattr(self, 'plot_as_bars_checkbox'):
                     is_plot_as_bars = self.plot_as_bars_checkbox.isChecked()

                # Ensure working df is valid before plotting
                if not self.working_df_tab1.empty:
                     # Check grouping for multiple folders
                     if 'DataFolder' in self.df.columns and self.df['DataFolder'].nunique() > 1:
                          fig = go.Figure() # Create empty figure to add traces
                          domain_col = 'TIME' if 'TIME' in self.df.columns else 'FREQ'
                          temp_df_grouped = self.df.set_index(domain_col) # Use original df as reference for grouping
                          for folder, group in temp_df_grouped.groupby('DataFolder'):
                               # Apply filter to the specific group data if needed
                               group_data_for_env = group[[selected_column]].copy()
                               if self.filter_checkbox.isChecked():
                                    try:
                                        group_data_for_env[selected_column] = self.apply_butterworth_filter(group[selected_column])
                                    except Exception as e:
                                        print(f"Error applying filter to group {folder}: {e}")

                               # Generate envelope for the group
                               envelope_fig = rolling_min_max_envelope(
                                    group_data_for_env,
                                    opacity=0.6,
                                    desired_num_points=desired_num_points,
                                    plot_as_bars=is_plot_as_bars,
                                    # title set later
                               )
                               # Add traces from group envelope fig to main fig
                               for trace in envelope_fig.data:
                                    trace.name = folder # Set name for legend
                                    trace.legendgroup = folder
                                    fig.add_trace(trace)
                     else:
                          # Single folder or no folder column
                          fig = rolling_min_max_envelope(
                               self.working_df_tab1, # Use working df (potentially filtered)
                               opacity=0.6,
                               desired_num_points=desired_num_points,
                               plot_as_bars=is_plot_as_bars,
                               # title set later
                          )
                          # If single folder, name traces appropriately
                          if fig and 'DataFolder' in self.df.columns and self.df['DataFolder'].nunique() == 1:
                               unique_folder = self.df['DataFolder'].unique()[0]
                               for trace in fig.data:
                                    trace.name = unique_folder
                                    trace.legendgroup = unique_folder
                else:
                     print("Working DataFrame is empty, cannot create envelope plot.")
                     # fig remains None

            else:
                # --- Standard plot (without rolling envelope) ---
                 # Hide envelope controls
                if 'TIME' in self.df.columns:
                     if hasattr(self, 'plot_as_bars_checkbox'): self.plot_as_bars_checkbox.setVisible(False)
                     if hasattr(self, 'desired_num_points_label'): self.desired_num_points_label.setVisible(False)
                     if hasattr(self, 'desired_num_points_input'): self.desired_num_points_input.setVisible(False)

                # Ensure working df is valid before plotting
                if not self.working_df_tab1.empty:
                     fig = go.Figure()
                     # Check grouping for multiple folders
                     if 'DataFolder' in self.df.columns and self.df['DataFolder'].nunique() > 1:
                          # Use working_df for plotting values (filtered if applicable)
                          temp_df_grouped = self.working_df_tab1.copy()
                          temp_df_grouped['DataFolder'] = self.df['DataFolder'].values # Add folder column back
                          for folder, group in temp_df_grouped.groupby('DataFolder'):
                               fig.add_trace(go.Scatter(
                                    x=group.index,
                                    y=group[selected_column], # Use potentially filtered data
                                    mode='lines',
                                    name=folder,
                                    opacity = 0.6,
                                    hovertemplate=custom_hover
                               ))
                     else:
                          # Single folder or no folder column
                           fig.add_trace(go.Scatter(
                               x=self.working_df_tab1.index,
                               y=self.working_df_tab1[selected_column], # Use potentially filtered data
                               mode='lines',
                               name=selected_column, # Or folder name if single folder
                               hovertemplate=custom_hover
                          ))
                           # If single folder, name trace appropriately
                           if 'DataFolder' in self.df.columns and self.df['DataFolder'].nunique() == 1:
                                fig.data[0].name = self.df['DataFolder'].unique()[0]
                                fig.data[0].legendgroup = self.df['DataFolder'].unique()[0]

                else:
                    print("Working DataFrame is empty, cannot create standard plot.")
                    # fig remains None


            # --- Apply Final Layout and Load Magnitude Plot ---
            if fig is not None:
                # Update layout (common styling)
                fig.update_layout(
                    title=plot_title, # Apply title
                    margin=dict(l=40, r=20, t=50, b=40),
                    legend=dict(
                        font=dict(family='Open Sans', size=self.legend_font_size, color='black'),
                        x=legend_position['x'], y=legend_position['y'],
                        xanchor=legend_position.get('xanchor', 'auto'), yanchor=legend_position.get('yanchor', 'top'),
                        bgcolor='rgba(255,255,255,0.6)'
                    ),
                    hoverlabel=dict(bgcolor='rgba(240, 240, 240, 0.9)', font_size=self.hover_font_size),
                    hovermode=self.hover_mode,
                    font=dict(family='Open Sans', size=self.default_font_size, color='black'),
                    showlegend=self.legend_visible,
                    xaxis_title=x_label, # Use determined x_label
                    yaxis_title="Value"
                )
                # Replace the setHtml call with the helper function
                self.load_fig_to_webview(fig, self.regular_plot) # <--- MODIFIED
            else:
                 # Handle case where fig couldn't be created
                 self.regular_plot.setHtml(f"<html><body>Could not generate plot for {selected_column}.</body></html>")


            # --- Phase Plot ---
            if 'FREQ' in self.df.columns and hasattr(self, 'phase_plot'):
                phase_column = 'Phase_' + selected_column
                fig_phase = None # Initialize

                if phase_column in df_plot.columns: # Check if phase column exists in original data copy
                     # Create phase dataframe (no filtering applied to phase)
                     y_data_dict_phase = {phase_column: df_plot[phase_column]}
                     self.original_df_phase_tab1, self.working_df_phase_tab1 = self.create_dataframes(df_plot.index, x_label, y_data_dict_phase)

                     if not self.working_df_phase_tab1.empty:
                          fig_phase = go.Figure()
                          # Check grouping for multiple folders
                          if 'DataFolder' in self.df.columns and self.df['DataFolder'].nunique() > 1:
                               temp_df_grouped = self.working_df_phase_tab1.copy()
                               temp_df_grouped['DataFolder'] = self.df['DataFolder'].values
                               for folder, group in temp_df_grouped.groupby('DataFolder'):
                                    fig_phase.add_trace(go.Scatter(
                                         x=group.index,
                                         y=group[phase_column],
                                         mode='lines',
                                         name=folder,
                                         hovertemplate=custom_hover
                                    ))
                          else:
                               # Single folder or no folder column
                               fig_phase.add_trace(go.Scatter(
                                    x=self.working_df_phase_tab1.index,
                                    y=self.working_df_phase_tab1[phase_column],
                                    mode='lines',
                                    name=phase_column, # Or folder name if single folder
                                    hovertemplate=custom_hover
                               ))
                               if 'DataFolder' in self.df.columns and self.df['DataFolder'].nunique() == 1:
                                    fig_phase.data[0].name = self.df['DataFolder'].unique()[0]
                                    fig_phase.data[0].legendgroup = self.df['DataFolder'].unique()[0]

                     else:
                          print("Working phase DataFrame is empty.")
                          # fig_phase remains None

                else:
                     print(f"Phase column '{phase_column}' not found.")
                     # fig_phase remains None


                # --- Apply Final Layout and Load Phase Plot ---
                if fig_phase is not None:
                     # Update layout
                     fig_phase.update_layout(
                          title=f'Phase {selected_column} Plot',
                          margin=dict(l=40, r=20, t=50, b=40),
                          legend=dict(
                               font=dict(family='Open Sans', size=self.legend_font_size, color='black'),
                               x=legend_position['x'], y=legend_position['y'],
                               xanchor=legend_position.get('xanchor', 'auto'), yanchor=legend_position.get('yanchor', 'top'),
                               bgcolor='rgba(255,255,255,0.6)'
                          ),
                          hovermode=self.hover_mode,
                          hoverlabel=dict(bgcolor='rgba(240, 240, 240, 0.9)', font_size=self.hover_font_size),
                          font=dict(family='Open Sans', size=self.default_font_size, color='black'),
                          showlegend=self.legend_visible,
                          xaxis_title=x_label, # Use determined x_label
                          yaxis_title="Phase [deg]" # More specific title
                     )
                     # Replace the setHtml call with the helper function
                     self.load_fig_to_webview(fig_phase, self.phase_plot) # <--- MODIFIED
                else:
                     # Handle case where phase fig couldn't be created
                     self.phase_plot.setHtml(f"<html><body>Could not generate phase plot for {selected_column}.</body></html>")

            # Clear phase plot if not frequency domain
            elif hasattr(self, 'phase_plot'):
                  self.phase_plot.setHtml("")


        except Exception as e:
             print(f"Error updating plots in Tab 1: {e}")
             traceback.print_exc()
             # Show error in the relevant plot views
             error_html = f"<html><body>Error updating plot:<br><pre>{e}</pre><pre>{traceback.format_exc()}</pre></body></html>"
             if hasattr(self, 'regular_plot'): self.regular_plot.setHtml(error_html)
             if hasattr(self, 'phase_plot'): self.phase_plot.setHtml("") # Clear phase plot on error

    def update_plots_tab2(self):
        interface = self.interface_selector.currentText()
        if interface:
            self.populate_side_selector_tab_2(interface)
        selected_side = self.side_selector.currentText()
        self.update_T_and_R_plots(self.t_series_plot, self.r_series_plot, interface, selected_side)

        # Create dataframe containers for tab2 (T and R plots)
        side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')

        t_series_columns = [col for col in self.df.columns
                            if any(sub in col for sub in ['T1', 'T2', 'T3', 'T2/T3'])
                            and side_pattern.search(col)]

        r_series_columns = [col for col in self.df.columns
                                      if any(sub in col for sub in ['R1', 'R2', 'R3', 'R2/R3'])
                                      and side_pattern.search(col)]

        # Use helper function to get x_data and x_label
        x_data, x_label = self.get_x_axis_data_and_label()

        y_data_dict_t_series = {col: self.df[col] for col in t_series_columns}
        self.original_df_t_series_tab2, self.working_df_t_series_tab2 = self.create_dataframes(x_data, x_label,
                                                                                               y_data_dict_t_series)

        y_data_dict_r_series = {col: self.df[col] for col in r_series_columns}
        self.original_df_r_series_tab2, self.working_df_r_series_tab2 = self.create_dataframes(x_data, x_label,
                                                                                               y_data_dict_r_series)

    def update_plots_tab3(self):
        selected_side = self.side_filter_selector.currentText()

        exclude_t2_t3_r2_r3 = self.exclude_checkbox.isChecked()
        self.update_T_and_R_plots(self.t_series_plot_tab3, self.r_series_plot_tab3, side=selected_side,
                                  exclude_t2_t3_r2_r3=exclude_t2_t3_r2_r3)
        if 'FREQ' in self.df.columns:
            self.update_time_domain_plot()

        # Create dataframe containers for tab3 (T and R plots)
        side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')

        t_series_columns = [col for col in self.df.columns
                            if any(sub in col for sub in ['T1', 'T2', 'T3', 'T2/T3'])
                            and side_pattern.search(col)]

        r_series_columns = [col for col in self.df.columns
                                      if any(sub in col for sub in ['R1', 'R2', 'R3', 'R2/R3'])
                                      and side_pattern.search(col)]

        # Use helper function to get x_data and x_label
        x_data, x_label = self.get_x_axis_data_and_label()

        y_data_dict_t_series = {col: self.df[col] for col in t_series_columns}
        self.original_df_t_series_tab3, self.working_df_t_series_tab3 = self.create_dataframes(x_data, x_label,
                                                                                               y_data_dict_t_series)

        y_data_dict_r_series = {col: self.df[col] for col in r_series_columns}
        self.original_df_r_series_tab3, self.working_df_r_series_tab3 = self.create_dataframes(x_data, x_label,
                                                                                               y_data_dict_r_series)

    def update_compare_plots(self):
        try:
            selected_column = self.compare_column_selector.currentText()
            if selected_column and self.df_compare is not None:
                is_freq_data = 'FREQ' in self.df.columns
                x_data = self.df['FREQ'] if is_freq_data else self.df['TIME']
                x_label = 'Freq [Hz]' if 'FREQ' in self.df.columns else 'Time [s]'
                x_data_compare = self.df_compare['FREQ'] if is_freq_data else self.df_compare['TIME']
                custom_hover = ('%{fullData.name}<br>' + (
                    'Hz: ' if is_freq_data else 'Time: ') + '%{x}<br>Value: %{y:.3f}<extra></extra>')

                # Regular plot
                y_data_dict = {
                    f'Original {selected_column}': self.df[selected_column],
                    f'Compare {selected_column}': self.df_compare[selected_column]
                }
                self.original_df_compare_tab, self.working_df_compare_tab = self.create_dataframes(x_data, x_label,
                                                                                                   y_data_dict)
                self.create_and_style_figure(
                    self.compare_regular_plot,
                    self.working_df_compare_tab,
                    x_data,
                    f'{selected_column} Comparison'
                )

                # Calculate differences
                results = self.calculate_differences(self.df, self.df_compare, [selected_column], is_freq_data)

                # Absolute difference plot
                y_data_dict = {f'Absolute Δ {selected_column}': results[0][1]}
                self.original_df_absolute_diff_tab, self.working_df_absolute_diff_tab = self.create_dataframes(x_data,
                                                                                                               x_label,
                                                                                                               y_data_dict)
                self.create_and_style_figure(
                    self.compare_absolute_diff_plot,
                    self.working_df_absolute_diff_tab,
                    x_data,
                    f'{selected_column} Absolute Difference'
                )

                # Relative difference plot
                y_data_dict = {f'Relative Δ {selected_column} (%)': 100 * results[0][1] / self.df[selected_column]}
                self.original_df_relative_diff_tab, self.working_df_relative_diff_tab = self.create_dataframes(x_data,
                                                                                                               x_label,
                                                                                                               y_data_dict)
                self.create_and_style_figure(
                    self.compare_percent_diff_plot,
                    self.working_df_relative_diff_tab,
                    x_data,
                    f'{selected_column} Relative Difference'
                )
        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred while updating compare plots: {str(e)}")

    def update_compare_part_loads_plots(self):
        try:
            selected_side = self.side_filter_selector_for_compare.currentText()
            if not selected_side or self.df_compare is None:
                return

            exclude_t2_t3_r2_r3 = self.exclude_checkbox_compare.isChecked()
            side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')

            def should_exclude(col):
                return any(re.search(r'\b' + sub + r'\b', col) and not re.search(rf'{sub}/', col) for sub in
                           ["T2", "T3", "R2", "R3"])

            t_series_columns = [col for col in self.df.columns if side_pattern.search(col) and any(
                sub in col for sub in ["T1", "T2", "T3"]) and not col.startswith('Phase_') and not (
                    exclude_t2_t3_r2_r3 and should_exclude(col))]
            r_series_columns = [col for col in self.df.columns if side_pattern.search(col) and any(
                sub in col for sub in ["R1", "R2", "R3"]) and not col.startswith('Phase_') and not (
                    exclude_t2_t3_r2_r3 and should_exclude(col))]

            if not t_series_columns and not r_series_columns:
                QMessageBox.warning(self, "Warning", "No matching columns found for the selected part side.")
                return

            is_freq_data = 'FREQ' in self.df.columns
            x_data = self.df['FREQ'] if is_freq_data else self.df['TIME']
            x_label = 'Freq [Hz]' if 'FREQ' in self.df.columns else 'Time [s]'
            custom_hover = '%{fullData.name}<br>' + (
                'Hz: ' if is_freq_data else 'Time: ') + '%{x}<br>Value: %{y:.3f}<extra></extra>'

            results_t = self.calculate_differences(self.df, self.df_compare, t_series_columns, is_freq_data)
            results_r = self.calculate_differences(self.df, self.df_compare, r_series_columns, is_freq_data)

            # T Series plot
            y_data_dict_t_series = {f'Δ {col}': magnitude_diff for col, magnitude_diff in results_t}
            self.original_df_compare_t_series_tab, self.working_df_compare_t_series_tab = self.create_dataframes(x_data,
                                                                                                                 x_label,
                                                                                                                 y_data_dict_t_series)
            self.create_and_style_figure(
                self.compare_t_series_plot,
                self.working_df_compare_t_series_tab,
                x_data,
                f'T Plot (Δ) - {selected_side}'
            )

            # R Series plot
            y_data_dict_r_series = {f'Δ {col}': magnitude_diff for col, magnitude_diff in results_r}
            self.original_df_compare_r_series_tab, self.working_df_compare_r_series_tab = self.create_dataframes(x_data,
                                                                                                                 x_label,
                                                                                                                 y_data_dict_r_series)
            self.create_and_style_figure(
                self.compare_r_series_plot,
                self.working_df_compare_r_series_tab,
                x_data,
                f'R Plot (Δ) - {selected_side}'
            )
        except KeyError as e:
            QMessageBox.critical(None, 'Error', f"KeyError: {str(e)}")
        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred while updating compare part loads plots: {str(e)}")

    def update_plots_for_selected_side_tab_2(self, selected_side):
        try:
            # If a side or interface is not selected yet, don't do anything
            if not selected_side:
                return

            interface = self.interface_selector.currentText()
            if not interface:
                return

            pattern = re.compile(r'^' + re.escape(interface) + r'([-\s]|$)')
            side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')

            t_series_columns = [
                col for col in self.df.columns
                if pattern.match(col) and any(sub in col for sub in ["T1", "T2", "T3", "T2/T3"]) and not col.startswith(
                    'Phase_')
                   and side_pattern.search(col)
            ]
            r_series_columns = [
                col for col in self.df.columns
                if pattern.match(col) and any(sub in col for sub in ["R1", "R2", "R3", "R2/R3"]) and not col.startswith(
                    'Phase_')
                   and side_pattern.search(col)
            ]

            x_data = self.df['FREQ'] if 'FREQ' in self.df.columns else self.df['TIME']
            self.create_and_style_figure(self.t_series_plot, self.df[t_series_columns], x_data, 'T Plot')
            self.create_and_style_figure(self.r_series_plot, self.df[r_series_columns], x_data, 'R Plot')

        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred white updating tab2 plots: {str(e)}")

    def update_time_domain_plot(self):
        try:
            freq = float(self.data_point_selector.currentText())
        except ValueError:
            return

        theta = np.linspace(0, 360, 361)
        x_data = np.radians(theta)

        selected_side = self.side_filter_selector.currentText()
        side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')
        displayed_columns = [col for col in self.df.columns if
                             side_pattern.search(col) and
                             any(sub in col for sub in ["T1", "T2", "T3", "T2/T3", "R1", "R2", "R3", "R2/R3"]) and
                             not col.startswith('Phase_')]

        y_data_list = []
        for col in displayed_columns:
            amplitude_col = col
            phase_col = 'Phase_' + col
            amplitude = self.df.loc[self.df['FREQ'] == freq, amplitude_col].values[0]
            phase = self.df.loc[self.df['FREQ'] == freq, phase_col].values[0]
            y_data = amplitude * np.cos(x_data - np.radians(phase))
            y_data_list.append(y_data)
            self.current_plot_data[col] = {'theta': theta, 'y_data': y_data}

        plot_title = f'Time Domain Representation at {str(freq)} Hz - {selected_side}'
        custom_hover = '%{fullData.name}: %{y:.2f}<extra></extra>'
        y_data_dict = {col: self.current_plot_data[col]['y_data'] for col in displayed_columns}
        self.original_df_time_domain_tab4, self.working_df_time_domain_tab4 = self.create_dataframes(theta,
                                                                                                     'Theta [deg]',
                                                                                                     y_data_dict)
        self.create_and_style_figure(
            self.time_domain_plot,
            self.working_df_time_domain_tab4,
            theta,
            plot_title
        )

    def update_side_selection_tab_2(self):
        selected_side = self.side_selector.currentText()
        self.update_plots_for_selected_side_tab_2(selected_side)
        if 'FREQ' in self.df.columns:
            self.update_time_domain_plot()

    def clear_plot_cache(self, show_message=True):
        """Clears temporary plot files and associated web views."""
        print("Clearing plot cache...")
        files_to_remove = list(self.temp_files) # Copy list for safe iteration
        cleared_count = 0
        errors = 0

        # Clear Web Views first (optional, but good practice)
        plot_web_views = [
            getattr(self, name, None) for name in [
                'regular_plot', 'phase_plot', 'spectrum_plot',
                't_series_plot', 'r_series_plot',
                't_series_plot_tab3', 'r_series_plot_tab3',
                'time_domain_plot', 'compare_regular_plot',
                'compare_absolute_diff_plot', 'compare_percent_diff_plot',
                'compare_t_series_plot', 'compare_r_series_plot'
            ] if hasattr(self, name) # Ensure the attribute exists
        ]

        for view in plot_web_views:
            if view: # Check if the view object is valid
                try:
                    # Using about:blank is a standard way to clear web views
                    view.setUrl(QtCore.QUrl("about:blank"))
                except Exception as e:
                    print(f"Could not clear web view {view}: {e}")


        # Delete Files
        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleared_count += 1
                    # print(f"Removed temp file: {file_path}") # Uncomment for debugging
                # Remove from the original list even if it didn't exist (cleanup)
                if file_path in self.temp_files:
                    self.temp_files.remove(file_path)
            except OSError as e:
                print(f"Error removing temp file {file_path}: {e}")
                errors += 1
                # Keep file in self.temp_files if removal failed, maybe retry later?

        print(f"Cache cleared: {cleared_count} files removed, {errors} errors.")

        if show_message:
            if errors > 0:
                QMessageBox.warning(self, "Cache Cleared with Errors",
                                     f"Cleared {cleared_count} plot cache file(s).\n"
                                     f"Could not remove {errors} file(s). See console for details.")
            elif cleared_count > 0:
                QMessageBox.information(self, "Cache Cleared",
                                        f"Cleared {cleared_count} plot cache file(s).")
            else:
                QMessageBox.information(self, "Cache Empty",
                                        "The plot cache was already empty.")

    def closeEvent(self, event):
        """Clean up temporary files on application close."""
        print("Cleaning up temporary plot files...")
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # print(f"Removed temp file: {file_path}") # Uncomment for debugging
            except OSError as e:
                print(f"Error removing temp file {file_path}: {e}")
        self.temp_files.clear() # Clear the list
        event.accept() # Proceed with closing

    def validate_section_min(self):
        text = self.section_min_input.text().strip()
        try:
            val = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a numeric Min Time.")
            self.section_min_input.clear()
            return

        times = self.df['TIME']
        overall_min = times.min()
        overall_max = times.max()
        if val < overall_min or val > overall_max:
            QMessageBox.warning(
                self,
                "Out of Range",
                f"Min Time must be between {overall_min} and {overall_max}."
            )
            self.section_min_input.clear()
            return

        if hasattr(self, 'section_max_value') and val >= self.section_max_value:
            QMessageBox.warning(self, "Invalid Range", "Min Time must be less than Max Time.")
            self.section_min_input.clear()
            return

        closest_idx = (np.abs(times - val)).idxmin()
        closest = times.iloc[closest_idx]
        self.section_min_value = float(closest)
        self.section_min_input.setText(f"{closest}")
        self.update_plots_tab3()

    def validate_section_max(self):
        text = self.section_max_input.text().strip()
        try:
            val = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a numeric Max Time.")
            self.section_max_input.clear()
            return

        times = self.df['TIME']
        overall_min = times.min()
        overall_max = times.max()
        if val < overall_min or val > overall_max:
            QMessageBox.warning(
                self,
                "Out of Range",
                f"Max Time must be between {overall_min} and {overall_max}."
            )
            self.section_max_input.clear()
            return

        if hasattr(self, 'section_min_value') and val <= self.section_min_value:
            QMessageBox.warning(self, "Invalid Range", "Max Time must be greater than Min Time.")
            self.section_max_input.clear()
            return

        closest_idx = (np.abs(times - val)).idxmin()
        closest = times.iloc[closest_idx]
        self.section_max_value = float(closest)
        self.section_max_input.setText(f"{closest}")
        self.update_plots_tab3()

    # endregion

#######################################
# endregion

# region Run the main script
if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    main = WE_load_plotter(raw_data_folder=selected_folder)
    main.show()
    sys.exit(app.exec_())
# endregion

#TODO - The last time point of each partition should be the first time point of the following partition. Its data value should be zero.
