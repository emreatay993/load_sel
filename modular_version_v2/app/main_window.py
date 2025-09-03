# File: app/main_window.py

import os
import re
import pandas as pd
from natsort import natsorted

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QMenuBar, QMenu, QAction, QMessageBox)
from PyQt5.QtGui import QIcon

# Import internal classes
from .ui.directory_tree_dock import DirectoryTreeDock
from .ui.tab_single_data import SingleDataTab
from .ui.tab_interface_data import InterfaceDataTab
from .ui.tab_part_loads import PartLoadsTab
from .ui.tab_time_domain_represent import TimeDomainRepresentTab
from .ui.tab_compare_data import CompareDataTab
from .ui.tab_compare_part_loads import ComparePartLoadsTab
from .ui.tab_settings import SettingsTab
from .plotting.plotter import Plotter
from . import config_manager
from .controllers.plot_controller import PlotController
from .controllers.action_handler import ActionHandler


class MainWindow(QMainWindow):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        
        # --- Core application state ---
        self.df = None
        self.df_compare = None
        self.data_domain = None
        self.raw_data_folder = None
        
        # --- Core components ---
        self.plotter = Plotter()

        # --- UI and Controllers ---
        self._setup_ui()
        self.action_handler = ActionHandler(self, self.data_manager)
        self.plot_controller = PlotController(self)
        self._connect_signals()

    def keyPressEvent(self, event):
        key = event.key()

        # Your custom logic for specific keys
        if key == QtCore.Qt.Key_K:
            self.plotter.cycle_legend_position()
            self.plot_controller.update_all_plots_from_settings()

        elif key == QtCore.Qt.Key_L:
            self.plotter.toggle_legend_visibility()
            self.plot_controller.update_all_plots_from_settings()

        # The "else" is for every other key not recognized
        else:
            # If program does not know what I key does,
            # it will pass it back to the default handler.
            super().keyPressEvent(event)

    def _setup_ui(self):
        self.setWindowTitle("WE MechLoad Viewer")
        self.setMinimumSize(1200, 800)
        icon_path = os.path.join("resources", "icon.ico")
        if os.path.exists(icon_path): self.setWindowIcon(QIcon(icon_path))

        # --- Menu Bar ---
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        file_menu = menu_bar.addMenu("File")
        view_menu = menu_bar.addMenu("View")
        self.open_action = QAction("Open New Data", self)
        file_menu.addAction(self.open_action)

        # --- Dock Widget ---
        self.dock = DirectoryTreeDock(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock)
        view_menu.addAction(self.dock.toggleViewAction())

        # --- Tab Widgets ---
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        self.tab_single_data = SingleDataTab()
        self.tab_interface_data = InterfaceDataTab()
        self.tab_part_loads = PartLoadsTab()
        self.tab_time_domain_represent = TimeDomainRepresentTab()
        self.tab_compare_data = CompareDataTab()
        self.tab_compare_part_loads = ComparePartLoadsTab()
        self.tab_settings = SettingsTab()

        self.tab_widget.addTab(self.tab_single_data, "Single Data")
        self.tab_widget.addTab(self.tab_interface_data, "Interface Data")
        self.tab_widget.addTab(self.tab_part_loads, "Part Loads")
        self.tab_widget.addTab(self.tab_compare_data, "Compare Data")
        self.tab_widget.addTab(self.tab_compare_part_loads, "Compare Data (Part Loads)")
        self.tab_widget.addTab(self.tab_settings, "Settings")

        # --- Apply Styles ---
        self.dock.tree_view.setStyleSheet(config_manager.TREEVIEW_STYLE)
        self.tab_widget.setStyleSheet(config_manager.TABWIDGET_STYLE)

    def _connect_signals(self):
        # --- Data Loading Signals ---
        self.data_manager.dataLoaded.connect(self.on_data_loaded)
        self.data_manager.comparisonDataLoaded.connect(self.on_comparison_data_loaded)
        self.dock.directories_selected.connect(self._on_directories_selected)
        self.open_action.triggered.connect(self.data_manager.load_data_from_directory)

        # --- Plot Update Signals (Connected to PlotController) ---
        self.tab_single_data.plot_parameters_changed.connect(self.plot_controller.update_single_data_plots)
        self.tab_single_data.spectrum_parameters_changed.connect(self.plot_controller.update_spectrum_plot_only)
        self.tab_interface_data.plot_parameters_changed.connect(self.plot_controller.update_interface_data_plots)
        self.tab_part_loads.plot_parameters_changed.connect(self.plot_controller.update_part_loads_plots)
        self.tab_time_domain_represent.plot_parameters_changed.connect(self.plot_controller.update_time_domain_represent_plot)
        self.tab_compare_data.plot_parameters_changed.connect(self.plot_controller.update_compare_data_plots)
        self.tab_compare_part_loads.plot_parameters_changed.connect(self.plot_controller.update_compare_part_loads_plots)
        self.tab_settings.settings_changed.connect(self.plot_controller.update_all_plots_from_settings)
        # The time domain plot also needs to update when part loads side changes
        self.tab_part_loads.plot_parameters_changed.connect(self.plot_controller.update_time_domain_represent_plot)


        # --- Action Signals (Connected to ActionHandler) ---
        self.tab_compare_data.select_compare_data_requested.connect(self.action_handler.handle_compare_data_selection)
        self.tab_part_loads.export_to_ansys_requested.connect(self.action_handler.handle_ansys_export)
        self.tab_time_domain_represent.extract_data_requested.connect(self.action_handler.handle_time_domain_represent_export)

    def _handle_time_domain_tab_visibility(self):
        is_present = self.tab_widget.indexOf(self.tab_time_domain_represent) != -1
        if self.data_domain == 'FREQ' and not is_present:
            self.tab_widget.insertTab(3, self.tab_time_domain_represent, "Time Domain Rep.")
        elif self.data_domain == 'TIME' and is_present:
            self.tab_widget.removeTab(self.tab_widget.indexOf(self.tab_time_domain_represent))

    def _populate_all_selectors(self):
        if self.df is None: return

        # Single Data Tab
        regular_cols = [c for c in self.df.columns if 'Phase_' not in c and c not in ['FREQ', 'TIME', 'NO', 'DataFolder']]
        self.tab_single_data.column_selector.clear()
        self.tab_single_data.column_selector.addItems(regular_cols)

        # Interface Data Tab
        interfaces = natsorted(list(set(re.match(r'I\d+[A-Za-z]?', c.split(' ')[0]).group(0) for c in self.df.columns if re.match(r'I\d+[A-Za-z]?', c.split(' ')[0]))))
        self.tab_interface_data.interface_selector.clear()
        self.tab_interface_data.interface_selector.addItems(interfaces)

        # Part Loads & Compare Part Loads Tabs
        sides = sorted(list(set(m.group(1).strip() for c in self.df.columns if not c.startswith('Phase_') and (m := re.search(r'(?<=\s-)(.*?)(?=\s*\()', c)))))
        self.tab_part_loads.side_filter_selector.clear()
        self.tab_part_loads.side_filter_selector.addItems(sides)
        self.tab_compare_part_loads.side_filter_selector.clear()
        self.tab_compare_part_loads.side_filter_selector.addItems(sides)

        # Time Domain Tab
        if self.data_domain == 'FREQ':
            freq_items = [str(freq) for freq in sorted(self.df['FREQ'].unique())]
            self.tab_time_domain_represent.data_point_selector.clear()
            self.tab_time_domain_represent.data_point_selector.addItem("Select a frequency [Hz] to plot")
            self.tab_time_domain_represent.data_point_selector.addItems(freq_items)

    @QtCore.pyqtSlot(pd.DataFrame, str, str)
    def on_data_loaded(self, data, data_domain, folder_path):
        self.df, self.data_domain, self.raw_data_folder = data, data_domain, folder_path
        self.tab_interface_data.set_dataframe(self.df)

        num_folders = self.df['DataFolder'].nunique()
        if num_folders > 1:
            title = f"WE MechLoad Viewer - ({num_folders} Data Folders Loaded)"
        else:
            parent = os.path.basename(os.path.dirname(self.raw_data_folder))
            selected = os.path.basename(self.raw_data_folder)
            title = f"WE MechLoad Viewer - (Directory: {parent} | Data Folder: {selected})"
        self.setWindowTitle(title)

        self.dock.set_root_path(self.raw_data_folder)

        # Enable/disable tabs based on folder count
        enable_single_folder_tabs = (num_folders == 1)
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_interface_data), enable_single_folder_tabs)
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_part_loads), enable_single_folder_tabs)
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_compare_data), enable_single_folder_tabs)
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_compare_part_loads), enable_single_folder_tabs)
        if self.tab_widget.indexOf(self.tab_time_domain_represent) != -1:
            self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_time_domain_represent), enable_single_folder_tabs)

        self._handle_time_domain_tab_visibility()
        self._populate_all_selectors()

        is_time_domain = self.data_domain == 'TIME'
        self.tab_single_data.set_time_domain_features_visibility(is_time_domain)
        self.tab_part_loads.set_time_domain_features_visibility(is_time_domain)
        self.tab_settings.rolling_min_max_checkbox.setEnabled(is_time_domain)
        if not is_time_domain:
            self.tab_settings.rolling_min_max_checkbox.setChecked(False)

        self.plot_controller.update_all_plots_from_settings()

    @QtCore.pyqtSlot(pd.DataFrame)
    def on_comparison_data_loaded(self, df_compare):
        if self.df is None:
            QMessageBox.warning(self, "Error", "Please load the primary data first.")
            return

        if self.data_domain not in df_compare.columns:
            QMessageBox.critical(self, "Domain Mismatch", f"Comparison data needs a '{self.data_domain}' column.")
            return

        self.df_compare = df_compare
        
        regular_cols = [c for c in self.df.columns if 'Phase_' not in c and c not in ['FREQ', 'TIME', 'NO']]
        self.tab_compare_data.compare_column_selector.clear()
        self.tab_compare_data.compare_column_selector.addItems(regular_cols)

        QMessageBox.information(self, "Success", "Comparison data loaded successfully.")
        
        self.plot_controller.update_compare_data_plots()
        self.plot_controller.update_compare_part_loads_plots()

    @QtCore.pyqtSlot(list)
    def _on_directories_selected(self, folder_paths):
        """Tells the DataManager to load newly selected folders from the dock."""
        if (self.df is not None and self.df['DataFolder'].nunique() == 1 and len(folder_paths) == 1 and
                os.path.normpath(folder_paths[0]) == os.path.normpath(self.raw_data_folder)):
            return

        if not self.data_domain:
            QMessageBox.warning(self, "No Initial Data", "Please open a primary data set first using 'File -> Open'.")
            return

        self.data_manager.load_data_from_paths(folder_paths)
