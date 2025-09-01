# File: app/main_window.py

import os
import sys
import re
import pandas as pd
import numpy as np
from natsort import natsorted
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QWidget, QMainWindow, QTabWidget, QMenuBar, QMenu, QAction, QDialog, QVBoxLayout, QHBoxLayout,
                             QListWidget, QListWidgetItem, QPushButton, QAbstractItemView, QMessageBox, QFileDialog)
import plotly.graph_objects as go
from PyQt5.QtGui import QIcon
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt
from collections import OrderedDict

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
from .analysis.ansys_exporter import AnsysExporter
from . import config_manager

class MainWindow(QMainWindow):
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.df = None
        self.df_compare = None  # Attribute to hold comparison data
        self.data_domain = None
        self.raw_data_folder = None
        self.df_compare = None  # For comparison data
        self.plotter = Plotter()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        self.setWindowTitle("WE MechLoad Viewer")
        self.setMinimumSize(1200, 800)
        icon_path = os.path.join("resources", "icon.ico")
        if os.path.exists(icon_path): self.setWindowIcon(QIcon(icon_path))

        menu_bar = QMenuBar(self);
        self.setMenuBar(menu_bar)
        file_menu = menu_bar.addMenu("File");
        view_menu = menu_bar.addMenu("View")
        self.open_action = QAction("Open New Data", self);
        clear_cache_action = QAction("Clear Plot Cache", self)
        file_menu.addAction(self.open_action);
        file_menu.addSeparator();
        file_menu.addAction(clear_cache_action)

        self.dock = DirectoryTreeDock(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.dock)
        view_menu.addAction(self.dock.toggleViewAction())

        self.tab_widget = QTabWidget();
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

        # Set styles for each UI element
        self.dock.tree_view.setStyleSheet(config_manager.TREEVIEW_STYLE)
        self.tab_widget.setStyleSheet(config_manager.TABWIDGET_STYLE)

    def _connect_signals(self):
        # Data Loading
        self.data_manager.dataLoaded.connect(self.on_data_loaded)
        self.data_manager.comparisonDataLoaded.connect(self.on_comparison_data_loaded)
        self.dock.directories_selected.connect(self._on_directories_selected)
        self.open_action.triggered.connect(self.data_manager.load_data_from_directory)

        # Plots
        self.tab_single_data.plot_parameters_changed.connect(self._update_single_data_plots)
        self.tab_interface_data.plot_parameters_changed.connect(self._update_interface_data_plots)
        self.tab_part_loads.plot_parameters_changed.connect(self._update_part_loads_plots)
        self.tab_time_domain_represent.plot_parameters_changed.connect(self._update_time_domain_represent_plot)
        self.tab_settings.settings_changed.connect(self._update_all_plots_from_settings)
        self.tab_compare_data.plot_parameters_changed.connect(self._update_compare_data_plots)
        self.tab_compare_part_loads.plot_parameters_changed.connect(self._update_compare_part_loads_plots)

        # Actions from UI tabs
        self.tab_part_loads.export_to_ansys_requested.connect(self._handle_ansys_export)
        self.tab_compare_data.select_compare_data_requested.connect(self._handle_compare_data_selection)
        self.tab_part_loads.export_to_ansys_requested.connect(self._handle_ansys_export)
        self.tab_time_domain_represent.extract_data_requested.connect(self._handle_time_domain_represent_export)

    def _handle_time_domain_tab_visibility(self):
        is_present = self.tab_widget.indexOf(self.tab_time_domain_represent) != -1
        if self.data_domain == 'FREQ' and not is_present:
            self.tab_widget.insertTab(3, self.tab_time_domain_represent, "Time Domain Rep.")
        elif self.data_domain == 'TIME' and is_present:
            self.tab_widget.removeTab(self.tab_widget.indexOf(self.tab_time_domain_represent))

    def _handle_time_domain_represent_export(self):
        """
        Handles the request to extract and save the reconstructed time-domain data.
        """
        try:
            interval_text = self.tab_time_domain_represent.interval_selector.currentText()
            if "Select an Interval" in interval_text:
                QMessageBox.warning(self, "Selection Required", "Please select a valid interval.")
                return
            interval = int(interval_text)

            # Check if there is data to export
            if not hasattr(self.tab_time_domain_represent, 'current_plot_data') or \
                           not self.tab_time_domain_represent.current_plot_data:
                QMessageBox.warning(self, "No Data",
                                    "No plot data available to extract. Please select a frequency first.")
                return

            # Generate the theta points based on the selected interval
            num_points = 360 // interval
            theta_points = np.arange(0, 361, interval)

            data_dict = {'Theta': theta_points}

            # Sample the high-resolution plot data at the desired theta points
            for col, plot_data in self.tab_time_domain_represent.current_plot_data.items():
                full_theta = plot_data['theta']  # This is the original 0-360 range
                full_y_data = plot_data['y_data']  # The full sine wave data

                # Use numpy's interpolation to find the y-values at our new theta points
                sampled_y_data = np.interp(theta_points, full_theta, full_y_data)
                data_dict[col] = sampled_y_data

            df_to_export = pd.DataFrame(data_dict)

            # Ask user where to save the file
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Extracted Data", "extracted_time_represent_data.csv",
                                                       "CSV Files (*.csv)")

            if save_path:
                df_to_export.to_csv(save_path, index=False)
                QMessageBox.information(self, "Export Successful", f"Data successfully saved to:\n{save_path}")
                directory = os.path.dirname(save_path)
                os.startfile(directory)

        except (ValueError, KeyError) as e:
            QMessageBox.critical(self, "Error", f"An error occurred during data extraction: {e}")

    def _populate_all_selectors(self):
        """Helper method to clear and fill all the comboboxes with data."""
        if self.df is None: return

        # --- Single Data Tab ---
        regular_cols = [c for c in self.df.columns if 'Phase_' not in c and c not in
                        ['FREQ', 'TIME', 'NO', 'DataFolder']]
        self.tab_single_data.column_selector.clear()
        self.tab_single_data.column_selector.addItems(regular_cols)

        # --- Interface Data Tab ---
        interface_pattern = re.compile(r'I\d{1,6}[A-Za-z]?')
        interfaces = natsorted(
            list(set(m.group(0) for c in self.df.columns if (m := interface_pattern.match(c.split(' ')[0])))))
        self.tab_interface_data.interface_selector.clear()
        self.tab_interface_data.interface_selector.addItems(interfaces)

        # --- Part Loads & Compare Part Loads Tabs ---
        side_pattern = re.compile(r'(?<=\s-)(.*?)(?=\s*\()')
        sides = sorted(list(set(m.group(1).strip() for c in self.df.columns if
                                not c.startswith('Phase_') and (m := side_pattern.search(c)))))

        self.tab_part_loads.side_filter_selector.clear()
        self.tab_part_loads.side_filter_selector.addItems(sides)

        self.tab_compare_part_loads.side_filter_selector.clear()
        self.tab_compare_part_loads.side_filter_selector.addItems(sides)

        # --- Time Domain Tab ---
        if self.data_domain == 'FREQ':
            freq_items = [str(freq) for freq in sorted(self.df['FREQ'].unique())]
            self.tab_time_domain_represent.data_point_selector.clear()
            self.tab_time_domain_represent.data_point_selector.addItem("Select a frequency [Hz] to plot")
            self.tab_time_domain_represent.data_point_selector.addItems(freq_items)

    def _update_all_plots_from_settings(self):
        if self.df is None: return

        # Update plotter settings from the UI
        settings_tab = self.tab_settings
        self.plotter.legend_font_size = int(settings_tab.legend_font_size_selector.currentText())
        self.plotter.default_font_size = int(settings_tab.default_font_size_selector.currentText())
        self.plotter.hover_font_size = int(settings_tab.hover_font_size_selector.currentText())
        self.plotter.hover_mode = settings_tab.hover_mode_selector.currentText()

        # Trigger a refresh of all visible plots
        self._update_single_data_plots()
        self._update_interface_data_plots()
        self._update_part_loads_plots()
        self._update_time_domain_represent_plot()
        self._update_compare_data_plots()
        self._update_compare_part_loads_plots()

    def _get_sides_for_export(self):
        """Creates and shows a dialog to select multiple sides for export."""
        all_sides = [self.tab_part_loads.side_filter_selector.itemText(i) for i in
                     range(self.tab_part_loads.side_filter_selector.count())]
        current_side = self.tab_part_loads.side_filter_selector.currentText()

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Parts to Export")
        layout = QVBoxLayout(dialog)
        list_widget = QListWidget()
        list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        for side in all_sides:
            item = QListWidgetItem(side)
            list_widget.addItem(item)
            if side == current_side:
                item.setSelected(True)

        button_layout = QHBoxLayout()
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(cancel_button)

        layout.addWidget(list_widget)
        layout.addLayout(button_layout)

        if dialog.exec_() == QDialog.Accepted:
            return [item.text() for item in list_widget.selectedItems()]
        return None

    def _calculate_differences(self, columns):
        """
        Calculates the absolute difference between self.df and self.df_compare
        for a list of given columns. Handles both TIME and FREQ domains.

        Returns a DataFrame of the differences, with columns named 'Δ {col}'.
        """
        diff_dict = {}
        for col in columns:
            if col not in self.df.columns or col not in self.df_compare.columns:
                continue  # Skip if column is missing in one dataset

            mag1 = self.df[col]
            mag2 = self.df_compare[col]

            if self.data_domain == 'FREQ':
                phase_col = f'Phase_{col}'
                if phase_col in self.df.columns and phase_col in self.df_compare.columns:
                    p1_rad = np.deg2rad(self.df[phase_col])
                    p2_rad = np.deg2rad(self.df_compare[phase_col])
                    # Calculate difference of complex numbers
                    diff = np.abs((mag1 * np.exp(1j * p1_rad)) - (mag2 * np.exp(1j * p2_rad)))
                else:
                    diff = np.abs(mag1 - mag2)  # Fallback if phase data is missing
            else:  # Time domain
                diff = np.abs(mag1 - mag2)

            diff_dict[f'Δ {col}'] = diff

        return pd.DataFrame(diff_dict)

    def _get_plot_df(self, cols, source_df=None):
        if source_df is None:
            source_df = self.df

        x_label = 'Time [s]' if self.data_domain == 'TIME' else 'Freq [Hz]'
        x_data = self.df[self.data_domain]
        df = self.df[cols].copy();
        df.index = x_data;
        df.index.name = x_label
        return df

    @QtCore.pyqtSlot(pd.DataFrame, str, str)
    def on_data_loaded(self, data, data_domain, folder_path):
        self.df, self.data_domain, self.raw_data_folder = data, data_domain, folder_path
        self.tab_interface_data.set_dataframe(self.df)

        ## Update title based on number of folders
        num_folders = self.df['DataFolder'].nunique()
        if num_folders > 1:
            title_text = f"WE MechLoad Viewer - v0.98.0    |    ({num_folders} Data Folders Loaded)"
        else:
            parent_folder = os.path.basename(os.path.dirname(self.raw_data_folder))
            selected_folder = os.path.basename(self.raw_data_folder)
            title_text = f"WE MechLoad Viewer - v0.98.0 | (Directory: {parent_folder} | Data Folder: {selected_folder})"
        self.setWindowTitle(title_text)

        self.dock.set_root_path(self.raw_data_folder)

        ## ADDED: Manage tab visibility based on folder count
        enable_advanced_tabs = (num_folders == 1)
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_interface_data), enable_advanced_tabs)
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_part_loads), enable_advanced_tabs)
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_compare_data), enable_advanced_tabs)
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_compare_part_loads), enable_advanced_tabs)
        if self.tab_widget.indexOf(self.tab_time_domain_represent) != -1:
            self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.tab_time_domain_represent), enable_advanced_tabs)

        self._handle_time_domain_tab_visibility()
        self._populate_all_selectors()

        # Update UI visibility based on the new data domain
        is_time_domain = self.data_domain == 'TIME'
        self.tab_single_data.set_time_domain_features_visibility(is_time_domain)
        self.tab_part_loads.set_time_domain_features_visibility(is_time_domain)

        # The rolling min-max checkbox is enabled only if the data is in the TIME domain.
        self.tab_settings.rolling_min_max_checkbox.setEnabled(is_time_domain)

        # NOTE: If the data is FREQ domain, the checkbox is disabled.
        if not is_time_domain:
            self.tab_settings.rolling_min_max_checkbox.setChecked(False)

        self._update_all_plots_from_settings()

    @QtCore.pyqtSlot()
    def _handle_compare_data_selection(self):
        """Handles the request from the CompareDataTab to load comparison data."""
        self.data_manager.load_comparison_data()

    @QtCore.pyqtSlot(pd.DataFrame)
    def on_comparison_data_loaded(self, df_compare):
        """Slot to receive and handle the loaded comparison dataframe from the DataManager."""
        if self.df is None:
            QMessageBox.warning(self, "Error", "Please load the primary data first.")
            return

        # Basic check for domain consistency
        domain_col_exists = self.data_domain in df_compare.columns
        if not domain_col_exists:
            QMessageBox.critical(self, "Domain Mismatch",
                                 f"Comparison data does not have a '{self.data_domain}' column and cannot be compared.")
            return

        self.df_compare = df_compare

        # Populate the selector in the compare tab
        self.tab_compare_data.compare_column_selector.clear()
        regular_cols = [c for c in self.df.columns if 'Phase_' not in c and c not in ['FREQ', 'TIME', 'NO']]
        self.tab_compare_data.compare_column_selector.addItems(regular_cols)

        QMessageBox.information(self, "Success", "Comparison data loaded successfully.")

        # Trigger updates for any tabs that use the comparison data
        self._update_compare_data_plots()
        self._update_compare_part_loads_plots()

    @QtCore.pyqtSlot()
    def _update_single_data_plots(self):
        if self.df is None: return
        tab = self.tab_single_data
        selected_col = tab.column_selector.currentText()
        if not selected_col: return

        num_folders = self.df['DataFolder'].nunique()
        is_multi_folder = num_folders > 1

        # This dictionary will hold the data for each trace.
        dfs_for_plot = {}

        # Group data by folder and prepare for plotting
        for folder_name, group_df in self.df.groupby('DataFolder'):
            plot_df_group = self._get_plot_df([selected_col], source_df=group_df)

            if self.data_domain == 'TIME' and tab.filter_checkbox.isChecked():
                try:
                    fs = 1 / group_df['TIME'].diff().mean()
                    cutoff = float(tab.cutoff_frequency_input.text())
                    order = tab.filter_order_input.value()
                    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
                    plot_df_group[selected_col] = filtfilt(b, a, plot_df_group[selected_col])
                except (ValueError, ZeroDivisionError) as e:
                    print(f"Could not apply filter for {folder_name}: {e}")

            trace_name = folder_name if is_multi_folder else selected_col
            dfs_for_plot[trace_name] = plot_df_group

        # --- REGULAR OR ENVELOPE PLOT ---
        plot_title = f"{selected_col} Plot"
        if self.tab_settings.rolling_min_max_checkbox.isChecked() and self.data_domain == 'TIME':
            try:
                points = int(self.tab_settings.desired_num_points_input.text())
                as_bars = self.tab_settings.plot_as_bars_checkbox.isChecked()
                fig = self.plotter.create_rolling_envelope_figure(dfs_for_plot, plot_title, points, as_bars)
            except ValueError:
                fig = self.plotter.create_standard_figure(dfs_for_plot, title=f"{plot_title} (Invalid Points)")
        else:
            fig = self.plotter.create_standard_figure(dfs_for_plot, title=plot_title)
        tab.display_regular_plot(fig)

        # --- PHASE PLOT (only for single-folder FREQ domain) ---
        if self.data_domain == 'FREQ' and not is_multi_folder:
            phase_col = f'Phase_{selected_col}'
            if phase_col in self.df.columns:
                phase_df = self._get_plot_df([phase_col])
                phase_fig = self.plotter.create_standard_figure({phase_col: phase_df}, f'Phase of {selected_col}',
                                                                'Phase [deg]')
                tab.set_phase_plot_visibility(True)
                tab.display_phase_plot(phase_fig)
            else:
                tab.set_phase_plot_visibility(False)
        else:
            tab.set_phase_plot_visibility(False)

        # --- SPECTRUM PLOT (only for single-folder TIME domain) ---
        if self.data_domain == 'TIME' and tab.spectrum_checkbox.isChecked() and not is_multi_folder:
            try:
                num_slices = int(tab.num_slices_input.text())
                plot_type = tab.plot_type_selector.currentText()
                colorscale = tab.colorscale_selector.currentText()
                spectrum_df = list(dfs_for_plot.values())[0]
                fig_spec = self.plotter.create_spectrum_figure(spectrum_df, num_slices, plot_type, freq_max=None,
                                                               colorscale=colorscale)
                tab.set_spectrum_plot_visibility(True)
                tab.display_spectrum_plot(fig_spec)
            except (ValueError, IndexError) as e:
                print(f"Could not generate spectrum: {e}")
                tab.set_spectrum_plot_visibility(False)
        else:
            tab.set_spectrum_plot_visibility(False)

    @QtCore.pyqtSlot()
    def _update_interface_data_plots(self):
        if self.df is None: return
        interface = self.tab_interface_data.interface_selector.currentText()
        side = self.tab_interface_data.side_selector.currentText()
        if not interface or not side: return

        t_cols = [c for c in self.df.columns if c.startswith(interface) and side in c and any(
            s in c for s in ['T1', 'T2', 'T3', 'T2/T3']) and 'Phase_' not in c]
        r_cols = [c for c in self.df.columns if c.startswith(interface) and side in c and any(
            s in c for s in ['R1', 'R2', 'R3', 'R2/R3']) and 'Phase_' not in c]

        plot_df_t = self._get_plot_df(t_cols)
        plot_df_r = self._get_plot_df(r_cols)

        # Pass data as a dictionary
        self.tab_interface_data.display_t_series_plot(self.plotter.create_standard_figure(plot_df_t, f'Translational - {side}'))
        self.tab_interface_data.display_r_series_plot(self.plotter.create_standard_figure(plot_df_r, f'Rotational - {side}'))


    @QtCore.pyqtSlot()
    def _update_part_loads_plots(self):
        if self.df is None: return
        side = self.tab_part_loads.side_filter_selector.currentText()
        if not side: return

        exclude = self.tab_part_loads.exclude_checkbox.isChecked()

        def filter_cols(cols, components):
            filtered = [c for c in cols if side in c and any(s in c for s in components) and 'Phase_' not in c]
            if exclude:
                return [c for c in filtered if not any(s in c for s in [' T2', ' T3', ' R2', ' R3'])]
            return filtered

        t_cols = filter_cols(self.df.columns, ['T1', 'T2', 'T3', 'T2/T3'])
        r_cols = filter_cols(self.df.columns, ['R1', 'R2', 'R3', 'R2/R3'])

        plot_df_t = self._get_plot_df(t_cols)
        plot_df_r = self._get_plot_df(r_cols)

        # Pass data as a dictionary
        self.tab_part_loads.display_t_series_plot(self.plotter.create_standard_figure(plot_df_t, f'Translational - {side}'))
        self.tab_part_loads.display_r_series_plot(self.plotter.create_standard_figure(plot_df_r, f'Rotational - {side}'))

        self._update_time_domain_represent_plot()

    @QtCore.pyqtSlot()
    def _update_time_domain_represent_plot(self):
        if self.df is None or self.data_domain != 'FREQ':
            return

        try:
            freq_text = self.tab_time_domain_represent.data_point_selector.currentText()
            if not freq_text or "Select a frequency" in freq_text:
                return
            freq = float(freq_text)

            # Get the currently selected side from the Part Loads tab
            selected_side = self.tab_part_loads.side_filter_selector.currentText()
            if not selected_side:
                self.tab_time_domain_represent.display_plot(go.Figure()) # Clear plot if no side is selected
                return

            side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')
            plot_cols = [
                c for c in self.df.columns if
                side_pattern.search(c) and not c.startswith('Phase_')
                and any(s in c for s in ['T1', 'T2', 'T3', 'R1', 'R2', 'R3', 'T2/T3', 'R2/R3'])
            ]

            theta = np.linspace(0, 360, 361)
            rads = np.radians(theta)
            time_domain_represent_data_for_plot = {}
            # Create a separate dictionary for the export data
            plot_data_for_export = {}

            data_at_freq = self.df[self.df['FREQ'] == freq].iloc[0]

            for col in plot_cols:
                phase_col = f'Phase_{col}'
                if phase_col in data_at_freq:
                    amplitude = data_at_freq[col]
                    phase_deg = data_at_freq[phase_col]
                    y_data = amplitude * np.cos(rads - np.radians(phase_deg))
                    time_domain_represent_data_for_plot[col] = y_data
                    # Store both theta and y_data for the exporter
                    plot_data_for_export[col] = {'theta': theta, 'y_data': y_data}

            # Save the complete data structure for the exporter
            self.tab_time_domain_represent.current_plot_data = plot_data_for_export

            # Use the plot-specific data to create the DataFrame
            df_time_domain = pd.DataFrame(time_domain_represent_data_for_plot, index=theta)
            df_time_domain.index.name = "Theta [deg]"

            title = f'Time Domain Representation at {freq} Hz for {selected_side}'
            fig = self.plotter.create_standard_figure(df_time_domain, title)
            self.tab_time_domain_represent.display_plot(fig)

        except (ValueError, IndexError) as e:
            print(f"Could not update time domain representation plot: {e}")
            self.tab_time_domain_represent.display_plot(go.Figure())

    @QtCore.pyqtSlot()
    def _update_compare_part_loads_plots(self):
        if self.df is None or self.df_compare is None: return
        tab = self.tab_compare_part_loads
        selected_side = tab.side_filter_selector.currentText()
        if not selected_side: return

        # Column Filtering
        exclude = tab.exclude_checkbox.isChecked()
        side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')

        def get_cols(components):
            cols = [c for c in self.df.columns if
                    side_pattern.search(c) and any(s in c for s in components) and not c.startswith('Phase_')]
            if exclude:
                return [c for c in cols if not any(s in c for s in [' T2', ' T3', ' R2', ' R3'])]
            return cols

        t_cols = get_cols(["T1", "T2", "T3"]);
        r_cols = get_cols(["R1", "R2", "R3"])

        x_label = 'Time [s]' if self.data_domain == 'TIME' else 'Freq [Hz]'
        x_data = self.df[self.data_domain]

        # Calculate Differences using the new helper
        t_diff_df = self._calculate_differences(t_cols)
        r_diff_df = self._calculate_differences(r_cols)

        # Plotting
        t_diff_df.index = x_data;
        t_diff_df.index.name = x_label
        fig_t = self.plotter.create_standard_figure(t_diff_df, f'Translational Difference (Δ) - {selected_side}')
        tab.display_t_series_plot(fig_t)

        r_diff_df.index = x_data;
        r_diff_df.index.name = x_label
        fig_r = self.plotter.create_standard_figure(r_diff_df, f'Rotational Difference (Δ) - {selected_side}')
        tab.display_r_series_plot(fig_r)

    @QtCore.pyqtSlot()
    def _update_compare_data_plots(self):
        if self.df is None or self.df_compare is None: return
        tab = self.tab_compare_data
        selected_column = tab.compare_column_selector.currentText()
        if not selected_column: return

        # Data Preparation
        x_label = 'Time [s]' if self.data_domain == 'TIME' else 'Freq [Hz]'
        x_data = self.df[self.data_domain]
        df1 = self.df[[selected_column]].copy();
        df1.index = x_data;
        df1.index.name = x_label
        df2 = self.df_compare[[selected_column]].copy();
        df2.index = x_data;
        df2.index.name = x_label

        # 1. Create the main comparison plot
        fig_compare = self.plotter.create_comparison_figure(df1, df2, selected_column, f'{selected_column} Comparison')
        tab.display_comparison_plot(fig_compare)

        # 2. Calculate differences using the new helper
        diff_df = self._calculate_differences([selected_column])
        if diff_df.empty: return  # Stop if calculation failed

        abs_diff_series = diff_df.iloc[:, 0]  # Get the first (and only) column of differences

        # 3. Create the absolute difference plot
        abs_diff_df = pd.DataFrame({'Absolute Difference': abs_diff_series})
        abs_diff_df.index = x_data;
        abs_diff_df.index.name = x_label
        fig_abs_diff = self.plotter.create_standard_figure(abs_diff_df, f'{selected_column} Absolute Difference')
        tab.display_absolute_diff_plot(fig_abs_diff)

        # 4. Create the relative difference plot
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.divide(100 * abs_diff_series, np.abs(self.df[selected_column]))
            relative_diff.fillna(0, inplace=True)

        rel_diff_df = pd.DataFrame({'Relative Difference (%)': relative_diff})
        rel_diff_df.index = x_data;
        rel_diff_df.index.name = x_label
        fig_rel_diff = self.plotter.create_standard_figure(rel_diff_df, f'{selected_column} Relative Difference (%)',
                                                           "Percent (%)")
        tab.display_relative_diff_plot(fig_rel_diff)

    @QtCore.pyqtSlot()
    def _handle_ansys_export(self):
        """Controller slot to manage the Ansys export process."""
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load data before exporting.")
            return

        selected_sides = self._get_sides_for_export()
        if not selected_sides:
            return  # User cancelled

        # --- 1. Prepare the initial combined DataFrame for all selected sides ---
        cols_to_keep = [self.data_domain]
        for side in selected_sides:
            side_pattern = re.compile(rf'\b{re.escape(side)}\b')
            cols_to_keep.extend(
                [c for c in self.df.columns if side_pattern.search(c) and not any(s in c for s in ['T2/T3', 'R2/R3'])])

        # This is now the master DataFrame for this operation
        df_processed = self.df[list(OrderedDict.fromkeys(cols_to_keep))].copy()

        # --- 2. Apply Data Processing ONCE to the combined DataFrame ---
        if self.data_domain == 'TIME':
            part_loads_tab = self.tab_part_loads

            # Apply Data Sectioning if enabled
            if part_loads_tab.section_checkbox.isChecked():
                try:
                    t_min = float(part_loads_tab.section_min_input.text())
                    t_max = float(part_loads_tab.section_max_input.text())
                    if t_min < t_max:
                        df_processed = df_processed[
                            (df_processed['TIME'] >= t_min) & (df_processed['TIME'] <= t_max)
                            ].copy()
                    else:
                        QMessageBox.warning(self, "Invalid Range", "Min Time must be less than Max Time.")
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input",
                                        "Please enter valid numeric values for Min and Max Time.")

            # Apply Tukey Window if enabled
            if part_loads_tab.tukey_checkbox.isChecked():
                if len(df_processed) > 1:
                    alpha = part_loads_tab.tukey_alpha_spin.value()
                    window = tukey(len(df_processed), alpha)
                    load_cols = [c for c in df_processed.columns if c != 'TIME']
                    df_processed.loc[:, load_cols] = df_processed[load_cols].multiply(window, axis=0)
                else:
                    print("Warning: Cannot apply Tukey window to a dataset with one or zero points.")

        # --- 3. Perform CSV Exports FROM THE PROCESSED DATA ---
        df_combined_converted = pd.DataFrame()
        for side in selected_sides:
            # Prepare DataFrame for the individual side from the PROCESSED data
            side_pattern = re.compile(rf'\b{re.escape(side)}\b')
            side_cols_to_keep = [self.data_domain]
            side_cols_to_keep.extend([c for c in df_processed.columns if side_pattern.search(c)])
            df_part_processed = df_processed[list(OrderedDict.fromkeys(side_cols_to_keep))]

            # Save the processed data in original units
            original_file_path = f"extracted_data_for_{side}_in_original_units.csv"
            self.data_manager.save_dataframe_to_csv(df_part_processed, original_file_path)

            # Perform unit conversion
            df_part_converted = df_part_processed.copy()
            for col in df_part_converted.columns:
                if col not in [self.data_domain, 'NO'] and not col.startswith('Phase_'):
                    df_part_converted[col] *= 1000

            # Save the converted units CSV
            converted_file_path = f"extracted_{side}_loads_multiplied_by_1000.csv"
            self.data_manager.save_dataframe_to_csv(df_part_converted, converted_file_path)

            # Combine for the final summary CSV
            if df_combined_converted.empty:
                df_combined_converted = df_part_converted
            else:
                df_to_concat = df_part_converted.drop(columns=[self.data_domain])
                df_combined_converted = pd.concat([df_combined_converted, df_to_concat], axis=1)

        # Save the final combined CSV
        combined_file_path = "extracted_loads_of_all_selected_parts_in_converted_units.csv"
        self.data_manager.save_dataframe_to_csv(df_combined_converted, combined_file_path)

        # --- 4. Call the Ansys Exporter with the processed data (in original units) ---
        exporter = AnsysExporter()
        if self.data_domain == 'FREQ':
            # df_processed is already in freq domain, no changes needed
            exporter.create_harmonic_template(df_processed, self.data_domain)
        elif self.data_domain == 'TIME':
            # Recalculate sample rate based on original data for accuracy
            time_diffs = self.df['TIME'].diff().dropna()
            sample_rate = 1 / time_diffs.mean() if not time_diffs.empty else 0
            exporter.create_transient_template(df_processed, self.data_domain, sample_rate)

    @QtCore.pyqtSlot(list)
    def _on_directories_selected(self, folder_paths):
        """
        When the user selects folders in the dock, this tells the DataManager
        to load them. The current data domain is passed for validation.
        """
        # Logic to prevent reloading the same single folder
        if (self.df is not None and self.df['DataFolder'].nunique() == 1 and
                len(folder_paths) == 1):
            selected_path = folder_paths[0]
            if os.path.normpath(selected_path) == os.path.normpath(self.raw_data_folder):
                return  # Exit early if the selected folder is already loaded

        if not self.data_domain:
            QMessageBox.warning(self, "No Initial Data", "Please open a primary data set first using 'File -> Open'.")
            return

        # This will trigger the dataLoaded signal when done
        self.data_manager.load_data_from_paths(folder_paths)
