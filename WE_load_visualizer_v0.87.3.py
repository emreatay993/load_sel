import sys
import csv
import pandas as pd
import re
from time import sleep
from collections import OrderedDict
from natsort import natsorted
from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
                             QSplitter, QComboBox, QLabel, QSizePolicy, QPushButton, QCheckBox)
import plotly.graph_objects as go
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox


def select_directory(title):
    folder = QFileDialog.getExistingDirectory(None, title)
    if not folder:
        QMessageBox.critical(None, 'Error', f"No folder selected for {title.lower()}! Exiting.")
        return None
    return folder


def get_file_path(folder, file_suffix):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(file_suffix)]


def read_max_pld_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            cleaned_row = [cell.strip() for cell in row if cell.strip()]
            if cleaned_row and not cleaned_row[0].startswith('_'):
                if len(cleaned_row) >= 2:
                    data.append([cleaned_row[0], cleaned_row[1]])
    df = pd.DataFrame(data[1:])
    return df.T


def insert_phase_columns(df):
    transformed_columns = []
    for i in range(len(df.columns)):
        col_index = i * 2
        phase_index = col_index + 1
        original_col = df.iloc[:, i]
        phase_label = f"Phase_{df.iloc[0, i]}"
        phase_col = [phase_label] + ['deg'] * (len(df) - 1)
        transformed_columns.append(pd.DataFrame({col_index: original_col}))
        transformed_columns.append(pd.DataFrame({phase_index: phase_col}))
    new_df = pd.concat(transformed_columns, axis=1)
    return new_df


def read_pld_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        headers = [h.strip() for h in lines[1].strip().split('|')[1:-1]]
        processed_data = []
        for line in lines[2:]:
            line = line.strip()
            if not line.startswith('|'):
                line = '|' + line
            if not line.endswith('|'):
                line = line + '|'
            data_cells = []
            for cell in line.split('|')[1:-1]:
                cell_cleaned = re.sub('[^0-9.Ee-]', '', cell.strip())
                try:
                    data_cells.append(float(cell_cleaned))
                except ValueError:
                    break
            else:
                processed_data.append(data_cells)
    return pd.DataFrame(processed_data, columns=headers)


def main():
    try:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        folder_selected_raw_data = select_directory('Please select a directory for raw data')
        folder_selected_headers_data = select_directory('Please select a directory for data headers')

        if not folder_selected_raw_data or not folder_selected_headers_data:
            sys.exit()

        file_path_full_data = get_file_path(folder_selected_raw_data, 'full.pld')
        file_path_headers_data = get_file_path(folder_selected_headers_data, 'max.pld')

        if not file_path_full_data or not file_path_headers_data:
            QMessageBox.critical(None, 'Error', "No required files found! Exiting.")
            sys.exit()

        dfs = [read_pld_file(file_path) for file_path in file_path_full_data]
        df = pd.concat(dfs, ignore_index=True)

        df_intf_before = read_max_pld_file(file_path_headers_data[0])
        if df.columns[1] == 'FREQ':
            df_intf = insert_phase_columns(df_intf_before)
            df_intf_labels = pd.DataFrame(df_intf.iloc[0]).T
            new_columns = ['NO'] + ['FREQ'] + df_intf_labels.iloc[0].tolist()
        elif df.columns[1] == 'TIME':
            df_intf = df_intf_before
            df_intf_labels = pd.DataFrame(df_intf.iloc[0]).T
            new_columns = ['NO'] + ['TIME'] + df_intf_labels.iloc[0].tolist()

        additional_columns_needed = len(df.columns) - len(new_columns)
        if additional_columns_needed > 0:
            extended_new_columns = new_columns + [f"Extra_Column_{i}" for i in range(1, additional_columns_needed + 1)]
            df.columns = extended_new_columns
        else:
            df.columns = new_columns[:len(df.columns)]

        df.columns = new_columns[:len(df.columns)]

        print(df)

        return df
    except Exception as e:
        QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")
        sys.exit()


if __name__ == "__main__":
    data = main()
    data.to_csv("full_data.csv", index=False)


class WE_load_plotter(QWidget):
    def __init__(self, parent=None):
        super(WE_load_plotter, self).__init__(parent)
        self.legend_visible = True
        self.legend_positions = ['default', 'top left', 'top right', 'bottom right', 'bottom left']
        self.current_legend_position = 0
        self.df = pd.read_csv('full_data.csv')
        self.df_compare = None
        self.result_df_full_part_load = None
        self.side_filter_selector_for_compare = QComboBox()
        self.current_plot_data = {}
        self.app_ansys = None
        self.mechanical = None

        self.default_font_size = 12
        self.legend_font_size = 10
        self.hover_font_size = 15
        self.hover_mode = 'closest'

        self.initUI()

    def initUI(self):
        tab_widget = QTabWidget(self)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        compare_tab = QWidget()
        compare_part_loads_tab = QWidget()
        settings_tab = QWidget()
        main_layout = QVBoxLayout(self)

        self.setupTab1(self.tab1)
        self.setupTab2(self.tab2)
        self.setupTab3(self.tab3)
        if self.df.columns[1] == 'FREQ':
            self.setupTab4(self.tab4)
        self.setupCompareTab(compare_tab)
        self.setupComparePartLoadsTab(compare_part_loads_tab)
        self.setupSettingsTab(settings_tab)
        if self.df.columns[1] == 'FREQ':
            self.time_domain_plot.show()

        tab_widget.addTab(self.tab1, "Single Data")
        tab_widget.addTab(self.tab2, "Interface Data")
        tab_widget.addTab(self.tab3, "Part Loads")
        if self.df.columns[1] == 'FREQ':
            tab_widget.addTab(self.tab4, "Time Domain Representation")
        tab_widget.addTab(compare_tab, "Compare Data")
        tab_widget.addTab(compare_part_loads_tab, "Compare Data (Part Loads)")
        tab_widget.addTab(settings_tab, "Settings")

        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)
        self.setWindowTitle("WE Load Visualizer - v0.87")
        self.showMaximized()

    def setupTab1(self, tab):
        splitter = QSplitter(QtCore.Qt.Vertical)
        self.regular_plot = QtWebEngineWidgets.QWebEngineView()
        splitter.addWidget(self.regular_plot)
        if self.df.columns[1] == 'FREQ':
            self.phase_plot = QtWebEngineWidgets.QWebEngineView()
            splitter.addWidget(self.phase_plot)
            splitter.setSizes([self.height() // 2, self.height() // 2])

        self.column_selector = QComboBox()
        self.column_selector.setEditable(False)
        regular_columns = [col for col in self.df.columns if
                           'Phase_' not in col and col != 'FREQ' and col != 'TIME' and col != 'NO']
        self.column_selector.addItems(regular_columns)
        self.column_selector.currentIndexChanged.connect(self.update_plots)

        layout = QVBoxLayout(tab)
        layout.addWidget(self.column_selector)
        layout.addWidget(splitter)

    def setupTab2(self, tab):
        layout = QVBoxLayout(tab)
        self.setupInterfaceSelector(layout)
        self.setupSideSelection(layout)
        self.setupPlots(layout)

    def setupTab3(self, tab):
        layout = QVBoxLayout(tab)
        upper_layout = QHBoxLayout()

        self.setupSideFilterSelector(upper_layout)

        self.exclude_checkbox = QCheckBox(" Filter out T2, T3, R2, and R3 from graphs")
        self.exclude_checkbox.stateChanged.connect(self.update_plots_tab3)
        upper_layout.addWidget(self.exclude_checkbox)

        layout.addLayout(upper_layout)
        self.setupSideFilterPlots(layout)

        self.data_point_selector_tab3 = QComboBox()
        self.data_point_selector_tab3.setEditable(True)
        if self.df.columns[1] == 'FREQ':
            self.data_point_selector_tab3.addItem("Select a frequency [Hz] to extract the raw data")
            self.data_point_selector_tab3.addItems([str(freq_point) for freq_point in sorted(self.df['FREQ'].unique())])
        if self.df.columns[1] == 'TIME':
            self.data_point_selector_tab3.addItem("Select a time point [sec] to extract the raw data")
            self.data_point_selector_tab3.addItems([str(time_point) for time_point in sorted(self.df['TIME'].unique())])
        data_point_selector_tab3_layout = QHBoxLayout()
        data_point_selector_tab3_layout.addWidget(self.data_point_selector_tab3)

        if self.df.columns[1] == 'FREQ':
            self.extract_data_button = QPushButton("Extract Data for Selected Frequency")
            self.extract_all_data_button = QPushButton("Extract Load Input for Selected Part")
            self.extract_data_button.clicked.connect(self.extract_data_point)
            data_point_selector_tab3_layout.addWidget(self.extract_data_button)
            self.extract_all_data_button.clicked.connect(self.extract_all_data_points)
            data_point_selector_tab3_layout.addWidget(self.extract_all_data_button)
        if self.df.columns[1] == 'TIME':
            self.extract_data_button = QPushButton("Extract Data for Selected Time")

        layout.addLayout(data_point_selector_tab3_layout)

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
        self.extract_button.clicked.connect(self.extract_time_data)
        interval_selector_layout.addWidget(self.extract_button)

        layout.addLayout(interval_selector_layout)
        tab.setLayout(layout)

    def setupSettingsTab(self, tab):
        layout = QVBoxLayout(tab)

        legend_font_size_layout = QHBoxLayout()
        legend_font_size_label = QLabel("Legend Font Size")
        self.legend_font_size_selector = QComboBox()
        self.legend_font_size_selector.addItems([str(size) for size in range(4, 21)])
        self.legend_font_size_selector.setCurrentText(str(self.legend_font_size))
        self.legend_font_size_selector.currentIndexChanged.connect(self.update_font_settings)
        legend_font_size_layout.addWidget(legend_font_size_label)
        legend_font_size_layout.addWidget(self.legend_font_size_selector)

        default_font_size_layout = QHBoxLayout()
        default_font_size_label = QLabel("Default Font Size")
        self.default_font_size_selector = QComboBox()
        self.default_font_size_selector.addItems([str(size) for size in range(8, 21)])
        self.default_font_size_selector.setCurrentText(str(self.default_font_size))
        self.default_font_size_selector.currentIndexChanged.connect(self.update_font_settings)
        default_font_size_layout.addWidget(default_font_size_label)
        default_font_size_layout.addWidget(self.default_font_size_selector)

        hover_font_size_layout = QHBoxLayout()
        hover_font_size_label = QLabel("Hover Font Size")
        self.hover_font_size_selector = QComboBox()
        self.hover_font_size_selector.addItems([str(size) for size in range(4, 21)])
        self.hover_font_size_selector.setCurrentText(str(self.hover_font_size))
        self.hover_font_size_selector.currentIndexChanged.connect(self.update_font_settings)
        hover_font_size_layout.addWidget(hover_font_size_label)
        hover_font_size_layout.addWidget(self.hover_font_size_selector)

        hover_mode_layout = QHBoxLayout()
        hover_mode_label = QLabel("Hover Mode")
        hover_mode_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.hover_mode_selector = QComboBox()
        self.hover_mode_selector.addItems(['closest', 'x', 'y', 'x unified', 'y unified'])
        self.hover_mode_selector.setCurrentText(self.hover_mode)
        self.hover_mode_selector.currentIndexChanged.connect(self.update_font_settings)
        hover_mode_layout.addWidget(hover_mode_label)
        hover_mode_layout.addWidget(self.hover_mode_selector)

        layout.addLayout(legend_font_size_layout)
        layout.addLayout(default_font_size_layout)
        layout.addLayout(hover_font_size_layout)
        layout.addLayout(hover_mode_layout)
        tab.setLayout(layout)

        contact_label = QLabel("Please reach K. Emre Atay (Compressor Module: Non-Stationary Parts Team) for bug reports / feature requests.")
        layout.addWidget(contact_label, alignment=QtCore.Qt.AlignBottom)

    def setupCompareTab(self, tab):
        splitter_main = QSplitter(QtCore.Qt.Vertical)
        splitter_upper = QSplitter(QtCore.Qt.Vertical)
        splitter_lower = QSplitter(QtCore.Qt.Vertical)

        self.compare_regular_plot = QtWebEngineWidgets.QWebEngineView()
        # self.compare_phase_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_absolute_diff_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_relative_diff_plot = QtWebEngineWidgets.QWebEngineView()

        splitter_upper.addWidget(self.compare_regular_plot)
        # splitter_upper.addWidget(self.compare_phase_plot)
        splitter_upper.setSizes([self.height() // 4, self.height() // 4])

        splitter_lower.addWidget(self.compare_absolute_diff_plot)
        splitter_lower.addWidget(self.compare_relative_diff_plot)
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

    def setupSideFilterSelector(self, layout):
        side_filter_layout = QHBoxLayout()
        self.side_filter_selector = QComboBox()
        self.side_filter_selector.setEditable(True)
        self.populate_side_filter_selector()
        self.side_filter_selector.currentIndexChanged.connect(self.update_plots_tab3)

        layout.addLayout(side_filter_layout)

    def setupSideFilterSelectorForCompare(self, layout):
        self.side_filter_selector_for_compare = QComboBox()
        self.side_filter_selector_for_compare.setEditable(True)
        self.populate_side_filter_selector()
        self.side_filter_selector_for_compare.currentIndexChanged.connect(self.update_compare_part_loads_plots)
        layout.addWidget(self.side_filter_selector_for_compare)

    def setupComparePartLoadsPlots(self, layout):
        splitter = QSplitter(QtCore.Qt.Vertical)
        self.compare_t_series_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_r_series_plot = QtWebEngineWidgets.QWebEngineView()
        splitter.addWidget(self.compare_t_series_plot)
        splitter.addWidget(self.compare_r_series_plot)
        splitter.setSizes([self.height() // 2, self.height() // 2])
        layout.addWidget(splitter)

    def select_compare_data(self):
        try:
            folder_selected_raw_data = select_directory('Please select a directory for raw data (Comparison)')
            folder_selected_headers_data = select_directory('Please select a directory for data headers (Comparison)')

            if not folder_selected_raw_data or not folder_selected_headers_data:
                return

            file_path_full_data = get_file_path(folder_selected_raw_data, 'full.pld')
            file_path_headers_data = get_file_path(folder_selected_headers_data, 'max.pld')

            if not file_path_full_data or not file_path_headers_data:
                QMessageBox.critical(None, 'Error', "No required files found! Exiting.")
                return

            dfs = [read_pld_file(file_path) for file_path in file_path_full_data]
            df_compare = pd.concat(dfs, ignore_index=True)

            df_intf_before = read_max_pld_file(file_path_headers_data[0])
            if self.df.columns[1] == 'FREQ':
                df_intf = insert_phase_columns(df_intf_before)
                df_intf_labels = pd.DataFrame(df_intf.iloc[0]).T
                new_columns = ['NO'] + ['FREQ'] + df_intf_labels.iloc[0].tolist()
            elif self.df.columns[1] == 'TIME':
                df_intf = df_intf_before
                df_intf_labels = pd.DataFrame(df_intf.iloc[0]).T
                new_columns = ['NO'] + ['TIME'] + df_intf_labels.iloc[0].tolist()

            additional_columns_needed = len(df_compare.columns) - len(new_columns)
            if (additional_columns_needed > 0):
                extended_new_columns = new_columns + [f"Extra_Column_{i}" for i in
                                                      range(1, additional_columns_needed + 1)]
                df_compare.columns = extended_new_columns
            else:
                df_compare.columns = new_columns[:len(df_compare.columns)]

            df_compare.columns = new_columns[:len(df_compare.columns)]

            # Ensure unique column labels
            df_compare = df_compare.loc[:, ~df_compare.columns.duplicated()]

            # Align columns with the main dataframe
            self.df_compare = df_compare.reindex(columns=self.df.columns, fill_value=np.nan)

            self.df_compare = df_compare
            self.compare_column_selector.clear()
            compare_columns = [col for col in self.df_compare.columns if
                               'Phase_' not in col and col != 'FREQ' and col != 'TIME' and col != 'NO']
            self.compare_column_selector.addItems(compare_columns)
            self.compare_column_selector.currentIndexChanged.connect(self.update_compare_plots)
        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")

    def update_font_settings(self):
        self.legend_font_size = int(self.legend_font_size_selector.currentText())
        self.default_font_size = int(self.default_font_size_selector.currentText())
        self.hover_font_size = int(self.hover_font_size_selector.currentText())
        self.hover_mode = self.hover_mode_selector.currentText()
        self.update_plots()
        self.update_plots_tab2()
        self.update_plots_tab3()
        if self.df.columns[1] == 'FREQ':
            self.update_time_domain_plot()
        self.update_compare_plots()
        self.update_compare_part_loads_plots()

    def update_time_domain_plot(self):
        try:
            freq = float(self.data_point_selector.currentText())
        except ValueError:
            return

        theta = np.linspace(0, 360, 361)
        x_data = np.radians(theta)

        fig = go.Figure()

        selected_side = self.side_filter_selector.currentText()
        side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')
        displayed_columns = [col for col in self.df.columns if
                             side_pattern.search(col) and
                             any(sub in col for sub in ["T1", "T2", "T3", "T2/T3", "R1", "R2", "R3", "R2/R3"]) and
                             not col.startswith('Phase_')]

        for col in displayed_columns:
            amplitude_col = col
            phase_col = 'Phase_' + col
            amplitude = self.df.loc[self.df['FREQ'] == freq, amplitude_col].values[0]
            phase = self.df.loc[self.df['FREQ'] == freq, phase_col].values[0]
            y_data = amplitude * np.cos(x_data - np.radians(phase))
            fig.add_trace(go.Scatter(x=theta, y=y_data, mode='lines', name=col,
                                     hoverinfo='name+x+y', hovertemplate='%{fullData.name}: %{y:.2f}<extra></extra>'))
            self.current_plot_data[col] = {'theta': theta, 'y_data': y_data}

        plot_title = f'Time Domain Representation at {str(freq)} Hz - {selected_side}'
        fig.update_layout(
            title=plot_title,
            xaxis_title='Theta (degrees)',
            yaxis_title='Amplitude',
            hovermode=self.hover_mode,
            margin=dict(l=20, r=20, t=35, b=35),
            legend=dict(
                font=dict(family='Open Sans', size=self.legend_font_size, color='black')
            ),
            hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=self.hover_font_size),
            font=dict(family='Open Sans', size=self.default_font_size, color='black')
        )
        html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
        self.time_domain_plot.setHtml(html_content)

    def extract_time_data(self):
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
            if col != 'Theta':
                nmm_data[col] = nmm_data[col].astype(float) * 1000
        nmm_data.to_csv("extracted_time_data_values_in_Nmm_units.csv", index=False)

    def extract_data_point(self):
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

    def extract_all_data_points(self):
        selected_side = self.side_filter_selector.currentText()
        if not selected_side:
            QMessageBox.information(self, "Selection Required", "Please select a valid side.")
            return

        try:
            side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')
            columns = ['FREQ']
            columns += [col for col in self.df.columns
                        if side_pattern.search(col)
                        and col != 'FREQ'
                        and "T2/T3" not in col
                        and "R2/R3" not in col]

            self.result_df_full_part_load = self.df[columns]

            original_file_path = f"extracted_data_for_{selected_side}_all_frequencies.csv"
            self.result_df_full_part_load.to_csv(original_file_path, index=False)

            for col in self.result_df_full_part_load.columns:
                if not col.startswith('Phase_') and col != 'FREQ':
                    self.result_df_full_part_load[col] = self.result_df_full_part_load[col] * 1000

            converted_file_path = f"extracted_data_for_{selected_side}_all_frequencies_multiplied_in_Nmm_units.csv"
            self.result_df_full_part_load.to_csv(converted_file_path, index=False)

            QMessageBox.information(self, "Extraction Complete",
                                    f"Data has been extracted and converted. \n\nOriginal data is saved to: \n\n{original_file_path}. \n\nConverted data is saved to: \n\n{converted_file_path}.")

            self.create_ansys_mechanical_input_template()

        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")

    def create_ansys_mechanical_input_template(self):
        # region Prepare interface data to be used as input loads for the selected part
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
        for col in self.result_df_full_part_load.columns:
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
        interface_dicts_full = {interface: create_full_interface_dict(self.result_df_full_part_load, cols) for
                                interface, cols in
                                full_interfaces.items()}

        # Create a separate list for FREQ
        list_of_all_frequencies = self.result_df_full_part_load["FREQ"].tolist()

        # region Create an ANSYS Mechanical template
        from ansys.mechanical.core import launch_mechanical
        from ansys.mechanical.core import global_variables

        from ansys.mechanical.core import App
        app = App()
        globals().update(global_variables(app))



        # Convert list of frequencies to a list of frequencies as quantities (in Hz)
        list_of_all_frequencies_as_quantity = []
        for frequency_value in list_of_all_frequencies:
            list_of_all_frequencies_as_quantity.append(Quantity(frequency_value, "Hz"))

        # Harmonic analysis setup
        analysis_HR = Model.AddHarmonicResponseAnalysis()
        analysis_settings_HR = analysis_HR.AnalysisSettings
        analysis_settings_HR.PropertyByName("HarmonicForcingFrequencyMax").InternalValue = self.result_df_full_part_load['FREQ'].max()
        analysis_settings_HR.PropertyByName("HarmonicForcingFrequencyIntervals").InternalValue = 1
        analysis_settings_HR.PropertyByName("HarmonicSolutionMethod").InternalValue = 1

        # List of interfaces for the selected part
        list_of_part_interface_names = list(interface_dicts_full.keys())

        # Check whether input lists are all zero for a set of lists
        def are_all_zeroes(*lists):
            return all(all(x == 0 for x in lst) for lst in lists)

        for interface_name in list_of_part_interface_names:

            # Add a reference coordinate system for each interface
            CS_interface = Model.CoordinateSystems.AddCoordinateSystem()
            CS_interface.Name = "CS_" + interface_name

            # Create remote points for each interface
            RP_interface = Model.AddRemotePoint()
            RP_interface.Name = "RP_" + interface_name
            RP_interface.CoordinateSystem = CS_interface

            # Create remote forces at each interface
            remote_force = analysis_HR.AddRemoteForce()
            remote_force.DefineBy = Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
            remote_force.Name = "RF_" + interface_name
            remote_force.PropertyByName("GeometryDefineBy").InternalValue = 2  # Scoped to remote point
            remote_force.Location = RP_interface

            # Create moments at each interface
            moment = analysis_HR.AddMoment()
            moment.DefineBy = Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
            moment.Name = "RM_" + interface_name
            moment.PropertyByName("GeometryDefineBy").InternalValue = 2  # Scoped to remote point
            moment.Location = RP_interface

            # region Define loads and their phase angles

            # Initialize lists of data to be used as tabular input
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

            # Create lists of quantities (for T1, T2, T3)
            for fx, fy, fz, angle_fx, angle_fy, angle_fz  in zip(interface_dicts_full[interface_name]["T1"],
                                                                 interface_dicts_full[interface_name]["T2"],
                                                                 interface_dicts_full[interface_name]["T3"],
                                                                 interface_dicts_full[interface_name]["Phase_T1"],
                                                                 interface_dicts_full[interface_name]["Phase_T2"],
                                                                 interface_dicts_full[interface_name]["Phase_T3"]):
                list_of_fx_values.append(Quantity(fx, "N"))
                list_of_fy_values.append(Quantity(fy, "N"))
                list_of_fz_values.append(Quantity(fz, "N"))
                list_of_angle_fx_values.append(Quantity(angle_fx, "deg"))
                list_of_angle_fy_values.append(Quantity(angle_fy, "deg"))
                list_of_angle_fz_values.append(Quantity(angle_fz, "deg"))

            # Create lists of quantities (for R1, R2, R3)
            for mx, my, mz, angle_mx, angle_my, angle_mz  in zip(interface_dicts_full[interface_name]["R1"],
                                                                 interface_dicts_full[interface_name]["R2"],
                                                                 interface_dicts_full[interface_name]["R3"],
                                                                 interface_dicts_full[interface_name]["Phase_R1"],
                                                                 interface_dicts_full[interface_name]["Phase_R2"],
                                                                 interface_dicts_full[interface_name]["Phase_R3"]):
                list_of_mx_values.append(Quantity(mx, "N mm"))
                list_of_my_values.append(Quantity(my, "N mm"))
                list_of_mz_values.append(Quantity(mz, "N mm"))
                list_of_angle_mx_values.append(Quantity(angle_mx, "deg"))
                list_of_angle_my_values.append(Quantity(angle_my, "deg"))
                list_of_angle_mz_values.append(Quantity(angle_mz, "deg"))

            # Define remote force frequencies
            remote_force.XComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            remote_force.YComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            remote_force.ZComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            remote_force.XPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            remote_force.YPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            remote_force.ZPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity

            # Define remote force forces and angles
            remote_force.XComponent.Output.DiscreteValues = list_of_fx_values
            remote_force.YComponent.Output.DiscreteValues = list_of_fy_values
            remote_force.ZComponent.Output.DiscreteValues = list_of_fz_values
            remote_force.XPhaseAngle.Output.DiscreteValues = list_of_angle_fx_values
            remote_force.YPhaseAngle.Output.DiscreteValues = list_of_angle_fy_values
            remote_force.ZPhaseAngle.Output.DiscreteValues = list_of_angle_fz_values

            # Define moment frequencies
            moment.XComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment.YComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment.ZComponent.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment.XPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment.YPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity
            moment.ZPhaseAngle.Inputs[0].DiscreteValues = list_of_all_frequencies_as_quantity

            # Define moments and angles
            moment.XComponent.Output.DiscreteValues = list_of_mx_values
            moment.YComponent.Output.DiscreteValues = list_of_my_values
            moment.ZComponent.Output.DiscreteValues = list_of_mz_values
            moment.XPhaseAngle.Output.DiscreteValues = list_of_angle_mx_values
            moment.YPhaseAngle.Output.DiscreteValues = list_of_angle_my_values
            moment.ZPhaseAngle.Output.DiscreteValues = list_of_angle_mz_values

            if are_all_zeroes(interface_dicts_full[interface_name]["R1"],
                              interface_dicts_full[interface_name]["R2"],
                              interface_dicts_full[interface_name]["R3"]):
                moment.Delete()    #    Delete moment object if no R1, R2, R3 components are all zero, making moment undefined

        app.save(os.path.join(os.getcwd(), "WE_Loading_Template.mechdat"))
        print('Mechdat file is created successfully.')
            # endregion

                # endregion

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

    def setupSideSelection(self, layout):
        side_selection_layout = QHBoxLayout()
        side_selection_label = QLabel("Part Side Filter")
        side_selection_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.side_selector = QComboBox()
        self.side_selector.setEditable(True)
        self.side_selector.currentIndexChanged.connect(self.update_side_selection)
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

    def populate_side_filter_selector(self):
        pattern = re.compile(r'(?<=\s-)(.*?)(?=\s*\()')
        sides = set()
        for col in self.df.columns:
            match = pattern.search(col)
            if match:
                sides.add(match.group(1).strip())
        self.side_filter_selector.addItems(sorted(sides))
        self.side_filter_selector_for_compare.addItems(sorted(sides))

    def setupSideFilterPlots(self, layout):
        splitter = QSplitter(QtCore.Qt.Vertical)
        self.t_series_plot_tab3 = QtWebEngineWidgets.QWebEngineView()
        self.r_series_plot_tab3 = QtWebEngineWidgets.QWebEngineView()
        splitter.addWidget(self.t_series_plot_tab3)
        splitter.addWidget(self.r_series_plot_tab3)
        splitter.setSizes([self.height() // 2, self.height() // 2])
        layout.addWidget(splitter)

    def setupPlots(self, layout):
        splitter = QSplitter(QtCore.Qt.Vertical)
        self.t_series_plot = QtWebEngineWidgets.QWebEngineView()
        self.r_series_plot = QtWebEngineWidgets.QWebEngineView()
        splitter.addWidget(self.t_series_plot)
        splitter.addWidget(self.r_series_plot)
        splitter.setSizes([self.height() // 2, self.height() // 2])
        layout.addWidget(splitter)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_L:
            self.legend_visible = not self.legend_visible
            self.update_plots()
            self.update_plots_tab2()
            self.update_plots_tab3()
            if self.df.columns[1] == 'FREQ':
                self.update_time_domain_plot()
            self.update_compare_plots()
            self.update_compare_part_loads_plots()
        elif event.key() == QtCore.Qt.Key_K:
            self.current_legend_position = (self.current_legend_position + 1) % len(self.legend_positions)
            self.update_plots()
            self.update_plots_tab2()
            self.update_plots_tab3()
            if self.df.columns[1] == 'FREQ':
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

    def update_side_selection(self):
        selected_side = self.side_selector.currentText()
        self.update_plots_for_selected_side(selected_side)
        if self.df.columns[1] == 'FREQ':
            self.update_time_domain_plot()

    def update_plots_for_selected_side(self, selected_side):
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

        self.update_plot(self.t_series_plot, t_series_columns, f'T Plot')
        self.update_plot(self.r_series_plot, r_series_columns, f'R Plot')

    def populate_side_selector(self, interface):
        pattern = re.compile(r'I\d+[a-zA-Z]?\s*-\s*(.*?)(?=\s*\()')
        relevant_columns = [col for col in self.df.columns if re.match(f"^{re.escape(interface)}(?=\D)", col)]
        sides = sorted(set(pattern.search(col).group(1).strip() for col in relevant_columns if pattern.search(col)))

        self.side_selector.clear()
        if sides:
            self.side_selector.addItems(sides)

    def update_plots_tab2(self):
        interface = self.interface_selector.currentText()
        selected_side = self.side_selector.currentText()
        if interface:
            pattern = re.compile(r'^' + re.escape(interface) + r'([-\s]|$)')
            if selected_side:
                side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')

            t_series_columns = [
                col for col in self.df.columns
                if pattern.match(col) and any(sub in col for sub in ["T1", "T2", "T3", "T2/T3"]) and not col.startswith(
                    'Phase_')
                   and (not selected_side or side_pattern.search(col))
            ]
            r_series_columns = [
                col for col in self.df.columns
                if pattern.match(col) and any(sub in col for sub in ["R1", "R2", "R3", "R2/R3"]) and not col.startswith(
                    'Phase_')
                   and (not selected_side or side_pattern.search(col))
            ]

            self.update_plot(self.t_series_plot, t_series_columns, 'T Series')
            self.update_plot(self.r_series_plot, r_series_columns, 'R Series')
            self.populate_side_selector(interface)

    def update_plots_tab3(self):
        selected_side = self.side_filter_selector.currentText()
        if not selected_side:
            return

        exclude_t2_t3_r2_r3 = self.exclude_checkbox.isChecked()
        side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')

        def should_exclude(col):
            if re.search(r'\bT2\b', col) and not re.search(r'T2/T3', col):
                return True
            if re.search(r'\bT3\b', col) and not re.search(r'T2/T3', col):
                return True
            if re.search(r'\bR2\b', col) and not re.search(r'R2/R3', col):
                return True
            if re.search(r'\bR3\b', col) and not re.search(r'R2/R3', col):
                return True
            return False

        t_series_columns = [col for col in self.df.columns if
                            side_pattern.search(col) and
                            any(sub in col for sub in ["T1", "T2", "T3"]) and not col.startswith('Phase_')
                            and not (exclude_t2_t3_r2_r3 and should_exclude(col))]
        r_series_columns = [col for col in self.df.columns if
                            side_pattern.search(col) and
                            any(sub in col for sub in ["R1", "R2", "R3"]) and not col.startswith('Phase_')
                            and not (exclude_t2_t3_r2_r3 and should_exclude(col))]

        self.update_plot(self.t_series_plot_tab3, t_series_columns, 'T Plot')
        self.update_plot(self.r_series_plot_tab3, r_series_columns, 'R Plot')
        if self.df.columns[1] == 'FREQ':
            self.update_time_domain_plot()

    def update_compare_part_loads_plots(self):
        try:
            selected_side = self.side_filter_selector_for_compare.currentText()
            if not selected_side or self.df_compare is None:
                return

            exclude_t2_t3_r2_r3 = self.exclude_checkbox_compare.isChecked()
            side_pattern = re.compile(rf'\b{re.escape(selected_side)}\b')

            def should_exclude(col):
                if re.search(r'\bT2\b', col) and not re.search(r'T2/T3', col):
                    return True
                if re.search(r'\bT3\b', col) and not re.search(r'T2/T3', col):
                    return True
                if re.search(r'\bR2\b', col) and not re.search(r'R2/R3', col):
                    return True
                if re.search(r'\bR3\b', col) and not re.search(r'R2/R3', col):
                    return True
                return False

            t_series_columns = [col for col in self.df.columns if
                                side_pattern.search(col) and
                                any(sub in col for sub in ["T1", "T2", "T3"]) and not col.startswith('Phase_')
                                and not (exclude_t2_t3_r2_r3 and should_exclude(col))]
            r_series_columns = [col for col in self.df.columns if
                                side_pattern.search(col) and
                                any(sub in col for sub in ["R1", "R2", "R3"]) and not col.startswith('Phase_')
                                and not (exclude_t2_t3_r2_r3 and should_exclude(col))]

            if not t_series_columns and not r_series_columns:
                QMessageBox.warning(self, "Warning", "No matching columns found for the selected part side.")
                return

            fig_numerical_diff_t = go.Figure()
            fig_numerical_diff_r = go.Figure()

            if self.df.columns[1] == 'FREQ':
                x_data = self.df['FREQ']
                x_data_compare = self.df_compare['FREQ']
            elif self.df.columns[1] == 'TIME':
                x_data = self.df['TIME']
                x_data_compare = self.df_compare['TIME']

            for col in t_series_columns:
                phase_col = f'Phase_{col}'
                if col in self.df.columns and col in self.df_compare.columns and phase_col in self.df.columns and phase_col in self.df_compare.columns:
                    magnitude1 = self.df[col]
                    phase1 = self.df[phase_col]
                    magnitude2 = self.df_compare[col]
                    phase2 = self.df_compare[phase_col]

                    complex1 = magnitude1 * np.exp(1j * phase1)
                    complex2 = magnitude2 * np.exp(1j * phase2)

                    complex_diff = complex1 - complex2

                    magnitude_diff = np.abs(complex_diff)

                    fig_numerical_diff_t.add_trace(
                        go.Scatter(x=x_data, y=magnitude_diff, mode='lines', name=f'Δ {col}',
                                   hovertemplate='%{fullData.name}<br>Hz: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>'))

            for col in r_series_columns:
                phase_col = f'Phase_{col}'
                if col in self.df.columns and col in self.df_compare.columns and phase_col in self.df.columns and phase_col in self.df_compare.columns:
                    magnitude1 = self.df[col]
                    phase1 = self.df[phase_col]
                    magnitude2 = self.df_compare[col]
                    phase2 = self.df_compare[phase_col]

                    complex1 = magnitude1 * np.exp(1j * phase1)
                    complex2 = magnitude2 * np.exp(1j * phase2)

                    complex_diff = complex1 - complex2

                    magnitude_diff = np.abs(complex_diff)

                    fig_numerical_diff_r.add_trace(
                        go.Scatter(x=x_data, y=magnitude_diff, mode='lines', name=f'Δ {col}',
                                   hovertemplate='%{fullData.name}<br>Hz: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>'))

            default_font = dict(family='Open Sans', size=self.default_font_size, color='black')
            legend_position = self.get_legend_position()

            fig_numerical_diff_t.update_layout(
                title=f'T Plot (Δ) - {selected_side}',
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
                font=default_font,
                showlegend=self.legend_visible
            )

            fig_numerical_diff_r.update_layout(
                title=f'R Plot (Δ) - {selected_side}',
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
                font=default_font,
                showlegend=self.legend_visible
            )

            html_t = fig_numerical_diff_t.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            self.compare_t_series_plot.setHtml(html_t)

            html_r = fig_numerical_diff_r.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            self.compare_r_series_plot.setHtml(html_r)
        except KeyError as e:
            QMessageBox.critical(None, 'Error', f"KeyError: {str(e)}")
        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred while updating compare part loads plots: {str(e)}")

    def update_hover_mode(self):
        hover_mode = self.hover_mode_selector.currentText()
        self.update_plots()
        self.update_plots_tab2()
        self.update_plots_tab3()
        self.update_compare_plots()
        self.update_compare_part_loads_plots()

    def update_plot(self, web_view, columns, title):
        if self.df.columns[1] == 'FREQ':
            x_data = self.df['FREQ']
        elif self.df.columns[1] == 'TIME':
            x_data = self.df['TIME']
        fig = go.Figure()

        for col in columns:
            fig.add_trace(go.Scatter(x=x_data, y=self.df[col], mode='lines', name=col))

        custom_hover = ('%{fullData.name}<br>Hz: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>')
        fig.update_traces(hovertemplate=custom_hover, meta=columns)

        legend_position = self.get_legend_position()
        default_font = dict(family='Open Sans', size=self.default_font_size, color='black')

        fig.update_layout(
            title=title,
            margin=dict(l=20, r=20, t=35, b=35),
            legend=dict(
                font=dict(family='Open Sans', size=self.legend_font_size, color='black'),
                orientation="h",
                x=legend_position['x'],
                y=legend_position['y'],
                xanchor=legend_position.get('xanchor', 'auto'),
                yanchor=legend_position.get('yanchor', 'top'),
                bgcolor='rgba(255, 255, 255, 0.5)'
            ),
            hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=self.hover_font_size),
            hovermode=self.hover_mode,
            font=default_font,
            showlegend=self.legend_visible

        )

        html_content = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
        web_view.setHtml(html_content)

    def update_compare_plots(self):
        try:
            selected_column = self.compare_column_selector.currentText()
            if selected_column and self.df_compare is not None:
                if self.df.columns[1] == 'FREQ':
                    x_data = self.df['FREQ']
                    x_data_compare = self.df_compare['FREQ']
                elif self.df.columns[1] == 'TIME':
                    x_data = self.df['TIME']
                    x_data_compare = self.df_compare['TIME']

                custom_hover = (selected_column + '<br>Hz: %{x}<br>Value: %{y:.3f}<extra></extra>')
                default_font = dict(family='Open Sans', size=self.default_font_size, color='black')
                legend_position = self.get_legend_position()

                fig_reg = go.Figure()
                fig_reg.add_trace(
                    go.Scatter(x=x_data, y=self.df[selected_column], mode='lines', name=f'Original {selected_column}',
                               hovertemplate=custom_hover))
                fig_reg.add_trace(go.Scatter(x=x_data_compare, y=self.df_compare[selected_column], mode='lines',
                                             name=f'Compare {selected_column}',
                                             hovertemplate=custom_hover))
                fig_reg.update_layout(margin=dict(l=20, r=20, t=35, b=35),
                                      legend=dict(
                                          font=dict(family='Open Sans', size=self.legend_font_size, color='black'),
                                          # orientation="h",
                                          x=legend_position['x'],
                                          y=legend_position['y'],
                                          xanchor=legend_position.get('xanchor', 'auto'),
                                          yanchor=legend_position.get('yanchor', 'top'),
                                          bgcolor='rgba(255, 255, 255, 0.5)'
                                      ),
                                      hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)',
                                                      font_size=self.hover_font_size),
                                      hovermode=self.hover_mode,
                                      font=default_font,
                                      showlegend=self.legend_visible)

                fig_absolute_diff = go.Figure()

                # Convert magnitude and phase to complex numbers and compute the difference
                phase_col = f'Phase_{selected_column}'
                if selected_column in self.df.columns and selected_column in self.df_compare.columns and phase_col in self.df.columns and phase_col in self.df_compare.columns:
                    magnitude1 = self.df[selected_column]
                    phase1 = self.df[phase_col]
                    magnitude2 = self.df_compare[selected_column]
                    phase2 = self.df_compare[phase_col]

                    complex1 = magnitude1 * np.exp(1j * phase1)
                    complex2 = magnitude2 * np.exp(1j * phase2)

                    complex_diff = complex1 - complex2

                    magnitude_diff = np.abs(complex_diff)

                    fig_absolute_diff.add_trace(
                        go.Scatter(x=x_data, y=magnitude_diff, mode='lines', name=f'Absolute Δ {selected_column}',
                                   hovertemplate=custom_hover))

                fig_absolute_diff.update_layout(margin=dict(l=20, r=20, t=35, b=35),
                                                legend=dict(
                                                    font=dict(family='Open Sans', size=self.legend_font_size,
                                                              color='black'),
                                                    # orientation="h",
                                                    x=legend_position['x'],
                                                    y=legend_position['y'],
                                                    xanchor=legend_position.get('xanchor', 'auto'),
                                                    yanchor=legend_position.get('yanchor', 'top'),
                                                    bgcolor='rgba(255, 255, 255, 0.5)'
                                                ),
                                                hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)',
                                                                font_size=self.hover_font_size),
                                                hovermode=self.hover_mode,
                                                font=default_font,
                                                showlegend=self.legend_visible)

                fig_relative_diff = go.Figure()
                relative_diff = 100 * magnitude_diff / self.df[selected_column]
                fig_relative_diff.add_trace(
                    go.Scatter(x=x_data, y=relative_diff, mode='lines', name=f'Relative Δ {selected_column} (%)',
                               hovertemplate=custom_hover))
                fig_relative_diff.update_layout(margin=dict(l=20, r=20, t=35, b=35),
                                                legend=dict(
                                                    font=dict(family='Open Sans', size=self.legend_font_size,
                                                              color='black'),
                                                    # orientation="h",
                                                    x=legend_position['x'],
                                                    y=legend_position['y'],
                                                    xanchor=legend_position.get('xanchor', 'auto'),
                                                    yanchor=legend_position.get('yanchor', 'top'),
                                                    bgcolor='rgba(255, 255, 255, 0.5)'
                                                ),
                                                hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)',
                                                                font_size=self.hover_font_size),
                                                hovermode=self.hover_mode,
                                                font=default_font,
                                                showlegend=self.legend_visible)

                html_reg = fig_reg.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
                self.compare_regular_plot.setHtml(html_reg)
                html_abs_diff = fig_absolute_diff.to_html(full_html=False, include_plotlyjs='cdn',
                                                          config={'responsive': True})
                self.compare_absolute_diff_plot.setHtml(html_abs_diff)
                html_rel_diff = fig_relative_diff.to_html(full_html=False, include_plotlyjs='cdn',
                                                          config={'responsive': True})
                self.compare_relative_diff_plot.setHtml(html_rel_diff)
        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred while updating compare plots: {str(e)}")

    def update_plots(self):
        selected_column = self.column_selector.currentText()
        if selected_column:
            if self.df.columns[1] == 'FREQ':
                x_data = self.df['FREQ']
            elif self.df.columns[1] == 'TIME':
                x_data = self.df['TIME']
            custom_hover = (selected_column + '<br>Hz: %{x}<br>Value: %{y:.3f}<extra></extra>')
            default_font = dict(family='Open Sans', size=self.default_font_size, color='black')

            fig_reg = go.Figure(go.Scatter(x=x_data, y=self.df[selected_column], mode='lines', name=selected_column,
                                           hovertemplate=custom_hover))
            fig_reg.update_layout(margin=dict(l=20, r=20, t=35, b=35), legend=dict(font=default_font),
                                  hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=self.hover_font_size),
                                  hovermode=self.hover_mode,
                                  font=default_font,
                                  showlegend=self.legend_visible)

            html_reg = fig_reg.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            self.regular_plot.setHtml(html_reg)

            if self.df.columns[1] == 'FREQ':
                phase_column = 'Phase_' + selected_column
                fig_phase = go.Figure(go.Scatter(x=x_data, y=self.df[phase_column], mode='lines', name=phase_column,
                                                 hovertemplate=custom_hover))
                fig_phase.update_layout(margin=dict(l=20, r=20, t=35, b=35), legend=dict(font=default_font),
                                        hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)',
                                                        font_size=self.hover_font_size),
                                        hovermode=self.hover_mode,
                                        font=default_font,
                                        showlegend=self.legend_visible)

                html_phase = fig_phase.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
                self.phase_plot.setHtml(html_phase)


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    main = WE_load_plotter()
    main.show()
    sys.exit(app.exec_())
