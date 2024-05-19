import sys
import csv
import pandas as pd
import re
from natsort import natsorted
from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
                             QSplitter, QComboBox, QLabel, QSizePolicy, QPushButton)
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
        headers = [h.strip() for h in lines[0].strip().split('|')[1:-1]]
        processed_data = []
        for line in lines[2:]:
            line = line.strip()
            if not line.startswith('|'):
                line = '|' + line
            if not line.endswith('|'):
                line = line + '|'
            data_cells = [float(re.sub('[^0-9.E-]', '', cell.strip())) for cell in line.split('|')[1:-1]]
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
        df_intf = insert_phase_columns(df_intf_before)

        df_intf_labels = pd.DataFrame(df_intf.iloc[0]).T
        new_columns = ['NO'] + ['FREQ'] + df_intf_labels.iloc[0].tolist()

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


class PlotlyGraphs(QWidget):
    def __init__(self, parent=None):
        super(PlotlyGraphs, self).__init__(parent)
        self.legend_visible = True
        self.legend_positions = ['default', 'top left', 'top right', 'bottom right', 'bottom left']
        self.current_legend_position = 0
        self.df = pd.read_csv('full_data.csv')
        self.df_compare = None
        self.side_filter_selector_for_compare = QComboBox()
        self.current_plot_data = {}

        self.default_font_size = 10
        self.legend_font_size = 12
        self.hover_font_size = 8
        self.hover_mode = 'x unified'

        self.initUI()

    def initUI(self):
        tab_widget = QTabWidget(self)
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()
        tab4 = QWidget()
        compare_tab = QWidget()
        compare_part_loads_tab = QWidget()
        settings_tab = QWidget()
        main_layout = QVBoxLayout(self)

        self.setupTab1(tab1)
        self.setupTab2(tab2)
        self.setupTab3(tab3)
        self.setupTab4(tab4)
        self.setupCompareTab(compare_tab)
        self.setupComparePartLoadsTab(compare_part_loads_tab)
        self.setupSettingsTab(settings_tab)
        self.time_domain_plot.show()

        tab_widget.addTab(tab1, "Single Data")
        tab_widget.addTab(tab2, "Interface Data")
        tab_widget.addTab(tab3, "Part Loads")
        tab_widget.addTab(tab4, "Time Domain Representation")
        tab_widget.addTab(compare_tab, "Compare Data")
        tab_widget.addTab(compare_part_loads_tab, "Compare Data (Part Loads)")
        tab_widget.addTab(settings_tab, "Settings")

        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)
        self.setWindowTitle("WE Harmonic Load Plotter")
        self.showMaximized()

    def setupTab1(self, tab):
        splitter = QSplitter(QtCore.Qt.Vertical)
        self.regular_plot = QtWebEngineWidgets.QWebEngineView()
        self.phase_plot = QtWebEngineWidgets.QWebEngineView()
        splitter.addWidget(self.regular_plot)
        splitter.addWidget(self.phase_plot)
        splitter.setSizes([self.height() // 2, self.height() // 2])

        self.column_selector = QComboBox()
        self.column_selector.setEditable(False)
        regular_columns = [col for col in self.df.columns if 'Phase_' not in col and col != 'FREQ' and col != 'NO']
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
        self.setupSideFilterSelector(layout)
        self.setupSideFilterPlots(layout)

        self.frequency_selector_tab3 = QComboBox()
        self.frequency_selector_tab3.setEditable(True)
        self.frequency_selector_tab3.addItem("Select a frequency [Hz] to extract the raw data")
        self.frequency_selector_tab3.addItems([str(freq) for freq in sorted(self.df['FREQ'].unique())])
        frequency_selector_tab3_layout = QHBoxLayout()
        frequency_selector_tab3_layout.addWidget(self.frequency_selector_tab3)

        self.extract_data_button = QPushButton("Extract Data for Selected Frequency")
        self.extract_data_button.clicked.connect(self.extract_frequency_data)
        frequency_selector_tab3_layout.addWidget(self.extract_data_button)

        layout.addLayout(frequency_selector_tab3_layout)

    def setupTab4(self, tab):
        layout = QVBoxLayout(tab)

        self.frequency_selector = QComboBox()
        self.frequency_selector.setEditable(True)
        self.frequency_selector.addItem("Select a frequency [Hz] to plot the time domain data")
        self.frequency_selector.addItems([str(freq) for freq in sorted(self.df['FREQ'].unique())])
        self.frequency_selector.currentIndexChanged.connect(self.update_time_domain_plot)
        frequency_selector_layout = QHBoxLayout()
        frequency_selector_label = QLabel("Select a Frequency")
        frequency_selector_layout.addWidget(frequency_selector_label)
        frequency_selector_layout.addWidget(self.frequency_selector)
        layout.addLayout(frequency_selector_layout)

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
        self.extract_button.clicked.connect(self.extract_values)
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

    def setupCompareTab(self, tab):
        splitter_main = QSplitter(QtCore.Qt.Vertical)
        splitter_upper = QSplitter(QtCore.Qt.Vertical)
        splitter_lower = QSplitter(QtCore.Qt.Vertical)

        self.compare_regular_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_phase_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_absolute_diff_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_relative_diff_plot = QtWebEngineWidgets.QWebEngineView()

        splitter_upper.addWidget(self.compare_regular_plot)
        splitter_upper.addWidget(self.compare_phase_plot)
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
        self.setupSideFilterSelectorForCompare(layout)
        self.setupComparePartLoadsPlots(layout)

    def setupSideFilterSelector(self, layout):
        self.side_filter_selector = QComboBox()
        self.side_filter_selector.setEditable(True)
        self.populate_side_filter_selector()
        self.side_filter_selector.currentIndexChanged.connect(self.update_plots_tab3)
        layout.addWidget(self.side_filter_selector)

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
            df_intf = insert_phase_columns(df_intf_before)

            df_intf_labels = pd.DataFrame(df_intf.iloc[0]).T
            new_columns = ['NO'] + ['FREQ'] + df_intf_labels.iloc[0].tolist()

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
                               'Phase_' not in col and col != 'FREQ' and col != 'NO']
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
        self.update_time_domain_plot()
        self.update_compare_plots()
        self.update_compare_part_loads_plots()

    def update_time_domain_plot(self):
        try:
            freq = float(self.frequency_selector.currentText())
        except ValueError:
            return

        theta = np.linspace(0, 360, 361)
        x_data = np.radians(theta)

        fig = go.Figure()

        selected_side = self.side_filter_selector.currentText()
        side_pattern = re.compile(re.escape(selected_side))
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

    def extract_values(self):
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
            extracted_data.to_csv("extracted_values.csv", index=False)
            self.convert_to_Nmm_units(extracted_data)
            QMessageBox.information(None, "Extraction Complete",
                                    "Data has been extracted and saved to extracted_values.csv and extracted_values_in_Nmm_units.csv.")
        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")

    def convert_to_Nmm_units(self, data):
        nmm_data = data.copy()
        for col in nmm_data.columns:
            if col != 'Theta':
                nmm_data[col] = nmm_data[col].astype(float) * 1000
        nmm_data.to_csv("extracted_values_in_Nmm_units.csv", index=False)

    def extract_frequency_data(self):
        selected_frequency_tab3 = self.frequency_selector_tab3.currentText()
        selected_side = self.side_filter_selector.currentText()

        if selected_frequency_tab3 == "Select a Frequency":
            QMessageBox.information(self, "Selection Required", "Please select a valid frequency.")
            return

        try:
            freq = float(selected_frequency_tab3)
            filtered_df = self.df[self.df['FREQ'] == freq]

            side_pattern = re.compile(re.escape(selected_side))
            columns = ['FREQ']
            columns += [col for col in filtered_df.columns if side_pattern.search(col) and col != 'FREQ']

            result_df = filtered_df[columns]

            original_file_path = f"extracted_data_for_{selected_side}_at_{selected_frequency_tab3}_Hz.csv"
            result_df.to_csv(original_file_path, index=False)

            for col in result_df.columns:
                if not col.startswith('Phase_') and col != 'FREQ':
                    result_df[col] = result_df[col] * 1000

            converted_file_path = f"extracted_data_for_{selected_side}_at_{selected_frequency_tab3}_Hz_multiplied_by_1000.csv"
            result_df.to_csv(converted_file_path, index=False)

            QMessageBox.information(self, "Extraction Complete",
                                    f"Data has been extracted and converted. Original data saved to {original_file_path}. Converted data saved to {converted_file_path}.")
        except Exception as e:
            QMessageBox.critical(None, 'Error', f"An error occurred: {str(e)}")

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
            self.update_time_domain_plot()
            self.update_compare_plots()
            self.update_compare_part_loads_plots()
        elif event.key() == QtCore.Qt.Key_K:
            self.current_legend_position = (self.current_legend_position + 1) % len(self.legend_positions)
            self.update_plots()
            self.update_plots_tab2()
            self.update_plots_tab3()
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
        self.update_time_domain_plot()

    def update_plots_for_selected_side(self, selected_side):
        if not selected_side:
            return

        interface = self.interface_selector.currentText()
        if not interface:
            return

        pattern = re.compile(r'^' + re.escape(interface) + r'([-\s]|$)')
        side_pattern = re.compile(re.escape(selected_side))

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
                side_pattern = re.compile(re.escape(selected_side))

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

        side_pattern = re.compile(re.escape(selected_side))

        t_series_columns = [col for col in self.df.columns if
                            side_pattern.search(col) and
                            any(sub in col for sub in ["T1", "T2", "T3", "T2/T3"]) and not col.startswith('Phase_')]
        r_series_columns = [col for col in self.df.columns if
                            side_pattern.search(col) and
                            any(sub in col for sub in ["R1", "R2", "R3", "R2/R3"]) and not col.startswith('Phase_')]

        self.update_plot(self.t_series_plot_tab3, t_series_columns, 'T Plot')
        self.update_plot(self.r_series_plot_tab3, r_series_columns, 'R Plot')
        self.update_time_domain_plot()

    def update_compare_part_loads_plots(self):
        try:
            selected_side = self.side_filter_selector_for_compare.currentText()
            if not selected_side or self.df_compare is None:
                return

            side_pattern = re.compile(re.escape(selected_side))

            t_series_columns = [col for col in self.df.columns if
                                side_pattern.search(col) and
                                any(sub in col for sub in ["T1", "T2", "T3", "T2/T3"]) and not col.startswith('Phase_')]
            r_series_columns = [col for col in self.df.columns if
                                side_pattern.search(col) and
                                any(sub in col for sub in ["R1", "R2", "R3", "R2/R3"]) and not col.startswith('Phase_')]

            if not t_series_columns and not r_series_columns:
                QMessageBox.warning(self, "Warning", "No matching columns found for the selected part side.")
                return

            fig_absolute_diff_t = go.Figure()
            fig_absolute_diff_r = go.Figure()

            x_data = self.df['FREQ']
            x_data_compare = self.df_compare['FREQ']

            for col in t_series_columns:
                if col in self.df.columns and col in self.df_compare.columns:
                    abs_diff = abs(self.df[col] - self.df_compare[col])
                    fig_absolute_diff_t.add_trace(
                        go.Scatter(x=x_data, y=abs_diff, mode='lines', name=f'Absolute Diff {col}',
                                   hovertemplate='%{fullData.name}<br>Hz: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>'))

            for col in r_series_columns:
                if col in self.df.columns and col in self.df_compare.columns:
                    abs_diff = abs(self.df[col] - self.df_compare[col])
                    fig_absolute_diff_r.add_trace(
                        go.Scatter(x=x_data, y=abs_diff, mode='lines', name=f'Absolute Diff {col}',
                                   hovertemplate='%{fullData.name}<br>Hz: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>'))

            default_font = dict(family='Open Sans', size=self.default_font_size, color='black')
            legend_position = self.get_legend_position()

            fig_absolute_diff_t.update_layout(
                title=f'Absolute Difference T Plot - {selected_side}',
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

            fig_absolute_diff_r.update_layout(
                title=f'Absolute Difference R Plot - {selected_side}',
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

            html_t = fig_absolute_diff_t.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            self.compare_t_series_plot.setHtml(html_t)

            html_r = fig_absolute_diff_r.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
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
        x_data = self.df['FREQ']
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
                x_data = self.df['FREQ']
                x_data_compare = self.df_compare['FREQ']

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
                abs_diff = abs(self.df[selected_column] - self.df_compare[selected_column])
                fig_absolute_diff.add_trace(
                    go.Scatter(x=x_data, y=abs_diff, mode='lines', name=f'Absolute Diff {selected_column}',
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
                relative_diff = 100 * abs_diff / self.df[selected_column]
                fig_relative_diff.add_trace(
                    go.Scatter(x=x_data, y=relative_diff, mode='lines', name=f'Relative Diff {selected_column} (%)',
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
            x_data = self.df['FREQ']
            custom_hover = (selected_column + '<br>Hz: %{x}<br>Value: %{y:.3f}<extra></extra>')
            default_font = dict(family='Open Sans', size=self.default_font_size, color='black')

            fig_reg = go.Figure(go.Scatter(x=x_data, y=self.df[selected_column], mode='lines', name=selected_column,
                                           hovertemplate=custom_hover))
            fig_reg.update_layout(margin=dict(l=20, r=20, t=35, b=35), legend=dict(font=default_font),
                                  hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=self.hover_font_size),
                                  hovermode=self.hover_mode,
                                  font=default_font,
                                  showlegend=self.legend_visible)

            phase_column = 'Phase_' + selected_column
            fig_phase = go.Figure(go.Scatter(x=x_data, y=self.df[phase_column], mode='lines', name=phase_column,
                                             hovertemplate=custom_hover))
            fig_phase.update_layout(margin=dict(l=20, r=20, t=35, b=35), legend=dict(font=default_font),
                                    hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=self.hover_font_size),
                                    hovermode=self.hover_mode,
                                    font=default_font,
                                    showlegend=self.legend_visible)

            html_reg = fig_reg.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            self.regular_plot.setHtml(html_reg)
            html_phase = fig_phase.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
            self.phase_plot.setHtml(html_phase)


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    main = PlotlyGraphs()
    main.show()
    sys.exit(app.exec_())
