import sys
import pandas as pd
import re
from natsort import natsorted
from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
                             QSplitter, QComboBox, QLabel, QSizePolicy)
import plotly.graph_objects as go
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox


def select_directory(title):
    app = QApplication(sys.argv)
    folder = QFileDialog.getExistingDirectory(None, title)
    if not folder:
        QMessageBox.critical(None, 'Error', f"No folder selected for {title.lower()}! Exiting.")
        sys.exit()
    return folder

def get_file_path(folder, file_suffix):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(file_suffix)]

def read_max_pld_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data_rows = [line.strip().split('|')[1:3] for line in lines[2:] if line.strip()]
    df_intf = pd.DataFrame(data_rows, columns=['Interface Labels', 'Units'])
    return df_intf.T

def insert_phase_columns(df):
    new_df = pd.DataFrame()
    for col in df.columns:
        new_df[col] = df[col]
        new_col_label = 'Phase_' + df[col].iloc[0].strip()
        new_df[new_col_label] = [new_col_label, 'deg']
    return new_df

def read_pld_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        headers = [h.strip() for h in lines[0].strip().split('|')[1:-1]]
        processed_data = []
        for line in lines[2:]:  # Skip header rows
            line = line.strip()
            if not line.startswith('|'):
                line = '|' + line  # Prepend '|' if missing
            if not line.endswith('|'):
                line = line + '|'  # Append '|' if missing
            # Now process the line assuming it is correctly formatted
            data_cells = [float(re.sub('[^0-9.E-]', '', cell.strip())) for cell in line.split('|')[1:-1]]
            processed_data.append(data_cells)
    return pd.DataFrame(processed_data, columns=headers)

def main():
    folder_selected_raw_data = select_directory('Please select a directory for raw data')
    folder_selected_headers_data = select_directory('Please select a directory for data headers')

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
    df.columns = new_columns[:len(df.columns)]  # Adjust columns if mismatched
    df.columns = [col.strip() for col in new_columns[:len(df.columns)]]

    print(df)

    return df

if __name__ == "__main__":
    data=main()

data.to_csv("full_data.csv", index=False)


class PlotlyGraphs(QWidget):
    def __init__(self, parent=None):
        super(PlotlyGraphs, self).__init__(parent)
        self.legend_visible = True  # Keep track of legend visibility state
        self.df = pd.read_csv('full_data.csv')
        self.initUI()

    def initUI(self):
        tab_widget = QTabWidget(self)
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()  # New tab for side filter data
        main_layout = QVBoxLayout(self)

        # Setup individual tabs
        self.setupTab1(tab1)
        self.setupTab2(tab2)
        self.setupTab3(tab3)  # Setup for the new tab

        # Add tabs to the tab widget
        tab_widget.addTab(tab1, "Single Data")
        tab_widget.addTab(tab2, "Interface Data")
        tab_widget.addTab(tab3, "Part Loads")  # Add the third tab to the widget

        # Set up the main layout
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
        self.column_selector.setEditable(True)
        regular_columns = [col for col in self.df.columns if 'Phase_' not in col and col != 'FREQ' and col !='No']
        self.column_selector.addItems(regular_columns)
        self.column_selector.currentIndexChanged.connect(self.update_plots)

        layout = QVBoxLayout(tab)
        layout.addWidget(self.column_selector)
        layout.addWidget(splitter)

    def setupTab2(self, tab):
        layout = QVBoxLayout(tab)
        self.setupInterfaceSelector(layout)
        self.setupHoverMode(layout)
        self.setupSideSelection(layout)
        self.setupPlots(layout)

    def setupTab3(self, tab):
        layout = QVBoxLayout(tab)
        self.setupSideFilterSelector(layout)
        self.setupSideFilterPlots(layout)

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

    def setupHoverMode(self, layout):
        hover_mode_layout = QHBoxLayout()
        hover_mode_label = QLabel("Hover Mode")
        hover_mode_label.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))
        self.hover_mode_selector = QComboBox()
        self.hover_mode_selector.addItems(['x unified', 'closest', 'x'])
        self.hover_mode_selector.currentIndexChanged.connect(self.update_hover_mode)
        hover_mode_layout.addWidget(hover_mode_label)
        hover_mode_layout.addWidget(self.hover_mode_selector)
        layout.addLayout(hover_mode_layout)

    def setupSideSelection(self, layout):
        side_selection_layout = QHBoxLayout()
        side_selection_label = QLabel("Part Side Filter")
        side_selection_label.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))
        self.side_selector = QComboBox()
        self.side_selector.setEditable(True)
        self.side_selector.currentIndexChanged.connect(self.update_side_selection)
        side_selection_layout.addWidget(side_selection_label)
        side_selection_layout.addWidget(self.side_selector)
        layout.addLayout(side_selection_layout)

    def setupSideFilterSelector(self, layout):
        self.side_filter_selector = QComboBox()
        self.side_filter_selector.setEditable(True)
        self.populate_side_filter_selector()  # Populate the combobox with unique sides
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
        # Check if the 'L' key was pressed
        if event.key() == QtCore.Qt.Key_L:
            # Toggle the legend visibility
            self.legend_visible = not self.legend_visible
            self.update_plots()
            self.update_plots_tab2()
            self.update_plots_tab3()  # Assuming you want to toggle legends across all tabs

    def update_side_selection(self):
        selected_side = self.side_selector.currentText()
        self.update_plots_for_selected_side(selected_side)

    def update_plots_for_selected_side(self, selected_side):
        # This method assumes that selected_side is valid and there is a need to update plots based on this selection.
        if not selected_side:  # Ensure there is a side selected to filter by.
            return

        interface = self.interface_selector.currentText()
        if not interface:  # Ensure there is an interface selected to work with.
            return

        # Compile a regex that matches the exact interface at the start of the column names
        pattern = re.compile(r'^' + re.escape(interface) + r'([-\s]|$)')

        # Further refine the selection by checking the selected side if it is not empty
        side_pattern = re.compile(re.escape(selected_side))

        # Filter columns for T and R series based on the selected interface and side
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

        # Update plots for both T and R series
        self.update_plot(self.t_series_plot, t_series_columns, f'T Series Filtered by {selected_side}')
        self.update_plot(self.r_series_plot, r_series_columns, f'R Series Filtered by {selected_side}')

    def populate_side_selector(self, interface):
        # Pattern to capture text between the second "-" and the first "("
        #pattern = re.compile(r'(?<=I\d+-\s)(.*?)(?=\s*\()')
        # Pattern to capture text between the first "-" and the first "("
        pattern = re.compile(r'(?<=I\d+[a-zA-Z]?-\s*)(.*?)(?=\s*\()')

        # Filter columns to those relevant to the selected interface and search for side descriptions
        relevant_columns = [col for col in self.df.columns if col.startswith(interface)]
        sides = sorted(set(m.group(1).strip() for col in relevant_columns if (m := pattern.search(col))))

        # Clear the ComboBox and add new items if any sides are found
        self.side_selector.clear()
        if sides:
            self.side_selector.addItems(sides)

    def update_plots_tab2(self):
        interface = self.interface_selector.currentText()
        selected_side = self.side_selector.currentText()
        if interface:
            # Compile a regex that matches the exact interface at the start of the column names
            pattern = re.compile(r'^' + re.escape(interface) + r'([-\s]|$)')

            # Further refine the selection by checking the selected side if it is not empty
            if selected_side:
                side_pattern = re.compile(re.escape(selected_side))

            # Filter columns for T and R series based on the selected interface and optionally by selected side
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

            # Update T series plot
            self.update_plot(self.t_series_plot, t_series_columns, 'T Series')

            # Update R series plot
            self.update_plot(self.r_series_plot, r_series_columns, 'R Series')

            # Populate the side selector if interface changes
            self.populate_side_selector(interface)

    def update_plots_tab3(self):
        selected_side = self.side_filter_selector.currentText()
        if not selected_side:
            return

        # Define regex to match the selected side in column names
        side_pattern = re.compile(re.escape(selected_side))

        # Filter columns for T and R series based on the selected side
        t_series_columns = [col for col in self.df.columns if
                            side_pattern.search(col) and
                            any(sub in col for sub in ["T1", "T2", "T3", "T2/T3"]) and not col.startswith('Phase_')]
        r_series_columns = [col for col in self.df.columns if
                            side_pattern.search(col) and
                            any(sub in col for sub in ["R1", "R2", "R3", "R2/R3"]) and not col.startswith('Phase_')]

        # Update plots with the filtered columns
        self.update_plot(self.t_series_plot_tab3, t_series_columns, 'Filtered T Series')
        self.update_plot(self.r_series_plot_tab3, r_series_columns, 'Filtered R Series')

    def update_hover_mode(self):
        hover_mode = self.hover_mode_selector.currentText()
        self.update_plots()
        self.update_plots_tab2()
        self.update_plots_tab3()

    def update_plot(self, web_view, columns, title):
        x_data = self.df['FREQ']
        fig = go.Figure()
        hover_mode = self.hover_mode_selector.currentText()

        for col in columns:
            fig.add_trace(go.Scatter(x=x_data, y=self.df[col], mode='lines', name=col))

        # Define the custom hover template
        custom_hover = ('%{fullData.name}<br>Hz: %{x:.3f}<br>Value: %{y:.3f}<extra></extra>')
        fig.update_traces(hovertemplate=custom_hover, meta=columns)

        # Define a default font for the whole figure
        default_font = dict(family='Open Sans', size=10, color='black')

        # Set default font and hover label for the layout
        fig.update_layout(
            title=title,
            margin=dict(l=20, r=20, t=35, b=35),
            legend=dict(
                font=default_font,
                orientation="h",
                x=1, y=0,  # Coordinates for bottom right positioning
                xanchor='auto', yanchor='auto',
                bgcolor='rgba(255, 255, 255, 0.5)'
            ),
            hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=8),
            hovermode=hover_mode,
            font=default_font,
            showlegend=self.legend_visible
        )

        # Convert figure to HTML
        html_content = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
        web_view.setHtml(html_content)

    def update_plots(self):
        selected_column = self.column_selector.currentText()
        if selected_column:
            x_data = self.df['FREQ']

            # Define the custom hover template
            custom_hover = (selected_column + '<br>Hz: %{x}<br>Value: %{y:.3f}<extra></extra>')

            # Define a default font for the whole figure
            default_font = dict(family='Open Sans', size=10, color='black')

            # Create the regular data plot with custom hover
            fig_reg = go.Figure(go.Scatter(x=x_data, y=self.df[selected_column], mode='lines', name=selected_column,
                                           hovertemplate=custom_hover))
            fig_reg.update_layout(margin=dict(l=20, r=20, t=35, b=35), legend=dict(font=default_font),
                                  hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=8),
                                  hovermode='x unified',
                                  font=default_font,
                                  showlegend=self.legend_visible)  # Set default font for the layout

            # Create the phase data plot with custom hover
            phase_column = 'Phase_' + selected_column
            fig_phase = go.Figure(go.Scatter(x=x_data, y=self.df[phase_column], mode='lines', name=phase_column,
                                             hovertemplate=custom_hover))
            fig_phase.update_layout(margin=dict(l=20, r=20, t=35, b=35), legend=dict(font=default_font),
                                    hoverlabel=dict(bgcolor='rgba(255, 255, 255, 0.8)', font_size=8),
                                    hovermode='x unified',
                                    font=default_font,
                                    showlegend=self.legend_visible)  # Set default font for the layout

            # Convert figures to HTML
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
