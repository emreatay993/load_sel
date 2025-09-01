# File: app/ui/tab_compare_part_loads.py

from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSplitter, QComboBox, QCheckBox
from ..plotting.plotter import load_fig_to_webview


class ComparePartLoadsTab(QtWidgets.QWidget):
    plot_parameters_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        # --- Controls ---
        self.side_filter_selector = QComboBox()
        self.side_filter_selector.setEditable(True)
        self.exclude_checkbox = QCheckBox("Filter out T2, T3, R2, and R3 from graphs")

        # --- Plots ---
        self.t_series_plot = QtWebEngineWidgets.QWebEngineView()
        self.r_series_plot = QtWebEngineWidgets.QWebEngineView()
        splitter = QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.t_series_plot)
        splitter.addWidget(self.r_series_plot)

        # --- Layouts ---
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(self.side_filter_selector)
        upper_layout.addWidget(self.exclude_checkbox)
        upper_layout.addStretch()

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(upper_layout)
        main_layout.addWidget(splitter)

        # --- Connections ---
        self.side_filter_selector.currentIndexChanged.connect(self.plot_parameters_changed)
        self.exclude_checkbox.stateChanged.connect(self.plot_parameters_changed)

    def display_t_series_plot(self, fig):
        load_fig_to_webview(fig, self.t_series_plot)

    def display_r_series_plot(self, fig):
        load_fig_to_webview(fig, self.r_series_plot)