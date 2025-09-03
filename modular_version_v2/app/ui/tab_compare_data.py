# File: app/ui/tab_compare_data.py

from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSplitter, QComboBox, QPushButton
from ..plotting.plotter import load_fig_to_webview
from .. import config_manager


class CompareDataTab(QtWidgets.QWidget):
    plot_parameters_changed = QtCore.pyqtSignal()
    # Signal to open the file dialog for comparison data
    select_compare_data_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        # Plots
        self.compare_regular_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_absolute_diff_plot = QtWebEngineWidgets.QWebEngineView()
        self.compare_percent_diff_plot = QtWebEngineWidgets.QWebEngineView()

        splitter_upper = QSplitter(QtCore.Qt.Vertical)
        splitter_upper.addWidget(self.compare_regular_plot)

        splitter_lower = QSplitter(QtCore.Qt.Vertical)
        splitter_lower.addWidget(self.compare_absolute_diff_plot)
        splitter_lower.addWidget(self.compare_percent_diff_plot)

        splitter_main = QSplitter(QtCore.Qt.Vertical)
        splitter_main.addWidget(splitter_upper)
        splitter_main.addWidget(splitter_lower)

        # Controls
        self.compare_column_selector = QComboBox()
        self.compare_column_selector.setEditable(False)
        self.compare_button = QPushButton("Select Data for Comparison")

        # Layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.compare_column_selector)
        main_layout.addWidget(splitter_main)
        main_layout.addWidget(self.compare_button)

        # Styles
        self.compare_button.setStyleSheet(config_manager.COMPARE_BUTTON_STYLE)

        # Connections
        self.compare_button.clicked.connect(self.select_compare_data_requested)
        self.compare_column_selector.currentIndexChanged.connect(self.plot_parameters_changed)

    def display_comparison_plot(self, fig):
        load_fig_to_webview(fig, self.compare_regular_plot)

    def display_absolute_diff_plot(self, fig):
        load_fig_to_webview(fig, self.compare_absolute_diff_plot)

    def display_relative_diff_plot(self, fig):
        load_fig_to_webview(fig, self.compare_percent_diff_plot)