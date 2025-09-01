# File: app/ui/tab_time_domain_represent.py

from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel
from ..plotting.plotter import load_fig_to_webview


class TimeDomainRepresentTab(QtWidgets.QWidget):
    plot_parameters_changed = QtCore.pyqtSignal()
    # Signal for data extraction
    extract_data_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_plot_data = {}
        self._setup_ui()

    def _setup_ui(self):
        # --- Widgets ---
        self.data_point_selector = QComboBox()
        self.data_point_selector.setEditable(True)
        self.data_point_selector.addItem("Select a frequency [Hz] to plot")

        self.time_domain_plot = QtWebEngineWidgets.QWebEngineView()

        self.interval_selector = QComboBox()
        self.interval_selector.setEditable(True)
        self.interval_selector.addItem("Select an Interval")
        for i in range(1, 361):
            if 360 % i == 0:
                self.interval_selector.addItem(str(i))

        self.extract_button = QPushButton("Extract Data at Each Interval as CSV file")

        # --- Layouts ---
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select a Frequency:"))
        selector_layout.addWidget(self.data_point_selector)

        extract_layout = QHBoxLayout()
        extract_layout.addWidget(QLabel("Select an Interval:"))
        extract_layout.addWidget(self.interval_selector)
        extract_layout.addWidget(self.extract_button)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(selector_layout)
        main_layout.addWidget(self.time_domain_plot)
        main_layout.addLayout(extract_layout)

        # --- Connections ---
        self.data_point_selector.currentIndexChanged.connect(self.plot_parameters_changed)
        self.extract_button.clicked.connect(self.extract_data_requested)

    def display_plot(self, fig):
        load_fig_to_webview(fig, self.time_domain_plot)