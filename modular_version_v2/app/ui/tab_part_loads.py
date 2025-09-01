# File: app/ui/tab_part_loads.py

from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QSplitter, QComboBox,
                             QPushButton, QCheckBox, QDoubleSpinBox, QLineEdit, QLabel)
from ..plotting.plotter import load_fig_to_webview

class PartLoadsTab(QtWidgets.QWidget):
    plot_parameters_changed = QtCore.pyqtSignal()
    export_to_ansys_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        # --- Upper Controls ---
        self.side_filter_selector = QComboBox()
        self.side_filter_selector.setEditable(True)

        self.exclude_checkbox = QCheckBox(r"Filter out T2, T3, R2, and R3 from graphs")

        self.tukey_checkbox = QCheckBox("Apply Tukey Window")
        self.tukey_checkbox.setVisible(False)
        self.tukey_alpha_spin = QDoubleSpinBox()
        self.tukey_alpha_spin.setRange(0.1, 0.5)
        self.tukey_alpha_spin.setSingleStep(0.05)
        self.tukey_alpha_spin.setValue(0.1)
        self.tukey_alpha_spin.setVisible(False)

        self.section_checkbox = QCheckBox("Section Data")
        self.section_checkbox.setVisible(False)
        self.section_min_label = QLabel("Min Time [sec]")
        self.section_min_input = QLineEdit()
        self.section_min_label.setVisible(False)
        self.section_min_input.setVisible(False)

        self.section_max_label = QLabel("Max Time [sec]")
        self.section_max_input = QLineEdit()
        self.section_max_label.setVisible(False)
        self.section_max_input.setVisible(False)

        # --- Plots ---
        self.t_series_plot = QtWebEngineWidgets.QWebEngineView()
        self.r_series_plot = QtWebEngineWidgets.QWebEngineView()
        splitter = QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.t_series_plot)
        splitter.addWidget(self.r_series_plot)

        # --- Lower Controls ---
        self.data_point_selector = QComboBox()
        self.data_point_selector.setEditable(True)
        self.extract_data_button = QPushButton("Extract Data")
        self.extract_all_data_button = QPushButton("Extract Part Loads as FEA Input (ANSYS)")

        # --- Layouts ---
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(self.side_filter_selector)
        upper_layout.addWidget(self.exclude_checkbox)
        upper_layout.addWidget(self.tukey_checkbox)
        upper_layout.addWidget(self.tukey_alpha_spin)
        upper_layout.addWidget(self.section_checkbox)
        upper_layout.addWidget(self.section_min_label)
        upper_layout.addWidget(self.section_min_input)
        upper_layout.addWidget(self.section_max_label)
        upper_layout.addWidget(self.section_max_input)
        upper_layout.addStretch()

        lower_layout = QHBoxLayout()
        lower_layout.addWidget(self.data_point_selector)
        lower_layout.addWidget(self.extract_data_button)
        lower_layout.addWidget(self.extract_all_data_button)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(upper_layout)
        main_layout.addWidget(splitter)
        main_layout.addLayout(lower_layout)

        # Connect signals
        self.side_filter_selector.currentIndexChanged.connect(self.plot_parameters_changed)
        self.exclude_checkbox.stateChanged.connect(self.plot_parameters_changed)
        self.tukey_checkbox.stateChanged.connect(self._on_tukey_toggled)
        self.tukey_alpha_spin.valueChanged.connect(self.plot_parameters_changed)
        # We will connect sectioning later when we add validation
        self.section_checkbox.stateChanged.connect(
            lambda s: [w.setVisible(s == QtCore.Qt.Checked) for w in
                       (self.section_min_label, self.section_min_input,
                        self.section_max_label, self.section_max_input)]
        )
        self.extract_all_data_button.clicked.connect(self.export_to_ansys_requested)

    def set_time_domain_features_visibility(self, visible):
        """Shows or hides widgets that are only relevant for time-domain data."""
        self.tukey_checkbox.setVisible(visible)
        self.section_checkbox.setVisible(visible)

        # If the main features are being hidden, also hide their sub-options
        if not visible:
            self.tukey_alpha_spin.setVisible(False)
            self.section_min_label.setVisible(False)
            self.section_min_input.setVisible(False)
            self.section_max_label.setVisible(False)
            self.section_max_input.setVisible(False)

    def _on_tukey_toggled(self, state):
        self.tukey_alpha_spin.setVisible(state == QtCore.Qt.Checked)
        self.plot_parameters_changed.emit()

    def display_t_series_plot(self, fig):
        load_fig_to_webview(fig, self.t_series_plot)

    def display_r_series_plot(self, fig):
        load_fig_to_webview(fig, self.r_series_plot)