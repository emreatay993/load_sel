# File: app/ui/tab_interface_data.py

import re
from natsort import natsorted
from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSplitter, QComboBox, QLabel, QSizePolicy
from ..plotting.plotter import load_fig_to_webview

class InterfaceDataTab(QtWidgets.QWidget):
    plot_parameters_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None  # Holding a reference to the main frame
        self._setup_ui()

    def set_dataframe(self, df):
        """MainWindow provides the dataframe to this tab."""
        self.df = df

    def _setup_ui(self):
        # Widgets
        self.interface_selector = QComboBox()
        self.interface_selector.setEditable(True)
        self.side_selector = QComboBox()
        self.side_selector.setEditable(True)
        self.t_series_plot = QtWebEngineWidgets.QWebEngineView()
        self.r_series_plot = QtWebEngineWidgets.QWebEngineView()

        splitter = QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.t_series_plot)
        splitter.addWidget(self.r_series_plot)

        # Layouts
        side_layout = QHBoxLayout()
        side_selection_label = QLabel("Part Side Filter")
        side_selection_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        side_layout.addWidget(side_selection_label)
        side_layout.addWidget(self.side_selector)

        main_layout = QVBoxLayout(self)
        interface_selection_label = QLabel("Interface")
        main_layout.addWidget(interface_selection_label)
        main_layout.addWidget(self.interface_selector)
        main_layout.addLayout(side_layout)
        main_layout.addWidget(splitter)

        # Connections
        self.interface_selector.currentIndexChanged.connect(self._on_interface_changed)
        self.side_selector.currentIndexChanged.connect(self.plot_parameters_changed)

    # Private slots for internal logic
    def _on_interface_changed(self):
        self._populate_side_selector()
        self.plot_parameters_changed.emit()

    # Helper methods to call
    def _populate_side_selector(self):
        if self.df is None:
            return

        current_interface = self.interface_selector.currentText()
        if not current_interface:
            self.side_selector.clear()
            return

        pattern = re.compile(r'I\d+[a-zA-Z]?\s*-\s*(.*?)(?=\s*\()')
        relevant_cols = [col for col in self.df.columns if col.startswith(current_interface)]
        sides = sorted(set(pattern.search(col).group(1).strip() for col in relevant_cols if pattern.search(col)))

        self.side_selector.blockSignals(True)
        self.side_selector.clear()
        self.side_selector.addItems(sides)
        self.side_selector.blockSignals(False)

    def display_t_series_plot(self, fig):
        load_fig_to_webview(fig, self.t_series_plot)

    def display_r_series_plot(self, fig):
        load_fig_to_webview(fig, self.r_series_plot)