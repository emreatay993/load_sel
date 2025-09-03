# File: app/ui/tab_single_data.py

from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QSplitter, QComboBox,
                             QLabel, QSizePolicy, QCheckBox, QLineEdit, QSpinBox)
from ..plotting.plotter import load_fig_to_webview
from .. import tooltips


class SingleDataTab(QtWidgets.QWidget):
    plot_parameters_changed = QtCore.pyqtSignal()
    spectrum_parameters_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.splitter_sizes = None
        self._setup_ui()

    def _setup_ui(self):
        self.splitter = QSplitter(QtCore.Qt.Vertical)
        self.regular_plot = QtWebEngineWidgets.QWebEngineView()
        self.phase_plot = QtWebEngineWidgets.QWebEngineView()
        self.spectrum_plot = QtWebEngineWidgets.QWebEngineView()

        self.splitter.addWidget(self.regular_plot)

        self.column_selector = QComboBox()
        self.column_selector.setEditable(False)
        self.column_selector.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.column_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.spectrum_checkbox = QCheckBox("Show Spectrum Plot")
        self.plot_type_selector = QComboBox()
        self.plot_type_selector.addItems(['Heatmap', 'Surface', 'Waterfall', 'Animation', 'Peak', 'Lines'])

        self.colorscale_label = QLabel("Colorscale:")
        self.colorscale_selector = QComboBox()
        self.colorscale_selector.addItems(['Hot', 'Viridis', 'Plasma', 'Jet', 'Greys', 'Cividis', 'Inferno'])

        self.num_slices_label = QLabel("Spectrum Slices:")
        self.num_slices_input = QLineEdit("400")
        self.filter_checkbox = QCheckBox("Apply Low-Pass Filter")
        self.cutoff_frequency_label = QLabel("Cutoff Freq [Hz]:")
        self.cutoff_frequency_input = QLineEdit()
        self.filter_order_label = QLabel("Order:")
        self.filter_order_input = QSpinBox()
        self.filter_order_input.setRange(1, 10)
        self.filter_order_input.setValue(2)

        self.plot_type_selector.setVisible(False)
        self.num_slices_label.setVisible(False)
        self.num_slices_input.setVisible(False)
        self.colorscale_label.setVisible(False)
        self.colorscale_selector.setVisible(False)
        self.cutoff_frequency_label.setVisible(False)
        self.cutoff_frequency_input.setVisible(False)
        self.filter_order_label.setVisible(False)
        self.filter_order_input.setVisible(False)
        self.spectrum_checkbox.setVisible(False)
        self.filter_checkbox.setVisible(False)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(self.column_selector)
        selector_layout.addWidget(self.spectrum_checkbox)
        selector_layout.addWidget(self.plot_type_selector)
        selector_layout.addWidget(self.colorscale_label)
        selector_layout.addWidget(self.colorscale_selector)
        selector_layout.addWidget(self.num_slices_label)
        selector_layout.addWidget(self.num_slices_input)
        selector_layout.addWidget(self.filter_checkbox)
        selector_layout.addWidget(self.cutoff_frequency_label)
        selector_layout.addWidget(self.cutoff_frequency_input)
        selector_layout.addWidget(self.filter_order_label)
        selector_layout.addWidget(self.filter_order_input)
        selector_layout.addStretch(1)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(selector_layout)
        main_layout.addWidget(self.splitter)

        # Controls that affect the main plot
        self.column_selector.currentIndexChanged.connect(self.plot_parameters_changed)
        self.filter_checkbox.stateChanged.connect(self._on_filter_toggled)
        self.cutoff_frequency_input.textChanged.connect(self.plot_parameters_changed)
        self.filter_order_input.valueChanged.connect(self.plot_parameters_changed)
        self.spectrum_checkbox.stateChanged.connect(self._on_spectrum_toggled)

        # Controls that only affect the spectrum plot
        self.plot_type_selector.currentIndexChanged.connect(self.spectrum_parameters_changed)
        self.plot_type_selector.currentIndexChanged.connect(self._update_colorscale_visibility)
        self.colorscale_selector.currentIndexChanged.connect(self.spectrum_parameters_changed)
        self.num_slices_input.returnPressed.connect(self.spectrum_parameters_changed)

        # Set Tooltips
        self.num_slices_input.setToolTip(tooltips.SPECTRUM_SLICES)

    def display_regular_plot(self, fig):
        load_fig_to_webview(fig, self.regular_plot)

    def display_phase_plot(self, fig):
        load_fig_to_webview(fig, self.phase_plot)

    def display_spectrum_plot(self, fig):
        load_fig_to_webview(fig, self.spectrum_plot)

    def set_phase_plot_visibility(self, visible):
        is_visible = self.phase_plot.parent() is not None
        if visible and not is_visible:
            self.splitter.addWidget(self.phase_plot)
        elif not visible and is_visible:
            self.phase_plot.setParent(None)

    def set_spectrum_plot_visibility(self, visible):
        is_visible = self.spectrum_plot.parent() is not None
        if visible and not is_visible:
            self.splitter.addWidget(self.spectrum_plot)
        elif not visible and is_visible:
            self.spectrum_plot.setParent(None)

    def set_time_domain_features_visibility(self, visible):
        """Shows or hides widgets that are only relevant for time-domain data."""
        self.filter_checkbox.setVisible(visible)
        self.spectrum_checkbox.setVisible(visible)

        # If the main features are being hidden, also hide their sub-options
        if not visible:
            # Hide filter sub-options
            self.cutoff_frequency_label.setVisible(False)
            self.cutoff_frequency_input.setVisible(False)
            self.filter_order_label.setVisible(False)
            self.filter_order_input.setVisible(False)

            # Hide spectrum sub-options
            self.plot_type_selector.setVisible(False)
            self.num_slices_label.setVisible(False)
            self.num_slices_input.setVisible(False)
            self.colorscale_label.setVisible(False)
            self.colorscale_selector.setVisible(False)

            # Ensure the spectrum plot is also hidden
            self.set_spectrum_plot_visibility(False)

    def _on_filter_toggled(self, state):
        is_checked = state == QtCore.Qt.Checked
        self.cutoff_frequency_label.setVisible(is_checked)
        self.cutoff_frequency_input.setVisible(is_checked)
        self.filter_order_label.setVisible(is_checked)
        self.filter_order_input.setVisible(is_checked)
        self.plot_parameters_changed.emit()

    def _on_spectrum_toggled(self, state):
        is_checked = state == QtCore.Qt.Checked

        # Toggle visibility of all related controls
        self.plot_type_selector.setVisible(is_checked)
        self._update_colorscale_visibility()
        self.colorscale_label.setVisible(is_checked)
        self.colorscale_selector.setVisible(is_checked)
        self.num_slices_label.setVisible(is_checked)
        self.num_slices_input.setVisible(is_checked)

        # Manage the splitter layout
        if is_checked:
            # We are showing the spectrum plot
            self.set_spectrum_plot_visibility(True)
            # Restore the previous splitter sizes if they exist
            if self.splitter_sizes:
                self.splitter.setSizes(self.splitter_sizes)
            else:
                # If it is the first time, distribute sizes equally
                count = self.splitter.count()
                if count > 0:
                    equal_sizes = [self.splitter.height() // count] * count
                    self.splitter.setSizes(equal_sizes)
        else:
            # Save the current sizes before hiding the widget
            self.splitter_sizes = self.splitter.sizes()
            self.set_spectrum_plot_visibility(False)

        self.plot_parameters_changed.emit()

    def _update_colorscale_visibility(self):
        """Shows or hides the colorscale option based on the selected plot type."""
        # First, check if the spectrum plot itself is visible
        if not self.spectrum_checkbox.isChecked():
            self.colorscale_label.setVisible(False)
            self.colorscale_selector.setVisible(False)
            return

        # If it is visible, check the plot type
        current_plot_type = self.plot_type_selector.currentText()
        show_colorscale = current_plot_type in ['Heatmap', 'Surface']
        self.colorscale_label.setVisible(show_colorscale)
        self.colorscale_selector.setVisible(show_colorscale)