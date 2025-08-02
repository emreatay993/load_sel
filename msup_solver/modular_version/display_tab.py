# display_tab.py
"""
Module for the Display Tab of the FEA visualization application.
...
"""

# ---- Standard Library Imports ----
import os
import time
import gc

# ---- Third-Party Imports ----
import imageio
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor

# ---- PySide/PyQt Imports ----
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDoubleValidator, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (QAbstractItemView, QAction, QComboBox, QDialog,
                             QDoubleSpinBox, QFileDialog, QGroupBox, QApplication,
                             QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                             QMenu, QMessageBox, QProgressDialog, QPushButton,
                             QSpinBox, QTableView, QVBoxLayout, QWidget,
                             QWidgetAction)

# ---- Local Imports ----
from solver_engine import NP_DTYPE
from fea_utilities import generate_apdl_ic

# region Module Classes
class DisplayTab(QWidget):
    """
    GUI tab for 3D visualization of FEA results using PyVista.

    This class manages the PyVista plotter, UI controls for visualization,
    time-point analysis, animation, and interaction features like hotspot
    detection and point picking.
    """
    # Create signals
    node_picked_signal = pyqtSignal(int)
    time_point_update_requested = pyqtSignal(float, dict)
    animation_precomputation_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mesh = None  # Track current mesh for memory management
        self.current_actor = None  # Track the current actor for scalar range updates
        self.camera_state = None
        self.camera_widget = None
        self.last_hover_time = 0  # For frame rate throttling
        self.hover_annotation = None
        self.hover_observer = None  # Track hover callback observer
        self.anim_timer = None  # timer for animation
        self.time_text_actor = None
        self.current_anim_time = 0.0  # current time in the animation
        self.animation_paused = False
        self.temp_solver = None

        # Attributes for Precomputation
        self.precomputed_scalars = None  # (num_nodes, num_anim_steps)
        self.precomputed_coords = None  # (num_nodes, 3, num_anim_steps) or similar
        self.precomputed_anim_times = None  # (num_anim_steps,) - actual time values for each frame
        self.current_anim_frame_index = 0  # Index for accessing precomputed arrays
        self.data_column_name = "Stress"  # Default/placeholder name for scalars
        self.is_deformation_included_in_anim = False  # Track if deformation was computed
        self.original_node_coords = None

        self.highlight_actor = None  # This tracks the highlight sphere

        # Attributes for the dynamic "Go To Node" marker
        self.target_node_index = None         # Index of the node to track
        self.target_node_id = None            # ID of the node (for the label text)
        self.target_node_label_actor = None   # The label actor
        self.label_point_data = None          # A single-point PolyData to anchor the label
        self.marker_poly = None
        self.target_node_marker_actor = None  # glyph actor drawn in pixels
        self.last_goto_node_id = None
        self.freeze_tracked_node = False  # on/off
        self.freeze_baseline = None  # np.array([x0, y0, z0]) when mode enabled

        self.box_widget = None
        self.hotspot_dialog = None
        self.is_point_picking_active = False

        self.init_ui()

    def init_ui(self):
        # Style settings
        button_style = """
            QPushButton {
                background-color: #e7f0fd;
                border: 1px solid #5b9bd5;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #cce4ff;
            }
        """
        group_box_style = """
        QGroupBox {
            border: 1px solid #5b9bd5;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
        }
        """

        # Create file selection components
        self.file_button = QPushButton('Load Visualization File')
        self.file_button.setStyleSheet(button_style)
        self.file_button.clicked.connect(self.load_file)

        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.file_path.setStyleSheet("background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;")

        # Visualization controls
        self.point_size = QSpinBox()
        self.point_size.setRange(1, 100)
        self.point_size.setValue(5)
        self.point_size.setPrefix("Size: ")
        self.point_size.valueChanged.connect(self.update_point_size)

        # Scalar range controls
        self.scalar_min_spin = QDoubleSpinBox()
        self.scalar_max_spin = QDoubleSpinBox()
        self.scalar_min_spin.setPrefix("Min: ")
        self.scalar_max_spin.setPrefix("Max: ")
        self.scalar_min_spin.setDecimals(3)
        self.scalar_max_spin.setDecimals(3)
        self.scalar_min_spin.valueChanged.connect(self._update_scalar_range)
        self.scalar_max_spin.valueChanged.connect(self._update_scalar_range)
        self.scalar_min_spin.valueChanged.connect(lambda v: self.scalar_max_spin.setMinimum(v))
        self.scalar_max_spin.valueChanged.connect(lambda v: self.scalar_min_spin.setMaximum(v))

        # Create PyVista widget
        self.plotter = QtInteractor(parent=self)
        self.plotter.set_background('#FFFFFF')

        # Add Custom Context Menu
        self.plotter.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plotter.customContextMenuRequested.connect(self.show_context_menu)

        # Layout
        layout = QVBoxLayout()

        # File controls
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_path)

        # Visualization controls
        self.graphics_control_layout = QHBoxLayout()
        self.graphics_control_layout.addWidget(QLabel("Node Point Size:"))
        self.graphics_control_layout.addWidget(self.point_size)
        self.graphics_control_layout.addWidget(QLabel("Legend Range:"))
        self.graphics_control_layout.addWidget(self.scalar_min_spin)
        self.graphics_control_layout.addWidget(self.scalar_max_spin)
        self.graphics_control_layout.addStretch()

        self.graphics_control_group = QGroupBox("Visualization Controls")
        self.graphics_control_group.setStyleSheet(group_box_style)
        self.graphics_control_group.setLayout(self.graphics_control_layout)

        # Contour Time-point controls layout:
        self.selected_time_label = QLabel("Initialize / Display results for a selected time point:")
        self.selected_time_label.setStyleSheet("margin: 10px;")
        # Initially hide the checkbox, and show it once the required files are loaded.

        self.time_point_spinbox = QDoubleSpinBox()
        self.time_point_spinbox.setDecimals(5)
        self.time_point_spinbox.setPrefix("Time (seconds): ")
        # Range will be updated later from the modal coordinate file's time values.
        self.time_point_spinbox.setRange(0, 0)

        self.update_time_button = QPushButton("Update")
        self.update_time_button.clicked.connect(self.update_time_point_results)

        self.save_time_button = QPushButton("Save Time Point as CSV")
        self.save_time_button.clicked.connect(self.save_time_point_results)

        self.extract_ic_button = QPushButton("Export Velocity as Initial Condition in APDL")
        self.extract_ic_button.setStyleSheet(button_style)
        self.extract_ic_button.clicked.connect(self.extract_initial_conditions)
        self.extract_ic_button.setVisible(False)  # Hidden by default

        # Put the new widgets in a horizontal layout
        self.time_point_layout = QHBoxLayout()
        self.time_point_layout.addWidget(self.selected_time_label)
        self.time_point_layout.addWidget(self.time_point_spinbox)
        self.time_point_layout.addWidget(self.update_time_button)
        self.time_point_layout.addWidget(self.save_time_button)
        self.time_point_layout.addWidget(self.extract_ic_button)
        self.time_point_layout.addStretch()

        self.time_point_group = QGroupBox("Initialization && Time Point Controls")
        self.time_point_group.setStyleSheet(group_box_style)
        self.time_point_group.setLayout(self.time_point_layout)
        self.time_point_group.setVisible(False)

        # Animation Control Layout
        self.anim_layout = QHBoxLayout()
        # Add a spin box for animation frame interval (in milliseconds)
        self.anim_interval_spin = QSpinBox()
        self.anim_interval_spin.setRange(5, 10000)  # Allow between 5 ms and 10,000 ms delay
        self.anim_interval_spin.setValue(100)  # Default delay is 100 ms
        self.anim_interval_spin.setPrefix("Interval (ms): ")
        self.anim_layout.addWidget(self.anim_interval_spin)
        # Label and spinbox for animation start time
        self.anim_start_label = QLabel("Time Range:")
        self.anim_start_spin = QDoubleSpinBox()
        self.anim_start_spin.setPrefix("Start: ")
        self.anim_start_spin.setDecimals(5)
        self.anim_start_spin.setMinimum(0)
        self.anim_start_spin.setValue(0)
        # Label and spinbox for animation end time
        self.anim_end_spin = QDoubleSpinBox()
        self.anim_end_spin.setPrefix("End: ")
        self.anim_end_spin.setDecimals(5)
        self.anim_end_spin.setMinimum(0)
        self.anim_end_spin.setValue(1)
        # Ensure valid range by connecting valueChanged signals
        self.anim_start_spin.valueChanged.connect(self._update_anim_range_min)
        self.anim_end_spin.valueChanged.connect(self._update_anim_range_max)
        # Play and Stop buttons
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.play_button.clicked.connect(self.start_animation)
        self.pause_button.clicked.connect(self.pause_animation)
        self.stop_button.clicked.connect(self.stop_animation)
        # Add Time Step Mode ComboBox and Custom Step SpinBox
        self.time_step_mode_combo = QComboBox()
        self.time_step_mode_combo.addItems(["Custom Time Step", "Actual Data Time Steps"])
        self.time_step_mode_combo.setCurrentIndex(1)
        self.custom_step_spin = QDoubleSpinBox()
        self.custom_step_spin.setDecimals(5)
        self.custom_step_spin.setRange(0.000001, 10)
        self.custom_step_spin.setValue(0.01)
        self.custom_step_spin.setPrefix("Step (secs): ")

        self.actual_interval_spin = QSpinBox()
        self.actual_interval_spin.setRange(1, 1)  # Set max later after loading time_values
        self.actual_interval_spin.setValue(1)
        self.actual_interval_spin.setPrefix("Every nth: ")
        self.actual_interval_spin.setVisible(False)  # Hidden by default

        # Connect the combo box's text change signal
        self.time_step_mode_combo.currentTextChanged.connect(self._update_step_spinbox_state)
        self._update_step_spinbox_state(self.time_step_mode_combo.currentText())

        # Deformation Scale Factor
        self.deformation_scale_label = QLabel("Deformation Scale Factor:")
        self.deformation_scale_edit = QLineEdit("1")
        # Create and set a QDoubleValidator. This will allow numbers in standard or scientific notation.
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.deformation_scale_edit.setValidator(validator)
        # Connect the editingFinished signal so that pressing Enter (or losing focus) triggers validation.
        self.deformation_scale_edit.editingFinished.connect(self._validate_deformation_scale)
        # Store the last valid input – starting at 1.0.
        self.last_valid_deformation_scale = 1.0
        self.deformation_scale_label.setVisible(False)
        self.deformation_scale_edit.setVisible(False)
        self.graphics_control_layout.addWidget(self.deformation_scale_label)
        self.graphics_control_layout.addWidget(self.deformation_scale_edit)

        # Add Save Animation Button ---
        self.save_anim_button = QPushButton("Save as Video/GIF")
        self.save_anim_button.setStyleSheet(button_style)  # Apply the style
        self.save_anim_button.clicked.connect(self.save_animation)
        self.save_anim_button.setEnabled(False)  # Initially disabled
        self.save_anim_button.setToolTip(
            "Save the precomputed animation frames as MP4 or GIF.\nRequires 'imageio' and 'ffmpeg' (for MP4).")

        # Add widgets to the animation layout
        self.anim_layout.addWidget(self.time_step_mode_combo)
        self.anim_layout.addWidget(self.custom_step_spin)
        self.anim_layout.addWidget(self.actual_interval_spin)
        self.anim_layout.addWidget(self.anim_interval_spin)
        self.anim_layout.addWidget(self.anim_start_label)
        self.anim_layout.addWidget(self.anim_start_spin)
        self.anim_layout.addWidget(self.anim_end_spin)
        self.anim_layout.addWidget(self.play_button)
        self.anim_layout.addWidget(self.pause_button)
        self.anim_layout.addWidget(self.stop_button)
        self.anim_layout.addWidget(self.save_anim_button)
        # self.anim_layout.addStretch()

        self.anim_group = QGroupBox("Animation Controls")
        self.anim_group.setStyleSheet(group_box_style)
        self.anim_group.setLayout(self.anim_layout)
        self.anim_group.setVisible(False)

        layout.addLayout(file_layout)
        layout.addWidget(self.graphics_control_group)
        layout.addWidget(self.time_point_group)
        layout.addWidget(self.anim_group)
        layout.addWidget(self.plotter)
        self.setLayout(layout)

    @pyqtSlot(object)
    def _setup_initial_view(self, initial_data):
        """
        Receives the initial data payload (time_values, node_coords, df_node_ids)
        and sets up the UI controls and initial point cloud view.
        """
        time_values, node_coords, df_node_ids, deformation_is_loaded = initial_data

        # Store some intial data as an instance attribute, such as time data, to be used later in other methods
        self.time_values = time_values
        self.original_node_coords = node_coords

        # --- Update Time and Animation UI controls ---
        min_time, max_time = np.min(time_values), np.max(time_values)
        self.time_point_spinbox.setRange(min_time, max_time)
        self.time_point_spinbox.setValue(min_time)

        # Compute the average sampling interval (dt)
        if len(time_values) > 1:
            avg_dt = np.mean(np.diff(time_values))
        else:
            avg_dt = 1.0  # Fallback if only one time value exists
        self.time_point_spinbox.setSingleStep(avg_dt)

        self.anim_start_spin.setRange(min_time, max_time)
        self.anim_end_spin.setRange(min_time, max_time)
        self.anim_start_spin.setValue(min_time)
        self.anim_end_spin.setValue(max_time)
        self.actual_interval_spin.setMaximum(len(time_values))
        self.actual_interval_spin.setValue(1)

        self.anim_group.setVisible(True)
        self.time_point_group.setVisible(True)
        self.deformation_scale_label.setVisible(True)
        self.deformation_scale_edit.setVisible(True)

        # --- Initialize plotter with points ---
        if node_coords is not None:
            mesh = pv.PolyData(node_coords)
            if df_node_ids is not None:
                mesh["NodeID"] = df_node_ids.astype(int)
            mesh["Index"] = np.arange(mesh.n_points)

            self.current_mesh = mesh
            self.update_visualization()
            self.plotter.reset_camera()

        # Logic for the deformation scale factor ---
        if deformation_is_loaded:
            self.deformation_scale_edit.setEnabled(True)
            self.deformation_scale_edit.setText(
                str(self.last_valid_deformation_scale))  # Restore last valid value
        else:
            self.deformation_scale_edit.setEnabled(False)
            self.deformation_scale_edit.setText("0")  # Set to "0" when disabled

    def _update_time_point_range(self):
        """
        Check whether both the modal coordinate file (MCF) and the modal stress file have been loaded.
        If so, update the range of the time_point_spinbox (using the global time_values array),
        set its singleStep to the average sampling interval, and make the
        'Display results for a selected time point' checkbox visible.
        """
        if "modal_coord" in globals() and "modal_sx" in globals() and "time_values" in globals():
            min_time = np.min(time_values)
            max_time = np.max(time_values)
            self.time_point_spinbox.setRange(min_time, max_time)
            self.time_point_spinbox.setValue(min_time)
            # Compute the average sampling interval (dt)
            if len(time_values) > 1:
                avg_dt = np.mean(np.diff(time_values))
            else:
                avg_dt = 1.0  # Fallback if only one time value exists
            self.time_point_spinbox.setSingleStep(avg_dt)

            # Update animation time range controls
            self.anim_start_spin.setRange(min_time, max_time)
            self.anim_end_spin.setRange(min_time, max_time)
            self.anim_start_spin.setValue(min_time)
            self.anim_end_spin.setValue(max_time)
            self.actual_interval_spin.setMaximum(len(time_values))  # max is number of time points
            self.actual_interval_spin.setValue(1)  # default to every point

            self.anim_group.setVisible(True)
            self.time_point_group.setVisible(True)
            self.deformation_scale_label.setVisible(True)
            self.deformation_scale_edit.setVisible(True)

            # Check whether modal deformations file is loaded
            has_deforms = "modal_ux" in globals()

            # Enable/disable the scale factor based on whether deformation data is loaded
            if has_deforms:
                self.deformation_scale_edit.setEnabled(True)
                self.deformation_scale_edit.setText(
                    str(self.last_valid_deformation_scale))  # Restore last valid value (e.g., "1")
            else:
                self.deformation_scale_edit.setEnabled(False)
                self.deformation_scale_edit.setText("0")  # Set to "0" when disabled

            # Initialize plotter with points
            if "node_coords" in globals() and node_coords is not None:
                mesh = pv.PolyData(node_coords)
                if "df_node_ids" in globals() and df_node_ids is not None:
                    mesh["NodeID"] = df_node_ids.astype(int)
                mesh["Index"] = np.arange(mesh.n_points)

                self.current_mesh = mesh
                self.data_column = "Index"
                self.update_visualization()
                self.plotter.reset_camera()
        else:
            # Hide the controls if the required data is not available.
            self.anim_group.setVisible(False)
            self.time_point_group.setVisible(False)
            self.deformation_scale_label.setVisible(False)
            self.deformation_scale_edit.setVisible(False)

    def save_time_point_results(self):
        """
        Saves the currently displayed results (node coordinates and the computed scalar field)
        into a CSV file. The saved column name is aware of the currently displayed output type.
        """
        # 1. Check if there is a mesh with data to save.
        if self.current_mesh is None:
            QMessageBox.warning(self, "No Data", "No visualization data to save.")
            return

        # 2. Get the name of the currently active data array from the mesh object.
        active_scalar_name = self.current_mesh.active_scalars_name
        if not active_scalar_name:
            QMessageBox.warning(self, "No Active Data",
                                "The current mesh does not have an active scalar field to save.")
            return

        # 3. Create a smart, descriptive default filename.
        base_name = active_scalar_name.split(' ')[0]  # Extracts "SVM", "S1", "Velocity", etc.
        selected_time = self.time_point_spinbox.value()
        # Format the filename, replacing illegal characters like '.' in the time value
        default_filename = f"{base_name}_T_{selected_time:.5f}.csv".replace('.', '_')

        # 4. Open the file dialog with the suggested filename.
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Time Point Results", default_filename,
                                                   "CSV Files (*.csv)")

        if not file_name:
            return  # User cancelled the dialog

        try:
            # 5. Get all necessary data arrays from the mesh.
            coords = self.current_mesh.points
            scalar_data = self.current_mesh[active_scalar_name]

            # 6. Create the output DataFrame, starting with NodeID if available.
            df_out = pd.DataFrame()
            if 'NodeID' in self.current_mesh.array_names:
                df_out['NodeID'] = self.current_mesh['NodeID']

            # 7. Add the coordinate and scalar data with their correct headers.
            df_out['X'] = coords[:, 0]
            df_out['Y'] = coords[:, 1]
            df_out['Z'] = coords[:, 2]
            df_out[active_scalar_name] = scalar_data

            # 8. Save the complete DataFrame to the chosen CSV file.
            df_out.to_csv(file_name, index=False)
            QMessageBox.information(self, "Save Successful", f"Time point results saved successfully to:\n{file_name}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the file: {e}")

    def _update_anim_range_min(self, value):
        # Ensure that the animation end spin box cannot be set to a value less than the start
        self.anim_end_spin.setMinimum(value)

    def _update_anim_range_max(self, value):
        # Ensure that the animation start spin box cannot exceed the end value
        self.anim_start_spin.setMaximum(value)

    def _validate_deformation_scale(self):
        """
        Validate the deformation scale factor input.
        If the input can be converted to a float, update the last valid value.
        Otherwise, revert the text to the last valid input.
        """
        text = self.deformation_scale_edit.text()
        try:
            value = float(text)
            self.last_valid_deformation_scale = value
        except ValueError:
            # Revert to the last valid input if the current text is invalid.
            self.deformation_scale_edit.setText(str(self.last_valid_deformation_scale))

    def update_time_point_results(self):
        """Gathers UI selections and emits a signal to request a time point calculation."""
        # 1. Access the main window's control panel to read the checkboxes
        main_tab = self.main_window.batch_solver_tab
        # Output mode validation check
        if main_tab.damage_index_checkbox.isChecked() or main_tab.time_history_checkbox.isChecked():
            QMessageBox.warning(self, "Invalid Selection",
                                "Damage Index and Time History Mode are not valid for single time point visualization.")
            return

        # 2. Package the user's selections into a dictionary
        options = {
            'compute_von_mises': main_tab.von_mises_checkbox.isChecked(),
            'compute_max_principal': main_tab.max_principal_stress_checkbox.isChecked(),
            'compute_min_principal': main_tab.min_principal_stress_checkbox.isChecked(),
            'compute_deformation_contour': main_tab.deformation_checkbox.isChecked(),
            'compute_velocity': main_tab.velocity_checkbox.isChecked(),
            'compute_acceleration': main_tab.acceleration_checkbox.isChecked(),
            'display_deformed_shape': main_tab.deformations_checkbox.isChecked(),
            'include_steady': main_tab.steady_state_checkbox.isChecked(),
            'skip_n_modes': int(main_tab.skip_modes_combo.currentText()),
            'scale_factor': float(self.deformation_scale_edit.text())
        }

        # 3. Get the requested time value
        selected_time = self.time_point_spinbox.value()

        # 4. Emit the signal with the request
        print(f"DisplayTab: Requesting update for time {selected_time} with options: {options}")
        self.time_point_update_requested.emit(selected_time, options)

    def extract_initial_conditions(self):
        """
        Extracts velocity data from the current mesh, generates an APDL
        script for initial conditions, and prompts the user to save it.
        """
        if self.current_mesh is None:
            QMessageBox.warning(self, "No Data", "No visualization data available to extract.")
            return

        required_arrays = ['NodeID', 'vel_x', 'vel_y', 'vel_z']
        if not all(arr in self.current_mesh.array_names for arr in required_arrays):
            QMessageBox.warning(self, "Incomplete Data",
                                "The current visualization does not contain the required "
                                "velocity components (vel_x, vel_y, vel_z). "
                                "Please re-run the time point update for Velocity to "
                                "generate this data.")
            return

        try:
            node_ids = self.current_mesh['NodeID']
            vel_x = self.current_mesh['vel_x']
            vel_y = self.current_mesh['vel_y']
            vel_z = self.current_mesh['vel_z']
        except Exception as e:
            QMessageBox.critical(self, "Data Extraction Error", f"Failed to get velocity data from the mesh: {e}")
            return

        apdl_script_content = generate_apdl_ic(node_ids, vel_x, vel_y, vel_z)

        default_file_name = "initial_conditions.inp"
        file_name, _ = QFileDialog.getSaveFileName(self, "Save APDL Initial Conditions", default_file_name,
                                                   "APDL Input Files (*.inp);;All Files (*)")

        if not file_name:
            return

        try:
            with open(file_name, 'w', encoding="utf-8") as f:
                f.write(apdl_script_content)
            QMessageBox.information(self,
                                    "Save Successful", f"APDL initial conditions saved successfully to:\n{file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the file: {e}")

    def start_animation(self):
        """Gathers animation parameters and requests a precomputation via a signal."""
        # The 'global' statements have been removed as they are no longer needed
        # and were causing the NameError.

        if self.current_mesh is None:
            QMessageBox.warning(self, "No Data",
                                "Please load or initialize the mesh before animating.")
            return

        # Preserve tracked node state before cleanup
        # Read the current tracking state into local variables
        tracked_node_index = self.target_node_index
        tracked_node_id = self.target_node_id
        is_frozen = self.freeze_tracked_node

        # Hide marker/label if frozen mode is active when animation starts/resumes
        if tracked_node_index is not None and is_frozen:
            if self.target_node_marker_actor:
                self.target_node_marker_actor.SetVisibility(False)
            if self.target_node_label_actor:
                self.target_node_label_actor.SetVisibility(False)
            self.plotter.render() # Immediately hide them

        # region Resume Logic
        if self.animation_paused:
            if self.precomputed_scalars is None:        # NEW LINE: Disable play button immediately when animation starts or resumes
                QMessageBox.warning(self, "Resume Error",
                                    "Cannot resume. Precomputed data is missing. Please stop and start again.")
                self.stop_animation()
                return
            print("Resuming animation...")
            self.animation_paused = False
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.play_button.setEnabled(False)  # keep Play disabled while running
            self.deformation_scale_edit.setEnabled(False)
            if self.anim_timer:
                self.anim_timer.start(self.anim_interval_spin.value())
            else:
                self.anim_timer = QTimer(self)
                self.anim_timer.timeout.connect(self._animate_frame)
                self.anim_timer.start(self.anim_interval_spin.value())
            return
        # endregion

        # Start fresh by stopping any existing animation
        self.stop_animation()

        # Disable play button immediately when animation starts or resumes
        self.play_button.setEnabled(False)

        # Restore tracked node state after cleanup
        # If a node was being tracked, restore its state so the new animation can use it.
        if tracked_node_index is not None:
            self.target_node_index = tracked_node_index
            self.target_node_id = tracked_node_id
            self.freeze_tracked_node = is_frozen
            # The markers themselves will be re-created by update_visualization
            # or moved by the animation loop, so we don't need to restore the actor objects.

            # NEW BLOCK: Re-establish freeze_baseline if freezing was active
            if self.freeze_tracked_node and self.original_node_coords is not None:
                # The _setup_initial_view or _visualize_data sets self.current_mesh.points
                # to original_node_coords when the mesh is first loaded or reset.
                # So, if original_node_coords exists, use it to set the baseline for the first frame.
                # The actual mesh points might be deformed in on_animation_data_ready,
                # but freeze_baseline needs to be relative to the coordinate system
                # in which the node's position is *initially* defined for the first frame.
                if self.current_mesh is not None and self.current_mesh.n_points > tracked_node_index:
                    self.freeze_baseline = self.current_mesh.points[self.target_node_index].copy()
                else:
                    print("Warning: Cannot set freeze_baseline, current_mesh or original_node_coords not ready.")
                    self.freeze_tracked_node = False # Disable feature if baseline cannot be set.


        # region 1. Get Animation Parameters & Time Steps
        anim_times, _, error_msg = self._get_animation_time_steps()
        if error_msg:
            QMessageBox.warning(self, "Animation Setup Error", error_msg)
            return
        if anim_times is None or len(anim_times) == 0:
            QMessageBox.warning(self, "Animation Setup Error", "No time steps generated for the animation.")
            return
        # endregion

        # region 2. Gather UI selections and parameters
        main_tab = self.window().batch_solver_tab
        if not (main_tab.von_mises_checkbox.isChecked() or main_tab.max_principal_stress_checkbox.isChecked() or
                main_tab.min_principal_stress_checkbox.isChecked() or main_tab.deformation_checkbox.isChecked() or
                main_tab.velocity_checkbox.isChecked() or main_tab.acceleration_checkbox.isChecked()):
            QMessageBox.warning(self, "No Selection", "No valid output is selected for animation.")
            return

        params = {
            'compute_von_mises': main_tab.von_mises_checkbox.isChecked(),
            'compute_max_principal': main_tab.max_principal_stress_checkbox.isChecked(),
            'compute_min_principal': main_tab.min_principal_stress_checkbox.isChecked(),
            'compute_deformation_anim': main_tab.deformations_checkbox.isChecked(),
            'compute_deformation_contour': main_tab.deformation_checkbox.isChecked(),
            'compute_velocity': main_tab.velocity_checkbox.isChecked(),
            'compute_acceleration': main_tab.acceleration_checkbox.isChecked(),
            'include_steady': main_tab.steady_state_checkbox.isChecked(),
            'skip_n_modes': int(main_tab.skip_modes_combo.currentText()),
            'scale_factor': float(self.deformation_scale_edit.text())
        }
        # endregion

        # region 3. Delegate Calculation by Emitting Signal
        # This replaces the entire local computation block.
        # The slot in 'main_app.py' will now handle the calculation.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        print("DisplayTab: Delegating animation precomputation by emitting a signal...")
        self.animation_precomputation_requested.emit(params)
        # The animation will begin playing once the 'on_animation_data_ready' slot
        # receives the computed data back from the main GUI.
        # endregion

    def pause_animation(self):
        """Pause the animation (resumes from the current frame when Play is clicked)."""
        if self.anim_timer is not None and self.anim_timer.isActive():
            self.anim_timer.stop()
            self.animation_paused = True  # Set the flag
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)

            # Make marker and label visible when paused, if a node is being tracked and frozen
            if self.target_node_index is not None and self.freeze_tracked_node:
                if self.target_node_marker_actor:
                    self.target_node_marker_actor.SetVisibility(True)
                if self.target_node_label_actor:
                    self.target_node_label_actor.SetVisibility(True)
                self.plotter.render() # Crucial to immediately update visibility


            print("\nAnimation paused.")
        else:
            print("\nPause command ignored: Animation timer not active.")

    def stop_animation(self):
        """Stop the animation, release precomputed data, and reset state."""
        # Get the initial node coordinates from initial view so that nodes can go back to those positions
        node_coords = self.original_node_coords

        # Check if there is an animation to stop before printing.
        is_stoppable = self.anim_timer is not None or self.precomputed_scalars is not None

        self._clear_goto_node_markers()

        if is_stoppable:
            print("\nStopping animation and releasing resources...")

        if self.anim_timer is not None:
            self.anim_timer.stop()
            # Optional: disconnect to be sure it doesn't trigger again accidentally
            try:
                self.anim_timer.timeout.disconnect(self._animate_frame)
            except TypeError:  # Already disconnected
                pass
            self.anim_timer = None  # Allow timer to be garbage collected

        # --- Release Precomputed Data ---
        print(" ")
        if self.precomputed_scalars is not None:
            del self.precomputed_scalars
            self.precomputed_scalars = None
            print("Released precomputed scalars.")
        if self.precomputed_coords is not None:
            del self.precomputed_coords
            self.precomputed_coords = None
            print("Released precomputed coordinates.")
        if self.precomputed_anim_times is not None:
            del self.precomputed_anim_times
            self.precomputed_anim_times = None
            print("Released precomputed times.")

        # Explicitly trigger garbage collection
        gc.collect()
        # --- End Release ---

        # Reset state variables
        self.current_anim_frame_index = 0
        self.animation_paused = False
        self.is_deformation_included_in_anim = False

        # Reset UI elements
        self.deformation_scale_edit.setEnabled(True)  # Re-enable editing
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.save_anim_button.setEnabled(False)

        # Remove the animation time text actor
        if hasattr(self, 'time_text_actor') and self.time_text_actor is not None:
            self.plotter.remove_actor(self.time_text_actor)
            self.time_text_actor = None

        # Optional: Reset mesh to original state (if node_coords exist)
        if self.current_mesh and self.original_node_coords is not None:
            print("Resetting mesh to original coordinates.")
            try:
                # Check if the mesh still has points data assigned
                if self.current_mesh.points is not None:
                    # Only reset if the number of points matches
                    if self.current_mesh.n_points == node_coords.shape[0]:
                        self.current_mesh.points = node_coords.copy()  # Use copy to be safe
                        # Optionally reset scalars to 0 or initial state if desired
                        self.current_mesh[self.data_column_name] = np.zeros(self.current_mesh.n_points)
                        self.plotter.render()  # Render the reset state
                    else:
                        print("Warning: Cannot reset mesh points, point count mismatch.")
                else:
                    print("Warning: Cannot reset mesh points, mesh points data is missing.")
            except Exception as e:
                print(f"Error resetting mesh points: {e}")

        if is_stoppable:
            print("\nAnimation stopped.")

    def _animate_frame(self, update_index=True):
        """Update the display using the next precomputed animation frame."""
        # --- Use helper to update mesh ---
        # Check if data exists before calling helper
        if self.precomputed_scalars is None or self.precomputed_anim_times is None:
            print("Animation frame skipped: Precomputed data not available.")
            self.stop_animation()
            return

        if not self._update_mesh_for_frame(self.current_anim_frame_index):
            print(f"Animation frame skipped: Failed to update mesh for index {self.current_anim_frame_index}.")
            # Attempt to stop gracefully if data seems inconsistent
            self.stop_animation()
            return

        # --- Render ---
        # Render happens *after* mesh update now
        self.plotter.render()

        # --- Increment Frame Index for Next Call ---
        if update_index:
            num_frames = len(self.precomputed_anim_times) if self.precomputed_anim_times is not None else 0
            if num_frames > 0:
                self.current_anim_frame_index += 1
                if self.current_anim_frame_index >= num_frames:
                    self.current_anim_frame_index = 0  # Loop animation
            else:
                # Should not happen if called correctly, but safety check
                self.stop_animation()

    def _update_mesh_for_frame(self, frame_index):
        """Updates the mesh data (scalars and optionally points) for a given frame index."""
        if self.precomputed_scalars is None or self.precomputed_anim_times is None or self.current_mesh is None:
            print("Error: Cannot update mesh - precomputed data or mesh missing.")
            return False

        num_frames = len(self.precomputed_anim_times)
        if frame_index < 0 or frame_index >= num_frames:
            print(f"Error: Invalid frame index {frame_index} for {num_frames} frames.")
            return False  # Invalid index

        try:
            current_scalars = self.precomputed_scalars[:, frame_index]
            current_time = self.precomputed_anim_times[frame_index]

            # MODIFICATION 1: Initialize current_coords with a copy of the base coordinates
            # This line needs to be ADDED.
            current_coords = self.current_mesh.points.copy()

            # Get deformed coordinates if available and enabled
            if self.is_deformation_included_in_anim and self.precomputed_coords is not None:
                if frame_index >= self.precomputed_coords.shape[2]:
                    print(f"Error: Frame index {frame_index} out of bounds for precomputed coordinates.")
                    return False
                # MODIFICATION 2: Ensure precomputed_coords is copied when used
                # Change: current_coords = self.precomputed_coords[:, :, frame_index]
                # To:
                current_coords = self.precomputed_coords[:, :, frame_index].copy() # Make a copy to modify

            # region Keep tracked node fixed by translating the mesh
            if self.freeze_tracked_node and self.freeze_baseline is not None:
                # Position of the tracked node AFTER the initial coordinate update (deformation or original)
                tracked_now = current_coords[self.target_node_index]
                shift = tracked_now - self.freeze_baseline

                if np.any(shift):
                    # MODIFICATION 3: Apply the shift to the `current_coords` array directly
                    # Remove the 'if' block that was here before, so `current_coords -= shift` is always executed
                    # (The 'if self.is_deformation_included_in_anim' and its 'else' block are removed)
                    current_coords -= shift

                    # 2) move the marker + label anchors
                    if self.marker_poly is not None:
                        self.marker_poly.points[:] -= shift
                        self.marker_poly.Modified()
            # endregion

            # MODIFICATION 4: Now, update the actual mesh points with the potentially shifted coordinates
            # This block was previously inside an 'if self.is_deformation_included_in_anim' and its 'else'
            # Now it's always executed after current_coords has been determined and potentially shifted.
            if self.current_mesh is not None:
                self.current_mesh.points = current_coords  # Update node positions
                # Tell VTK/PyVista that the coordinates array changed
                try:
                    self.current_mesh.points_modified()  # PyVista >= 0.37
                except AttributeError:
                    self.current_mesh.GetPoints().Modified()  # older versions
            else:
                print("Error: Current mesh is None, cannot update points.")
                return False

            # Update scalars on the mesh
            if self.current_mesh is not None:
                self.current_mesh[self.data_column_name] = current_scalars
                # Ensure the active scalar is set correctly
                if self.current_mesh.active_scalars_name != self.data_column_name:
                    self.current_mesh.set_active_scalars(self.data_column_name)
            else:
                print("Error: Current mesh is None, cannot update scalars.")
                return False

            # Update the scalar bar range if necessary (using fixed range from UI)
            fixed_min = self.scalar_min_spin.value()
            fixed_max = self.scalar_max_spin.value()
            if self.current_actor and hasattr(self.current_actor, 'mapper') and self.current_actor.mapper:
                # Check if range needs setting
                current_range = self.current_actor.mapper.GetScalarRange()
                # Use a tolerance for float comparison
                if abs(current_range[0] - fixed_min) > 1e-6 or abs(current_range[1] - fixed_max) > 1e-6:
                    self.current_actor.mapper.SetScalarRange(fixed_min, fixed_max)
                    # Update scalar bar title if needed
                    if hasattr(self.plotter, 'scalar_bar') and self.plotter.scalar_bar:
                        self.plotter.scalar_bar.SetTitle(self.data_column_name)

            # --- Update Time Text ---
            time_text = f"Time: {current_time:.5f} s"
            if hasattr(self, 'time_text_actor') and self.time_text_actor is not None:
                # Check if actor still exists in renderer before trying to set input
                try:
                    # VTKPythonCore could throw errors if the underlying VTK object is deleted
                    if self.time_text_actor.GetViewProps() is not None:
                        self.time_text_actor.SetInput(time_text)
                    else:  # Underlying object gone, recreate
                        # Safer to attempt removal by ref first
                        self.plotter.remove_actor(self.time_text_actor, render=False)
                        self.time_text_actor = self.plotter.add_text(time_text, position=(0.8, 0.9), viewport=False,
                                                                     font_size=10)
                except (
                        AttributeError,
                        ReferenceError):  # Actor might have been garbage collected or VTK object deleted
                    # Attempt removal if reference still exists
                    try:
                        self.plotter.remove_actor(self.time_text_actor, render=False)
                    except:
                        pass
                    self.time_text_actor = self.plotter.add_text(time_text, position=(0.8, 0.9), viewport=False,
                                                                 font_size=10)
            else:
                self.time_text_actor = self.plotter.add_text(time_text, position=(0.8, 0.9), viewport=False,
                                                             font_size=10)

            # Update "Go To Node" Marker Position During Animation
            # This block ensures the marker moves with the mesh on every frame.
            if self.target_node_index is not None:
                # Get the new coordinates of the tracked node from the deformed mesh
                new_coords = self.current_mesh.points[self.target_node_index]

                # If camera is NOT frozen, update the label's position.
                # If camera IS frozen, the label should be hidden (handled in toggle_freeze_node)
                if not self.freeze_tracked_node and self.target_node_label_actor and self.label_point_data:
                    self.label_point_data.points[0, :] = new_coords
                    self.label_point_data.Modified()  # Notify VTK that the data has changed

                # Move pixel glyph
                if self.target_node_marker_actor and self.marker_poly:
                    self.marker_poly.points[0] = new_coords
                    self.marker_poly.Modified()

            return True

        except IndexError as e:
            print(f"Error: Index {frame_index} out of bounds during mesh update. {e}")
            return False
        except Exception as e:
            # Log the error type and message
            print(f"Error updating mesh for frame {frame_index}: {type(e).__name__}: {e}")
            # Optionally show a QMessageBox here for critical errors
            # QMessageBox.critical(self, "Animation Error", f"Failed to update mesh for frame {frame_index}: {str(e)}")
            return False

    def _close_enough(self, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
        """Return True if vectors a and b are within 'tol' (Euclidean) distance."""
        return np.linalg.norm(a - b) < tol

    def _capture_animation_frame(self, frame_index):
        """Updates the plotter for the given frame index and returns a screenshot (NumPy array)."""
        # Update mesh data, scalar bar, time text for the target frame
        if not self._update_mesh_for_frame(frame_index):
            print(f"Warning: Failed to update mesh for frame {frame_index} before capture.")
            return None  # Indicate failure

        # Render the scene *after* updating
        self.plotter.render()
        # Allow the event loop to process the render command, might help prevent blank frames
        QApplication.processEvents()
        time.sleep(0.01)  # Small delay, sometimes helps ensure rendering completes before screenshot

        # Capture screenshot
        try:
            # Use window_size=None to capture the current interactive window size
            # Ensure plotter and renderer are valid
            if self.plotter and self.plotter.renderer:
                frame_image = self.plotter.screenshot(transparent_background=False, return_img=True, window_size=None)
                if frame_image is None:
                    print(f"Warning: Screenshot returned None for frame {frame_index}.")
                    return None
                return frame_image
            else:
                print(f"Error: Plotter or renderer invalid for frame {frame_index}.")
                return None
        except Exception as e:
            print(f"Error taking screenshot for frame {frame_index}: {type(e).__name__}: {e}")
            return None

    def save_animation(self):
            print("Animation saving process aborted.")

    def _get_save_path_and_format(self):
        """
        Opens a file dialog for the user to choose a save location and format.

        Returns:
            tuple[str | None, str | None]: A tuple containing the full file path and the
                                           format ('MP4' or 'GIF'), or (None, None) if cancelled.
        """
        default_dir = ""
        if hasattr(self.window(), 'project_directory') and self.window().project_directory:
            default_dir = self.window().project_directory
        elif self.file_path.text():
            default_dir = os.path.dirname(self.file_path.text())

        file_name, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Animation",
            default_dir,
            "MP4 Video (*.mp4);;Animated GIF (*.gif)",
            "MP4 Video (*.mp4)",  # Default filter
        )

        if not file_name:
            return None, None  # User cancelled

        if "MP4" in selected_filter:
            file_format = "MP4"
            if not file_name.lower().endswith(".mp4"):
                file_name += ".mp4"
        elif "GIF" in selected_filter:
            file_format = "GIF"
            if not file_name.lower().endswith(".gif"):
                file_name += ".gif"
        else:
            QMessageBox.warning(self, "Cannot Determine Format",
                                  "Could not determine file format. Please use .mp4 or .gif extension.")
            return None, None

        return file_name, file_format

    def _write_animation_to_file(self, file_path, file_format):
        """
        Handles the core loop of capturing and writing each animation frame to a file.

        Args:
            file_path (str): The full path to save the file to.
            file_format (str): The format of the file ('MP4' or 'GIF').

        Returns:
            bool: True if saving was completed successfully, False otherwise.
        """
        num_frames = len(self.precomputed_anim_times)
        fps = 1000.0 / self.anim_interval_spin.value()
        print(
            f"---Saving animation...---\n"
            f"Attempting to save {num_frames} frames to '{file_path}' as {file_format} "
            f"at {fps:.2f} FPS."
        )

        progress = QProgressDialog("Saving animation frames...", "Cancel", 0, num_frames, self.window())
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Saving Animation")
        progress.setMinimumDuration(1000)

        writer = None
        try:
            # --- Preserve Camera State (with fallback) ---
            initial_camera_state = None
            if self.plotter and self.plotter.camera:
                try:
                    # Preferred method for newer PyVista/VTK
                    initial_camera_state = self.plotter.camera.GetState()
                except AttributeError:
                    # Fallback for older versions
                    print("Camera.GetState() not found. Using fallback to save position/focal_point/up.")
                    initial_camera_state = (
                        self.plotter.camera.position,
                        self.plotter.camera.focal_point,
                        self.plotter.camera.up,
                    )

            # --- Setup imageio writer ---
            writer_kwargs = {'fps': fps, 'macro_block_size': None}
            if file_format == 'MP4':
                writer_kwargs.update({'pixelformat': 'yuv420p', 'quality': 7})
            else:  # GIF
                writer_kwargs.update({'loop': 0, 'palettesize': 256})

            writer = imageio.get_writer(file_path, format=file_format, mode='I', **writer_kwargs)

            # --- Frame-by-frame capture loop ---
            for i in range(num_frames):
                if progress.wasCanceled():
                    print("Save cancelled by user.")
                    return False

                # --- Restore Camera State before each capture (with fallback) ---
                if initial_camera_state:
                    try:
                        # Check if state is a dict (from GetState) or a tuple (from our fallback)
                        if isinstance(initial_camera_state, dict):
                            self.plotter.camera.SetState(initial_camera_state)
                        elif isinstance(initial_camera_state, tuple) and len(initial_camera_state) == 3:
                            self.plotter.camera.position = initial_camera_state[0]
                            self.plotter.camera.focal_point = initial_camera_state[1]
                            self.plotter.camera.up = initial_camera_state[2]
                    except Exception as cam_err:
                        print(f"Warning: Could not restore camera state for frame {i}: {cam_err}")


                frame_image = self._capture_animation_frame(i)
                if frame_image is None:
                    print(f"Warning: Skipping frame {i} due to capture failure.")
                    continue

                writer.append_data(frame_image)
                progress.setValue(i + 1)
                QApplication.processEvents()

            return True  # Success

        except ImportError as e:
            error_msg = f"ImportError: {e}. Cannot save animation.\n\n"
            if file_format == 'MP4':
                error_msg += "Saving MP4 requires 'ffmpeg'. Please install it.\nTry: pip install imageio[ffmpeg]"
            QMessageBox.critical(self, "Missing Dependency", error_msg)
            return False
        except Exception as e:
            error_msg = f"Failed to save animation:\n{type(e).__name__}: {e}\n\nCheck write permissions and ensure dependencies (like ffmpeg) are installed."
            QMessageBox.critical(self, "Save Error", error_msg)
            return False
        finally:
            if writer:
                writer.close()
            progress.close()

    def _restore_plotter_state(self, original_state):
        """
        Restores the plotter and animation timer to their state before the save operation.

        Args:
            original_state (dict): A dictionary containing the pre-save state.
        """
        print("Restoring plotter state...")
        try:
            self._update_mesh_for_frame(original_state["frame_index"])
            self.plotter.render()
        except Exception as restore_err:
            print(f"Warning: Could not fully restore plotter state: {restore_err}")

        # Restore animation timer if it was running and not paused before saving
        if original_state["was_timer_active"] and not original_state["is_paused"]:
            if self.anim_timer:
                self.anim_timer.start(self.anim_interval_spin.value())
                print("Live animation timer restarted.")
            else:  # Recreate if it was somehow destroyed
                self.anim_timer = QTimer(self)
                self.anim_timer.timeout.connect(self._animate_frame)
                self.anim_timer.start(self.anim_interval_spin.value())

        self.animation_paused = original_state["is_paused"]

    def _get_animation_time_steps(self):
        """
        Determines the time values and corresponding indices from global time_values
        needed for the animation based on user settings.

        Returns:
            tuple: (
                anim_times: np.array or None - The actual time values for each animation frame.
                anim_indices: np.array or None - The indices in global time_values corresponding to anim_times.
                error_message: str or None - An error message if inputs are invalid or no steps generated, otherwise None.
            )
        """
        time_values = self.time_values
        # --- Input Validation ---
        if time_values is None or len(time_values) == 0:
            return None, None, "Global time_values not loaded or empty."

        start_time = self.anim_start_spin.value()
        end_time = self.anim_end_spin.value()

        if start_time >= end_time:
            return None, None, "Animation start time must be less than end time."

        # Initialize as standard Python lists
        anim_times_list = []
        anim_indices_list = []

        # --- Logic based on Time Step Mode ---
        if self.time_step_mode_combo.currentText() == "Custom Time Step":
            step = self.custom_step_spin.value()
            if step <= 0:
                return None, None, "Custom time step must be positive."

            current_t = start_time
            last_added_idx = -1  # Keep track of the last index added

            # Loop through custom time steps
            while current_t <= end_time:
                # Find the index of the closest actual time point in the data
                idx = np.argmin(np.abs(time_values - current_t))

                # Ensure the found index corresponds to a time within the overall bounds
                # (Handles cases where closest time might be outside start/end due to large steps)
                # And ensure we don't add duplicate indices consecutively
                if time_values[idx] >= start_time and time_values[idx] <= end_time and idx != last_added_idx:
                    anim_indices_list.append(idx)
                    anim_times_list.append(time_values[idx])  # Use the actual data time point
                    last_added_idx = idx  # Update last added index

                # Prevent infinite loops for very small steps, break if time doesn't advance significantly
                if current_t + step <= current_t:
                    print("Warning: Custom time step is too small, breaking loop.")
                    break
                current_t += step

            # Ensure the time point closest to the requested end_time is included, if not already the last one
            end_idx = np.argmin(np.abs(time_values - end_time))
            if time_values[end_idx] >= start_time and time_values[end_idx] <= end_time:
                if not anim_indices_list or end_idx != anim_indices_list[-1]:
                    anim_indices_list.append(end_idx)
                    anim_times_list.append(time_values[end_idx])

        else:  # "Actual Data Time Steps"
            nth = self.actual_interval_spin.value()
            if nth <= 0:
                return None, None, "Actual data step interval (Every nth) must be positive."

            # Find indices of actual data points within the requested time range
            valid_indices = np.where((time_values >= start_time) & (time_values <= end_time))[0]

            if len(valid_indices) == 0:
                return None, None, "No actual data time steps found within the specified range."

            # Select every nth index from the valid ones
            selected_indices_np = valid_indices[::nth]

            # Convert to list for easier manipulation and checking
            selected_indices_list = selected_indices_np.tolist()

            # Ensure the very first point in range is included if skipped by nth
            first_valid_idx = valid_indices[0]
            if first_valid_idx not in selected_indices_list:
                selected_indices_list.insert(0, first_valid_idx)

            # Ensure the very last point in range is included if skipped by nth
            last_valid_idx = valid_indices[-1]
            if last_valid_idx not in selected_indices_list:
                # Check if the list is empty before trying to access last element
                if not selected_indices_list or last_valid_idx != selected_indices_list[-1]:
                    selected_indices_list.append(last_valid_idx)

            # Use the final list of indices
            anim_indices_list = selected_indices_list
            # Get the corresponding time values from the global array
            anim_times_list = time_values[anim_indices_list].tolist()  # Convert result to list

        # --- Final Check and Return ---
        # Perform the emptiness check ON THE LISTS before returning/converting
        if not anim_times_list:
            return None, None, "No animation frames generated for the selected time range and step."

        # If the lists are not empty, THEN convert to NumPy arrays and return
        # Use np.unique to remove potential duplicates introduced by start/end point logic, preserving order
        unique_indices, order_indices = np.unique(anim_indices_list, return_index=True)
        final_indices = unique_indices[np.argsort(order_indices)]
        final_times = time_values[final_indices]

        return np.array(final_times), np.array(final_indices, dtype=int), None

    def _estimate_animation_ram(self, num_nodes, num_anim_steps, include_deformation):
        """
        Estimates the peak RAM needed in GB for precomputing animation data.
        This revised version considers the intermediate arrays needed during calculation.
        """
        element_size = np.dtype(NP_DTYPE).itemsize  # Size of NP_DTYPE selected

        # RAM for the 6 intermediate normal/shear stress arrays (sx, sy, sz, sxy, syz, sxz)
        # These are needed to compute the final scalars (Von Mises or S1)
        # Shape: (num_nodes, num_anim_steps) for each of the 6 components.
        normal_stress_ram = num_nodes * 6 * num_anim_steps * element_size

        # RAM for the final stored scalar array (Von Mises or S1)
        # Shape: (num_nodes, num_anim_steps)
        scalar_ram = num_nodes * 1 * num_anim_steps * element_size

        # RAM for deformation calculation and storage (if requested)
        intermediate_displacement_ram = 0
        final_coordinate_ram = 0
        if include_deformation:
            # RAM for intermediate displacement arrays (ux_anim, uy_anim, uz_anim)
            # Shape: (num_nodes, num_anim_steps) for each of the 3 components.
            intermediate_displacement_ram = num_nodes * 3 * num_anim_steps * element_size

            # RAM for the final stored deformed coordinate array (precomputed_coords)
            # Stores X, Y, Z coordinates for each node at each step.
            # Shape: (num_nodes, 3, num_anim_steps) -> num_nodes * 3 * num_anim_steps elements.
            final_coordinate_ram = num_nodes * 3 * num_anim_steps * element_size

        # Total estimated peak RAM is the sum of intermediate stresses, final scalars,
        # and potentially intermediate displacements and final coordinates.
        # We assume the peak occurs when most of these arrays exist simultaneously
        # before intermediate ones are deleted by garbage collection.
        total_ram_bytes = (normal_stress_ram +
                           scalar_ram +
                           intermediate_displacement_ram +
                           final_coordinate_ram)

        # Add a safety buffer (e.g., 20%) for Python overhead, temporary copies, etc.
        # Increased buffer slightly as the calculation involves several large steps.
        total_ram_bytes *= 1.20

        # Convert bytes to Gigabytes (GB)
        return total_ram_bytes / (1024 ** 3)

    def load_file(self):
        """Load and visualize new data file"""
        try:
            # Clear previous data
            self._clear_visualization()

            file_name, _ = QFileDialog.getOpenFileName(
                self, 'Open Visualization File', '', 'CSV Files (*.csv)'
            )
            if not file_name:
                return

            self.file_path.setText(file_name)
            self._visualize_data(file_name)

        except Exception as e:
            QMessageBox.critical(self, "Loading Error", f"Failed to load file: {str(e)}")

    def _visualize_data(self, filename):
        """Handle data visualization with robust data cleaning, surface reconstruction, and interpolation."""
        try:
            # 1. Load the data and clean it as before
            df = pd.read_csv(filename)
            if df.empty:
                raise ValueError("The selected CSV file is empty.")

            df.columns = [col.strip() for col in df.columns]
            x_col = next((c for c in df.columns if c.upper() == 'X'), None)
            y_col = next((c for c in df.columns if c.upper() == 'Y'), None)
            z_col = next((c for c in df.columns if c.upper() == 'Z'), None)
            nodeid_col = next((c for c in df.columns if c.upper() == 'NODEID'), None)

            if not all([x_col, y_col, z_col]):
                raise ValueError("CSV file must contain X, Y, and Z columns.")

            df_clean = df.dropna(subset=[x_col, y_col, z_col])
            coords = df_clean[[x_col, y_col, z_col]].values

            potential_data_cols = [c for c in df_clean.columns if c.upper() not in ['NODEID', 'X', 'Y', 'Z']]
            if not potential_data_cols:
                raise ValueError("No data column found in the CSV file.")
            self.data_column = potential_data_cols[0]

            scalar_values = df_clean[self.data_column].fillna(0).values

            # 2. Create the ORIGINAL point cloud
            point_cloud = pv.PolyData(coords)

            # 3. Add the scalar data to the ORIGINAL point cloud FIRST
            point_cloud[self.data_column] = scalar_values

            # 3. Add NodeID and scalar data directly to the point cloud
            if nodeid_col:
                point_cloud['NodeID'] = df_clean[nodeid_col].values

            point_cloud.set_active_scalars(self.data_column)

            # 4. Assign this point cloud as the current mesh (no reconstruction)
            self.current_mesh = point_cloud

            # 5. Update UI controls
            data_min, data_max = self.current_mesh.get_data_range(self.data_column)
            self.scalar_min_spin.blockSignals(True)
            self.scalar_max_spin.blockSignals(True)
            self.scalar_min_spin.setRange(data_min, data_max)
            self.scalar_min_spin.setValue(data_min)
            self.scalar_max_spin.setRange(data_min, 1e30)
            self.scalar_max_spin.setValue(data_max)
            self.scalar_min_spin.blockSignals(False)
            self.scalar_max_spin.blockSignals(False)

            # 6. Finalize visualization
            if not self.camera_widget:
                self.camera_widget = self.plotter.add_camera_orientation_widget()
                self.camera_widget.EnabledOn()

            self.update_visualization()
            self.plotter.reset_camera()
            self.plotter.camera.zoom(1)

        except Exception as e:
            self._clear_visualization()
            QMessageBox.critical(self, "Visualization Error", f"Failed to visualize data:\n\n{str(e)}")

    def _update_scalar_range(self):
        """Update the scalar range of the current visualization based on spin box values."""
        if self.current_actor is None:
            return
        min_val = self.scalar_min_spin.value()
        max_val = self.scalar_max_spin.value()
        self.current_actor.mapper.SetScalarRange(min_val, max_val)
        self.plotter.render()

    def _update_step_spinbox_state(self, text):
        """Enable/disable the step spinbox based on the selected time step mode."""
        if text == "Actual Data Time Steps":
            self.custom_step_spin.setVisible(False)
            self.actual_interval_spin.setVisible(True)
        else:
            self.custom_step_spin.setVisible(True)
            self.actual_interval_spin.setVisible(False)

    def update_visualization(self):
        """Update plotter with current settings"""
        if not self.current_mesh:
            return

        # Store current camera state before clearing
        self.camera_state = {
            'position': self.plotter.camera.position,
            'focal_point': self.plotter.camera.focal_point,
            'view_up': self.plotter.camera.up,
            'view_angle': self.plotter.camera.view_angle
        }

        self.plotter.clear()
        self.data_column = self.current_mesh.array_names[0] if self.current_mesh.array_names else None

        self.current_actor = self.plotter.add_mesh(
            self.current_mesh,
            scalars=self.data_column,
            cmap='jet',  # Changed colormap to 'jet' to mimic ANSYS Mechanical
            point_size=self.point_size.value(),
            render_points_as_spheres=True,
            below_color='gray',
            above_color='magenta',
            scalar_bar_args={
                'title': self.data_column,
                'fmt': '%.4f',
                'position_x': 0.04,  # Left edge (5% from left)
                'position_y': 0.35,  # Vertical position (35% from bottom)
                'width': 0.05,  # Width of the scalar bar (5% of window)
                'height': 0.5,  # Height of the scalar bar (50% of window)
                'vertical': True,  # Force vertical orientation
                'title_font_size': 14,
                'label_font_size': 12,
                'shadow': True,  # Optional: Add shadow for readability
                'n_labels': 10  # Number of labels to display
            }
        )

        # Re-create the 'Go To Node' markers if a node is being tracked ---
        # This ensures the markers persist after the plotter is cleared for a time point update.
        if self.target_node_index is not None:
            try:
                # Get the current coordinates for the tracked node from the newly loaded mesh
                point_coords = self.current_mesh.points[self.target_node_index]

                # Re-create the marker actor
                self.marker_poly = pv.PolyData([point_coords])
                self.target_node_marker_actor = self.plotter.add_points(
                    self.marker_poly,
                    color='black',
                    point_size=self.point_size.value() * 2,
                    render_points_as_spheres=True,
                    opacity=0.3,
                )

                # Re-create the dynamic anchor point and the label actor
                self.label_point_data = pv.PolyData([point_coords])
                self.target_node_label_actor = self.plotter.add_point_labels(
                    self.label_point_data, [f"Node {self.target_node_id}"],
                    name="target_node_label",
                    font_size=16, text_color='red',
                    always_visible=True, show_points=False
                )
            except IndexError:
                # This can happen if the new mesh has fewer points than the old one.
                # In this case, the tracked index is invalid, so we clear the markers.
                print(f"Warning: Tracked node index {self.target_node_index} is out of bounds for the new mesh. Clearing marker.")
                self._clear_goto_node_markers()

        self._setup_hover_annotation()

        # Restore camera state if available
        if self.camera_state:
            self.plotter.camera.position = self.camera_state['position']
            self.plotter.camera.focal_point = self.camera_state['focal_point']
            self.plotter.camera.up = self.camera_state['view_up']
            self.plotter.camera.view_angle = self.camera_state['view_angle']

        # Ensure the camera widget is re-enabled if it was removed.
        if not self.camera_widget:
            self.camera_widget = self.plotter.add_camera_orientation_widget()
            self.camera_widget.EnabledOn()

    def _setup_hover_annotation(self):
        """Set up hover callback to display node ID and value"""
        if not self.current_mesh or 'NodeID' not in self.current_mesh.array_names:
            return

        # Clean up previous hover elements
        self._clear_hover_elements()

        # Create new annotation
        self.hover_annotation = self.plotter.add_text(
            "", position='upper_right', font_size=8,
            color='black', name='hover_annotation'
        )

        # Create picker and callback with throttling
        picker = vtk.vtkPointPicker() # pylint: disable=no-member
        picker.SetTolerance(0.01)

        def hover_callback(obj, event):
            now = time.time()
            if (now - self.last_hover_time) < 0.033:  # 30 FPS throttle
                return

            iren = obj
            pos = iren.GetEventPosition()
            picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
            point_id = picker.GetPointId()

            if point_id != -1 and point_id < self.current_mesh.n_points:
                node_id = self.current_mesh['NodeID'][point_id]
                value = self.current_mesh[self.data_column][point_id]
                self.hover_annotation.SetText(2, f"Node ID: {node_id}\n{self.data_column}: {value:.5f}")
            else:
                self.hover_annotation.SetText(2, "")

            iren.GetRenderWindow().Render()
            self.last_hover_time = now

        # Add and track new observer
        self.hover_observer = self.plotter.iren.add_observer('MouseMoveEvent', hover_callback)

    def _clear_hover_elements(self):
        """Dedicated hover element cleanup"""
        if self.hover_annotation:
            self.plotter.remove_actor(self.hover_annotation)
            self.hover_annotation = None

        if self.hover_observer:
            self.plotter.iren.remove_observer(self.hover_observer)
            self.hover_observer = None

    def update_point_size(self):
        """
        Handles dynamic point size updates efficiently by modifying the actor directly
        while also correctly re-initializing hover annotations. This avoids clearing
        the entire scene.
        """
        # We need a mesh and an actor to be present to do anything
        if self.current_mesh and self.current_actor:
            # 1. Clear the old hover annotations and their observers
            self._clear_hover_elements()

            # 2. Directly modify the properties of the existing actor
            new_size = self.point_size.value()
            self.current_actor.prop.point_size = new_size
            self.current_actor.prop.render_points_as_spheres = True

            # 3. Re-create the hover annotations for the updated plot
            self._setup_hover_annotation()

            # 4. Render the changes to the screen.
            self.plotter.render()

    def _clear_visualization(self):
        """Properly clear existing visualization"""
        self.stop_animation()
        self._clear_hover_elements()
        self._clear_goto_node_markers()

        # Manually disable and remove the box widget if it exists
        if self.box_widget:
            self.box_widget.Off()
            self.box_widget = None

        if self.camera_widget:
            self.camera_widget.EnabledOff()
            self.camera_widget = None

        self.plotter.clear()
        if self.current_mesh:
            self.current_mesh.clear_data()
            self.current_mesh = None

        self.current_actor = None
        self.scalar_min_spin.clear()
        self.scalar_max_spin.clear()

        self.file_path.clear()

    def show_context_menu(self, position):
        """Creates and displays the right-click context menu."""
        # Do nothing if the scene is empty – prevents right-click menu entirely
        if self.current_mesh is None:
            return

        context_menu = QMenu(self)

        context_menu.setStyleSheet("""
            QMenu {
                background-color: #e7f0fd;      /* Main background - matches buttons */
                color: black;                   /* Text color */
                border: 1px solid #5b9bd5;      /* Border color - matches group boxes */
                border-radius: 5px;             /* Rounded corners */
                padding: 5px;                   /* Padding around the whole menu */
            }
            QMenu::item {
                background-color: transparent;  /* Make items transparent by default */
                padding: 5px 25px 5px 20px;     /* Set padding for each item */
                margin: 2px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #cce4ff;      /* Highlight color on hover - matches button hover */
                color: black;
            }
            QMenu::item:disabled {
                color: #808080;                 /* Gray text color for disabled items */
                background-color: transparent;  /* Ensure it has no background */
            }
            QMenu::separator {
                height: 1px;
                background-color: #5b9bd5;      /* Color of the separator line */
                margin: 5px 0px;                /* Space above and below the line */
            }
        """)

        title_style = """
            font-weight: bold; color: #333; 
            text-decoration: underline; padding: 4px 0px 6px 7px;
        """

        # Title for the Selection Tools (such as Box) Group
        box_title_label = QLabel("Selection Tools")
        box_title_label.setStyleSheet(title_style)
        box_title_action = QWidgetAction(context_menu)
        box_title_action.setDefaultWidget(box_title_label)
        context_menu.addAction(box_title_action)

        # Add/Remove Box
        if self.box_widget is None:
            box_action_text = "Add Selection Box"
        else:
            box_action_text = "Remove Selection Box"
        toggle_box_action = QAction(box_action_text, self)
        toggle_box_action.triggered.connect(self.toggle_selection_box)
        context_menu.addAction(toggle_box_action)

        # Pick Center
        pick_action = QAction("Pick Box Center", self)
        pick_action.setCheckable(True)
        pick_action.setChecked(self.is_point_picking_active)
        pick_action.setEnabled(self.current_mesh is not None)
        pick_action.triggered.connect(self.toggle_point_picking_mode)
        context_menu.addAction(pick_action)

        context_menu.addSeparator()

        # Title for Hotspot Analysis
        hotspot_title_label = QLabel("Hotspot Analysis")
        hotspot_title_label.setStyleSheet(title_style)
        hotspot_title_action = QWidgetAction(context_menu)
        hotspot_title_action.setDefaultWidget(hotspot_title_label)
        context_menu.addAction(hotspot_title_action)

        # Action for finding hotspots on the whole view
        hotspot_action = QAction("Find Hotspots (on current view)", self)
        hotspot_action.setEnabled(self.current_mesh and self.current_mesh.active_scalars is not None)
        hotspot_action.triggered.connect(self._find_hotspots_on_view)
        context_menu.addAction(hotspot_action)

        # Find in Box
        find_in_box_action = QAction("Find Hotspots in Selection", self)
        find_in_box_action.setEnabled(self.box_widget is not None)
        find_in_box_action.triggered.connect(self.find_hotspots_in_box)
        context_menu.addAction(find_in_box_action)

        context_menu.addSeparator()

        # Title for Point-Based Analysis
        point_analysis_title_label = QLabel("Point-Based Analysis")
        point_analysis_title_label.setStyleSheet(title_style)
        point_analysis_title_action = QWidgetAction(context_menu)
        point_analysis_title_action.setDefaultWidget(point_analysis_title_label)
        context_menu.addAction(point_analysis_title_action)

        plot_point_history_action = QAction("Plot Time History for Selected Node", self)
        plot_point_history_action.triggered.connect(self.enable_time_history_picking)
        context_menu.addAction(plot_point_history_action)

        context_menu.addSeparator()

        # Title for View Control
        view_title_label = QLabel("View Control")
        view_title_label.setStyleSheet(title_style)
        view_title_action = QWidgetAction(context_menu)
        view_title_action.setDefaultWidget(view_title_label)
        context_menu.addAction(view_title_action)

        # Go To Node action
        go_to_node_action = QAction("Go To Node", self)
        # Determine if the action should be enabled
        has_mesh_and_node_ids = (self.current_mesh is not None and
                                 'NodeID' in self.current_mesh.array_names)

        is_animation_running_and_frozen = (self.anim_timer is not None and
                                           self.anim_timer.isActive() and
                                           self.freeze_tracked_node)

        can_go_to_node = has_mesh_and_node_ids and not is_animation_running_and_frozen

        go_to_node_action.setEnabled(can_go_to_node)

        go_to_node_action.triggered.connect(self.go_to_node)
        context_menu.addAction(go_to_node_action)

        # Lock Camera action
        freeze_action = QAction("Lock Camera for Animation (freeze node)", self)
        freeze_action.setCheckable(True)
        freeze_action.setChecked(self.freeze_tracked_node)
        freeze_action.setEnabled(self.target_node_index is not None)
        freeze_action.triggered.connect(self.toggle_freeze_node)
        context_menu.addAction(freeze_action)

        # Reset Camera action
        reset_camera_action = QAction("Reset Camera", self)
        reset_camera_action.triggered.connect(self.plotter.reset_camera)
        context_menu.addAction(reset_camera_action)

        context_menu.exec_(self.plotter.mapToGlobal(position))

    def _find_and_show_hotspots(self, mesh_to_analyze):
        """A helper function to run hotspot analysis on a given mesh."""
        if not mesh_to_analyze or mesh_to_analyze.n_points == 0:
            QMessageBox.information(self, "No Nodes Found", "No nodes were found in the selected area.")
            return

        # Ask user for Top N
        num_hotspots, ok = QInputDialog.getInt(self, "Number of Hotspots", "How many top nodes to find?", 10, 1, 1000)
        if not ok:
            return

        # Get data from the provided mesh
        try:
            node_ids = mesh_to_analyze['NodeID']
            scalar_values = mesh_to_analyze.active_scalars
            scalar_name = mesh_to_analyze.active_scalars_name
            if scalar_name is None:
                scalar_name = "Result"

            df = pd.DataFrame({'NodeID': node_ids, scalar_name: scalar_values})
            df_hotspots = df.sort_values(by=scalar_name, ascending=False).head(num_hotspots).copy()
            df_hotspots.insert(0, 'Rank', range(1, 1 + len(df_hotspots)))
            df_hotspots.reset_index(drop=True, inplace=True)

            # If a dialog is already open, close it before creating a new one
            if self.hotspot_dialog is not None:
                self.hotspot_dialog.close()

            # Create and launch the dialog
            dialog = HotspotDialog(df_hotspots, self)
            dialog.node_selected.connect(self._highlight_and_focus_on_node)
            dialog.finished.connect(self._cleanup_hotspot_analysis)  # Clean up when closed

            if self.box_widget is not None:
                self.box_widget.Off() # Disable the widget to lock its position and size

            self.hotspot_dialog = dialog
            self.hotspot_dialog.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to find hotspots: {e}")

    def _find_hotspots_on_view(self):
        """Finds hotspots only on the points currently visible to the camera."""
        if not self.current_mesh:
            QMessageBox.warning(self, "No Data", "There is no mesh loaded to find hotspots on.")
            return

        # 1. Create the VTK filter for selecting visible points
        selector = vtk.vtkSelectVisiblePoints()

        # 2. Configure the selector with the input mesh and the plotter's renderer
        selector.SetInputData(self.current_mesh)
        selector.SetRenderer(self.plotter.renderer)
        selector.Update()  # Execute the filter

        # 3. Get the result and wrap it as a PyVista PolyData object
        visible_mesh = pv.wrap(selector.GetOutput())

        # Check if any points were actually visible
        if visible_mesh.n_points == 0:
            QMessageBox.information(self, "No Visible Points", "No points are visible in the current camera view.")
            return

        # Pass the new, filtered mesh to your existing analysis function
        self._find_and_show_hotspots(visible_mesh)

    def _highlight_and_focus_on_node(self, node_id):
        if self.current_mesh is None:
            QMessageBox.warning(self, "No Mesh", "Cannot highlight node because no mesh is loaded.")
            return

        # --- THIS IS THE FIX ---
        # 1. If a highlight actor from a previous selection exists, remove it
        #    directly from the VTK renderer.
        if self.highlight_actor:
            self.plotter.renderer.RemoveActor(self.highlight_actor)
            self.highlight_actor = None
        # --- END OF FIX ---

        try:
            # 1. Find the node's index and coordinates (this part is the same)
            node_indices = np.where(self.current_mesh['NodeID'] == node_id)[0]
            if len(node_indices) == 0:
                print(f"Node ID {node_id} not found in the current mesh.")
                return

            point_index = node_indices[0]
            point_coords = self.current_mesh.points[point_index]

            # 2. Create the label text
            label_text = f"Node {node_id}"

            # 3. Add the point label actor instead of a sphere
            #    This creates a visible point and text label at the coordinates.
            self.highlight_actor = self.plotter.add_point_labels(
                point_coords, [label_text],
                name="hotspot_label",
                font_size=16,
                point_color='black',
                shape_opacity=0.5,
                point_size=self.point_size.value() * 2,
                text_color='purple',
                always_visible=True # Ensures the label is not hidden by the mesh
            )

            # 4. Move the camera to focus on the point
            self.plotter.fly_to(point_coords)

        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Could not highlight node {node_id}: {e}")

    def toggle_selection_box(self):
        """Adds or removes the box widget from the plotter."""
        if self.box_widget is None:
            # Add the widget and store a reference to it
            self.box_widget = self.plotter.add_box_widget(callback=self._dummy_callback)

            # Set the initial size of the box to be 75% of the dataset's bounds.
            # A smaller box will have smaller handles.
            self.box_widget.SetPlaceFactor(0.75)

            # Get the property for the handles and change their color
            handle_property = self.box_widget.GetHandleProperty()
            handle_property.SetColor(0.8, 0.4, 0.2) # Set to a less obtrusive orange color
            handle_property.SetPointSize(1)

            # Get the property for the currently selected handle
            selected_handle_property = self.box_widget.GetSelectedHandleProperty()
            selected_handle_property.SetColor(1.0, 0.5, 0.0) # Set to a bright orange when selected

        else:
            # Use the recommended way to remove all widgets of this type
            self.plotter.clear_box_widgets()
            self.box_widget = None
        # We need to render to see the change
        self.plotter.render()

    def find_hotspots_in_box(self):
        """Clips the mesh to the box bounds and runs the hotspot analysis."""
        if self.box_widget is None:
            return  # Should not happen if the menu is disabled, but a good safety check

        # Create a vtk.vtkPolyData object to store the box's geometry
        box_geometry = vtk.vtkPolyData() # pylint: disable=no-member
        # Ask the widget to populate our object with its current geometry
        self.box_widget.GetPolyData(box_geometry)
        # Now, get the bounds from the geometry object, which has the GetBounds() method
        bounds = box_geometry.GetBounds()

        # Clip the main mesh using these bounds
        clipped_mesh = self.current_mesh.clip_box(bounds, invert=False)

        # Call the existing helper function with the clipped mesh
        self._find_and_show_hotspots(clipped_mesh)

    def _dummy_callback(self, *args):
        """A do-nothing callback function to satisfy the widget's requirement."""
        pass

    def toggle_point_picking_mode(self, checked):
        """Toggles the point picking mode on the plotter."""
        self.is_point_picking_active = checked
        if checked:
            # Disables other interactions and sets up our callback
            self.plotter.enable_point_picking(
                callback=self._on_point_picked_for_box,
                show_message=False,  # Don't show the default PyVista message box
                use_picker=True,  # Ensures we pick a point on the mesh
                left_clicking = True
            )
            self.plotter.setCursor(Qt.CrossCursor)  # Give user visual feedback
        else:
            self.plotter.disable_picking()
            self.plotter.setCursor(Qt.ArrowCursor)

    def _on_point_picked_for_box(self, *args):
        """Callback executed when a point is picked on the mesh."""
        # Use *args to robustly handle different PyVista versions
        # Check if args is empty or if the coordinate array has a size of 0
        if not args or args[0].size == 0:
            return

        center = args[0]

        # If the box widget doesn't exist yet, create it now
        if self.box_widget is None:
            self.box_widget = self.plotter.add_box_widget(callback=self._dummy_callback)
            # Apply our custom properties
            self.box_widget.GetHandleProperty().SetColor(0.8, 0.4, 0.2)
            self.box_widget.GetSelectedHandleProperty().SetColor(1.0, 0.5, 0.0)
            self.box_widget.GetHandleProperty().SetPointSize(10)
            self.box_widget.GetSelectedHandleProperty().SetPointSize(15)

            # Define a default size for the new box
            size = self.current_mesh.length * 0.1
            bounds = [
                center[0] - size / 2.0, center[0] + size / 2.0,
                center[1] - size / 2.0, center[1] + size / 2.0,
                center[2] - size / 2.0, center[2] + size / 2.0,
            ]
        else:
            # If the box already exists, get its current size
            box_geometry = vtk.vtkPolyData() # pylint: disable=no-member
            self.box_widget.GetPolyData(box_geometry)
            current_bounds = box_geometry.GetBounds()

            x_size = current_bounds[1] - current_bounds[0]
            y_size = current_bounds[3] - current_bounds[2]
            z_size = current_bounds[5] - current_bounds[4]
            # Calculate new bounds centered on the picked point
            bounds = [
                center[0] - x_size / 2.0, center[0] + x_size / 2.0,
                center[1] - y_size / 2.0, center[1] + y_size / 2.0,
                center[2] - z_size / 2.0, center[2] + z_size / 2.0,
            ]

        # Move the box widget to the new bounds directly
        self.box_widget.PlaceWidget(bounds)
        self.plotter.render()

        # Turn off picking mode after one use
        self.toggle_point_picking_mode(False)

    def _cleanup_hotspot_analysis(self):
        """Removes all highlight labels and re-enables the box widget."""
        # Remove the text label actor
        if hasattr(self, 'highlight_actor') and self.highlight_actor:
            self.plotter.remove_actor("hotspot_label", reset_camera=False)
            self.highlight_actor = None

        self._clear_goto_node_markers()

        # Re-enable the box widget if it still exists
        if self.box_widget:
            self.box_widget.On()

        # Clear the reference to the now-closed dialog
        self.hotspot_dialog = None

        self.plotter.render()

    def enable_time_history_picking(self):
        """Activates point‑history plotting, first offering the tracked node (if any)."""

        # --- 1. Offer tracked node directly -----------------------------------
        if self.target_node_index is not None and self.target_node_id is not None:
            reply = QMessageBox.question(
                self,
                "Use Tracked Node?",
                f"Plot time history for tracked node {self.target_node_id}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.node_picked_signal.emit(self.target_node_id)  # plot immediately
                return
            # If the user clicks “No”, fall through to visual picking

        # --- 2. Fall back to normal picking -----------------------------------
        if not self.current_mesh or 'NodeID' not in self.current_mesh.array_names:
            QMessageBox.warning(self, "No Data",
                                "Cannot pick a point. Please load data with NodeIDs first.")
            return

        print("Picking mode enabled: Click on a node to plot its time history.")
        self.plotter.enable_point_picking(
            callback=self._on_point_picked_for_history,
            show_message=False,
            use_picker=True,
            left_clicking=True
        )
        self.plotter.setCursor(Qt.CrossCursor)

    def _on_point_picked_for_history(self, *args):
        """Callback for when a point is picked. Emits the node ID signal."""
        # Disable picking mode immediately to make it a one-shot action.
        self.plotter.disable_picking()
        self.plotter.setCursor(Qt.ArrowCursor)

        # The picked point's coordinates are passed in the callback's arguments.
        # We must use these arguments to find the closest point index.
        if not args or len(args) == 0 or not isinstance(args[0], (np.ndarray, list, tuple)):
            print("Picking cancelled or missed the mesh.")
            return

        picked_coords = args[0]
        if len(picked_coords) == 0:
            print("Picking cancelled or missed the mesh.")
            return

        # find_closest_point() correctly returns a single integer index.
        picked_point_index = self.current_mesh.find_closest_point(picked_coords)

        # Check if the pick was successful. A value of -1 indicates a miss.
        if picked_point_index != -1 and picked_point_index < self.current_mesh.n_points:
            try:
                # A valid point on the mesh was picked.
                node_id = self.current_mesh['NodeID'][picked_point_index]
                print(f"Node {node_id} picked. Emitting signal...")
                self.node_picked_signal.emit(node_id)
            except (KeyError, IndexError) as e:
                print(f"Could not retrieve NodeID for picked point index {picked_point_index}: {e}")
        else:
            # The user clicked on empty space or the pick was otherwise invalid.
            print("Picking cancelled or missed the mesh.")

    def go_to_node(self):
        """Prompts for a Node ID, sets up dynamic markers, and flies the camera to it."""
        if not self.current_mesh or 'NodeID' not in self.current_mesh.array_names:
            QMessageBox.warning(self, "Action Unavailable", "No mesh with NodeIDs is currently loaded.")
            return

        default_val = self.last_goto_node_id if self.last_goto_node_id is not None else 0
        node_id, ok = QInputDialog.getInt(self, "Go To Node", "Enter Node ID:", value=default_val)
        if not ok:
            return

        try:
            # Clear any previously tracked node first
            self._clear_goto_node_markers()

            node_indices = np.where(self.current_mesh['NodeID'] == node_id)[0]
            if not node_indices.size:
                QMessageBox.warning(self, "Not Found", f"Node ID {node_id} was not found.")
                return

            # Remember node id for a possible re-run of "Go To Node" command
            self.last_goto_node_id = node_id

            # --- Store information for dynamic tracking ---
            self.target_node_index = node_indices[0]
            self.target_node_id = node_id
            point_coords = self.current_mesh.points[self.target_node_index]

            # 1. Create pixel glyph marker
            self.marker_poly = pv.PolyData([point_coords])
            self.target_node_marker_actor = self.plotter.add_points(
                self.marker_poly,
                color='black',
                point_size=self.point_size.value() * 2,  # 2 × current point size
                render_points_as_spheres=True,
                opacity=0.3,
            )

            # 2. Create a dynamic anchor point for the label
            self.label_point_data = pv.PolyData([point_coords])

            # 3. Create the label using the dynamic anchor
            self.target_node_label_actor = self.plotter.add_point_labels(
                self.label_point_data, [f"Node {self.target_node_id}"],
                name="target_node_label",
                font_size=16, text_color='red',
                always_visible=True, show_points=False
            )

            self.plotter.fly_to(point_coords)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not go to node {node_id}: {e}")

    def _clear_goto_node_markers(self):
        """Removes all actors and resets state for the 'Go To Node' feature."""
        if self.target_node_marker_actor:
            self.plotter.remove_actor(self.target_node_marker_actor)
        if self.target_node_label_actor:
            self.plotter.remove_actor(self.target_node_label_actor)

        self.target_node_index = None
        self.target_node_id = None
        self.marker_poly = None
        self.target_node_marker_actor = None
        self.target_node_label_actor = None

        self.freeze_tracked_node = False
        self.freeze_baseline = None

    def toggle_freeze_node(self, checked: bool):
        """
        Activates / de‑activates the mesh‑translation trick that keeps
        the tracked node visually fixed.
        """
        if self.target_node_index is None or self.current_mesh is None:
            QMessageBox.warning(self, "No Tracked Node",
                                "Use 'Go To Node' first, then lock the camera.")
            return

        self.freeze_tracked_node = checked
        if checked:
            # store reference position AFTER the mesh has already been updated
            self.freeze_baseline = self.current_mesh.points[
                self.target_node_index].copy()

            # Hide the label and marker, ONLY if animation is active
            if self.anim_timer is not None and self.anim_timer.isActive():
                if self.target_node_label_actor:
                    self.target_node_label_actor.SetVisibility(False)
                if self.target_node_marker_actor:
                    self.target_node_marker_actor.SetVisibility(False)
        else: # If unchecked (unfreezing)
            self.freeze_baseline = None

            # Always show the label and marker when unfreezing, regardless of animation state
            if self.target_node_label_actor:
                self.target_node_label_actor.SetVisibility(True)
            if self.target_node_marker_actor:
                self.target_node_marker_actor.SetVisibility(True)

        self.plotter.render()  # Render the changes to visibility??

    @pyqtSlot(object, str, float, float)
    def update_view_with_results(self, mesh, scalar_bar_title, data_min, data_max):
        """Receives a calculated mesh and updates the PyVista view."""
        print("DisplayTab: Received calculated results. Updating view.")

        # Show or hide the APDL export button based on the result type
        if "Velocity" in scalar_bar_title:
            self.extract_ic_button.setVisible(True)
        else:
            self.extract_ic_button.setVisible(False)

        # 1. Update the scalar range spin boxes
        self.scalar_min_spin.blockSignals(True)
        self.scalar_max_spin.blockSignals(True)
        self.scalar_min_spin.setRange(data_min, data_max)
        self.scalar_max_spin.setRange(data_min, 1e30)
        self.scalar_min_spin.setValue(data_min)
        self.scalar_max_spin.setValue(data_max)
        self.scalar_min_spin.blockSignals(False)
        self.scalar_max_spin.blockSignals(False)

        # 2. Update the visualization
        self.current_mesh = mesh
        self.data_column = scalar_bar_title
        self.update_visualization()
        self.file_path.clear()

    @pyqtSlot(object)
    def on_animation_data_ready(self, precomputed_data):
        """Receives the precomputed animation data and starts the playback timer."""
        QApplication.restoreOverrideCursor()  # Restore cursor

        if precomputed_data is None:
            print("Animation precomputation failed or was cancelled. See console for details.")
            self.stop_animation()  # Reset UI
            return

        print("DisplayTab: Received precomputed animation data. Starting playback.")

        # Unpack the data
        self.precomputed_scalars, self.precomputed_coords, self.precomputed_anim_times, self.data_column_name, self.is_deformation_included_in_anim = precomputed_data

        # Set/Refresh the legend title before the animation begins.
        if hasattr(self.plotter, 'scalar_bar') and self.plotter.scalar_bar:
            self.plotter.scalar_bar.SetTitle(self.data_column_name)

        # If a node is being tracked, its actors were cleared by stop_animation.
        # We must re-create them here before the first frame is rendered.
        if self.target_node_index is not None:
            try:
                # Use the coordinates from the very first animation frame
                point_coords = self.precomputed_coords[self.target_node_index, :, 0] if self.precomputed_coords is not None else self.current_mesh.points[self.target_node_index]

                self.marker_poly = pv.PolyData([point_coords])
                self.target_node_marker_actor = self.plotter.add_points(
                    self.marker_poly,
                    color='black',
                    point_size=self.point_size.value() * 2,
                    render_points_as_spheres=True,
                    opacity=0.3
                )

                self.label_point_data = pv.PolyData([point_coords])
                self.target_node_label_actor = self.plotter.add_point_labels(
                    self.label_point_data, [f"Node {self.target_node_id}"],
                    name="target_node_label",
                    font_size=16, text_color='red',
                    always_visible=True, show_points=False
                )

                # If the user had chosen to lock camera, keep both actors hidden
                if self.freeze_tracked_node:
                    if self.target_node_marker_actor:
                        self.target_node_marker_actor.SetVisibility(False)
                    if self.target_node_label_actor:
                        self.target_node_label_actor.SetVisibility(False)
                    self.plotter.render()

            except IndexError:
                print("Warning: Could not re-create tracked node marker for animation.")
                self._clear_goto_node_markers()

        # --- Start the animation playback ---
        self.current_anim_frame_index = 0
        self.animation_paused = False

        try:
            self._animate_frame(update_index=False)  # Render the first frame
        except Exception as e:
            QMessageBox.critical(self, "Animation Error", f"Failed initial frame render: {str(e)}")
            self.stop_animation()
            return

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_frame)
        self.anim_timer.start(self.anim_interval_spin.value())

        # Update UI state
        self.deformation_scale_edit.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.save_anim_button.setEnabled(True)

    def __del__(self):
        """Ensure proper cleanup"""
        self._clear_visualization()


class HotspotDialog(QDialog):
    """
    A dialog window to display and interact with hotspot analysis results.

    This dialog presents a table of nodes with the highest scalar values,
    allowing the user to click on a specific node to highlight it in the
    main 3D visualization window.
    """
    # Signal to be emitted when a node is selected from the table
    # It will carry the integer Node ID.
    node_selected = pyqtSignal(int)

    def __init__(self, hotspot_df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hotspot Analysis Results")
        self.setMinimumSize(300, 300)

        self.table_view = QTableView()
        self.model = QStandardItemModel(self)
        self.table_view.setModel(self.model)

        # Make the table non-editable and select whole rows at a time
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)

        # Populate the table with the data
        self._populate_table(hotspot_df)

        # When a row is clicked, trigger our handler
        self.table_view.clicked.connect(self._on_row_clicked)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Click a row to navigate to the node in the Display tab."))
        layout.addWidget(self.table_view)
        self.setLayout(layout)

    def _populate_table(self, df):
        """Populates the table, formatting floats to 4 decimal places."""
        self.model.setHorizontalHeaderLabels(df.columns)

        for index, row in df.iterrows():
            items = []
            for col_name, val in row.items():
                # Keep Rank and NodeID as integers
                if col_name in ['Rank', 'NodeID']:
                    items.append(QStandardItem(str(int(float(val)))))
                # Format all other columns as floats with 4 decimal places
                else:
                    items.append(QStandardItem(f"{val:.4f}"))
            self.model.appendRow(items)

        self.table_view.resizeColumnsToContents()

    def _on_row_clicked(self, index):
        # Get the row of the clicked cell
        row = index.row()
        # Assume 'NodeID' is the second column (index 1)
        node_id_item = self.model.item(row, 1)
        if node_id_item:
            node_id = int(float(node_id_item.text()))
            # Emit the signal with the node ID
            self.node_selected.emit(node_id)
# endregion

# TODO - Lock Camera should work for Update Single Time Point as well
