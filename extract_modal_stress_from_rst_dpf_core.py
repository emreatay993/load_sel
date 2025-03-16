# region Import Libraries
from ansys.dpf import core as dpf
import os
import csv
import sys
import traceback
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QLabel, QCheckBox, QHBoxLayout,
                             QComboBox, QPushButton, QFileDialog, QMessageBox)
# endregion

# region Global functions and classes
class NamedSelectionDialog(QDialog):
    def __init__(self, list_of_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Named Selection or Import Nodes")
        self.setMinimumSize(500, 200)
        self.file_path = None

        layout = QVBoxLayout()

        # Checkbox for file import
        self.file_checkbox = QCheckBox("Get node list from exported text file")
        self.file_checkbox.stateChanged.connect(self.toggle_file_input)
        layout.addWidget(self.file_checkbox)

        # File selection widgets
        self.file_layout = QHBoxLayout()
        self.file_label = QLabel("Selected file:")
        self.file_display = QLabel("None")
        self.file_button = QPushButton("Browse...")
        self.file_button.clicked.connect(self.select_file)
        self.file_layout.addWidget(self.file_label)
        self.file_layout.addWidget(self.file_display)
        self.file_layout.addWidget(self.file_button)
        self.file_layout.setEnabled(False)
        layout.addLayout(self.file_layout)

        # Named selection dropdown
        self.ns_label = QLabel("Select a Named Selection:")
        layout.addWidget(self.ns_label)

        self.combo = QComboBox()
        for name in list_of_names:
            self.combo.addItem(name)
        layout.addWidget(self.combo)

        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        layout.addWidget(self.ok_btn)

        self.setLayout(layout)

    def toggle_file_input(self, state):
        enable_file = state == 2  # Qt.Checked
        self.file_layout.setEnabled(enable_file)
        self.combo.setEnabled(not enable_file)
        self.ns_label.setEnabled(not enable_file)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Node List File", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.file_path = file_path
            self.file_display.setText(os.path.basename(file_path))


def show_message(message, title="Information"):
    msg = QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.setIcon(QMessageBox.Information)
    msg.exec_()


def show_error(message, title="Error"):
    msg = QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.setIcon(QMessageBox.Critical)
    msg.exec_()


def show_exception():
    exc_type, exc_value, exc_tb = sys.exc_info()
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    show_error(error_message, "An error occurred")
# endregion

# Initialize Qt application
app = QApplication(sys.argv)

try:
    # Get RST file through Qt dialog
    rst_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Result File",
        "",
        "Result Files (*.rst);;All Files (*)"
    )

    if not rst_path:
        show_message("No file selected. Exiting.")
        sys.exit()

    folder = os.path.dirname(rst_path)
    filename = os.path.basename(rst_path)

    # region DPF Operators Setup
    op_SX = dpf.operators.result.stress_X()
    op_SY = dpf.operators.result.stress_Y()
    op_SZ = dpf.operators.result.stress_Z()
    op_SXY = dpf.operators.result.stress_XY()
    op_SYZ = dpf.operators.result.stress_YZ()
    op_SXZ = dpf.operators.result.stress_XZ()
    # endregion

    # Define Data Source Object
    data_src = dpf.DataSources(rst_path)

    # region Time Scoping
    time_scoping = dpf.Scoping()
    model = dpf.Model(rst_path)
    number_sets = model.metadata.time_freq_support.n_sets
    time_scoping.ids = range(1, number_sets + 1)
    # endregion

    # region Named Selection Handling
    list_of_names = model.metadata.available_named_selections

    if not list_of_names:
        show_error("No named selections found!")
        sys.exit()

    ns_dialog = NamedSelectionDialog(list_of_names)
    if not ns_dialog.exec_():
        show_message("No named selection selected. Exiting.")
        sys.exit()

    # Handle either file import or named selection
    if ns_dialog.file_checkbox.isChecked() and ns_dialog.file_path:
        # Read node IDs from text file
        try:
            with open(ns_dialog.file_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                headers = next(reader)

                # Find Node Number column
                if "Node Number" not in headers:
                    raise ValueError("File missing 'Node Number' column")

                node_idx = headers.index("Node Number")
                IDs_of_NS_of_selected_nodes = [int(row[node_idx]) for row in reader]

        except Exception as e:
            show_error(f"Error reading node file: {str(e)}")
            sys.exit()
    else:
        # Use traditional named selection
        selected_NS = ns_dialog.combo.currentText()
        obj_of_NS_of_selected_nodes = model.metadata.named_selection(selected_NS)
        IDs_of_NS_of_selected_nodes = obj_of_NS_of_selected_nodes.ids

    my_mesh_scoping = dpf.Scoping(ids=IDs_of_NS_of_selected_nodes)
    my_mesh_scoping.location = dpf.locations.nodal
    # endregion

    # region Operator Connections
    operators = [op_SX, op_SY, op_SZ, op_SXY, op_SYZ, op_SXZ]
    for op in operators:
        op.inputs.data_sources.connect(data_src)
        op.inputs.time_scoping.connect(time_scoping)
        op.inputs.mesh_scoping.connect(my_mesh_scoping)
    # endregion

    # region Coordinate Extraction
    node_coords = {}
    try:
        # Use time set 1 to obtain node IDs.
        field_tmp = op_SX.outputs.fields_container.get_data().get_field_by_time_id(1)
        my_node_ids = field_tmp.scoping.ids
        mesh = model.metadata.meshed_region
        for nid in my_node_ids:
            if my_node_ids is not None:
                node = mesh.nodes.node_by_id(nid)
                node_coords[nid] = (node.coordinates[0], node.coordinates[1], node.coordinates[2])
            else:
                node_coords[nid] = (None, None, None)
                print(f"Node {nid} not found in mesh")
    except Exception as e:
        show_error(f"Coordinate extraction failed: {str(e)}")
        node_coords = None
    # endregion

    # region CSV Output
    # We will build a dictionary (one row per node) that contains all stress values
    # with headers concatenated by mode (i.e., _Mode1, _Mode2, etc.)
    stress_data = { comp: { nid: [] for nid in my_node_ids } for comp in ['SX', 'SY', 'SZ', 'SXY', 'SYZ', 'SXZ'] }

    # Loop through each time scoping id (mode) one by one
    for time_id in range(1, number_sets + 1):
        # Create a new scoping for the current time set only.
        single_time_scoping = dpf.Scoping()
        single_time_scoping.ids = [time_id]

        # Reconnect operators with the current single time scoping.
        for op in operators:
            op.inputs.time_scoping.connect(single_time_scoping)

        # region Data Collection for current time set
        # For the current mode, get the stress fields from each operator.
        stress_components = {
            'SX': op_SX.outputs.fields_container.get_data(),
            'SY': op_SY.outputs.fields_container.get_data(),
            'SZ': op_SZ.outputs.fields_container.get_data(),
            'SXY': op_SXY.outputs.fields_container.get_data(),
            'SYZ': op_SYZ.outputs.fields_container.get_data(),
            'SXZ': op_SXZ.outputs.fields_container.get_data()
        }
        # endregion

        # Retrieve field data for each stress component at the current mode.
        field_SX = stress_components['SX'].get_field_by_time_id(time_id)
        field_SY = stress_components['SY'].get_field_by_time_id(time_id)
        field_SZ = stress_components['SZ'].get_field_by_time_id(time_id)
        field_SXY = stress_components['SXY'].get_field_by_time_id(time_id)
        field_SYZ = stress_components['SYZ'].get_field_by_time_id(time_id)
        field_SXZ = stress_components['SXZ'].get_field_by_time_id(time_id)

        # Assume node ordering is the same for each field.
        node_ids = field_SX.scoping.ids

        # Append the stress values for each node for this mode.
        for idx, nid in enumerate(node_ids):
            stress_data['SX'][nid].append(field_SX.data[idx])
            stress_data['SY'][nid].append(field_SY.data[idx])
            stress_data['SZ'][nid].append(field_SZ.data[idx])
            stress_data['SXY'][nid].append(field_SXY.data[idx])
            stress_data['SYZ'][nid].append(field_SYZ.data[idx])
            stress_data['SXZ'][nid].append(field_SXZ.data[idx])

    # Build a pandas DataFrame where each row corresponds to a node.
    # The columns are: NodeID, X, Y, Z, then for each mode, the stress components with suffix _ModeX.
    rows = []
    for nid in my_node_ids:
        row = {"NodeID": nid}
        if node_coords:
            row["X"], row["Y"], row["Z"] = node_coords.get(nid, (None, None, None))
        else:
            row["X"], row["Y"], row["Z"] = None, None, None
        # For each mode, add the stress values.
        for mode in range(1, number_sets + 1):
            row[f'sx_Mode{mode}'] = stress_data['SX'][nid][mode - 1]
            row[f'sy_Mode{mode}'] = stress_data['SY'][nid][mode - 1]
            row[f'sz_Mode{mode}'] = stress_data['SZ'][nid][mode - 1]
            row[f'sxy_Mode{mode}'] = stress_data['SXY'][nid][mode - 1]
            row[f'syz_Mode{mode}'] = stress_data['SYZ'][nid][mode - 1]
            row[f'sxz_Mode{mode}'] = stress_data['SXZ'][nid][mode - 1]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(folder, "modal_stress.csv"), index=False)
    show_message(f"Results successfully saved to:\n{os.path.join(folder, 'modal_stress.csv')}")
    # endregion

except Exception as e:
    show_exception()
    sys.exit(1)  # Error exit
else:
    sys.exit(0)  # Success exit
