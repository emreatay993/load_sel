# region Import Libraries
import mech_dpf
import Ans.DataProcessing as dpf
import os
import csv
import clr
import sys
import traceback
clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")
from System.Windows.Forms import (Application, Form, ComboBox, Button, Label, DialogResult,
                                 MessageBox, MessageBoxButtons, MessageBoxIcon)
from System.Drawing import Size, Point
# endregion

# region Global functions and classes
# Function to handle the ComboBox selection
def on_select(combo, event):
    selected_NS = combo.SelectedItem
    print("Selected Named Selection: %s" % selected_NS)
    global obj_of_NS_of_selected_nodes
    obj_of_NS_of_selected_nodes = DataModel.GetObjectsByName(selected_NS)[0]  # Set selected object here

# Create a Form (GUI) for ComboBox selection
class MyForm(Form):
    def __init__(self, list_of_ns):
        self.Text = "Select Named Selection"
        
        # Set initial size of the form and minimum size
        self.Size = Size(400, 200)  # Wider initial size
        self.MinimumSize = Size(400, 200)  # Minimum size to prevent too much shrinking
        
        # Label
        label = Label()
        label.Text = "Select a Named Selection:"
        label.Location = Point(10, 20)
        label.Width = 350  # Widen the label to fit the text
        self.Controls.Add(label)
        
        # ComboBox
        self.combo = ComboBox()
        self.combo.Location = Point(10, 50)
        self.combo.Width = 360  # Widen the ComboBox to fit the initial form size
        self.combo.Anchor = (
            System.Windows.Forms.AnchorStyles.Top 
            | System.Windows.Forms.AnchorStyles.Left 
            | System.Windows.Forms.AnchorStyles.Right
        )  # Allow the ComboBox to stretch horizontally with form resizing
        
        # Add items to the ComboBox one by one
        for ns in list_of_ns:
            self.combo.Items.Add(ns.Name)
        
        self.combo.SelectedIndexChanged += on_select
        self.Controls.Add(self.combo)
        
        # Button to confirm selection
        btn = Button()
        btn.Text = "OK"
        btn.Location = Point(10, 90)
        btn.Width = 360  # Match button width with ComboBox
        btn.Anchor = (
            System.Windows.Forms.AnchorStyles.Top 
            | System.Windows.Forms.AnchorStyles.Left 
            | System.Windows.Forms.AnchorStyles.Right
        )  # Button also stretches horizontally with form resizing
        btn.Click += self.on_ok
        self.Controls.Add(btn)
        
    def on_ok(self, sender, event):
        if self.combo.SelectedItem:
            self.DialogResult = DialogResult.OK  # Close form when OK is pressed

# Function to handle the GUI of message boxes
def show_message_box(message, title="Message", icon=MessageBoxIcon.Information):
    MessageBox.Show(message, title, MessageBoxButtons.OK, icon)

# Function to handle errors and show them in a message box
def show_exception_in_message_box():
    exc_type, exc_value, exc_tb = sys.exc_info()
    error_message = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    show_message_box(error_message, "An error occurred", MessageBoxIcon.Error)
# endregion

try:
    folder = DataModel.GetObjectsByName("Modal")[0].WorkingDir
    filename = "file.rst"
    filepath = os.path.join(folder, filename)
    
    op_SX  = dpf.operators.result.stress_X()
    op_SY  = dpf.operators.result.stress_Y()
    op_SZ  = dpf.operators.result.stress_Z()
    op_SXY = dpf.operators.result.stress_XY()
    op_SYZ = dpf.operators.result.stress_YZ()
    op_SXZ = dpf.operators.result.stress_XZ()
    
    dataSources = dpf.DataSources()
    dataSources.SetResultFilePath(filepath)
    
    # region Define scoping on selected result point
    time_scoping = dpf.Scoping()
    model = dpf.Model(dataSources)
    
    # Get the number of result sets inside result file (number of modes extracted)
    number_sets = model.TimeFreqSupport.NumberSets
    print("My number sets" + str(number_sets))
    time_scoping.Ids = range(1, number_sets + 1)
    # endregion
    
    # # region Define scoping on a named selection (should exist inside rst already)
    # scoping_on_ns = dpf.operators.scoping.on_named_selection()
    # scoping_on_ns.inputs.requested_location.Connect('Nodal')
    # scoping_on_ns.inputs.named_selection_name.Connect('NS_Modal_Stress')
    # scoping_on_ns.inputs.data_sources.Connect(dataSources)
    # my_mesh_scoping = scoping_on_ns.outputs.mesh_scoping.GetData()
    # # endregion
    
    # Get the list of nodal named selections that exist in the model
    list_of_obj_of_all_NS = DataModel.Project.GetChildren(DataModelObjectCategory.NamedSelection,True)
    # Filter out any NS other than nodal NSs
    list_of_obj_of_all_nodal_NS = [obj for obj in list_of_obj_of_all_NS
                                   if "Node" in obj.PropertyByName("GeometrySelection").StringValue]
    
    # Create and show the GUI
    form = MyForm(list_of_obj_of_all_nodal_NS)
    if form.ShowDialog() == DialogResult.OK:
        # obj_of_NS_of_selected_nodes will be set in the on_select function
        IDs_of_NS_of_selected_nodes = obj_of_NS_of_selected_nodes.Location.Ids
        my_mesh_scoping = dpf.MeshScopingFactory.NodalScoping(IDs_of_NS_of_selected_nodes)
    
    # region Connect input pins
    op_SX.inputs.data_sources.Connect(dataSources)
    op_SX.inputs.time_scoping.Connect(time_scoping)
    op_SX.inputs.mesh_scoping.Connect(my_mesh_scoping)
    
    op_SY.inputs.data_sources.Connect(dataSources)
    op_SY.inputs.time_scoping.Connect(time_scoping)
    op_SY.inputs.mesh_scoping.Connect(my_mesh_scoping)
    
    op_SZ.inputs.data_sources.Connect(dataSources)
    op_SZ.inputs.time_scoping.Connect(time_scoping)
    op_SZ.inputs.mesh_scoping.Connect(my_mesh_scoping)
    
    op_SXY.inputs.data_sources.Connect(dataSources)
    op_SXY.inputs.time_scoping.Connect(time_scoping)
    op_SXY.inputs.mesh_scoping.Connect(my_mesh_scoping)
    
    op_SYZ.inputs.data_sources.Connect(dataSources)
    op_SYZ.inputs.time_scoping.Connect(time_scoping)
    op_SYZ.inputs.mesh_scoping.Connect(my_mesh_scoping)
    
    op_SXZ.inputs.data_sources.Connect(dataSources)
    op_SXZ.inputs.time_scoping.Connect(time_scoping)
    op_SXZ.inputs.mesh_scoping.Connect(my_mesh_scoping)
    # endregion
    
    # region Collect node vs modal stress data
    # Collect Modal Stress vs Nodes data in a Field Container
    my_fields_container_SX = op_SX.outputs.fields_container.GetData()
    my_fields_container_SY = op_SY.outputs.fields_container.GetData()
    my_fields_container_SZ = op_SZ.outputs.fields_container.GetData()
    my_fields_container_SXY = op_SXY.outputs.fields_container.GetData()
    my_fields_container_SYZ = op_SYZ.outputs.fields_container.GetData()
    my_fields_container_SXZ = op_SXZ.outputs.fields_container.GetData()
    
    # Get node IDs from the first field
    node_ids = my_fields_container_SX.GetFieldByTimeId(1).ScopingIds
    
    # Initialize a dictionary to store stress data per node
    stress_data_per_node_SX = {}
    stress_data_per_node_SY = {}
    stress_data_per_node_SZ = {}
    stress_data_per_node_SXY = {}
    stress_data_per_node_SYZ = {}
    stress_data_per_node_SXZ = {}
    
    # Initialize the stress_data_per_node dictionary with node IDs as keys
    for node_id in node_ids:
        stress_data_per_node_SX[node_id]  = []
        stress_data_per_node_SY[node_id]  = []
        stress_data_per_node_SZ[node_id]  = []
        stress_data_per_node_SXY[node_id] = []
        stress_data_per_node_SYZ[node_id] = []
        stress_data_per_node_SXZ[node_id] = []
    
    # Loop over each mode and collect stress data
    for field_no in time_scoping.Ids:
        # Get stress data for the current mode
        field_SX  = my_fields_container_SX.GetFieldByTimeId(field_no)
        field_SY  = my_fields_container_SY.GetFieldByTimeId(field_no)
        field_SZ  = my_fields_container_SZ.GetFieldByTimeId(field_no)
        field_SXY = my_fields_container_SXY.GetFieldByTimeId(field_no)
        field_SYZ = my_fields_container_SYZ.GetFieldByTimeId(field_no)
        field_SXZ = my_fields_container_SXZ.GetFieldByTimeId(field_no)
        
        field_node_ids = field_SX.ScopingIds
        
        stress_values_SX  = field_SX.Data
        stress_values_SY  = field_SY.Data
        stress_values_SZ  = field_SZ.Data
        stress_values_SXY = field_SXY.Data
        stress_values_SYZ = field_SYZ.Data
        stress_values_SXZ = field_SXZ.Data
    
        # Map stress values to node IDs
        for idx, node_id in enumerate(field_node_ids):
            stress_value_SX  = stress_values_SX[idx]
            stress_value_SY  = stress_values_SY[idx]
            stress_value_SZ  = stress_values_SZ[idx]
            stress_value_SXY = stress_values_SXY[idx]
            stress_value_SYZ = stress_values_SYZ[idx]
            stress_value_SXZ = stress_values_SXZ[idx]
            
            stress_data_per_node_SX[node_id].append(stress_value_SX)
            stress_data_per_node_SY[node_id].append(stress_value_SY)
            stress_data_per_node_SZ[node_id].append(stress_value_SZ)
            stress_data_per_node_SXY[node_id].append(stress_value_SXY)
            stress_data_per_node_SYZ[node_id].append(stress_value_SYZ)
            stress_data_per_node_SXZ[node_id].append(stress_value_SXZ)
    # endregion
    
    # region Create output files
    # Prepare CSV header: 'NodeID', 'Mode1', 'Mode2', ..., 'ModeN'
    header_SX = ['NodeID'] + ['Mode%d_SX' % field_no for field_no in time_scoping.Ids]
    header_SY = ['NodeID'] + ['Mode%d_SY' % field_no for field_no in time_scoping.Ids]
    header_SZ = ['NodeID'] + ['Mode%d_SZ' % field_no for field_no in time_scoping.Ids]
    header_SXY = ['NodeID'] + ['Mode%d_SXY' % field_no for field_no in time_scoping.Ids]
    header_SYZ = ['NodeID'] + ['Mode%d_SYZ' % field_no for field_no in time_scoping.Ids]
    header_SXZ = ['NodeID'] + ['Mode%d_SXZ' % field_no for field_no in time_scoping.Ids]
    
    # Path to save the CSV file
    filename_output_csv_SX  = "modal_stress_SX.csv"
    filename_output_csv_SY  = "modal_stress_SY.csv"
    filename_output_csv_SZ  = "modal_stress_SZ.csv"
    filename_output_csv_SXY = "modal_stress_SXY.csv"
    filename_output_csv_SYZ = "modal_stress_SYZ.csv"
    filename_output_csv_SXZ = "modal_stress_SXZ.csv"
    
    filepath_output_csv_SX  = os.path.join(folder, filename_output_csv_SX)
    filepath_output_csv_SY  = os.path.join(folder, filename_output_csv_SY)
    filepath_output_csv_SZ  = os.path.join(folder, filename_output_csv_SZ)
    filepath_output_csv_SXY = os.path.join(folder, filename_output_csv_SXY)
    filepath_output_csv_SYZ = os.path.join(folder, filename_output_csv_SYZ)
    filepath_output_csv_SXZ = os.path.join(folder, filename_output_csv_SXZ)
    
    # Write data to CSV file
    with open(filepath_output_csv_SX, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(header_SX)
    
        # Write data rows
        for node_id in node_ids:
            row = [node_id] + stress_data_per_node_SX[node_id]
            csvwriter.writerow(row)
            
    with open(filepath_output_csv_SY, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(header_SY)
    
        # Write data rows
        for node_id in node_ids:
            row = [node_id] + stress_data_per_node_SY[node_id]
            csvwriter.writerow(row)
            
    with open(filepath_output_csv_SZ, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(header_SZ)
    
        # Write data rows
        for node_id in node_ids:
            row = [node_id] + stress_data_per_node_SZ[node_id]
            csvwriter.writerow(row)
    
    with open(filepath_output_csv_SXY, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(header_SXY)
    
        # Write data rows
        for node_id in node_ids:
            row = [node_id] + stress_data_per_node_SXY[node_id]
            csvwriter.writerow(row)
    
    with open(filepath_output_csv_SYZ, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(header_SYZ)
    
        # Write data rows
        for node_id in node_ids:
            row = [node_id] + stress_data_per_node_SYZ[node_id]
            csvwriter.writerow(row)
    
    with open(filepath_output_csv_SXZ, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(header_SXZ)
    
        # Write data rows
        for node_id in node_ids:
            row = [node_id] + stress_data_per_node_SXZ[node_id]
            csvwriter.writerow(row)
    
    # Path to save the merged CSV file
    filename_output_csv_merged = "modal_stress.csv"
    filepath_output_csv_merged = os.path.join(folder, filename_output_csv_merged)
    
    # Prepare the merged CSV header
    merged_header = ['NodeID']
    merged_header += ['sx_Mode%s' % mode for mode in time_scoping.Ids]
    merged_header += ['sy_Mode%s' % mode for mode in time_scoping.Ids]
    merged_header += ['sz_Mode%s' % mode for mode in time_scoping.Ids]
    merged_header += ['sxy_Mode%s' % mode for mode in time_scoping.Ids]
    merged_header += ['syz_Mode%s' % mode for mode in time_scoping.Ids]
    merged_header += ['sxz_Mode%s' % mode for mode in time_scoping.Ids]
    
    # Write merged data to CSV file
    with open(filepath_output_csv_merged, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(merged_header)
    
        # Write data rows
        # Write data rows
        for node_id in node_ids:
            row = [node_id]
            row.extend(stress_data_per_node_SX[node_id])
            row.extend(stress_data_per_node_SY[node_id])
            row.extend(stress_data_per_node_SZ[node_id])
            row.extend(stress_data_per_node_SXY[node_id])
            row.extend(stress_data_per_node_SYZ[node_id])
            row.extend(stress_data_per_node_SXZ[node_id])
            csvwriter.writerow(row)
    
    show_message_box("Modal stress data has been successfully saved to \n %s" % filepath_output_csv_merged)
    
    # Delete output files temporarily created
    os.remove(filepath_output_csv_SX)
    os.remove(filepath_output_csv_SY)
    os.remove(filepath_output_csv_SZ)
    os.remove(filepath_output_csv_SXY)
    os.remove(filepath_output_csv_SYZ)
    os.remove(filepath_output_csv_SXZ)

except:
    show_exception_in_message_box()
