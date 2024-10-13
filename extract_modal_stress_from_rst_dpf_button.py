import mech_dpf
import Ans.DataProcessing as dpf
import os
import csv

folder = DataModel.GetObjectsByName("Modal")[0].WorkingDir
filename = "file.rst"
filepath = os.path.join(folder, filename)

op = dpf.operators.result.stress_XZ()

dataSources = dpf.DataSources()
dataSources.SetResultFilePath(filepath)

# region Define scoping on selected result point
time_scoping = dpf.Scoping()
model = dpf.Model(dataSources)

# Get the number of result sets inside result file (number of modes extracted)
number_sets = model.TimeFreqSupport.NumberSets
time_scoping.Ids = range(1, number_sets + 1)
# endregion

# # region Define scoping on a named selection (should exist inside rst already)
# scoping_on_ns = dpf.operators.scoping.on_named_selection()
# scoping_on_ns.inputs.requested_location.Connect('Nodal')
# scoping_on_ns.inputs.named_selection_name.Connect('NS_Modal_Stress')
# scoping_on_ns.inputs.data_sources.Connect(dataSources)
# my_mesh_scoping = scoping_on_ns.outputs.mesh_scoping.GetData()
# # endregion

# region Define scoping on a named selection (should exist inside rst already)
obj_of_NS_of_selected_nodes = DataModel.GetObjectsByName("NS_Modal_Expansion")[0]
IDs_of_NS_of_selected_nodes = obj_of_NS_of_selected_nodes.Location.Ids
my_mesh_scoping = dpf.MeshScopingFactory.NodalScoping(IDs_of_NS_of_selected_nodes)
# endregion

# region Connect input pins
op.inputs.data_sources.Connect(dataSources)
op.inputs.time_scoping.Connect(time_scoping)
op.inputs.mesh_scoping.Connect(my_mesh_scoping)
# endregion

# Collect Modal Stress vs Nodes data in a Field Container
my_fields_container = op.outputs.fields_container.GetData()

# Get node IDs from the first field (assuming node IDs are consistent across modes)
node_ids = my_fields_container.GetFieldByTimeId(1).ScopingIds

# Initialize a dictionary to store stress data per node
stress_data_per_node = {}

# Initialize the stress_data_per_node dictionary with node IDs as keys
for node_id in node_ids:
    stress_data_per_node[node_id] = []

# Loop over each mode and collect stress data
for field_no in time_scoping.Ids:
    # Get stress data for the current mode
    field = my_fields_container.GetFieldByTimeId(field_no)
    stress_values = field.Data
    field_node_ids = field.ScopingIds

    # Map stress values to node IDs
    for idx, node_id in enumerate(field_node_ids):
        stress_value = stress_values[idx]
        stress_data_per_node[node_id].append(stress_value)

# Prepare CSV header: 'NodeID', 'Mode1', 'Mode2', ..., 'ModeN'
header = ['NodeID'] + ['Mode%d' % field_no for field_no in time_scoping.Ids]

# Path to save the CSV file
filename_output_csv = "modal_stress.csv"
filepath_output_csv = os.path.join(folder, filename_output_csv)

# Write data to CSV file
with open(filepath_output_csv, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(header)

    # Write data rows
    for node_id in node_ids:
        row = [node_id] + stress_data_per_node[node_id]
        csvwriter.writerow(row)

print("Modal stress data has been successfully saved to %s" % filepath_output_csv)
