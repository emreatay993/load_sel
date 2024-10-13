import mech_dpf
import Ans.DataProcessing as dpf
import os
import csv

folder = DataModel.GetObjectsByName("Modal")[0].WorkingDir
filename = "file.rst"
filepath = os.path.join(folder, filename)

# Define stress operators manually
stress_operators = {
    "sx": dpf.operators.result.stress_X(),
    "sy": dpf.operators.result.stress_Y(),
    "sz": dpf.operators.result.stress_Z(),
    "sxy": dpf.operators.result.stress_XY(),
    "syz": dpf.operators.result.stress_YZ(),
    "sxz": dpf.operators.result.stress_XZ()
}

dataSources = dpf.DataSources()
dataSources.SetResultFilePath(filepath)

# region Define scoping on selected result point
time_scoping = dpf.Scoping()
model = dpf.Model(dataSources)

# Get the number of result sets inside result file (number of modes extracted)
number_sets = model.TimeFreqSupport.NumberSets
time_scoping.Ids = range(1, number_sets + 1)
# endregion

# region Define scoping on a named selection (should exist inside rst already)
obj_of_NS_of_selected_nodes = DataModel.GetObjectsByName("NS_Modal_Expansion")[0]
IDs_of_NS_of_selected_nodes = obj_of_NS_of_selected_nodes.Location.Ids
my_mesh_scoping = dpf.MeshScopingFactory.NodalScoping(IDs_of_NS_of_selected_nodes)
# endregion

# region Connect input pins
for op in stress_operators.values():
    op.inputs.data_sources.Connect(dataSources)
    op.inputs.time_scoping.Connect(time_scoping)
    op.inputs.mesh_scoping.Connect(my_mesh_scoping)
# endregion

# Collect Modal Stress vs Nodes data in a Field Container
fields_containers = {comp: op.outputs.fields_container.GetData() for comp, op in stress_operators.items()}

# Get node IDs from the first field (assuming node IDs are consistent across modes)
node_ids = fields_containers["sx"].GetFieldByTimeId(1).ScopingIds

# Initialize a dictionary to store stress data per node
stress_data_per_node = {node_id: {comp: [] for comp in stress_components} for node_id in node_ids}

# Loop over each mode and collect stress data
for field_no in time_scoping.Ids:
    for comp in stress_components:
        field = fields_containers[comp].GetFieldByTimeId(field_no)
        field_node_ids = field.ScopingIds
        stress_values = field.Data

        # Map stress values to node IDs
        for idx, node_id in enumerate(field_node_ids):
            stress_data_per_node[node_id][comp].append(stress_values[idx])

# Prepare CSV header: 'NodeID', 'SX_Mode1', 'SY_Mode1', ..., 'SX_ModeN', 'SY_ModeN', etc.
csv_header = ["NodeID"]
for field_no in time_scoping.Ids:
    for comp in stress_components:
        csv_header.append(comp + "_Mode" + str(field_no))

# Path to save the combined CSV file
combined_csv_filename = "modal_stress.csv"
combined_csv_filepath = os.path.join(folder, combined_csv_filename)

# Write combined data to a single CSV file
with open(combined_csv_filepath, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(csv_header)

    # Write data rows
    for node_id in node_ids:
        row = [node_id]
        for mode_idx in range(len(time_scoping.Ids)):
            for comp in stress_components:
                row.append(stress_data_per_node[node_id][comp][mode_idx])
        csvwriter.writerow(row)

print("Combined modal stress data has been successfully saved to " + combined_csv_filepath)
