# Import libraries
import os
import csv
import mech_dpf
import Ans.DataProcessing as dpf

# Define the solution path
modal_solution = DataModel.GetObjectsByName("Modal")[0].Solution
obj_of_NS = DataModel.GetObjectsByName("NS_Modal_Expansion")[0]

folder = modal_solution.WorkingDir
filename = "file.rst"
filepath = os.path.join(folder, filename)
data_src = dpf.DataSources()
data_src.SetResultFilePath(filepath)
model = dpf.Model(data_src)

# Get the number of modes inside the rst file
num_modes = model.TimeFreqSupport.NumberSets

# ==================== Modal Directional Deformation Extraction ====================
# Create the Directional Deformation object using AddDirectionalDeformation()
directional_deformation = modal_solution.AddDirectionalDeformation()
directional_deformation.ScopingMethod = GeometryDefineByType.Component
directional_deformation.Location = obj_of_NS

# Define deformation orientations (using the same NormalOrientationType values for X, Y, and Z)
deformation_orientations = [
    ("XAxis", NormalOrientationType.XAxis),
    ("YAxis", NormalOrientationType.YAxis),
    ("ZAxis", NormalOrientationType.ZAxis)
]

# Use Mode=1, Orientation=XAxis to obtain the unit for deformations
directional_deformation.SetNumber = 1
directional_deformation.NormalOrientation = NormalOrientationType.XAxis
modal_solution.EvaluateAllResults()

deformation_unit = directional_deformation.PlotData.Dependents["Values"].Unit
if deformation_unit == "m":
    deformation_conversion = 1e3  # Convert meters to millimeters (adjust as needed)
else:
    deformation_conversion = 1.0

# Dictionary to store deformation results for each mode and orientation
directional_deformation_results = {}

# Loop through modes and orientations to extract deformations in X, Y, and Z directions
for mode in range(1, num_modes + 1):
    directional_deformation.SetNumber = mode
    for orientation_name, orientation_type in deformation_orientations:
        directional_deformation.NormalOrientation = orientation_type
        modal_solution.EvaluateAllResults()
        values = [v * deformation_conversion for v in directional_deformation.PlotData.Dependents["Values"]]
        directional_deformation_results[(mode, orientation_name)] = values

# -------------------- Extract Node IDs and Coordinates --------------------
# Get node IDs from the directional deformation independents
IDs_of_NS_of_selected_nodes = [node for node in directional_deformation.PlotData.Independents['Node']]

my_mesh_scoping = dpf.Scoping(ids=IDs_of_NS_of_selected_nodes)
my_mesh_scoping.Location = dpf.locations.nodal

node_coords = {}
try:
    for nid in IDs_of_NS_of_selected_nodes:
        node = model.Mesh.NodeById(nid)
        if node is not None:
            node_coords[nid] = (node.X, node.Y, node.Z)
        else:
            node_coords[nid] = (None, None, None)
            print("Node {} not found in mesh".format(nid))
except Exception as e:
    print("Coordinate extraction failed: {}".format(str(e)))
    node_coords = None

# -------------------- Write Modal Directional Deformation Results to CSV --------------------
deformation_csv_path = os.path.join(modal_solution.WorkingDir, "modal_directional_deformation_w_coords.csv")
num_rows = len(directional_deformation_results[(1, "XAxis")])  # Assumes same number of nodes for all result lists

with open(deformation_csv_path, mode='wb') as file:
    writer = csv.writer(file)
    # Write header with node info followed by deformation components for each mode
    header = ["NodeID", "X", "Y", "Z"]
    for mode in range(1, num_modes + 1):
        header.extend([
            "UX_mode{}".format(mode),
            "UY_mode{}".format(mode),
            "UZ_mode{}".format(mode)
        ])
    writer.writerow(header)
    
    # Write each row: node info and deformation values for each mode
    for i in range(num_rows):
        row = []
        node_id = IDs_of_NS_of_selected_nodes[i]
        if node_coords is not None and node_id in node_coords:
            coords = node_coords[node_id]
        else:
            coords = (None, None, None)
        row.extend([node_id, coords[0], coords[1], coords[2]])
        for mode in range(1, num_modes + 1):
            row.append(directional_deformation_results[(mode, "XAxis")][i])
            row.append(directional_deformation_results[(mode, "YAxis")][i])
            row.append(directional_deformation_results[(mode, "ZAxis")][i])
        writer.writerow(row)

print("Directional Deformation CSV file saved at: " + deformation_csv_path)
