# Import libraries
import os
import csv
import mech_dpf
import Ans.DataProcessing as dpf

# Define the solution path
modal_solution = DataModel.GetObjectsByName("Modal")[0].Solution
obj_of_NS = DataModel.GetObjectsByName("NS_Modal_Stress")[0]

folder = modal_solution.WorkingDir
filename = "file.rst"
filepath = os.path.join(folder, filename)
data_src = dpf.DataSources()
data_src.SetResultFilePath(filepath)
model = dpf.Model(data_src)

# Get the number of modes inside the rst file
num_modes = model.TimeFreqSupport.NumberSets

# Add Normal Stress and Shear Stress objects
normal_stress = modal_solution.AddNormalStress()
shear_stress = modal_solution.AddShearStress()

# Apply the selected named selection on both objects as scoping
normal_stress.ScopingMethod = GeometryDefineByType.Component
normal_stress.Location = obj_of_NS
shear_stress.ScopingMethod = GeometryDefineByType.Component
shear_stress.Location = obj_of_NS

# Define orientations for Normal Stress
normal_orientations = [
    ("XAxis", NormalOrientationType.XAxis),
    ("YAxis", NormalOrientationType.YAxis),
    ("ZAxis", NormalOrientationType.ZAxis)
]

# Define orientations for Shear Stress
shear_orientations = [
    ("XYAxis", ShearOrientationType.XYPlane),
    ("YZAxis", ShearOrientationType.YZPlane),
    ("XZAxis", ShearOrientationType.XZPlane)
]

# Pick Mode=1, Orientation=XAxis just to get the unit
normal_stress.SetNumber = 1
normal_stress.NormalOrientation = NormalOrientationType.XAxis
modal_solution.EvaluateAllResults()

# Check the unit from PlotData
stress_unit = normal_stress.PlotData.Dependents["Values"].Unit
if stress_unit == "Pa":
    conversion_factor = 1e-6  # Convert Pa -> MPa
else:
    conversion_factor = 1.0

# Dictionaries to store results
normal_stress_results = {}
shear_stress_results = {}

# Loop through modes
for mode in range(1, num_modes + 1):
    # ---------------- Normal Stress -------------------
    normal_stress.SetNumber = mode
    for orientation_name, orientation_type in normal_orientations:
        normal_stress.NormalOrientation = orientation_type
        modal_solution.EvaluateAllResults()
        stress_values_list = [v * conversion_factor for v in normal_stress.PlotData.Dependents["Values"]]
        normal_stress_results[(mode, orientation_name)] = stress_values_list

    # ---------------- Shear Stress -------------------
    shear_stress.SetNumber = mode
    for orientation_name, orientation_type in shear_orientations:
        shear_stress.ShearOrientation = orientation_type
        modal_solution.EvaluateAllResults()
        stress_values_list = [v * conversion_factor for v in shear_stress.PlotData.Dependents["Values"]]
        shear_stress_results[(mode, orientation_name)] = stress_values_list


# -------------------- Extract Node IDs and Coordinates --------------------
# Get node IDs from the Normal Stress independents
IDs_of_NS_of_selected_nodes = [node for node in normal_stress.PlotData.Independents['Node']]

my_mesh_scoping = dpf.Scoping(ids=IDs_of_NS_of_selected_nodes)
my_mesh_scoping.Location = dpf.locations.nodal

node_coords = {}
try:
    # Use the node IDs to obtain coordinates (using time set 1 for example)
    my_node_ids = IDs_of_NS_of_selected_nodes
    for nid in my_node_ids:
        node = model.Mesh.NodeById(nid)
        if node is not None:
            node_coords[nid] = (node.X, node.Y, node.Z)
        else:
            node_coords[nid] = (None, None, None)
            print("Node {} not found in mesh".format(nid))
except Exception as e:
    print("Coordinate extraction failed: {}".format(str(e)))
    node_coords = None


# -------------------- Write all results to CSV --------------------
file_path = modal_solution.WorkingDir + r"\\modal_stress_tensor_w_coords.csv"  

# Determine number of rows (assuming same number of nodes as stress results length)
num_rows = len(normal_stress_results[(1, "XAxis")])  # Assumes same length for all stress result lists

with open(file_path, mode='wb') as file:
    writer = csv.writer(file)
    
    # Write header with node info as the first four columns
    header = ["NodeID", "X", "Y", "Z"]
    for mode in range(1, num_modes + 1):
        # Normal Stress Headers
        header.extend([
            "sx_Mode{}".format(mode),
            "sy_Mode{}".format(mode),
            "sz_Mode{}".format(mode)
        ])
        # Shear Stress Headers
        header.extend([
            "sxy_Mode{}".format(mode),
            "syz_Mode{}".format(mode),
            "sxz_Mode{}".format(mode)
        ])
    writer.writerow(header)
    
    # Write each row: node info first, then stress values for each mode
    for i in range(num_rows):
        row = []
        # Get the node ID from the list (assuming same order as stress results)
        node_id = IDs_of_NS_of_selected_nodes[i]
        # Get node coordinates (X, Y, Z) from our dictionary
        if node_coords is not None and node_id in node_coords:
            coords = node_coords[node_id]
        else:
            coords = (None, None, None)
        row.extend([node_id, coords[0], coords[1], coords[2]])
        
        # Append stress values for each mode in order
        for mode in range(1, num_modes + 1):
            # Normal Stress values
            row.append(normal_stress_results[(mode, "XAxis")][i])
            row.append(normal_stress_results[(mode, "YAxis")][i])
            row.append(normal_stress_results[(mode, "ZAxis")][i])
            # Shear Stress values
            row.append(shear_stress_results[(mode, "XYAxis")][i])
            row.append(shear_stress_results[(mode, "YZAxis")][i])
            row.append(shear_stress_results[(mode, "XZAxis")][i])
        writer.writerow(row)

print("CSV file saved at: " + file_path)
