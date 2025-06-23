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

# ==================== Modal Stress Extraction ====================
# Create Normal Stress and Shear Stress objects
normal_stress = modal_solution.AddNormalStress()
shear_stress = modal_solution.AddShearStress()

# Apply the selected named selection on both objects as scoping
normal_stress.ScopingMethod = GeometryDefineByType.Component
normal_stress.Location = obj_of_NS
shear_stress.ScopingMethod = GeometryDefineByType.Component
shear_stress.Location = obj_of_NS

# Define orientations for Normal Stress and Shear Stress
normal_orientations = [
    ("XAxis", NormalOrientationType.XAxis),
    ("YAxis", NormalOrientationType.YAxis),
    ("ZAxis", NormalOrientationType.ZAxis)
]

shear_orientations = [
    ("XYAxis", ShearOrientationType.XYPlane),
    ("YZAxis", ShearOrientationType.YZPlane),
    ("XZAxis", ShearOrientationType.XZPlane)
]

# Pick Mode=1, Orientation=XAxis just to obtain the unit for stress
normal_stress.SetNumber = 1
normal_stress.NormalOrientation = NormalOrientationType.XAxis
modal_solution.EvaluateAllResults()

stress_unit = normal_stress.PlotData.Dependents["Values"].Unit
if stress_unit == "Pa":
    conversion_factor = 1e-6  # Convert Pa -> MPa
else:
    conversion_factor = 1.0

# Dictionaries to store stress results
normal_stress_results = {}
shear_stress_results = {}

# Loop over modes to extract normal and shear stresses
for mode in range(1, num_modes + 1):
    # ---------------- Normal Stress -------------------
    normal_stress.SetNumber = mode
    for orientation_name, orientation_type in normal_orientations:
        normal_stress.NormalOrientation = orientation_type
        modal_solution.EvaluateAllResults()
        values = [v * conversion_factor for v in normal_stress.PlotData.Dependents["Values"]]
        normal_stress_results[(mode, orientation_name)] = values

    # ---------------- Shear Stress -------------------
    shear_stress.SetNumber = mode
    for orientation_name, orientation_type in shear_orientations:
        shear_stress.ShearOrientation = orientation_type
        modal_solution.EvaluateAllResults()
        values = [v * conversion_factor for v in shear_stress.PlotData.Dependents["Values"]]
        shear_stress_results[(mode, orientation_name)] = values

# -------------------- Extract Node IDs and Coordinates --------------------
# Get node IDs from the Normal Stress independents
IDs_of_NS_of_selected_nodes = [node for node in normal_stress.PlotData.Independents['Node']]

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

# -------------------- Write Stress Results to CSV --------------------
stress_csv_path = os.path.join(modal_solution.WorkingDir, "modal_stress_tensor_w_coords.csv")
num_rows = len(normal_stress_results[(1, "XAxis")])  # Assumes same length for all result lists

with open(stress_csv_path, mode='wb') as file:
    writer = csv.writer(file)
    # Header: node info followed by stress components for each mode
    header = ["NodeID", "X", "Y", "Z"]
    for mode in range(1, num_modes + 1):
        header.extend([
            "sx_Mode{}".format(mode),
            "sy_Mode{}".format(mode),
            "sz_Mode{}".format(mode),
            "sxy_Mode{}".format(mode),
            "syz_Mode{}".format(mode),
            "sxz_Mode{}".format(mode)
        ])
    writer.writerow(header)
    
    # Write node data and stress values
    for i in range(num_rows):
        row = []
        node_id = IDs_of_NS_of_selected_nodes[i]
        if node_coords is not None and node_id in node_coords:
            coords = node_coords[node_id]
        else:
            coords = (None, None, None)
        row.extend([node_id, coords[0], coords[1], coords[2]])
        for mode in range(1, num_modes + 1):
            row.append(normal_stress_results[(mode, "XAxis")][i])
            row.append(normal_stress_results[(mode, "YAxis")][i])
            row.append(normal_stress_results[(mode, "ZAxis")][i])
            row.append(shear_stress_results[(mode, "XYAxis")][i])
            row.append(shear_stress_results[(mode, "YZAxis")][i])
            row.append(shear_stress_results[(mode, "XZAxis")][i])
        writer.writerow(row)

print("Stress CSV file saved at: " + stress_csv_path)

# ==================== Modal Directional Deformation Extraction ====================
# Create the Directional Deformation object using AddDirectionalDeformation()
directional_deformation = modal_solution.AddDirectionalDeformation()
directional_deformation.ScopingMethod = GeometryDefineByType.Component
directional_deformation.Location = obj_of_NS

# Define deformation orientations (using the same NormalOrientationType values)
deformation_orientations = [
    ("XAxis", NormalOrientationType.XAxis),
    ("YAxis", NormalOrientationType.YAxis),
    ("ZAxis", NormalOrientationType.ZAxis)
]

# Pick Mode=1, Orientation=XAxis to obtain the unit for deformations
directional_deformation.SetNumber = 1
directional_deformation.NormalOrientation = NormalOrientationType.XAxis
modal_solution.EvaluateAllResults()

deformation_unit = directional_deformation.PlotData.Dependents["Values"].Unit
if deformation_unit == "m":
    deformation_conversion = 1e3  # For example, convert meters to millimeters
else:
    deformation_conversion = 1.0

# Dictionary to store directional deformation results
directional_deformation_results = {}

# Loop over modes to extract deformations in X, Y, and Z directions
for mode in range(1, num_modes + 1):
    directional_deformation.SetNumber = mode
    for orientation_name, orientation_type in deformation_orientations:
        directional_deformation.NormalOrientation = orientation_type
        modal_solution.EvaluateAllResults()
        values = [v * deformation_conversion for v in directional_deformation.PlotData.Dependents["Values"]]
        directional_deformation_results[(mode, orientation_name)] = values

# -------------------- Write Directional Deformation Results to CSV --------------------
deformation_csv_path = os.path.join(modal_solution.WorkingDir, "modal_directional_deformation_w_coords.csv")
num_rows = len(directional_deformation_results[(1, "XAxis")])  # Assumes same length for all result lists

with open(deformation_csv_path, mode='wb') as file:
    writer = csv.writer(file)
    # Header: node info followed by deformation components for each mode
    header = ["NodeID", "X", "Y", "Z"]
    for mode in range(1, num_modes + 1):
        header.extend([
            "ux_Mode{}".format(mode),
            "uy_Mode{}".format(mode),
            "uz_Mode{}".format(mode)
        ])
    writer.writerow(header)
    
    # Write node data and deformation values
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
