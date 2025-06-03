"""
The directory should contain velocities extracted from the step that will be 
the initial condition for nonlinear transient analysis.

The directory should contain the following file names:

veloc_x*.txt
veloc_y*.txt
veloc_z*.txt

"""

import os
import glob

# Find all files starting with "veloc_"
files = glob.glob("veloc_*.txt")

# Store velocity data per axis: {'x': {'567': '2.2', ...}, 'y': {...}, 'z': {...}}
vel_data = {}

for path in files:
    axis = os.path.basename(path).split("_")[1]  # "x", "y", or "z"
    axis_dict = {}

    with open(path, "r") as f:
        header = f.readline().rstrip("\n")
        cols = header.split("\t")
        idx_node = cols.index("Node Number")
        idx_vel = cols.index("Directional Velocity (mm/s)")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(idx_node, idx_vel):
                continue
            node = parts[idx_node].strip()
            vel = parts[idx_vel].strip()
            if node:
                axis_dict[node] = vel

    vel_data[axis] = axis_dict

# Find node IDs present in all three axes
nodes_x = set(vel_data.get("x", {}))
nodes_y = set(vel_data.get("y", {}))
nodes_z = set(vel_data.get("z", {}))
common_nodes = sorted(nodes_x & nodes_y & nodes_z, key=int)

# Write APDL lines: three per node (VELX, VELY, VELZ)
with open("ic_commands.txt", "w") as fout:
    for node in common_nodes:
        fout.write(f"IC,{node},VELX,{vel_data['x'][node]}\n")
        fout.write(f"IC,{node},VELY,{vel_data['y'][node]}\n")
        fout.write(f"IC,{node},VELZ,{vel_data['z'][node]}\n")
