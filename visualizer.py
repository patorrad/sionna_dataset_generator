import numpy as np
import matplotlib.pyplot as plt
from data_manipulations import denormalize
data = np.load('data_lunar_mesh_ex_no_diffuse.npy')
print(data.shape)
# Testing denormalizing and normalizing data
viz_data = data[:,:,7:10].reshape((960 * 19, 3)) # data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
rssi_predicted = np.abs(data[:,:,0].reshape((960 * 19, 1))) # np.abs(denormalize(viz_data[:190, 0], -90, 0) )
pos = viz_data[:,:]

import trimesh
# mesh = trimesh.load_mesh("models/lunar_mesh_ex.ply")
mesh = trimesh.load_mesh("models/meshes_512/small_mesh0.ply")
# Extract Z heights of vertices
z_vals = mesh.vertices[:, 2]

# Normalize the height values to [0, 1]
z_min, z_max = z_vals.min(), z_vals.max()
z_norm = (z_vals - z_min) / (z_max - z_min)

# Choose a colormap (e.g., viridis)
cmap = plt.get_cmap('viridis')
colors = cmap(z_norm)[:, :3]  # RGBA -> RGB

# Convert to 0-255 range
vertex_colors = (colors * 255).astype(np.uint8)

# Assign vertex colors to mesh
mesh.visual.vertex_colors = vertex_colors
# mesh = trimesh.load_mesh("models/canyon.ply")
import numpy as np

radius = 5.0  # small radius for visibility
markers = []
for rssi, pose in zip(rssi_predicted, pos):
    marker = trimesh.creation.icosphere(radius=radius, color=[rssi[0], 0, 1 - rssi[0], 1])  # Normalize rssi to 0-1
    marker.apply_translation(pose.real)
    markers.append(marker)

scene = trimesh.Scene()
# scene.add_geometry(markers)
print(mesh.bounds)

scene.add_geometry(mesh)
scene.show(viewer="gl", resolution=(1000, 800))