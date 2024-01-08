import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Directory containing the TIFF files
directory = 'data/train/kidney_3_sparse/labels'

# List all TIFF files in the directory
tiff_files = sorted([f for f in os.listdir(directory) if f.endswith('.tif')])

# Load the images and convert them to numpy arrays
print("Loading images...")
images = [io.imread(os.path.join(directory, file)).astype(np.uint8) for file in tiff_files]

# Inspecting the first image dimensions and data type
first_image_shape = images[0].shape
first_image_dtype = images[0].dtype

first_image_shape, first_image_dtype, tiff_files

# Convert each image to a binary array and stack them to form a 3D array
print("Converting to binary...")
binary_images = [np.where(image > 0, 1, 0).astype(np.uint8) for image in images]  # Convert to binary
voxel_grid = np.stack(binary_images, axis=0)  # Stack along a new axis

# Delete images to save space
del images
del binary_images

# # Visualizing the 3D structure
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Plotting each voxel
print("Plotting...")
x, y, z = voxel_grid.nonzero()
# ax.scatter(z, x, y, zdir='z', c='blue', marker='.', alpha=0.5)

# # Setting labels and title
# ax.set_xlabel('Z Axis')
# ax.set_ylabel('X Axis')
# ax.set_zlabel('Y Axis')
# ax.set_title('3D Voxel Visualization')

# # Show plot
# plt.show()


from skimage.measure import marching_cubes
import open3d as o3d

# Generate a mesh using marching cubes algorithm
print("Applying marching cubes on the voxel grid...")
verts, faces, _, _ = marching_cubes(voxel_grid, level=0)

# Delete the voxel grid to save space
del voxel_grid

# Creating Open3D point cloud from non-zero voxel points
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))

# Creating Open3D mesh from the vertices and faces
print("Creating mesh...")
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)

# File paths for export
point_cloud_file = 'point_cloud_3_sparse.ply'
mesh_file = 'mesh_3_sparse.obj'

# Export point cloud and mesh
print(f"Exporting point cloud to {point_cloud_file}...")
o3d.io.write_point_cloud(point_cloud_file, point_cloud)
print(f"Exporting mesh to {mesh_file}...")
o3d.io.write_triangle_mesh(mesh_file, mesh)

print("Done!")
point_cloud_file, mesh_file
