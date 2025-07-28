import numpy as np
from stl import mesh

# Load the STL file
stl_mesh = mesh.Mesh.from_file('twsbr_env/envs/urdf/meshes/Robot_Body.stl')  # Ganti dengan jalur file STL Anda

# Ambil vertices dari mesh
vertices = stl_mesh.points

# Hitung dimensi asli
min_point = np.min(vertices, axis=0)
max_point = np.max(vertices, axis=0)

# Hitung ukuran (dimensi) objek
original_dimensions = max_point - min_point
print(f"Original Dimensions (X, Y, Z): {original_dimensions}")
print(f"STL MESH: {stl_mesh}")
print(f"min: {min_point}")
print(f"max: {max_point}")

