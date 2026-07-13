import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
mesh.compute_vertex_normals()

coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
o3d.visualization.draw_geometries([mesh, coord])

