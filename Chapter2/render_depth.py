import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
mesh.compute_vertex_normals()

coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
o3d.visualization.draw_geometries([mesh, coord])

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(mesh)

ctr = vis.get_view_control()
ctr.set_lookat([0, 0, 0])
ctr.set_front([-1, 0, 1])
ctr.set_zoom(2.0)
vis.poll_events()
depth = vis.capture_depth_float_buffer()

o3d.visualization.draw_geometries([o3d.geometry.Image(np.asarray(depth))])