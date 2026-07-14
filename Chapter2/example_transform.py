import open3d as o3d
import numpy as np
from copy import deepcopy

pcd = o3d.io.read_point_cloud("cat.ply")


R = pcd.get_rotation_matrix_from_xyz((0, 0, theta))

T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = [0.5, 0.1, 0.2]

pcd_transformed = deepcopy(pcd).transform(T)

o3d.visualization.draw_geometries([pcd, pcd_transformed])
