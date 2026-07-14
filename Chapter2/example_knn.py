import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("cat.ply")

pcd_tree = o3d.geometry.KDTreeFlann(pcd)

query_indices = [1200, 1400, 2500, 4000, 9000]
query_points = np.array(pcd.points)[query_indices]

all_neighbors = []
query_clouds = []
neighbor_clouds = []

for q in query_points:
    k = 200


    all_neighbors.append(idx)

    # 쿼리 포인트 (빨강)
    qc = o3d.geometry.PointCloud()
    qc.points = o3d.utility.Vector3dVector([q])
    qc.paint_uniform_color([1, 0, 0])
    query_clouds.append(qc)

    # 이웃 포인트 (초록)
    nc = pcd.select_by_index(idx)
    nc.paint_uniform_color([0, 1, 0])
    neighbor_clouds.append(nc)

o3d.visualization.draw_geometries([pcd] + neighbor_clouds + query_clouds)
