import open3d as o3d

dataset = o3d.data.DemoICPPointClouds()
pcd = o3d.io.read_point_cloud(dataset.paths[0])
o3d.visualization.draw_geometries([pcd])

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
a, b, c, d = plane_model
print(f"Plane: {a:.4f} x + {b:.4f} y + {c:.4f} z + {d:.4f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

inlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])
outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])