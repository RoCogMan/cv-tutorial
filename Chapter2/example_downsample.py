import open3d as o3d

pcd = o3d.io.read_point_cloud("cat.ply")

voxel_sizes = [0.001, 0.005, 0.01]

for v in voxel_sizes:
    down = pcd.voxel_down_sample(voxel_size=v)
    print(f"Voxel size {v} -> points: {len(down.points)}")
    o3d.visualization.draw_geometries([down], window_name=f"voxel={v}")