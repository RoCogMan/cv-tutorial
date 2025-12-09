import open3d as o3d

# 1. PLY 파일 읽기
pcd = o3d.io.read_point_cloud("output.ply")

# 2. 시각화
o3d.visualization.draw_geometries([pcd])

