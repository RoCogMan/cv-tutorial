import cv2
import numpy as np

color = cv2.imread("dataset/00000_colors.png")
depth = cv2.imread("dataset/00000_depth.png", cv2.IMREAD_UNCHANGED)
print("Depth shape:", depth.shape, "dtype:", depth.dtype)

fx = 582
fy = 582
cx = 313
cy = 238
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)

height, width = depth.shape
u, v = np.meshgrid(np.arange(width), np.arange(height))

Z = depth.astype(np.float32) / 1000.0 # mm -> m

X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy

mask = (Z > 0)

color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
colors = color_rgb.reshape(-1, 3)
colors = colors[mask.reshape(-1)] / 255.0

points = np.stack((X, Y, Z), axis=-1)
points = points.reshape(-1, 3)
points = points[mask.reshape(-1)]
print("Valid points:", points.shape)

import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
