import cv2
import numpy as np
import open3d as o3d

# --- [1]
pcd = o3d.io.read_point_cloud("dataset/00000_cloud.ply")
# --- [2]
fx = 5.8264e+2
fy = 5.8269e+2
cx = 3.1304e+2
cy = 2.3844e+2

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)
# --- [3]
H, W = 480, 640
depth = np.zeros((H, W), dtype=np.float32)

# --- [4]
points = np.asarray(pcd.points)
X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]

valid = Z > 0
X, Y, Z = X[valid], Y[valid], Z[valid]
# --- [5]
u = (X * fx / Z + cx).astype(np.int32)
v = (Y * fy / Z + cy).astype(np.int32)

mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
u, v, Z = u[mask], v[mask], Z[mask]

for ui, vi, zi in zip(u, v, Z):
    if depth[vi, ui] == 0 or zi < depth[vi, ui]:
        depth[vi, ui] = zi

