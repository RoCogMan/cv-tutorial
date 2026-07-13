import open3d as o3d
import numpy as np
import time

# 0
N = 100_000
ITERS = 2

# 1
points = np.random.rand(N, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
query = np.array([0.5, 0.5, 0.5])

# 2
def brute_force_knn(points, query):
    diff = points - query
    dist2 = np.sum(diff*diff, axis=1)
    return np.argmin(dist2)

# 3
start = time.perf_counter()
for _ in range(ITERS):
    idx = brute_force_knn(points, query)
brute_avg = (time.perf_counter() - start) / ITERS


# 4
kdtree = o3d.geometry.KDTreeFlann(pcd)
start = time.perf_counter()
for _ in range(ITERS):
    k, idx, dist = kdtree.search_knn_vector_3d(query, 1)
kdtree_avg = (time.perf_counter() - start) / ITERS

#
print("====== 정확한 시간 비교 (평균) ======")
print(f"Points: {N:,}")
print(f"Iterations: {ITERS}")
print(f"\nBrute Force: {brute_avg * 1e6:.2f} us ({brute_avg:.6f} sec)")
print(f"KD-Tree: {kdtree_avg * 1e6:.2f} us ({kdtree_avg:.6f} sec)")
print(f"\nSpeed-up: {brute_avg / kdtree_avg:.1f}x faster")
