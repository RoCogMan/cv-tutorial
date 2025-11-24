import open3d as o3d
import numpy as np
import os
import sys

# 아웃라이어 생성
def make_outliers(target, num_outliers=500):
    num_outliers = np.int16(num_outliers)
    
    bbox = target.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound

    outliers = np.random.uniform(low=min_bound - 0.1, high=max_bound + 0.1, size=(num_outliers, 3))

    outlier_pcd = o3d.geometry.PointCloud()
    outlier_pcd.points = o3d.utility.Vector3dVector(outliers)

    target_with_outliers = target + outlier_pcd
    target = target_with_outliers
    
    return target

# 포인트에 perturbation 추가
def add_noise_to_points(target, noise_sigma=0.002):
    points = np.asarray(target.points)
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=points.shape)
    points += noise
    target.points = o3d.utility.Vector3dVector(points)
    return target


mode = None
if len(sys.argv) > 1:
    mode = sys.argv[1].lower()
if len(sys.argv) > 2:
    try:
        param = np.float32(sys.argv[2])
    except ValueError:
        print("두 번째 인자는 숫자여야 합니다.")
        sys.exit(1)


# 포인트 클라우드 로드
current_dir = os.getcwd()
source = o3d.io.read_point_cloud(os.path.join(current_dir, "models", "cat_half.ply"))
target = o3d.io.read_point_cloud(os.path.join(current_dir, "models", "cat.ply"))


if mode == "noise":
    target = add_noise_to_points(target, param)
elif mode == "outlier":
    target = make_outliers(target, param)

source.paint_uniform_color([1.0, 0.0, 0.0])
target.paint_uniform_color([0.0, 0.0, 1.0])

# 포인트 다운샘플링
voxel_size = 0.01
src_down = source.voxel_down_sample(voxel_size)
tgt_down = target.voxel_down_sample(voxel_size)

# 노멀벡터 추출
src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# FPFH 추론
radius_feature = voxel_size * 5

src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    src_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    tgt_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# RANSAC 기반 정합
distance_threshold = voxel_size * 1.5

result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    src_down, tgt_down, src_fpfh, tgt_fpfh, distance_threshold, True,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    4,
    [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ],
    o3d.pipelines.registration.RANSACConvergenceCriteria(8000000, 1000))

print("RANSAC Initial Alignment")
print(result_ransac)
print(result_ransac.transformation)

# ICP 정밀 정합
distance_threshold_icp = voxel_size * 0.4

result_icp = o3d.pipelines.registration.registration_icp(
    src_down, tgt_down, distance_threshold_icp,
    result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

print("ICP refinement")
print(result_icp)
print(result_icp.transformation)

# 시각화
source_temp = source.transform(result_icp.transformation)
o3d.visualization.draw_geometries([source_temp, target])
