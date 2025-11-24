import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import sys
import cv2

roi = None
drawing = False
ix, iy = -1, -1
color_preview = None

# 마우스로 드래그하여 ROI 지정
def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, roi, color_preview

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img2 = color_preview.copy()
            cv2.rectangle(img2, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("ROI Select", img2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        print("Selected ROI:", roi)


# RealSense 초기 세팅
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intr.width, intr.height,
    intr.fx, intr.fy,
    intr.ppx, intr.ppy
)

camera_intrinsic = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])


# ROI 영역을 PointCloud로 변환
def frame_to_pointcloud_roi(color_frame, depth_frame, depth_scale, roi):
    x, y, w, h = roi

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

    depth[:y, :] = 0.0
    depth[y+h:, :] = 0.0
    depth[:, :x] = 0.0
    depth[:, x+w:] = 0.0

    o3d_color = o3d.geometry.Image(color)
    o3d_depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=1.0,
        depth_trunc=1.5,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return pcd


# FPFH 구하기
def preprocess_point_cloud(pcd_down, voxel_size):
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pcd_down.orient_normals_to_align_with_direction(np.array([0, 0, 1]))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh


# FPFH, RANSAC을 이용해 초기 포즈 추정
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2.0
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh, True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.9)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000)
    )
    return result


# 6D 포즈 시각화
def draw_axes_cv2(img, pose, K, length=0.05):
    R = pose[:3, :3]
    t = pose[:3, 3]
    axes = np.array([[length, 0, 0], [0, length, 0], [0, 0, length]]).T
    pts = R @ axes + t.reshape(3, 1)

    def proj(pt):
        return (int(K[0, 2] + K[0, 0] * pt[0] / (pt[2]+1e-6)),
                int(K[1, 2] + K[1, 1] * pt[1] / (pt[2]+1e-6)))

    origin = proj(t)
    cv2.line(img, origin, proj(pts[:, 0]), (0, 0, 255), 2)
    cv2.line(img, origin, proj(pts[:, 1]), (0, 255, 0), 2)
    cv2.line(img, origin, proj(pts[:, 2]), (255, 0, 0), 2)
    return img


# 바운딩 박스의 꼭지점 구하기
def get_3d_bbox_points(min_pt, max_pt):
    return np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]]
    ])

# 바운딩 박스의 꼭지점을 이미지에 사영하여 ROI 업데이트
def project_bbox_to_roi(global_pose, K, bbox_min, bbox_max):
    pts_3d = get_3d_bbox_points(bbox_min, bbox_max)
    pts_3d = (global_pose[:3, :3] @ pts_3d.T + global_pose[:3, 3:4]).T

    pts_2d = []
    for p in pts_3d:
        if p[2] <= 0.05:
            continue
        u = int(K[0, 0] * p[0] / p[2] + K[0, 2])
        v = int(K[1, 1] * p[1] / p[2] + K[1, 2])
        pts_2d.append((u, v))

    if len(pts_2d) == 0:
        return None

    xs = [p[0] for p in pts_2d]
    ys = [p[1] for p in pts_2d]

    x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    return (x, y, w, h)

# 로우패스필터 적용
def low_pass_pose_filter(prev_pose, new_pose, alpha=0.5):
    r_prev, _ = cv2.Rodrigues(prev_pose[:3, :3])
    r_new, _ = cv2.Rodrigues(new_pose[:3, :3])

    r_filt = (1 - alpha) * r_prev + alpha * r_new
    R_filt, _ = cv2.Rodrigues(r_filt)

    t_prev = prev_pose[:3, 3]
    t_new = new_pose[:3, 3]
    t_filt = (1 - alpha) * t_prev + alpha * t_new

    pose_filt = np.eye(4)
    pose_filt[:3, :3] = R_filt
    pose_filt[:3, 3] = t_filt
    return pose_filt


if len(sys.argv) > 1:
    try:
        param = np.float32(sys.argv[1])
    except ValueError:
        print("두 번째 인자는 숫자여야 합니다.")
        sys.exit(1)

# 고양이 모델 로드
voxel_size = 0.005
model = o3d.io.read_point_cloud("models/cat_half.ply")
model_down = model.voxel_down_sample(voxel_size)
model_down.paint_uniform_color([0.7, 0.7, 0.7])
model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    model_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
)

# 모델 바운딩박스 구하기
bbox = model_down.get_axis_aligned_bounding_box()
bbox_min = np.asarray(bbox.min_bound)
bbox_max = np.asarray(bbox.max_bound)

global_pose = np.eye(4)
initialized = False
auto_roi = False

cv2.namedWindow("ROI Select")
cv2.setMouseCallback("ROI Select", mouse_callback)

# 메인 루프
try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        color_img = np.asanyarray(color.get_data())

        # 첫 프레임에 마우스로 ROI 선택
        if not auto_roi:
            color_preview = color_img.copy()
            cv2.imshow("ROI Select", color_preview)

            if roi is None:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
        
        # 이후 포즈를 기반으로 자동 ROI 업데이트
        else:
            new_roi = project_bbox_to_roi(global_pose, camera_intrinsic, bbox_min, bbox_max)
            if new_roi is not None:
                roi = new_roi

        # ROI 영역 PointCloud 생성
        curr_pcd = frame_to_pointcloud_roi(color, depth, depth_scale, roi)
        curr_down = curr_pcd.voxel_down_sample(voxel_size)
        curr_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )

        # 초기 포즈를 RANSAC으로 구하기
        if not initialized:
            curr_down, curr_fpfh = preprocess_point_cloud(curr_down, voxel_size)
            result_ransac = execute_global_registration(
                model_down, curr_down, model_fpfh, curr_fpfh, voxel_size
            )
            global_pose = result_ransac.transformation
            initialized = True
            auto_roi = True     
            print("Initial Pose Set (RANSAC).")

        # ICP 트래킹 실행
        result_icp = o3d.pipelines.registration.registration_icp(
            model_down, curr_down,
            voxel_size * 0.5,
            global_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        new_pose = result_icp.transformation
        
        if not np.array_equal(global_pose, np.eye(4)):
            global_pose = low_pass_pose_filter(global_pose, new_pose, param)
        
        # 6D 포즈 시각화
        img_axes = draw_axes_cv2(color_img.copy(), global_pose, camera_intrinsic)
        cv2.imshow("6DoF Pose Visualization", img_axes)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except:
    print("Tracking failed.")

