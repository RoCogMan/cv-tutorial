import pyrealsense2 as rs
import cv2
import numpy as np
from target_detector import TargetDetector


def main():
    # 1. RealSense 파이프라인 설정 (컬러만)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # 2. 컬러 카메라 intrinsic 가져오기
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    cam_mat = np.array(
        [
            [intr.fx, 0,        intr.ppx],
            [0,       intr.fy,  intr.ppy],
            [0,       0,        1       ],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

    print("cameraMatrix:\n", cam_mat)
    print("distCoeffs:", dist_coeffs.ravel())

    # 3. TargetDetector 생성 
    detector = TargetDetector(cam_mat=cam_mat, dist_coeffs=dist_coeffs)

    print("Press Ctrl+C to stop")

    try:
        while True:
            # 4. 프레임 수신
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())  # BGR 이미지

            # 5. Charuco pose 추정 (그림 안 그리고 pose만)
            result = detector.estimate_pose(img, draw=False)
            if result is None:
                continue

            pose, _ = result
            rvec = pose.rvec.ravel()
            tvec = pose.tvec.ravel()
            print(f"rvec: {rvec}, tvec: {tvec}")

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
