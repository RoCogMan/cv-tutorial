import pyrealsense2 as rs
import cv2
import numpy as np
from target_detector import TargetDetector


def draw_board_bbox(img, rvec, tvec, detector, color=(0,255,0), thickness=2):
    """
    Charuco 보드의 외곽 bounding box를 이미지에 그리는 함수
    """
    img = img.copy()

    nx, ny = detector.board.getChessboardSize()
    s = detector.board.getSquareLength()

    # 보드 네 꼭짓점 (Charuco 좌표계 기준)
    obj_corners_3d = np.array([
        [0,           0,        0],
        [nx*s,    0,        0],
        [nx*s, ny*s,    0],
        [0,        ny*s,    0]
    ])

    # 투영
    corners_2d, _ = cv2.projectPoints(
        obj_corners_3d, rvec, tvec, detector.cam_mat, detector.dist_coeffs
    )
    corners_2d = corners_2d.reshape(-1,2).astype(int)

    # 선 연결하여 박스 그림
    cv2.polylines(img, [corners_2d], True, color, thickness)
    return img


def main():
    # RealSense 컬러 스트림
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # 카메라 intrinsic
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    cam_mat = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1],
    ], dtype=np.float32)

    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

    detector = TargetDetector(cam_mat=cam_mat, dist_coeffs=dist_coeffs)

    print("Press Ctrl+C to stop")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            # pose 추정 + charuco 시각화
            result = detector.estimate_pose(img, draw=True)
            if result is None:
                cv2.imshow("Charuco", img)
                if cv2.waitKey(1) == 27:
                    break
                continue

            pose, vis_img = result
            rvec = pose.rvec
            tvec = pose.tvec










    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
