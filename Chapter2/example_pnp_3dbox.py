import pyrealsense2 as rs
import cv2
import numpy as np
from target_detector import TargetDetector


def draw_board_box(img, rvec, tvec, detector,
                   height=0.03, color=(0,255,0), thickness=2):
    img = img.copy()

    nx, ny = detector.board.getChessboardSize()
    s = detector.board.getSquareLength()

    # 바닥면 4점
    p0 = np.array([0,           0,        0], dtype=np.float32)
    p1 = np.array([(nx)*s,    0,        0], dtype=np.float32)
    p2 = np.array([(nx)*s, (ny)*s,    0], dtype=np.float32)
    p3 = np.array([0,        (ny)*s,    0], dtype=np.float32)

    # 윗면 4점
    p4 = p0 + np.array([0, 0, -height], dtype=np.float32)
    p5 = p1 + np.array([0, 0, -height], dtype=np.float32)
    p6 = p2 + np.array([0, 0, -height], dtype=np.float32)
    p7 = p3 + np.array([0, 0, -height], dtype=np.float32)

    obj_points = np.array([p0,p1,p2,p3,p4,p5,p6,p7], dtype=np.float32)

    img_points, _ = cv2.projectPoints(
        obj_points, rvec, tvec, detector.cam_mat, detector.dist_coeffs
    )
    img_points = img_points.reshape(-1,2).astype(int)

    P0,P1,P2,P3,P4,P5,P6,P7 = img_points

    # 바닥면
    cv2.polylines(img, [np.array([P0,P1,P2,P3])], True, color, thickness)
    # 윗면
    cv2.polylines(img, [np.array([P4,P5,P6,P7])], True, color, thickness)

    # 연결선
    cv2.line(img, P0, P4, color, thickness)
    cv2.line(img, P1, P5, color, thickness)
    cv2.line(img, P2, P6, color, thickness)
    cv2.line(img, P3, P7, color, thickness)

    return img


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cam_mat = np.array([[intr.fx, 0, intr.ppx],
                        [0, intr.fy, intr.ppy],
                        [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

    detector = TargetDetector(cam_mat=cam_mat, dist_coeffs=dist_coeffs)

    print("Press ESC to quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frame = frames.get_color_frame()
            if not frame:
                continue

            img = np.asanyarray(frame.get_data())

            result = detector.estimate_pose(img, draw=True)
            if result is None:
                cv2.imshow("Charuco", img)
                if cv2.waitKey(1) == 27: break
                continue

            pose, vis_img = result
            rvec, tvec = pose.rvec, pose.tvec

            # 박스 높이 = 3cm
            vis_img = draw_board_box(vis_img, rvec, tvec, detector, height=0.03)

            cv2.imshow("Charuco", vis_img)
            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
