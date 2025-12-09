import pyrealsense2 as rs
import cv2
import numpy as np

# =========================
# 1. Charuco 보드 정의
# =========================
# 반드시 실제 출력한 보드와 동일하게 맞춰야 함
squaresX     = 5        # 가로 체스 칸 수
squaresY     = 7        # 세로 체스 칸 수
squareLength = 0.04     # 한 칸 크기 [m] (예: 4cm)
markerLength = 0.03     # 마커 한 변 [m] (예: 3cm)

aruco_dict_id = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
board = cv2.aruco.CharucoBoard_create(
    squaresX, squaresY, squareLength, markerLength, aruco_dict
)

# =========================
# 2. RealSense 컬러 스트림 설정
# =========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# 카메라 내부 파라미터 가져오기
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

cameraMatrix = np.array([[intr.fx,      0, intr.ppx],
                         [     0, intr.fy, intr.ppy],
                         [     0,      0,       1]], dtype=np.float32)
# RealSense 왜곡계수 (Brown 모델 기준 k1,k2,p1,p2,k3)
distCoeffs = np.array(intr.coeffs[:5], dtype=np.float32).reshape(-1, 1)

print("cameraMatrix:\n", cameraMatrix)
print("distCoeffs:", distCoeffs.ravel())

print("Press Ctrl+C to stop.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # numpy BGR 이미지로 변환
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # =========================
        # 3. ArUco 마커 검출
        # =========================
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        if ids is None or len(ids) == 0:
            # 마커 없음 → 이 프레임은 스킵
            continue

        # =========================
        # 4. Charuco 코너 보간
        # =========================
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        if charuco_corners is None or charuco_ids is None:
            continue
        if len(charuco_corners) < 4:
            # solvePnP 최소 4점 이상 필요
            continue

        # =========================
        # 5. solvePnP용 3D-2D 매칭
        # =========================
        obj_points = []
        img_points = []

        for corner, cid in zip(charuco_corners, charuco_ids):
            idx = int(cid)  # Charuco 코너 인덱스
            obj_points.append(board.chessboardCorners[idx])  # (3,)
            img_points.append(corner[0])                    # (2,)

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # =========================
        # 6. solvePnP로 pose 추정
        # =========================
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_points,
            imagePoints=img_points,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            continue

        # 매 프레임 pose 출력
        print("rvec:", rvec.ravel(), "  tvec:", tvec.ravel())

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    pipeline.stop()

