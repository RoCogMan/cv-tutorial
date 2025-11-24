import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 1. YOLO 모델 로드 (학습된 best.pt)
model = YOLO("runs/detect/train/weights/best.pt")

# 2. RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # 3. 프레임 받아오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # 4. YOLO 추론
        results = model(frame, imgsz=640, conf=0.5, verbose=False)
        annotated = results[0].plot()  # 박스가 그려진 이미지

        # 5. 화면 출력
        cv2.imshow("YOLO RealSense", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
