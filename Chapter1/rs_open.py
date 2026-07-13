import cv2 # OpenCV 라이브러리 가져오기

import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
pipeline.start()

while True:
    frames = pipeline.wait_for_frames()
    frame = frames.get_color_frame()
    frame = np.asanyarray(frame.get_data())
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Camera", frame)  # 프레임을 창에 표시

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()  # 창 닫기
