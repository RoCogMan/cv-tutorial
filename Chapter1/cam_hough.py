import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("영상을 가져올 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 캐니 엣지 검출
    edges = cv2.Canny(gray, 50, 150)

    # 허프 변환으로 직선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,minLineLength=50, maxLineGap=10)

    # 검출된 직선 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Hough Lines', frame)
    cv2.imshow('Edges', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()