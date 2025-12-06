import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False  # 녹화 여부
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):    # S 키: 녹화 토글 
        if not recording:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter("output.avi", fourcc, 30.0, (w, h))
            recording = True
            print("녹화 시작")
        else:
            recording = False
            out.release()
            print("녹화 종료")
    if recording:    # 녹화 중이면 프레임 저장
        out.write(frame)
    if key == ord('q'):
        break
if recording and out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()
