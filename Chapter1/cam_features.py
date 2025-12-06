import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 2. 특징점 검출기 생성
orb = cv2.ORB_create(nfeatures=500)
sift = cv2.SIFT_create(nfeatures=300) 
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) Harris 코너
    gray_f = np.float32(gray)
    harris = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    harris_vis = frame.copy()
    harris_vis[harris > 0.01 * harris.max()] = [0, 0, 255]

    # 2) ORB 특징점
    kp_orb, des_orb = orb.detectAndCompute(gray, None)
    orb_vis = cv2.drawKeypoints(frame, kp_orb, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    # 3) SIFT 특징점
    kp_sift, des_sift = sift.detectAndCompute(gray, None)

    sift_vis = cv2.drawKeypoints(frame, kp_sift, None, color=(255, 0, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Original", frame)
    cv2.imshow("Harris Corners", harris_vis)
    cv2.imshow("ORB Keypoints", orb_vis)
    cv2.imshow("SIFT Keypoints", sift_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

