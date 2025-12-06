import cv2

# 이미지 불러오기 (그레이스케일)
img = cv2.imread("lenna.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ORB 생성 및 특징점 검출
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

# 특징점 그리기
img_orb = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

cv2.imshow("ORB Keypoints", img_orb)
cv2.waitKey(0)
cv2.destroyAllWindows()