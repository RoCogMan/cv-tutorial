import cv2

# 이미지 불러오기 (그레이스케일)
img = cv2.imread("lenna.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT 생성 및 특징점 검출
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 특징점 그리기
img_sift = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

cv2.imshow("SIFT Keypoints", img_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()