import cv2
import numpy as np

# 이미지 불러오기 (그레이스케일)
img = cv2.imread("chess2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris 코너 검출


dst = cv2.dilate(dst, None)  # 강조

# 임계값 이상인 지점에 표시
img_harris = img.copy()
img_harris[dst > 0.01 * dst.max()] = [0, 255, 0]  # 초록색 표시

cv2.imshow("Harris Corners", img_harris)
cv2.waitKey(0)
cv2.destroyAllWindows()
