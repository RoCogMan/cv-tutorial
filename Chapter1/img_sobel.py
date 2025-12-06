import cv2
import numpy as np

# 1. 이미지 불러오기 (흑백 변환)
img = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)

# 2. 가우시안 블러 적용 (노이즈 제거)
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)


# 3. Sobel 필터 적용 (x방향, y방향)
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # x 방향
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # y 방향


# 4. 절대값 + 8비트로 변환
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# 5. x, y 방향을 합쳐 최종 에지 이미지 생성
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# 6. Otsu의 이진화 적용
ret, binary = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

# 7. 결과 출력
cv2.imshow("Original", img)
cv2.imshow("Sobel X", sobel_x)
cv2.imshow("Sobel Y", sobel_y)
cv2.imshow("Sobel Combined", sobel_combined)
cv2.imshow("Binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
