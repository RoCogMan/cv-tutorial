import cv2
import numpy as np

img = cv2.imread("shapes.jpg")
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

output = img.copy() 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# THRESH_BINARY_INV: 흰 도형 / 검은 배경 되도록 반전
_, binary = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 윤곽선 검출
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# 도형 개수 세기
num_tri = 0      # 삼각형 개수
num_rect = 0     # 사각형 개수
num_circle = 0   # 원(곡선) 개수

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area < 100:
        continue

    # 윤곽선 근사 (꼭지점 개수 보기)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        num_tri += 1
    elif vertices == 4:
        num_rect += 1
    else:
        num_circle += 1

text = f"Triangles: {num_tri}, Rectangles: {num_rect}, Circles: {num_circle}"

cv2.putText(output, text, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
cv2.imshow("Original", img)
cv2.imshow("Contours & Count", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
