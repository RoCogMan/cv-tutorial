import cv2



# 이미지 불러오기
img = cv2.imread("lenna.jpg")



# 대비(Contrast)와 밝기(Brightness) 설정
alpha = 2   # 대비 (1.0은 원본, 1.0보다 크면 더 선명)
beta = 30     # 밝기 (양수면 밝게, 음수면 어둡게)



# 밝기/대비 조절
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)



# 표시
cv2.imshow("Original", img)
cv2.imshow("Adjusted", adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()
