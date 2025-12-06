import cv2



# 1. 이미지 불러오기
img = cv2.imread("lenna.jpg")



# 2. 기본 임계값 적용 (Threshold = 127) R 채널 사용
ret, binary = cv2.threshold(img[:,:,2], 127, 255, cv2.THRESH_BINARY)



# 3. 결과 출력
cv2.imshow("R Channel Binary", img[:,:,2])
cv2.imshow("R Channel Binary Threshold", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
