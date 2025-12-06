import cv2



# 이미지 불러오기
img = cv2.imread("lenna.jpg")



# 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# 히스토그램 평활화
equalized = cv2.equalizeHist(gray)

# 출력
cv2.imshow("Gray", gray)
cv2.imshow("Equalized", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
