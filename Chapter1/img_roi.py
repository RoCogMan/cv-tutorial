import cv2



# 이미지 불러오기
img = cv2.imread("lenna.jpg")



# ROI 선택 (드래그해서 선택)
# 반환값: (x, y, w, h)
r = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)



x, y, w, h = r



# ROI 영역 잘라내기
roi = img[y:y+h, x:x+w]



# 표시
cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
