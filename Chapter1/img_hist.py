import cv2
import matplotlib.pyplot as plt



# 이미지 불러오기
img = cv2.imread("lenna.jpg")



# 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# 히스토그램 계산
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])



# 히스토그램 그래프 출력
plt.plot(hist, color='black')
plt.title("Grayscale Histogram")
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.show()
