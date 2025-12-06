import cv2
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 (BGR 그대로 불러온 뒤 B 채널 사용)
img = cv2.imread("lenna.jpg")


# 2. 오츠 알고리즘을 이용한 자동 임계값 이진화
channel = img[:, :, 2]  # R 채널 사용
ret, binary = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# 3. 히스토그램 계산 및 임계값 시각화
hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
print("오츠 알고리즘이 선택한 임계값 =", ret)

plt.plot(hist, color='black')
plt.axvline(x=ret, color='red', linestyle='dashed', linewidth=1)
plt.title("Histogram & Otsu Threshold")
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.show()


# 4. 결과 출력
cv2.imshow("Original", img)
cv2.imshow("R Channel Binary", channel)
cv2.imshow("Otsu Binary Threshold", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
