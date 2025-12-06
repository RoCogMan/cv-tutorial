import cv2



img = cv2.imread("lenna.jpg")



median = cv2.medianBlur(img, 9)  # 커널 크기 = 9



cv2.imshow("Original", img)
cv2.imshow("Median Blur", median)
cv2.waitKey(0)
cv2.destroyAllWindows()



