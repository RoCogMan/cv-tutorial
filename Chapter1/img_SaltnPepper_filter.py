import cv2
import numpy as np

img = cv2.imread("Salt & Pepper Noise.png")



# Gaussian Blur
gaussian = cv2.GaussianBlur(img, (9, 9), 5)



# Median Filter
median = cv2.medianBlur(img, 9)  

cv2.imshow("Original", img)
cv2.imshow("Salt & Pepper Noise + Gaussian Blur", gaussian)
cv2.imshow("Salt & Pepper Noise + Median", median)



cv2.waitKey(0)
cv2.destroyAllWindows()
