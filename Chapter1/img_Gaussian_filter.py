import cv2
import numpy as np



img = cv2.imread("Gaussian Noise.png")



# Gaussian Blur
gaussian = cv2.GaussianBlur(img, (11, 11), 5)



# Median Filter
median = cv2.medianBlur(img, 11)  



cv2.imshow("Original", img)
cv2.imshow("Gaussian Noise + Gaussian Blur", gaussian)
cv2.imshow("Gaussian Noise + Median", median)



cv2.waitKey(0)
cv2.destroyAllWindows()
