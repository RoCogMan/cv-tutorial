import cv2
import numpy as np

img16 = cv2.imread("dataset/00000_depth.png", -1)
img_norm = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX)

print(img_norm.dtype)

cv2.imshow("normalized", img_norm)
cv2.waitKey(0)
