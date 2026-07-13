import cv2
import numpy as np

img16 = cv2.imread("dataset/00000_depth.png", -1)
img16 = img16.astype(np.float32)
img_norm = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX)
img_norm_uint8 = img_norm.astype(np.uint8)

print(img_norm.dtype)
print(img_norm_uint8.dtype)

cv2.imshow("raw", img16)
cv2.imshow("normalized", img_norm)
cv2.imshow("normalized_8bit", img_norm_uint8)
cv2.waitKey(0)

