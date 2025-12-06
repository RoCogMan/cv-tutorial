import cv2
import numpy as np

image = cv2.imread('lenna.jpg')

def shear_image(img, shear_factor=0.5):
    rows, cols = img.shape[:2]
    M = np.array([[1, shear_factor, 0],[0, 1, 0]], dtype=np.float32)
    width = int(cols + rows * shear_factor)
    height = rows
    return cv2.warpAffine(img, M, (width, height))
sheared = shear_image(image)

# 화면에 결과 출력
cv2.imshow('sheared Image', sheared)
cv2.waitKey(0)
cv2.destroyAllWindows()


