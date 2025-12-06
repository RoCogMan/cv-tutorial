import cv2
import numpy as np

image = cv2.imread('lenna.jpg')

def scale_image(img, scale_factor=4.5):
    rows, cols = img.shape[:2]
    M = np.array([[scale_factor, 0, 0],[0, scale_factor, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (int(cols * scale_factor),
    int(rows * scale_factor)))
scaled = scale_image(image)

# 화면에 결과 출력
cv2.imshow('Scaled Image', scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()