import cv2
import numpy as np

image = cv2.imread('lenna.jpg')
def permute_image(img):
    rows, cols = img.shape[:2]
    M = np.array([[0, 1, 0],[1, 0, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (rows, cols))

permuted = permute_image(image)
# 화면에 결과 출력
cv2.imshow('Permuted Image', permuted)
cv2.waitKey(0)
cv2.destroyAllWindows()


