import cv2
import numpy as np

# 이미지 경로 설정
image = cv2.imread('lenna.jpg')
def translate_image(img, tx=700, ty=50):
    rows, cols = img.shape[:2]
    M = np.array([[1, 0, tx],
                  [0, 1, ty]], dtype=np.float32)
    new_cols = cols + abs(tx)
    new_rows = rows + abs(ty)
    return cv2.warpAffine(img, M, (new_cols, new_rows))

translated = translate_image(image)

cv2.namedWindow('Translated Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Translated Image', 800, 600)
cv2.imshow('Translated Image', translated)
cv2.waitKey(0)
cv2.destroyAllWindows()