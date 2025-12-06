import cv2
import numpy as np

image = cv2.imread('lenna.jpg')
def warp_image(img):
    rows, cols = img.shape[:2]
    # 원본 이미지에서 변환할 4점 좌표 (예시)
    pts1 = np.float32([[50, 50], [cols-50, 50], [50, rows-50], [cols-50, rows-50]])
    # 변환 후 도착할 4점 좌표 (비뚤어진 사각형)
    pts2 = np.float32([[200, 50], [cols-200, 50], [50, rows-50], [cols-50, rows-50]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (cols, rows))
    return warped
warped = warp_image(image)

cv2.namedWindow('Warped Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Warped Image', 800, 600)
cv2.imshow('Warped Image', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()