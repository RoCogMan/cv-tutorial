import cv2


image = cv2.imread('lenna.jpg')
def rotate_image(img, angle=45):
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (cols, rows))

rotated = rotate_image(image)

# 화면에 결과 출력
cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()


