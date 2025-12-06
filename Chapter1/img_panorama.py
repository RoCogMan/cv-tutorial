import sys
import cv2
img_names = ['panorama1.jpg', 'panorama2.jpg', 'panorama3.jpg', 'panorama4.jpg', 'panorama5.jpg']
imgs = []

for name in img_names:
    img = cv2.imread(name)
    if img is None:
        print(f'Image load failed! ({name})')
        sys.exit()
    imgs.append(img)
stitcher = cv2.Stitcher_create()
status, dst = stitcher.stitch(imgs)

cv2.imwrite('output.jpg', dst)
cv2.namedWindow('Panorama', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Panorama', 1000, 600)
cv2.imshow('Panorama', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()