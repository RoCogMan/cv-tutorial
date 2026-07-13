import cv2

img = cv2.imread("dataset/00000_depth.png")
print("Image shape:", img.shape, "dtype: ", img.dtype)
cv2.imshow("depth", img)
cv2.waitKey(0)
