import cv2

img = cv2.imread("dataset/00000_colors.png")
print("Image shape:", img.shape, "dtype: ", img.dtype)
cv2.imshow("color", img)
cv2.waitKey(0)
