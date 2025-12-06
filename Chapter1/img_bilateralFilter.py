import cv2



img = cv2.imread("lenna.jpg")



# d=9, sigmaColor=75, sigmaSpace=75
bilateral = cv2.bilateralFilter(img, 11, 80, 80)



cv2.imshow("Original", img)
cv2.imshow("Bilateral Filter", bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()



