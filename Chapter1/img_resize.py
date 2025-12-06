import cv2



img = cv2.imread('lenna.jpg')



# 50% 축소
small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)



# 2배 확대
big = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)



cv2.imshow("small", small)
cv2.imshow("big", big)
cv2.waitKey(0)
cv2.destroyAllWindows()
