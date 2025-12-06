import cv2
import numpy as np

display_scale = 0.2   # 원본사진이 너무크면 값을 작게
image_path = 'card.jpg'
image = cv2.imread(image_path)
orig_h, orig_w = image.shape[:2]
disp_img = cv2.resize(image, None, fx=display_scale, fy=display_scale)
points = []
def select_points(event, x, y, flags, param):
    global points, disp_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            cv2.circle(disp_img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select 4 Points", disp_img)
            real_x = int(x / display_scale)
            real_y = int(y / display_scale)
            points.append((real_x, real_y))
            print(f"점 선택됨 (표시 좌표 = {(x,y)}) → (실제 좌표 = {(real_x,real_y)})")
cv2.namedWindow("Select 4 Points")
cv2.setMouseCallback("Select 4 Points", select_points)
print("'좌상 → 우상 → 좌하 → 우하' 순으로 4개의 점을 클릭하세요.")

while True:
    cv2.imshow("Select 4 Points", disp_img)
    key = cv2.waitKey(1)
    if len(points) == 4:
        break

cv2.destroyAllWindows()
src = np.float32(points)
width, height = 400, 600
dst = np.float32([
    [0, 0],
    [width - 1, 0],
    [0, height - 1],
    [width - 1, height - 1]
])

M = cv2.getPerspectiveTransform(src, dst)
corrected = cv2.warpPerspective(image, M, (width, height))

cv2.imwrite("corrected_photo.jpg", corrected)
cv2.imshow("Corrected Result", corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()