import cv2
from glob import glob

# 사용자 정의 모듈 임포트
from target_detector import TargetDetector  # 코너 검출을 보조할 class


def main():
    detector = TargetDetector()  # 코너 검출을 보조할 class의 객체 생성

    img_paths = glob("cam_calib/**.png")  # cam_calib 안의 모든 png 파일의 경로를 획득

    # 대응쌍을 저장할 list 선언
    all_obj_points = []
    all_img_points = []
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)  # 이미지를 불러오기
        marker_corners, marker_ids = detector.detect_aruco(img)  # 마커 검출 시도
        charuco_corners, charuco_ids = detector.detect_charuco(
            img, marker_corners, marker_ids
        )  # checkerboard 코너 검출 시도

        img = detector.draw_aruco(
            img, marker_corners, marker_ids
        )  # 검출된 결과를 이미지에 표시 (마커)

        img = detector.draw_charuco(
            img, charuco_corners, charuco_ids
        )  # 검출된 결과를 이미지에 표시 (checkerboard 코너)

        obj_points, img_points = detector.board.matchImagePoints(
            charuco_corners, charuco_ids
        )  # 오브젝트 포인트 및 이미지 포인트를 반환

        # list에 대응쌍 및 이미지를 추가
        all_obj_points.append(obj_points)
        all_img_points.append(img_points)
        imgs.append(img)

        cv2.imshow("img", img)  # 해당 결과를 시각화
        cv2.waitKey(10)

    # 카메라 캘리브레이션 수행
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    reproj_err, cam_mat, dist_coeffs, _, _ = cv2.calibrateCamera(
        all_obj_points, all_img_points, img.shape[:2], None, None, criteria=criteria
    )

    # 캘리브레이션 수행 결과를 출력
    print(f"reproj_err: {reproj_err}")  # 재투영 오차
    print(f"cam_mat: \n{cam_mat}")  # camera matrix
    print(f"dist_coeffs: {dist_coeffs}")  # distortion coefficients

    for img in imgs:
        undistorted = cv2.undistort(
            img, cam_mat, dist_coeffs
        )  # 캘리브레이션 결과를 이용하여 왜곡 보정
        cv2.imshow("img", undistorted)  # 결과를 시각화
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
