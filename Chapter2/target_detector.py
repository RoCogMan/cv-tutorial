import cv2
import numpy as np
from typing import Optional

from dataclasses import dataclass


@dataclass
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray


# calibration target의 검출 및 검출 결과에 대한 시각화를 수행할 class에 해당합니다.
class TargetDetector:
    def __init__(
        self,
        cam_mat: np.ndarray = None,
        dist_coeffs: np.ndarray = None,
    ):
        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard(
            (8, 8),
            20.0 * 1e-3,
            15.0 * 1e-3,
            self.dict,
        )
        self.board.setLegacyPattern(True)  # calib.io이 legacy pattern을 사용
        self.update_intrinsic(cam_mat, dist_coeffs)

        self.charuco_corner_color = (0, 50, 200)

    # 입력받은 이미지 해상도에 대응하는 Calibration target을 그립니다.
    def generate_img(self, img_res: int = 500) -> np.ndarray:
        img = self.board.generateImage((img_res, img_res))
        return img

    # 내부 파라미터 및 왜곡 계수를 입력받아 Detector의 Charuco 파라미터를 업데이트 합니다.
    def update_intrinsic(self, cam_mat: np.ndarray, dist_coeffs: np.ndarray):
        self.params = cv2.aruco.CharucoParameters()
        self.params.cameraMatrix = cam_mat
        self.params.distCoeffs = dist_coeffs
        # self.params.minMarkers = 6

        self.marker_detector = cv2.aruco.ArucoDetector(self.dict)
        self.detector = cv2.aruco.CharucoDetector(self.board, self.params)

    def detect_aruco(
        self, img: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:

        if "4.10" in cv2.__version__:
            marker_corners, marker_ids, rejected_corners = cv2.aruco.detectMarkers(
                img, self.board.getDictionary()
            )
        else:
            marker_corners, marker_ids, rejected_corners = (
                self.marker_detector.detectMarkers(img)
            )
        return marker_corners, marker_ids

    def draw_aruco(
        self, img: np.ndarray, marker_corners: list[np.ndarray], marker_ids: np.ndarray
    ) -> np.ndarray:
        img = img.copy()
        if self._is_grayscale(img):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        return img

    def detect_charuco(
        self, img, marker_corners, marker_ids
    ) -> tuple[np.ndarray, np.ndarray]:
        charuco_corners, charuco_ids, _, _ = self.detector.detectBoard(
            img, markerCorners=marker_corners, markerIds=marker_ids
        )

        return charuco_corners, charuco_ids

    def draw_charuco(self, img, charuco_corners, charuco_ids) -> np.ndarray:
        img = img.copy()
        if self._is_grayscale(img):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                img, charuco_corners, charuco_ids, self.charuco_corner_color
            )
        return img

    def _is_grayscale(self, img: np.ndarray) -> bool:
        # (480, 640, 1)의 shape을 갖는 mono 이미지에 대응하기 위해 img.shape[-1] != 3의 조건을 추가
        return len(img.shape) != 3 or img.shape[-1] != 3

    def estimate_pose(
        self, img: np.ndarray, draw: bool = True
    ) -> Optional[tuple[Pose, Optional[np.ndarray]]]:
        """
        이미지를 입력받아 Calibration target의 위치를 추정합니다.
        추정에 성공한 경우 Pose와 추정 과정이 시각화 된 이미지(draw가 True)/None(draw가 False)를 반환합니다
        검출에 실패한 경우 None을 반환합니다.
        """
        cam_mat = self.params.cameraMatrix
        dist_coeffs = self.params.distCoeffs

        marker_corners, marker_ids = self.detect_aruco(img)
        charuco_corners, charuco_ids = self.detect_charuco(
            img, marker_corners, marker_ids
        )

        if charuco_ids is None:
            return None

        if len(charuco_ids) < 6:
            return None

        if "4.10" in cv2.__version__:
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self.board,
                cam_mat,
                dist_coeffs,
                None,
                None,
            )
        else:
            obj_points, img_points = self.board.matchImagePoints(
                charuco_corners, charuco_ids
            )
            valid, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, cam_mat, dist_coeffs
            )

        if not valid:
            return None

        pose = Pose(rvec=rvec, tvec=tvec)
        if not draw:
            return (pose, None)

        img = self.draw_aruco(img, marker_corners, marker_ids)
        img = self.draw_charuco(img, charuco_corners, charuco_ids)
        return (pose, img)

    @property
    def cam_mat(self) -> np.ndarray:
        return self.params.cameraMatrix

    @property
    def dist_coeffs(self) -> np.ndarray:
        return self.params.distCoeffs

    @property
    def square_length(self) -> float:
        return self.board.getSquareLength()


def test_target_detector():
    target_detector = TargetDetector(cam_mat=np.identity(3), dist_coeffs=np.zeros(5))
    img = target_detector.generate_img(500)

    cv2.imshow("img", img)
    cv2.waitKey(0)

    output = target_detector.estimate_pose(img, draw=True)
    if output:
        pose, detected_img = output
        cv2.drawFrameAxes(
            detected_img,
            target_detector.cam_mat,
            target_detector.dist_coeffs,
            pose.rvec,
            pose.tvec,
            target_detector.square_length,
        )

        cv2.imshow("img", detected_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    test_target_detector()
