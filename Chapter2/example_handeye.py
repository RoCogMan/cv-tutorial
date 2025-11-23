import cv2
import numpy as np

# 사용자 정의 모듈 임포트
from sim import Simulator  # mujoco 기반 simulator 클래스
from rs_wrapper import RealSenseWrapper  # realsense 사용을 보조하는 클래스
from target_detector import TargetDetector  # calibration target(ChArUco) 검출용 클래스
from utils import (
    tf2vecs,
)  # transform matrix를 rotation vector와 translation vector로 분할하는 함수


def main():
    cam = RealSenseWrapper()  # realsense를 사용하기 위한 객체 생성
    detector = TargetDetector(
        *cam.fetch_color_intrinsics()
    )  # calibration target 검출용 객체 생성

    sim = Simulator(
        launch_viewer=False
    )  # simulator 객체를 생성 (passive viewer는 실행하지 않음)

    # 핸드-아이 캘리브레이션 수행에 필요한 데이터를 보관하기 위한 list 선언
    rvecs_base2ee = []
    tvecs_base2ee = []
    rvecs_cam2target = []
    tvecs_cam2target = []

    while True:
        sim.step()  # simulation step을 수행

        img, _ = cam.read()  # 이미지를 획득
        ret = detector.estimate_pose(img)  # 획득한 이미지로부터 target 검출 시도

        if ret:  # target이 검출된 경우
            pose, img = ret  # 추정된 포즈 정보와 검출결과가 표시된 이미지
            sim.update_camera_pose(
                pose.rvec, pose.tvec
            )  # 추정된 포즈를 simulation에 반영
            detected = True  # target이 검출되었는 지 유무
        else:
            detected = False  # target이 검출되었는 지 유무

        scene_img = sim.render(
            img
        )  # Offscreen 렌더링 수행 후 overlay가 수행될 img를 함께 전달

        message = f"# poses: {len(rvecs_base2ee)}"  # 화면에 표시할 텍스트
        cv2.putText(
            scene_img, message, (30, 30), cv2.FONT_ITALIC, 1.0, (0, 255, 0), 3
        )  # 이미지에 텍스트를 추가

        cv2.imshow("scene_img", scene_img)  # 렌더링 된 이미지를 표시
        key = cv2.waitKey(1)  # 1ms간 키보드 입력 대기
        if key == ord("q"):  # 'q'를 입력받으면 종료
            break
        elif key == ord("c") and detected:  # 'c'를 입력받으면 데이터에 추가
            cv2.imwrite(f"calib_img{len(rvecs_base2ee)}.png", scene_img)
            rvec, tvec = tf2vecs(sim.base2ee)

            # 핸드-아이 캘리브레이션 수행을 위해 list에 추가
            rvecs_base2ee.append(rvec)
            tvecs_base2ee.append(tvec)
            rvecs_cam2target.append(pose.rvec)
            tvecs_cam2target.append(pose.tvec)

    # 수집된 결과를 통해 캘리브레이션 수행
    rmat, tvec = cv2.calibrateHandEye(
        rvecs_base2ee, tvecs_base2ee, rvecs_cam2target, tvecs_cam2target
    )
    # 결과를 출력
    print(rmat)
    print(tvec)

    np.set_printoptions(precision=3, suppress=True)
    tx = np.identity(4)
    tx[0:3, 0:3] = rmat
    tx[0:3, 3] = tvec.ravel()
    print(tx)

    print(sim.ee2cam)


if __name__ == "__main__":
    main()
