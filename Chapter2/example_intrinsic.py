import cv2
import numpy as np
import open3d as o3d
from copy import deepcopy

# 사용자 정의 모듈 임포트
from intrinsic_sim import IntrinsicSimulator

SCALE = 1.0


def main():
    intrinsic_sim = IntrinsicSimulator()  # 렌더링 수행을 보조할 객체 생성
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        0.1 * SCALE
    )  # 반지름이 0.1인 구 생성
    sphere.compute_vertex_normals()  # 법선 벡터 계산 수행

    sphere.translate(np.array([0.0, 0.0, 1.0]) * SCALE)  # 정면 방향으로 1.0만큼 이동
    right_sphere = deepcopy(sphere).translate(
        np.array([0.3, 0.0, 0.0]) * SCALE
    )  # 우측으로 0.3만큼 이동
    left_sphere = deepcopy(sphere).translate(
        np.array([-0.3, 0.0, 0.0]) * SCALE
    )  # 좌측으로 0.3만큼 이동
    intrinsic_sim.add_geoms([sphere, left_sphere, right_sphere])  # scene에 추가

    focal_length = 900.0
    cam_mat = np.array(
        [
            [focal_length, 0.0, 640.0],
            [0.0, focal_length, 360.0],
            [0.0, 0.0, 1.0],
        ]
    )  # camera matrix를 생성
    intrinsic_sim.update_params(cam_mat)  # 내부 파라미터 업데이트
    img = intrinsic_sim.render()  # 렌더링을 수행하여 이미지를 획득

    cv2.imshow("img", img)  # 출력된 이미지를 시각화
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
