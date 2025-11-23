import cv2
import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from time import time
from typing import Optional

# 사용자 정의 모듈 임포트
from target_detector import TargetDetector
from rs_wrapper import RealSenseWrapper
from utils import *


class Simulator:
    def __init__(
        self,
        launch_viewer: bool = False,
        realtime_factor: float = 0.5,
        mjcf_path: str = "universal_robots_ur5e/scene.xml",
    ):
        self.m = mj.MjModel.from_xml_path(mjcf_path)
        self.d = mj.MjData(self.m)
        self.ren = mj.Renderer(self.m, 720, 1280)
        self.aspect_ratio = self.ren.width / self.ren.height  # 1.777

        self.realtime_factor = realtime_factor
        self.launch_viewer = launch_viewer

        if self.launch_viewer:
            self.viewer = mjv.launch_passive(self.m, self.d)

        self.reset()

    def close_viewer(self):
        if self.launch_viewer and self.viewer.is_running():
            self.viewer.close()

    def viewer_is_running(self):
        return self.launch_viewer and self.viewer.is_running()

    def reset(self):
        mj.mj_resetData(self.m, self.d)
        mj.mj_forward(self.m, self.d)
        self.d.qpos[0:6] = [-3.77, -1.67, 1.86, -2.14, -1.54, -0.282]
        self.init_time = time()
        self.world2target = self._fetch_body_tf("checkerboard")

    def _fetch_body_tf(self, body_name: int) -> np.ndarray:
        tf = np.identity(4, self.d.qpos.dtype)
        tf[0:3, 0:3] = self.d.body(body_name).xmat.reshape(3, 3)
        tf[0:3, 3] = self.d.body(body_name).xpos.ravel()

        return tf

    def render_cam(self, cam_name: str = "color_cam") -> np.ndarray:
        self.ren.update_scene(self.d, cam_name)
        img = self.ren.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def render(
        self,
        cam_img: Optional[np.ndarray] = None,
        win_width: int = 300,
        spacing: int = 30,
    ) -> np.ndarray:
        scene_img = self.render_cam("remote_cam")
        eye_img = self.render_cam("color_cam")
        resize_res = (
            int(win_width * self.aspect_ratio),
            win_width,
        )
        eye_img = cv2.resize(eye_img, resize_res)
        cv2.rectangle(eye_img, (0, 0), resize_res, (0, 255, 0), 5)

        height, width = scene_img.shape[:2]

        sim_anchor = (
            height - resize_res[1] - spacing,
            width - resize_res[0] - spacing,
        )
        self._overlay_img(scene_img, eye_img, sim_anchor)

        if cam_img is not None:
            cam_img = cv2.resize(cam_img, resize_res)
            cv2.rectangle(cam_img, (0, 0), resize_res, (0, 0, 255), 5)

            real_anchor = (spacing, width - resize_res[0] - spacing)
            self._overlay_img(scene_img, cam_img, real_anchor)

        return scene_img

    def _overlay_img(
        self, background: np.ndarray, foreground: np.ndarray, anchor: tuple[int]
    ):
        rows, cols = foreground.shape[:2]

        background[anchor[0] : anchor[0] + rows, anchor[1] : anchor[1] + cols] = (
            foreground
        )

    def step(self):
        elapsed_time: float = time() - self.init_time

        while elapsed_time * self.realtime_factor > self.d.time:
            mj.mj_step(self.m, self.d)

        if self.launch_viewer:
            self.viewer.sync()

    def _update_cmd(self, rvec: np.ndarray, tvec: np.ndarray, id: int = 0):
        self.d.mocap_pos[id, :] = tvec.ravel()
        self.d.mocap_quat[id, :] = rvec2quat(rvec)

    def _compute_goal(self, cam2target: np.ndarray):
        world2cam = self.world2target @ np.linalg.inv(cam2target)
        return world2cam

    def update_camera_pose(self, rvec: np.ndarray, tvec: np.ndarray):
        tf = vecs2tf(rvec, tvec)
        world2cam = self._compute_goal(tf)
        rvec_des, tvec_des = tf2vecs(world2cam)

        self._update_cmd(rvec_des, tvec_des, id=0)

    @property
    def sim_time(self):
        return self.d.time

    @property
    def ee2cam(self):
        mj.mj_forward(self.m, self.d)
        base2ee = self._fetch_body_tf("tcp")
        base2cam = self._fetch_body_tf("rgb_imager")

        ee2cam = np.linalg.inv(base2ee) @ base2cam
        return ee2cam

    @property
    def base2ee(self):
        mj.mj_forward(self.m, self.d)
        return self._fetch_body_tf("tcp")


def test_viewer():
    rs = RealSenseWrapper()
    cam_mat, dist_coeffs = rs.fetch_color_intrinsics()
    detector = TargetDetector(cam_mat, dist_coeffs)

    sim = Simulator(launch_viewer=True, realtime_factor=0.7)
    while sim.viewer_is_running():
        sim.step()

        img, _ = rs.read(True, False)
        rst = detector.estimate_pose(img, draw=False)

        if rst:
            pose, _ = rst
            sim.update_camera_pose(pose.rvec, pose.tvec)


def test_offscreen():
    rs = RealSenseWrapper()
    cam_mat, dist_coeffs = rs.fetch_color_intrinsics()
    detector = TargetDetector(cam_mat, dist_coeffs)

    sim = Simulator(launch_viewer=False, realtime_factor=0.7)
    while True:
        sim.step()

        img, _ = rs.read(True, False)
        rst = detector.estimate_pose(img, draw=False)

        if rst:
            pose, _ = rst
            sim.update_camera_pose(pose.rvec, pose.tvec)

        sim_img = sim.render(cam_img=img)
        cv2.imshow("img", sim_img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    # test_viewer()
    test_offscreen()
