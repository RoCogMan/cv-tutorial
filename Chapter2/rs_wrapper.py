import pyrealsense2 as rs
import numpy as np
import cv2


class RealSenseWrapper:
    def __init__(self):
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        self.cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        self.cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.profile = self.pipe.start(self.cfg)

    def __del__(self):
        self.pipe.stop()

    def fetch_color_intrinsics(self):
        color_profile: rs.video_stream_profile = self.profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()

        cam_mat = np.identity(3)
        cam_mat[0, 0] = intrinsics.fx
        cam_mat[1, 1] = intrinsics.fy
        cam_mat[0, 2] = intrinsics.ppx
        cam_mat[1, 2] = intrinsics.ppy

        dist_coeffs = np.array(intrinsics.coeffs)

        return cam_mat, dist_coeffs

    def read(
        self, read_color: bool = True, read_depth: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = self.pipe.wait_for_frames()

        color = None
        depth = None

        if read_color:
            color_frame = frames.get_color_frame()
            color = np.asarray(color_frame.get_data())
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        if read_depth:
            depth_frame = frames.get_depth_frame()
            depth = np.asarray(depth_frame.get_data())

        return color, depth


def test_rs_wrapper():
    rs_wrapper = RealSenseWrapper()

    print(rs_wrapper.fetch_color_intrinsics())
    while True:
        color, depth = rs_wrapper.read(True, False)

        cv2.imshow("color", color)
        cv2.waitKey(10)


if __name__ == "__main__":
    test_rs_wrapper()
