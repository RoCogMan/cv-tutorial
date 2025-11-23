import os
import numpy as np
import open3d as o3d


class IntrinsicSimulator:
    def __init__(
        self,
        img_width: int = 1280,
        img_height: int = 720,
        background_color: list[float] = [0.3, 0.3, 0.3],
    ):
        os.environ["XDG_SESSION_TYPE"] = "x11"
        self.vis = o3d.visualization.Visualizer()

        self.vis.create_window(width=img_width, height=img_height, visible=False)
        self.vis.get_render_option().background_color = np.array(background_color)
        self.view_ctrl = self.vis.get_view_control()
        self.view_ctrl.set_constant_z_near(0.01)
        self.view_ctrl.set_constant_z_far(1000.0)

        self.pinhole_params = (
            self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        )

    def add_origin_axes(self, axes_length=0.1):
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_length)
        self.add_geom(axes, True)

    def add_geom(self, geom: o3d.geometry.Geometry, reset_bb: bool = True):
        self.vis.add_geometry(geom, reset_bb)

    def add_geoms(self, geoms: list[o3d.geometry.Geometry], reset_bb: bool = True):
        for geom in geoms:
            self.vis.add_geometry(geom, reset_bb)

    def change_background(self, color: np.ndarray):
        self.vis.get_render_option().background_color = color

    def render(self):
        img = self.vis.capture_screen_float_buffer(True)
        img = (np.array(img) * 255.0).astype(np.uint8)
        return img

    def update_params(self, cam_mat, extrinsic=np.identity(4)):
        self.pinhole_params.intrinsic.intrinsic_matrix = cam_mat
        self.pinhole_params.extrinsic = extrinsic

        self.vis.get_view_control().convert_from_pinhole_camera_parameters(
            self.pinhole_params, True
        )

        self.vis.poll_events()
        self.vis.update_renderer()

    def disp_params(self):
        print("intrinsic:")
        print(self.pinhole_params.intrinsic.intrinsic_matrix)
        print("extrinsic:")
        print(self.pinhole_params.extrinsic)
        print("cx, cy: ", end="")
        print(self.pinhole_params.intrinsic.get_principal_point())
