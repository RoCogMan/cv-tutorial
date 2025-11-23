import numpy as np
import mujoco as mj
import cv2


def vecs2tf(rvec: np.ndarray, tvec: np.ndarray):
    tf = np.identity(4)
    rmat, _ = cv2.Rodrigues(rvec)
    tf[0:3, 0:3] = rmat
    tf[0:3, 3] = tvec.ravel()

    return tf


def tf2vecs(tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(tf[0:3, 0:3])
    rvec = rvec.ravel()
    tvec = tf[0:3, 3].copy()

    return rvec, tvec


def rvec2quat(rvec: np.ndarray) -> np.ndarray:
    quat = np.zeros((4,))
    angle = np.linalg.norm(rvec)
    axis = rvec / angle
    mj.mju_axisAngle2Quat(quat, axis, angle)

    return quat
