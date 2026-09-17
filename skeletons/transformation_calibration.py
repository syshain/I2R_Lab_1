"""Capture paired robot + ArUco-board poses for hand-eye calibration.

Runs a live detection loop over the wrist-mounted camera; when the user hits
'c' it records the current T_base_ee alongside the detected T_cam_board, and
's' saves the collected pairs into ../data/calibration_data.npy.

The board geometry is read from ../data/board_config.json (a 3-D polyhedron of
markers), matching Labs 2 and 3. The corner coordinates are defined at 30 mm
marker scale in the config and scaled up by 4/3 to the physical 40 mm markers.
"""
import json

import numpy as np
import cv2
import cv2.aruco as aruco
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from lab_config import (
    ROBOT_IP, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    ARUCO_BOARD_SCALE, ARUCO_REPROJ_REJECT_PX,
)
from fk_lite6 import fk_lite6
import robot_io

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'

# Per-marker corner winding correction (matches Labs 2/3 detector).
_CORNER_ROLL = {0: 0, 2: 0, 3: 0, 4: 0, 5: 0}


def get_robot_end_effector_pose(arm):
    """Return (T_base_ee, joints_deg, raw_pose) from the arm's current state.

    T_base_ee is derived by forward kinematics from the joint angles read via
    robot_io.read_joints_rad(arm) and passed to fk_lite6 — NOT taken from the
    controller's cartesian get_position(). joints_deg is the 6-element degree
    list; raw_pose is the controller [x,y,z,r,p,y] kept as a cross-check (may be
    None if get_position fails).
    """
    raise NotImplementedError('TODO: get_robot_end_effector_pose')


class ArucoBoardDetector:
    """3-D ArUco polyhedron detector driven by board_config.json."""

    def __init__(self, camera_index=None):
        raise NotImplementedError('TODO: __init__')

    def _build_board_model(self):
        raise NotImplementedError('TODO: _build_board_model')

    def start_camera(self):
        raise NotImplementedError('TODO: start_camera')

    def detect_board(self, frame):
        raise NotImplementedError('TODO: detect_board')

    def draw_detection(self, frame, rvec, tvec, ids, corners, success):
        raise NotImplementedError('TODO: draw_detection')

    def get_robot_end_effector_pose(self):
        raise NotImplementedError('TODO: get_robot_end_effector_pose')

    def capture_calibration_pose(self):
        raise NotImplementedError('TODO: capture_calibration_pose')

    def run(self, mode='detect'):
        raise NotImplementedError('TODO: run')


if __name__ == '__main__':
    pass
