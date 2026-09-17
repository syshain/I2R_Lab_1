"""Hand-eye calibration via FK-derived robot poses + OpenCV solvers.

Loads captured pairs from ../data/calibration_data.npy. For each pose the base->EE
transform is derived by forward kinematics from the stored joint angles (fk_lite6),
falling back to the controller's cartesian get_position() only for legacy data that
has no joints. Several OpenCV hand-eye methods are then scored by reconstructed
board-position consistency; the best T_ee_cam is saved to ../data/.
"""
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from fk_lite6 import fk_lite6

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'


def fk_pose(joints_deg):
    """Base->EE transform (mm) from 6 joint angles in degrees via Craig-DH FK."""
    raise NotImplementedError('TODO: fk_pose')


def cartesian_pose(raw_pose):
    """Fallback Base->EE transform from controller [x,y,z,roll,pitch,yaw] (mm,deg)."""
    raise NotImplementedError('TODO: cartesian_pose')


def resolve_T_base_ee(d):
    """Pick the best available source for a pose's Base->EE transform.

    Prefers FK from stored joints; falls back to cartesian for legacy captures.
    Returns (T_base_ee, source_label).
    """
    raise NotImplementedError('TODO: resolve_T_base_ee')


def evaluate_consistency(T_ee_cam, data):
    """Reconstruct the board position in the base frame for every pose.

    Returns per-pose positions, their mean, per-axis std, max deviation, and the
    RMS error norm (root-mean-square of each pose's deviation from the mean).
    """
    raise NotImplementedError('TODO: evaluate_consistency')


def save_result(T_ee_cam, filename_npy='T_ee_cam.npy', filename_txt='T_ee_cam.txt'):
    raise NotImplementedError('TODO: save_result')


if __name__ == '__main__':
    pass
