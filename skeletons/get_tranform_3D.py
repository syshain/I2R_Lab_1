"""Hand-eye calibration by brute-force search over Euler orders + solvers.

Loads captured pairs from ../data/calibration_data.npy, tries every Euler-angle
convention against several OpenCV hand-eye methods, scores each by reconstructed
board-position consistency, and saves the best T_ee_cam to ../data/.
"""
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'

def robot_pose_to_matrix(raw_pose, euler_order='xyz'):
    raise NotImplementedError('TODO: robot_pose_to_matrix')

def evaluate_consistency(T_ee_cam, data, euler_order):
    raise NotImplementedError('TODO: evaluate_consistency')

def save_result(T_ee_cam, filename_npy='T_ee_cam.npy', filename_txt='T_ee_cam.txt'):
    raise NotImplementedError('TODO: save_result')
if __name__ == '__main__':
    pass
