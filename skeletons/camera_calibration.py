"""Chessboard camera calibration.

Detects chessboard corners across a set of captured images and estimates the
intrinsic matrix and distortion coefficients. Reads images from ../data/ and
writes camera_matrix.npy, dist_coeffs.npy, calibration_results.npz and
calibration_parameters.txt back into ../data/.
"""
import numpy as np
import cv2 as cv
import glob
import os
from pathlib import Path

from lab_config import CHESSBOARD_SIZE, SQUARE_SIZE_MM, CALIB_IMAGE_PATTERN

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'

def calibrate_camera(images_path=CALIB_IMAGE_PATTERN,
                     chessboard_size=CHESSBOARD_SIZE,
                     square_size_mm=SQUARE_SIZE_MM):
    raise NotImplementedError('TODO: calibrate_camera')
if __name__ == '__main__':
    pass
