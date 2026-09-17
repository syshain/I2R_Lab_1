"""Capture chessboard images for camera calibration.

Opens the camera, lets the user press SPACE to grab frames; saves them as
calib_NNN.jpg into ../data/. Press ESC to finish.
"""
import cv2
from pathlib import Path
from lab_config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'

def main():
    raise NotImplementedError('TODO: main')
if __name__ == '__main__':
    pass
