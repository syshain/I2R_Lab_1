"""Shared hardware configuration for Lab 1 scripts.

Every script that talks to the robot or opens the wrist camera reads its
settings from here, so there is one place to change them. Values can be
overridden without editing code via environment variables, which is handy on
the bench:

    export ROBOT_IP=192.168.1.153
    export CAMERA_INDEX=4
    python src/image_capture.py
"""
import os

# UFACTORY Lite 6 controller IP (same subnet as the PC; verify with ping).
ROBOT_IP = os.environ.get('ROBOT_IP', '192.168.1.153')

# USB device index of the wrist-mounted camera. Find it with:
#   Linux:  v4l2-ctl --list-devices
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '4'))

# Capture resolution used by the calibration pipelines.
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
