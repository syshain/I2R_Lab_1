"""Validate a hand-eye calibration with a RELOCATED board (live, keygated).

Workflow:
  1. Complete the normal calibration first (transformation_calibration.py +
     get_tranform_3D.py / ransac_calibration.py) so T_ee_cam.npy exists.
  2. Physically MOVE the ArUco board to a different spot on the table and hold
     it stationary there.
  3. Run this script. Move the arm through 8-10 different poses while keeping
     the board still; press 'c' to record each one. Press 's' to finish.

Why relocate? Validating against the same board location used for calibration
can let a systematic error in T_ee_cam partially cancel out. A fresh location
means any bias shows up as scatter across the new poses, so the reported RMS is
an honest measure of whether the transform generalises.

For every captured pose we reconstruct where the board sits in the base frame:
    T_base_board = T_base_ee @ T_ee_cam @ T_cam_board
Because the board is physically fixed, a correct calibration makes all these
reconstructions land on essentially one point. The RMS spread of those points
is the validation metric.

Requires the robot connected and the wrist camera open. Saves the collected
pairs to ../data/validation_data.npy.
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from transformation_calibration import ArucoBoardDetector
from lab_config import ROBOT_IP

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'

MIN_POSES = 8
MAX_POSES = 10


def reconstruct_board_positions(T_ee_cam, pairs):
    """Return the board centre in the base frame for each pair (mm)."""
    positions = []
    for d in pairs:
        T_base_board = d['T_base_ee'] @ T_ee_cam @ d['T_cam_board']
        positions.append(T_base_board[:3, 3])
    return np.array(positions)


def report(name, positions):
    """Print mean/std/max-deviation summary for an array of 3D positions."""
    mean = np.mean(positions, axis=0)
    std = np.std(positions, axis=0)
    max_dev = float(np.max(np.linalg.norm(positions - mean, axis=1)))
    rms = float(np.sqrt(np.mean(np.sum((positions - mean) ** 2, axis=1))))

    print(f"\n{name}")
    print("-" * 50)
    print(f"  Poses evaluated : {len(positions)}")
    print(f"  Mean position   : ({mean[0]:8.2f}, {mean[1]:8.2f}, {mean[2]:8.2f}) mm")
    print(f"  Std dev (xyz)   : ({std[0]:6.2f}, {std[1]:6.2f}, {std[2]:6.2f}) mm")
    print(f"  Max deviation   : {max_dev:.2f} mm from mean")
    print(f"  RMS scatter     : {rms:.2f} mm")

    if max_dev < 5:
        grade = "EXCELLENT (< 5 mm)"
    elif max_dev < 10:
        grade = "GOOD (< 10 mm)"
    elif max_dev < 20:
        grade = "ACCEPTABLE (< 20 mm)"
    else:
        grade = "POOR (>= 20 mm) - consider re-capturing with more variation"
    print(f"  Quality         : {grade}")
    return max_dev


def main():
    t_path = str(_DATA_DIR / 'T_ee_cam.npy')
    if not Path(t_path).exists():
        raise FileNotFoundError(
            f"No {t_path}. Run the calibration pipeline first so T_ee_cam.npy "
            f"is written before validating."
        )
    T_ee_cam = np.load(t_path)

    euler = R.from_matrix(T_ee_cam[:3, :3]).as_euler('xyz', degrees=True)
    print("=" * 50)
    print("HAND-EYE CALIBRATION VALIDATION (relocated board)")
    print("=" * 50)
    print(f"T_ee_cam translation (mm): {T_ee_cam[:3, 3]}")
    print(f"T_ee_cam rotation (deg)  : roll={euler[0]:.2f} "
          f"pitch={euler[1]:.2f} yaw={euler[2]:.2f}")
    print("\nMove the board to a NEW location and keep it STILL.")
    print(f"Then move the arm through {MIN_POSES}-{MAX_POSES} varied poses.")

    det = ArucoBoardDetector()
    if not det.start_camera():
        return
    det.calibration_data = []

    print("\nControls:")
    print(f"  'c' - Capture current pose (need {MIN_POSES}-{MAX_POSES})")
    print("  's' - Finish & compute RMS spread")
    print("  'q' - Quit without saving")
    print("=" * 60 + "\n")

    while True:
        ret, frame = det.cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        success, T_cam_board, rvec, tvec, _reproj = det.detect_board(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detector.detectMarkers(gray)
        frame = det.draw_detection(frame, rvec, tvec, ids, corners, success)

        n = len(det.calibration_data)
        cv2.putText(frame, f"Validation poses: {n}/{MAX_POSES}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.imshow('Calibration Validation', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if n >= MAX_POSES:
                print(f"Already at {MAX_POSES}; press 's' to finish.")
            elif not success:
                print("Board not detected - try again.")
            else:
                T_base_ee, raw_pose = det.get_robot_end_effector_pose()
                if T_base_ee is None:
                    print("Failed to read robot pose")
                else:
                    det.calibration_data.append({
                        'T_base_ee': T_base_ee,
                        'robot_pose_raw': raw_pose,
                        'T_cam_board': T_cam_board,
                        'timestamp': cv2.getTickCount(),
                    })
                    print(f"✓ Captured validation pose #{len(det.calibration_data)}")
        elif key == ord('s'):
            if n < MIN_POSES:
                print(f"Need at least {MIN_POSES} poses (have {n}).")
            else:
                break

    det.cap.release()
    cv2.destroyAllWindows()

    pairs = det.calibration_data
    if len(pairs) < MIN_POSES:
        print(f"\nAborted: only {len(pairs)} poses captured "
              f"(need {MIN_POSES}-to-{MAX_POSES}). No results saved.")
        return

    np.save(str(_DATA_DIR / 'validation_data.npy'), pairs)
    print(f"✓ Saved {len(pairs)} validation poses -> data/validation_data.npy")

    positions = reconstruct_board_positions(T_ee_cam, pairs)
    report("Reconstructed board position across validation captures", positions)

    print("\nPer-axis spread (mm):")
    for axis, label in enumerate(['X', 'Y', 'Z']):
        vals = positions[:, axis]
        print(f"  {label}: min={vals.min():8.2f}  max={vals.max():8.2f}  "
              f"range={np.ptp(vals):7.2f}")

    print("\nTip: large spread usually means the board moved during capture,")
    print("some frames were mis-detected, or T_ee_cam itself is inaccurate.")


if __name__ == "__main__":
    print(f"Connecting to UFACTORY Lite 6 at {ROBOT_IP}...")
    main()
