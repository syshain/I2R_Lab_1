"""Temporary diagnostic: camera calibration reprojection error with k3 = 0.

Runs the same corner detection + cv.calibrateCamera as src/camera_calibration.py,
but forces the radial coefficient k3 to zero (distortion model 'k1_k2_p' instead
of the default 'k1_k2_p_k3') and compares overall RMS reprojection error against
the stored full-model result. Headless — no windows opened, nothing written.

Run from anywhere:  python check_k3_zero.py
"""

import glob
import os
from pathlib import Path

import cv2 as cv
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR / 'src'
_DATA_DIR = _SCRIPT_DIR / 'data'

# Same constants the pipeline uses (mirrored here so this script is standalone).
CHESSBOARD_SIZE = (10, 7)
SQUARE_SIZE_MM = 15.0
IMAGE_PATTERN = 'calib_*.jpg'


def collect_corners():
    """Detect + refine chessboard corners in all calib images (headless)."""
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                           0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints, imgpoints, good_files = [], [], []
    for fname in sorted(glob.glob(str(_DATA_DIR / IMAGE_PATTERN))):
        img = cv.imread(fname)
        if img is None:
            print(f"✗ Could not read: {os.path.basename(fname)}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if not ret:
            print(f"✗ No chessboard found: {os.path.basename(fname)}")
            continue
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        good_files.append(os.path.basename(fname))

    return objpoints, imgpoints, good_files


def per_image_errors(rvecs, tvecs, K, dist, objpoints, imgpoints):
    """Per-image mean reprojection error (px) using projectPoints."""
    errs = []
    for i in range(len(imgpoints)):
        reproj, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = np.linalg.norm(reproj - imgpoints[i].reshape(-1, 1, 2), axis=2)
        errs.append(float(err.mean()))
    return errs


def main():
    objpoints, imgpoints, good_files = collect_corners()
    n_good = len(good_files)
    print(f"\nGood images: {n_good}/{len(list(glob.glob(str(_DATA_DIR / IMAGE_PATTERN))))}")
    if n_good < 5:
        print("ERROR: need at least 5 good images.")
        return

    first = cv.imread(str(_DATA_DIR / good_files[0]))
    h, w = first.shape[:2]

    # --- Full model (as in camera_calibration.py: default k1,k2,p1,p2,k3) ----
    rms_full, K_full, D_full, rv_f, tv_f = cv.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None)

    # --- With k3 fixed at 0 ----------------------------------------------------
    # Seed dist_coeffs with zeros and fix k3: calibrateCamera then optimises only
    # k1, k2, p1, p2 (extrinsics are always recomputed per iteration), which is
    # equivalent to a 'k1,k2,p1,p2' distortion model.
    dist_seed = np.zeros((5, 1), dtype=np.float64)
    rms_no_k3, K_nok3, D_nok3, rv_n, tv_n = cv.calibrateCamera(
        objpoints, imgpoints, (w, h), None, dist_seed,
        flags=cv.CALIB_FIX_K3)

    print("\n" + "=" * 64)
    print("REPROJECTION ERROR COMPARISON")
    print("=" * 64)
    print(f"Image size: {w} x {h}px   |   Good frames: {n_good}")
    print("-" * 64)
    print(f"Full model (k1,k2,p1,p2,k3):  RMS = {rms_full:.4f} px")
    print(f"k3 = 0      (k1,k2,p1,p2):    RMS = {rms_no_k3:.4f} px")
    delta = rms_no_k3 - rms_full
    print(f"Delta (no-k3 minus full):     {delta:+.4f} px "
          f"({100 * delta / rms_full:+.2f}% relative)")
    print("-" * 64)

    print("\nDistortion coefficients:")
    names = ['k1', 'k2', 'p1', 'p2', 'k3']
    d_full = D_full.flatten()
    d_nok3 = D_nok3.flatten()
    for j, name in enumerate(names):
        val_full = d_full[j] if j < len(d_full) else float('nan')
        val_nok3 = d_nok3[j] if j < len(d_nok3) else float('nan')
        print(f"  {name:>3}: full={val_full:+.6e}   k3=0 run={val_nok3:+.6e}")

    # Per-image breakdown for both fits
    e_full = per_image_errors(rv_f, tv_f, K_full, D_full, objpoints, imgpoints)
    e_nok3 = per_image_errors(rv_n, tv_n, K_nok3, D_nok3, objpoints, imgpoints)
    print("\nPer-image mean reprojection error (px):")
    print(f"{'image':<14} {'full':>8} {'k3=0':>8} {'diff':>8}")
    for i, name in enumerate(good_files):
        print(f"{name:<14} {e_full[i]:8.3f} {e_nok3[i]:8.3f} "
              f"{e_nok3[i]-e_full[i]:+8.3f}")

    # Interpretation
    print("\n" + "=" * 64)
    tol = 0.05  # px difference considered negligible
    if abs(delta) <= tol:
        print(f"CONCLUSION: dropping k3 changes RMS by only {delta:+.4f} px (< {tol}).")
        print("The fitted k3 carries almost no information -> a mild overfit of")
        print("the 5th distortion term; k3 ≈ 0 is safe to assume.")
    elif delta > tol:
        print(f"CONCLUSION: forcing k3 = 0 worsens RMS by {delta:+.4f} px (> {tol}).")
        print("The data does use the 5th radial term somewhat; keep the full")
        print("model unless downstream PnP behaves poorly.")
    else:
        print(f"CONCLUSION: forcing k3 = 0 improves RMS by {-delta:+.4f} px — unusual;")
        print("check image quality / outliers before trusting either fit.")
    print("=" * 64)


if __name__ == '__main__':
    main()
