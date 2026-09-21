"""Temporary: re-score the EXISTING validation data (no new capture).

Loads data/validation_data.npy and re-runs the reconstruction + scoring from
src/validate_calibration.py on whatever T_6_C results are present. Reuses the
real module's functions directly (reconstruct_artifact_positions, _stats,
report, write_validation_report) so numbers are computed identically to a live
run — only the live camera/robot/keygate loop is skipped.

NOTE: importing validate_calibration pulls in transformation_calibration, which
imports the xArm SDK; no connection is made at import time, but the package
must be installed. Nothing in src/ or data/ is modified except that the shared
report writer refreshes data/validation_results.txt (same format as a live run).

Run from anywhere:  python3 rerun_validation.py
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / 'src'))

from validate_calibration import (  # noqa: E402
    _DATA_DIR, _load_transform, reconstruct_artifact_positions,
    _stats, report, write_validation_report,
)


def main():
    pairs = np.load(str(_DATA_DIR / 'validation_data.npy'), allow_pickle=True)
    print(f"Loaded {len(pairs)} stored validation poses from "
          f"data/validation_data.npy\n")

    T_normal = _load_transform(str(_DATA_DIR / 'T_6_C_normal.npy'),
                               "Normal (direct)")
    T_ransac = _load_transform(str(_DATA_DIR / 'T_6_C_ransac.npy'),
                               "RANSAC")

    if T_normal is None and T_ransac is None:
        raise FileNotFoundError(
            "Neither T_6_C_normal.npy nor T_6_C_ransac.npy found in data/.")

    transforms, results = {}, {}
    if T_normal is not None:
        s_n = _stats(reconstruct_artifact_positions(T_normal, pairs))
        transforms['Normal (direct)'] = T_normal
        results['Normal (direct)'] = s_n
        report("NORMAL CALIBRATION — reconstructed artifact position", s_n)
    if T_ransac is not None:
        s_r = _stats(reconstruct_artifact_positions(T_ransac, pairs))
        transforms['RANSAC'] = T_ransac
        results['RANSAC'] = s_r
        report("RANSAC CALIBRATION — reconstructed artifact position", s_r)

    if len(results) == 2:
        a, b = list(results.keys())
        sa, sb = results[a], results[b]
        print("\n" + "=" * 60)
        print("SIDE-BY-SIDE COMPARISON (existing validation poses)")
        print("=" * 60)
        print(f"{'Metric':<25} {a[:12]:>12} {b[:12]:>12}")
        print("-" * 50)
        print(f"{'Max deviation (mm)':<25} {sa['max_dev']:>12.2f} {sb['max_dev']:>12.2f}")
        print(f"{'RMS scatter (mm)':<25} {sa['rms']:>12.2f} {sb['rms']:>12.2f}")
        for axis, label in enumerate(['X', 'Y', 'Z']):
            print(f"{'Std ' + label + ' (mm)':<25} {sa['std'][axis]:>12.2f} "
                  f"{sb['std'][axis]:>12.2f}")
        winner = b if sb['max_dev'] < sa['max_dev'] else \
                 (a if sa['max_dev'] < sb['max_dev'] else 'Tie')
        diff = abs(sa['max_dev'] - sb['max_dev'])
        print(f"\n  → {winner} is tighter by {diff:.2f} mm (max-deviation metric).")

    write_validation_report(transforms, results)


if __name__ == '__main__':
    main()
