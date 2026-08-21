import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ============================================================
# HELPERS
# ============================================================
def robot_pose_to_matrix(raw_pose, euler_order='xyz'):
    """
    raw_pose = [x, y, z, roll, pitch, yaw]
    """
    x, y, z, roll, pitch, yaw = raw_pose

    Rmat = R.from_euler(euler_order, [roll, pitch, yaw], degrees=True).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rmat
    T[:3, 3] = [x, y, z]
    return T


def evaluate_consistency(T_ee_cam, data, euler_order):
    """
    Rebuild T_base_ee from raw robot poses using the tested Euler order,
    then compute board positions in robot base frame.

    T_base_board = T_base_ee @ T_ee_cam @ T_cam_board
    """
    positions = []

    for d in data:
        raw_pose = d['robot_pose_raw']
        T_base_ee = robot_pose_to_matrix(raw_pose, euler_order=euler_order)
        T_cam_board = d['T_cam_board']

        T_base_board = T_base_ee @ T_ee_cam @ T_cam_board
        positions.append(T_base_board[:3, 3])

    positions = np.array(positions)
    mean_pos = np.mean(positions, axis=0)
    std_pos = np.std(positions, axis=0)
    max_dev = np.max(np.linalg.norm(positions - mean_pos, axis=1))
    mean_std = np.mean(std_pos)

    return mean_std, mean_pos, std_pos, max_dev, positions


def save_result(T_ee_cam, filename_npy='T_ee_cam.npy', filename_txt='T_ee_cam.txt'):
    np.save(filename_npy, T_ee_cam)

    with open(filename_txt, 'w') as f:
        f.write("Hand-Eye Calibration Result: T_ee_cam\n")
        f.write("=" * 60 + "\n\n")
        f.write("4x4 Transformation Matrix:\n")
        for row in T_ee_cam:
            f.write(f"{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}\n")

        f.write("\nTranslation [mm]:\n")
        f.write(f"X: {T_ee_cam[0,3]:.6f}\n")
        f.write(f"Y: {T_ee_cam[1,3]:.6f}\n")
        f.write(f"Z: {T_ee_cam[2,3]:.6f}\n")

        euler = R.from_matrix(T_ee_cam[:3, :3]).as_euler('xyz', degrees=True)
        f.write("\nEuler xyz [deg]:\n")
        f.write(f"Roll:  {euler[0]:.6f}\n")
        f.write(f"Pitch: {euler[1]:.6f}\n")
        f.write(f"Yaw:   {euler[2]:.6f}\n")


# ============================================================
# MAIN CALIBRATION
# ============================================================
if __name__ == "__main__":
    # Load captured data
    data = np.load("calibration_data.npy", allow_pickle=True).tolist()
    print(f"Loaded {len(data)} poses")

    # Keep only samples that have raw robot pose + T_cam_board
    valid_data = []
    for d in data:
        if 'robot_pose_raw' in d and 'T_cam_board' in d:
            valid_data.append(d)

    print(f"Valid poses with raw robot pose: {len(valid_data)}")

    if len(valid_data) < 5:
        print("Not enough valid data.")
        raise SystemExit

    # Euler conventions to test
    euler_orders = [
        'xyz', 'xzy',
        'yxz', 'yzx',
        'zxy', 'zyx'
    ]

    # Hand-eye methods to test
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
        (cv2.CALIB_HAND_EYE_PARK, "Park"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
    ]

    best = None
    best_score = float("inf")

    print("\n" + "=" * 90)
    print("TESTING ALL EULER CONVENTIONS AND HAND-EYE METHODS")
    print("=" * 90)

    for euler_order in euler_orders:
        # Rebuild robot transforms using this Euler interpretation
        R_gripper2base = []
        t_gripper2base = []

        for d in valid_data:
            T_base_ee = robot_pose_to_matrix(d['robot_pose_raw'], euler_order=euler_order)
            R_gripper2base.append(T_base_ee[:3, :3])
            t_gripper2base.append(T_base_ee[:3, 3])

        R_target2cam = [d['T_cam_board'][:3, :3] for d in valid_data]
        t_target2cam = [d['T_cam_board'][:3, 3] for d in valid_data]

        print(f"\nEuler order: {euler_order}")
        print("-" * 90)

        for method, method_name in methods:
            try:
                R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
                    R_gripper2base,
                    t_gripper2base,
                    R_target2cam,
                    t_target2cam,
                    method=method
                )

                T_ee_cam = np.eye(4, dtype=np.float64)
                T_ee_cam[:3, :3] = R_ee_cam
                T_ee_cam[:3, 3] = t_ee_cam.flatten()

                mean_std, mean_pos, std_pos, max_dev, positions = evaluate_consistency(
                    T_ee_cam, valid_data, euler_order
                )

                print(
                    f"{method_name:10s} | mean std = {mean_std:8.3f} mm "
                    f"| std = {std_pos} | max dev = {max_dev:8.3f} mm"
                )

                if mean_std < best_score:
                    best_score = mean_std
                    best = {
                        'euler_order': euler_order,
                        'method_name': method_name,
                        'T_ee_cam': T_ee_cam,
                        'mean_std': mean_std,
                        'mean_pos': mean_pos,
                        'std_pos': std_pos,
                        'max_dev': max_dev,
                        'positions': positions
                    }

            except Exception as e:
                print(f"{method_name:10s} | failed: {e}")

    if best is None:
        print("\nCalibration failed.")
        raise SystemExit

    # ============================================================
    # REPORT BEST RESULT
    # ============================================================
    T_ee_cam = best['T_ee_cam']
    euler_xyz = R.from_matrix(T_ee_cam[:3, :3]).as_euler('xyz', degrees=True)

    print("\n" + "=" * 90)
    print("BEST RESULT")
    print("=" * 90)
    print(f"Best Euler order : {best['euler_order']}")
    print(f"Best HE method   : {best['method_name']}")
    print(f"Mean std [mm]    : {best['mean_std']:.3f}")
    print(f"Std XYZ [mm]     : {best['std_pos']}")
    print(f"Max dev [mm]     : {best['max_dev']:.3f}")
    print("\nT_ee_cam =")
    print(T_ee_cam)
    print(f"\nTranslation [mm]: {T_ee_cam[:3, 3]}")
    print(f"Euler xyz [deg]: {euler_xyz}")

    # Quality message
    if np.max(best['std_pos']) < 5:
        print("\n✓ Calibration quality: EXCELLENT")
    elif np.max(best['std_pos']) < 10:
        print("\n✓ Calibration quality: GOOD")
    elif np.max(best['std_pos']) < 20:
        print("\n⚠ Calibration quality: ACCEPTABLE")
    else:
        print("\n✗ Calibration quality: POOR")

    # Save best result
    save_result(T_ee_cam)
    print("\nSaved:")
    print("  T_ee_cam.npy")
    print("  T_ee_cam.txt")