import os
import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'

# Shared hardware config lives with the Lab 1 src scripts.
sys.path.insert(0, str(_SCRIPT_DIR.parent / 'src'))
from lab_config import ROBOT_IP, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


def get_robot_pose_raw(arm):
    """Returns raw robot pose: [x, y, z, roll, pitch, yaw]"""
    code, pose = arm.get_position()
    if code != 0:
        print("❌ Robot error")
        return None
    return np.array(pose, dtype=np.float64)


def robot_pose_to_matrix(raw_pose, euler_order='xyz'):
    """
    Convert raw robot pose [x, y, z, roll, pitch, yaw]
    into homogeneous transform T_base_ee.
    """
    x, y, z, roll, pitch, yaw = raw_pose
    Rmat = R.from_euler(
        euler_order, [roll, pitch, yaw], degrees=True
    ).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rmat
    T[:3,  3] = [x, y, z]
    return T


# ============================================================
# 3D ARUCO BOARD DETECTOR
# ============================================================
class Aruco3DBoardDetector:
    """
    3D ArUco polyhedron detector.
    Loads corner coordinates from board_config.json.
    Scales from 30mm -> 40mm markers (scale = 4/3).
    Re-centers so origin = geometric center of polyhedron.
    """

    CORNER_ROLL = {
        0: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
    }

    def __init__(self, camera_index=None, config_path=None):
        if camera_index is None:
            camera_index = CAMERA_INDEX
        if config_path is None:
            config_path = str(_DATA_DIR / 'board_config.json')

        self.camera_matrix = np.load(str(_DATA_DIR / 'camera_matrix.npy'))
        self.dist_coeffs   = np.load(str(_DATA_DIR / 'dist_coeffs.npy'))

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(self.aruco_dict, params)

        self.cap          = cv2.VideoCapture(camera_index)
        self.camera_index = camera_index
        self.scale        = 4.0 / 3.0
        self.config_path  = config_path

        self._build_3d_model()

    # ─────────────────────────────────────────────────────────────────────
    def _build_3d_model(self):
        with open(self.config_path, 'r') as f:
            cfg = json.load(f)

        board   = cfg['toolList'][0]
        ids     = board['marker_ids']
        corners = board['marker_corners_mm']

        # Scale + winding correction
        scaled = {}
        for mid, raw in zip(ids, corners):
            pts  = np.array(raw, dtype=np.float64) * self.scale
            roll = self.CORNER_ROLL.get(mid, 0)
            scaled[mid] = np.roll(pts, -roll, axis=0)

        # Re-center: subtract centroid of all corners
        all_pts  = np.vstack([scaled[mid] for mid in ids])
        centroid = all_pts.mean(axis=0)

        self.marker_world    = {}
        self.marker_ids_list = ids
        self.centroid_offset = centroid

        for mid in ids:
            self.marker_world[mid] = scaled[mid] - centroid

        print(f"[Model] Loaded {len(ids)} markers  "
              f"scale={self.scale:.4f}  "
              f"centroid=[{centroid[0]:.2f},"
              f"{centroid[1]:.2f},{centroid[2]:.2f}]")

    # ─────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray):
        """
        Detect markers and solve for camera->object transform.

        Returns:
            success      : bool
            T_cam_board  : (4,4) homogeneous transform
            rvec         : (3,1)
            tvec         : (3,1)
            reproj_error : float (pixels)
            corners      : raw ArUco corners
            ids          : raw ArUco ids
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return False, None, None, None, None, corners, ids

        ids_flat      = ids.flatten()
        object_points = []
        image_points  = []
        matched_ids   = []

        for i, mid in enumerate(ids_flat):
            if mid in self.marker_world:
                roll      = self.CORNER_ROLL.get(mid, 0)
                world_pts = np.roll(self.marker_world[mid], -roll, axis=0)
                for j in range(4):
                    object_points.append(world_pts[j])
                    image_points.append(corners[i][0][j])
                matched_ids.append(int(mid))

        if len(object_points) < 4:
            return False, None, None, None, None, corners, ids

        object_points = np.array(object_points, dtype=np.float32)
        image_points  = np.array(image_points,  dtype=np.float32)

        # ── solvePnP ──────────────────────────────────────────────────────
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )

        if not success:
            return False, None, None, None, None, corners, ids

        # ── LM refinement ─────────────────────────────────────────────────
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                rvec, tvec
            )
        except Exception:
            pass

        # ── Reprojection error ─────────────────────────────────────────────
        proj, _ = cv2.projectPoints(
            object_points, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        reproj_error = float(np.mean(
            np.linalg.norm(
                proj.reshape(-1, 2) - image_points, axis=1
            )
        ))

        if reproj_error > 15.0:
            print(f"[REJECT] reproj_err={reproj_error:.2f}px > 15px")
            return False, None, None, None, reproj_error, corners, ids

        R_mat, _ = cv2.Rodrigues(rvec)
        T_cam_board         = np.eye(4, dtype=np.float64)
        T_cam_board[:3, :3] = R_mat
        T_cam_board[:3,  3] = tvec.flatten()

        return True, T_cam_board, rvec, tvec, reproj_error, corners, ids


# ============================================================
# SAFE AXIS DRAWING
# ============================================================
def draw_frame_axes_safe(img, camera_matrix, dist_coeffs,
                          rvec, tvec, length=20.0, margin=20):
    """Draw frame axes only if all points project within frame bounds."""
    axis_pts = np.float32([
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ]).reshape(-1, 3)

    origin = np.float32([[0, 0, 0]])

    img_origin, _ = cv2.projectPoints(
        origin, rvec, tvec, camera_matrix, dist_coeffs
    )
    img_pts, _ = cv2.projectPoints(
        axis_pts, rvec, tvec, camera_matrix, dist_coeffs
    )

    h, w = img.shape[:2]
    for pt in [img_origin[0][0]] + [p[0] for p in img_pts]:
        x, y = pt
        if not (margin <= x <= w - margin and margin <= y <= h - margin):
            return False

    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, length)
    return True


# ============================================================
# MAIN
# ============================================================
def main():
    print("3D ArUco Board Pose Capture")
    print("Using 3D polyhedron board (board_config.json)")
    print("Auto-save enabled after every capture")
    print("Press 'c' to capture")
    print("Press 's' to save captured data")
    print("Press 'q' to quit\n")

    arm = XArmAPI(ROBOT_IP)

    detector = Aruco3DBoardDetector()

    if not detector.cap.isOpened():
        print(f"Failed to open camera {detector.camera_index}")
        raise SystemExit

    detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    np.set_printoptions(precision=6, suppress=True)

    data = []
    pose_id = 0

    while True:
        ret, frame = detector.cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        success, T_cam_board, rvec, tvec, reproj_error, corners, ids = \
            detector.detect(frame)

        # ── Build IDs string first so it's available everywhere ────────────
        if ids is not None:
            ids_text = ",".join(map(str, ids.flatten().tolist()))
        else:
            ids_text = "none"

        # ── Draw all detected markers ──────────────────────────────────────
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(
                frame, f"Detected IDs: {ids_text}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
            )

        # ── Overlay ────────────────────────────────────────────────────────
        if success:
            drawn = draw_frame_axes_safe(
                frame,
                detector.camera_matrix,
                detector.dist_coeffs,
                rvec, tvec,
                length=20.0
            )

            if not drawn:
                origin_2d, _ = cv2.projectPoints(
                    np.float32([[0, 0, 0]]),
                    rvec, tvec,
                    detector.camera_matrix,
                    detector.dist_coeffs
                )
                cv2.circle(
                    frame,
                    tuple(origin_2d[0][0].astype(int)),
                    5, (0, 255, 0), -1
                )

            cv2.putText(
                frame,
                f"Board detected [{ids_text}] — press 'c' to capture",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Reproj err: {reproj_error:.2f} px",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )
            y_text = 90

        else:
            cv2.putText(
                frame, "Board not detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )
            y_text = 60

        cv2.putText(
            frame, f"Poses saved: {len(data)}",
            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 0), 2
        )

        cv2.imshow("3D ArUco Board Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── CAPTURE ───────────────────────────────────────────────────────
        if key == ord('c'):
            if not success:
                print("Board not detected — cannot capture.")
                continue

            if reproj_error is None or reproj_error > 4.0:
                print(f"Pose rejected: reproj error too high "
                      f"({reproj_error:.2f} px)")
                continue

            raw_pose = get_robot_pose_raw(arm)
            if raw_pose is None:
                print("Could not read robot pose. Check connection.")
                continue

            robot_euler_order_used = 'xyz'
            T_base_ee = robot_pose_to_matrix(
                raw_pose, euler_order=robot_euler_order_used
            )

            pose_id += 1
            sample = {
                'pose_id'               : pose_id,
                'robot_pose_raw'        : raw_pose,
                'robot_euler_order_used': robot_euler_order_used,
                'T_base_ee'             : T_base_ee,
                'T_cam_board'           : T_cam_board,
                'marker_ids_used'       : ids.flatten().tolist()
                                          if ids is not None else [],
                'reprojection_error_px' : float(reproj_error),
                'detected_ids'          : ids.flatten().tolist()
                                          if ids is not None else [],
            }

            data.append(sample)

            # ── Terminal log ───────────────────────────────────────────────
            print("\n"+"="*40)
            print(f"Pose {pose_id} captured and saved")
            print(f"Reprojection error : {reproj_error:.2f} px")
            print(f"Raw robot pose     : {raw_pose}")
            print("\n"+"="*40)

        # ── QUIT ──────────────────────────────────────────────────────────
        elif key == ord('q'):
            break

        elif key == ord('s'):
            if len(data) > 0:
                np.save(str(_DATA_DIR / 'calibration_data.npy'),
                        np.array(data, dtype=object))
                print(f"Saved {len(data)} calibration poses to {_DATA_DIR}")
            else:
                print("No calibration data to save. Press 'c' during run to capture poses")

    detector.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
