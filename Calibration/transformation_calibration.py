import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
arm = XArmAPI('192.168.1.153')

def get_robot_end_effector_pose(arm):
    """
    Gets the current end-effector pose from the UFACTORY Lite 6 robot
    and returns it as a 4x4 homogeneous transformation matrix T_base_ee.
    """
    # Retrieve the current pose in [x, y, z, roll, pitch, yaw] format
    # The SDK returns a tuple: (code, [x, y, z, roll, pitch, yaw])
    code, pose_data = arm.get_position()
    
    if code != 0:
        # Handle error based on your SDK version
        print(f"Error getting position: {code}")
        return None
        
    x, y, z, roll, pitch, yaw = pose_data

    # Create rotation matrix from Euler angles (XYZ fixed angles)
    # Convert degrees to radians
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rotation_matrix = r.as_matrix()
    
    # Build the 4x4 homogeneous transformation matrix
    T_base_ee = np.eye(4)
    T_base_ee[:3, :3] = rotation_matrix
    T_base_ee[:3, 3] = [x, y, z]
    
    return T_base_ee

def modified_dh_transform(alpha, a, d, theta):
    """
    Create the modified DH transformation matrix (Craig's convention)
    Parameters:
    alpha: twist angle (radians)
    a: link length
    d: link offset
    theta: joint angle (radians)
    Returns:
    4x4 transformation matrix from frame i-1 to frame i
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)
    T = np.array([
        [ct, -st, 0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0, 0, 0, 1]
    ])
    return T
   
    
class VerticalArucoBoardDetector:
    def __init__(self, camera_index=4):
        # Load camera calibration
        self.camera_matrix = np.load('camera_matrix.npy')
        self.dist_coeffs = np.load('dist_coeffs.npy')
        
        # ============================================================
        # ARUCO SETUP for OpenCV 4.13
        # ============================================================
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # ============================================================
        # BOARD DEFINITION (Vertical stack: 2 rows, 1 column)
        # ============================================================
        self.marker_length = 100.0      # mm
        self.marker_separation = 25.0   # mm (edge-to-edge gap)
        
        # Build object points for solvePnP
        self._build_board_points()
        
        # Camera setup
        self.cap = None
        self.camera_index = camera_index
        self.calibration_data = []
    
    def _build_board_points(self):
        """
        Build the 3D object points for all markers in the board.
        These points represent the corners of all markers in the board's
        coordinate system (origin at center between both markers).
        """
        # Calculate center positions (Y-axis: positive UP)
        half_total_height = (self.marker_length + self.marker_separation) / 2.0
        y_top = half_total_height      # Top marker center at +62.5mm
        y_bottom = -half_total_height  # Bottom marker center at -62.5mm
        
        # Get 3D corners for both markers
        self.corners_top = self._get_marker_corners_3d(0, y_top)
        self.corners_bottom = self._get_marker_corners_3d(0, y_bottom)
        
        # Combine all corners into a single array for solvePnP
        # Shape: (8, 3) - 8 corners total (4 per marker), each with X,Y,Z
        self.object_points = np.vstack([self.corners_top, self.corners_bottom])
        
        # Store marker IDs and corner indices for reference
        self.marker_ids = [0, 3]
    
    def _get_marker_corners_3d(self, center_x, center_y, z=0):
        """
        Get the 4 corners of a marker in 3D board coordinate system.
        
        Returns 4x3 numpy array (float32) with corners in order:
        top-left, top-right, bottom-right, bottom-left
        """
        half_size = self.marker_length / 2.0  # 50mm
        
        corners = np.array([
            [center_x - half_size, center_y + half_size, z],  # top-left
            [center_x + half_size, center_y + half_size, z],  # top-right
            [center_x + half_size, center_y - half_size, z],  # bottom-right
            [center_x - half_size, center_y - half_size, z]   # bottom-left
        ], dtype=np.float32)
        
        return corners
    
    def start_camera(self):
        """Open camera connection."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        print(f"✓ Camera {self.camera_index} opened")
        print(f"✓ OpenCV version: {cv2.__version__}")
        print(f"✓ Board: 2 markers (IDs 0 and 3) vertically stacked")
        print(f"✓ Marker size: {self.marker_length} mm")
        print(f"✓ Gap between markers: {self.marker_separation} mm")
        return True
    
    def detect_board(self, frame):
        """
        Detect ArUco board and estimate pose using solvePnP.
        
        Returns:
        - success: bool (True if enough markers detected)
        - T_cam_board: 4x4 transformation matrix (camera to board)
        - rvec, tvec: OpenCV rotation vector and translation vector
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is None or len(ids) < 2:
            return False, None, None, None
        
        # Check if we have both markers (0 and 3)
        ids_flat = ids.flatten()
        if not (0 in ids_flat and 3 in ids_flat):
            return False, None, None, None
        
        # ============================================================
        # Build image points for solvePnP
        # Collect 2D image corners for detected markers that belong to our board
        # ============================================================
        image_points = []
        object_points_subset = []
        
        for i, marker_id in enumerate(ids_flat):
            if marker_id == 0:
                # Marker 0 corners - use top marker's 3D points
                marker_corners = corners[i][0]  # Shape: (4, 2)
                for corner in marker_corners:
                    image_points.append(corner)
                for obj_corner in self.corners_top:
                    object_points_subset.append(obj_corner)
            elif marker_id == 3:
                # Marker 3 corners - use bottom marker's 3D points
                marker_corners = corners[i][0]  # Shape: (4, 2)
                for corner in marker_corners:
                    image_points.append(corner)
                for obj_corner in self.corners_bottom:
                    object_points_subset.append(obj_corner)
        
        # Convert to numpy arrays
        image_points = np.array(image_points, dtype=np.float32)
        object_points_subset = np.array(object_points_subset, dtype=np.float32)
        
        # Minimum points needed for solvePnP (need at least 4 points)
        if len(image_points) < 4:
            return False, None, None, None
        
        # ============================================================
        # SOLVEPNP: Estimate camera pose relative to board
        # This replaces the deprecated estimatePoseBoard function
        # ============================================================
        success, rvec, tvec = cv2.solvePnP(
            object_points_subset,  # 3D points in board coordinate system
            image_points,           # 2D points in image plane
            self.camera_matrix,     # Camera intrinsic matrix
            self.dist_coeffs,       # Distortion coefficients
            flags=cv2.SOLVEPNP_ITERATIVE  # Iterative method for better accuracy
        )
        
        if not success:
            return False, None, None, None
        
        # Convert to 4x4 transformation matrix
        R_mat, _ = cv2.Rodrigues(rvec)
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R_mat
        T_cam_board[:3, 3] = tvec.flatten()
        
        return True, T_cam_board, rvec, tvec
    
    def draw_detection(self, frame, rvec, tvec, ids, corners, success):
        """Draw detected markers and board axes."""
        # Draw all detected markers
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
        
        if success and rvec is not None:
            # Draw board coordinate frame (3D axes)
            cv2.drawFrameAxes(
                frame, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, self.marker_length
            )
            
            # Display text info
            cv2.putText(frame, "✓ Board Detected! (IDs 0 and 3)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            pos = tvec.flatten()
            cv2.putText(frame, f"X: {pos[0]:6.1f} mm", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Y: {pos[1]:6.1f} mm", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Z: {pos[2]:6.1f} mm", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "✗ Need both markers (0 and 3)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

    def get_robot_end_effector_pose(self):
        """
        ============================================================
        *** REPLACE THIS WITH YOUR ACTUAL ROBOT API CALL ***
        ============================================================

        Returns:
        - T_base_ee: 4x4 transformation matrix from robot base to end effector

        Examples for different robot types:
        """

        dh_params = arm.get_dh_params()[1]

#        T_base_ee = np.eye(4)
#
#        for i in range(7):
#            dh_set = dh_params[4*i:4*i+4]
#            T_temp = modified_dh_transform(dh_set[3],dh_set[2],dh_set[1],dh_set[0])
#            T_base_ee = np.dot(T_base_ee,T_temp)
        T_base_ee = get_robot_end_effector_pose(arm)

        return T_base_ee
        
    def capture_calibration_pose(self):
        """Capture current robot pose + board pose for hand-eye calibration."""
        if self.cap is None:
            self.start_camera()
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            return False
        
        success, T_cam_board, rvec, tvec = self.detect_board(frame)
        if not success:
            print("Failed to detect both markers (0 and 3)")
            return False
        
        T_base_ee = self.get_robot_end_effector_pose()
        
        self.calibration_data.append({
            'T_base_ee': T_base_ee,
            'T_cam_board': T_cam_board,
            'timestamp': cv2.getTickCount()
        })
        
        evec = T_base_ee[:3,3].flatten()
        print(f"✓ Captured pose pair #{len(self.calibration_data)}")
        print(f" Cam to board transformation")
        for i in range(4):
                print(f"{T_cam_board[i][0]} {T_cam_board[i][1]} {T_cam_board[i][2]} {T_cam_board[i][3]}")

        r1 = R.from_matrix(T_cam_board[:3,:3])
        euler_cam_board = r1.as_euler('zxy', degrees=True)
        print(f"  Board euler: ({euler_cam_board[0]:.1f}, {euler_cam_board[1]:.1f}, {euler_cam_board[2]:.1f}) mm")
#        print(f"  Board position: ({tvec[0][0]:.1f}, {tvec[1][0]:.1f}, {tvec[2][0]:.1f}) mm")

        r2 = R.from_matrix(T_base_ee[:3,:3])
        euler_base_ee = r2.as_euler('zxy', degrees=True)
        print(f"  EE euler: ({euler_base_ee[0]:.1f}, {euler_base_ee[1]:.1f}, {euler_base_ee[2]:.1f}) mm")
        print("Base to EE transformation")
        for i in range(4):
                print(f"{T_base_ee[i][0]} {T_base_ee[i][1]} {T_base_ee[i][2]} {T_base_ee[i][3]}")

#        print(f" EE position: ({evec[0]:.1f}, {evec[1]:.1f}, {evec[2]:.1f}) mm ")
        return True
    
    def run(self, mode='detect'):
        """Main loop."""
        if not self.start_camera():
            return
        
        self.calibration_data = []
        
        print("\n" + "="*60)
        print(f"ARUCO BOARD DETECTION - Mode: {mode.upper()}")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        
        if mode == 'calibrate':
            print("  'c' - Capture current pose (robot + marker)")
            print("  's' - Save captured data")
            print(f"  Currently captured: {len(self.calibration_data)} poses")
        else:
            print("  'c' - Print current transformation")
        
        print("="*60 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            success, T_cam_board, rvec, tvec = self.detect_board(frame)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            frame = self.draw_detection(frame, rvec, tvec, ids, corners, success)
            
            if mode == 'calibrate':
                cv2.putText(frame, f"Calibration poses: {len(self.calibration_data)}", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            
            cv2.imshow('ArUco Board Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                if mode == 'calibrate':
                    self.capture_calibration_pose()
                elif success:
                    print("\n" + "="*40)
                    print("T_cam_board (Camera to Board):")
                    print("="*40)
                    print(T_cam_board)
                    print(f"\nPosition: ({tvec[0][0]:.1f}, {tvec[1][0]:.1f}, {tvec[2][0]:.1f}) mm")
            elif key == ord('s') and mode == 'calibrate':
                if len(self.calibration_data) > 0:
                    np.save('calibration_data.npy', self.calibration_data)
                    print(f"✓ Saved {len(self.calibration_data)} calibration poses")
                else:
                    print("No calibration data to save. Press 'c' to capture poses first.")
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = VerticalArucoBoardDetector(camera_index=4)
    detector.run(mode='calibrate')
