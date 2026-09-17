import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
arm = XArmAPI('192.168.1.153')

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
 
def solve_hand_eye_calibration(calibration_data_file='calibration_data.npy'):
    """
    Solve for T_ee_cam (end effector to camera transform)
    
    The equation we solve: T_base_board = T_base_ee * T_ee_cam * T_cam_board
    where T_base_board is constant across all poses.
    
    This gives us: A * X = X * B
    where: A = T_base_ee, B = T_cam_board, X = T_ee_cam
    """
    
    # Load captured data
    print("="*60)
    print("HAND-EYE CALIBRATION SOLVER")
    print("="*60)
    
    data = np.load(calibration_data_file, allow_pickle=True)
    calibration_data = data.tolist()  # Convert from numpy array to list
    
    print(f"Loaded {len(calibration_data)} calibration poses")
    
    if len(calibration_data) < 3:
        print("ERROR: Need at least 3 poses for calibration!")
        return None
    
    # Extract rotation matrices and translation vectors
    R_base_ee = []  # A matrices (robot base to end effector)
    t_base_ee = []
    
    R_cam_board = []  # B matrices (camera to board)
    t_cam_board = []
    
    for i, pose in enumerate(calibration_data):
#        dh_params = arm.get_dh_params()[1]
#
#        T_base_ee = np.eye(4)
#    
#        for j in range(7):
#            dh_set = dh_params[4*j:4*j+4]
#            T_temp = modified_dh_transform(dh_set[3],dh_set[2],dh_set[1],dh_set[0])
#            T_base_ee = np.dot(T_base_ee,T_temp) 
#        if i == 0:
#            print(T_base_ee)

        T_cam_board = pose['T_cam_board']  # From ArUco detection
        T_base_ee = pose['T_base_ee']
        
        R_base_ee.append(T_base_ee[:3, :3])
        t_base_ee.append(T_base_ee[:3, 3])
        
        R_cam_board.append(T_cam_board[:3, :3])
        t_cam_board.append(T_cam_board[:3, 3])
        
        print(f"Pose {i+1}: Robot pos=({t_base_ee[-1][0]:.1f}, {t_base_ee[-1][1]:.1f}, {t_base_ee[-1][2]:.1f}) mm, "
              f"Board pos=({t_cam_board[-1][0]:.1f}, {t_cam_board[-1][1]:.1f}, {t_cam_board[-1][2]:.1f}) mm")
    
    print("\n" + "="*60)
    print("RUNNING HAND-EYE CALIBRATION...")
    print("="*60)
    
    # ============================================================
    # METHOD 1: OpenCV's calibrateHandEye (most common)
    # ============================================================
    print("\nMethod 1: OpenCV calibrateHandEye")
    
    R_cam_ee, t_cam_ee = cv2.calibrateHandEye(
        R_base_ee, t_base_ee,      # A = gripper2base (robot pose)
        R_cam_board, t_cam_board,  # B = target2cam (board in camera)
        method=cv2.CALIB_HAND_EYE_TSAI  # Tsai method (good for noisy data)
    )
    
    # Build T_ee_cam (end effector to camera)
    # Note: calibrateHandEye returns R_cam_ee, t_cam_ee (camera to end effector)
    # We want end effector to camera, so invert if needed
    T_cam_ee = np.eye(4)
    T_cam_ee[:3, :3] = R_cam_ee
    T_cam_ee[:3, 3] = t_cam_ee.flatten()
    
    # Invert to get T_ee_cam
    T_ee_cam_opencv = np.linalg.inv(T_cam_ee)
    
    print("\nT_ee_cam (End Effector to Camera) from OpenCV:")
    print(T_ee_cam_opencv)
    
    # ============================================================
    # METHOD 2: Manual least squares (alternative verification)
    # ============================================================
#    print("\n" + "="*60)
#    print("Method 2: Manual Least Squares Verification")
#    print("="*60)
#    
#    # Build the system of equations: A * X = X * B
#    # Rearranged to: (A - I) * X = X * B - I  (simplified approach)
#    
#    # Collect all equations into a linear system
#    A_matrices = []
#    B_matrices = []
#    
#    for i in range(len(calibration_data)):
#        T_base_ee = calibration_data[i]['T_base_ee']
#        T_cam_board = calibration_data[i]['T_cam_board']
#        
#        # For each pose, we have constraint: T_base_ee * X * T_cam_board = constant
#        # Which implies: T_base_ee * X = X * T_cam_board_inv
#        # Actually simpler: Use CV method, but let's verify consistency
#        
#        A_matrices.append(T_base_ee)
#        B_matrices.append(T_cam_board)
#    
#    # Alternative: Use scipy's minimization if needed
#    from scipy.optimize import minimize
#    
#    def objective_function(x_flat, A_list, B_list):
#        """Minimize the reprojection error of T_base_board consistency"""
#        X = x_flat.reshape(4, 4)
#        
#        total_error = 0
#        for A, B in zip(A_list, B_list):
#            # T_base_board should be constant = A * X * B
#            T_base_board = A @ X @ B
#            
#            # Add to list to compute variance later
#            total_error += np.linalg.norm(T_base_board[:3, 3])
#        
#        # We want consistent T_base_board, so minimize variance
#        # This is simplified - full implementation would compute actual variance
#        return total_error / len(A_list)
    
    # Try optimization (simpler to just use OpenCV method)
#    X_init = np.eye(4)
#    result = minimize(objective_function, X_init.flatten(), 
#                     args=(A_matrices, B_matrices), method='Nelder-Mead')
#    
#    T_ee_cam_manual = result.x.reshape(4, 4)
#    print("\nT_ee_cam from manual optimization:")
#    print(T_ee_cam_manual)
    
    # ============================================================
    # VERIFICATION: Check consistency of T_base_board
    # ============================================================
    print("\n" + "="*60)
    print("VERIFICATION: Computing board position in robot base frame")
    print("="*60)
    
    positions_opencv = []
#    positions_manual = []
    
    for i, pose in enumerate(calibration_data):
        T_base_ee = pose['T_base_ee']
        T_cam_board = pose['T_cam_board']
        
        # Using OpenCV result
        T_base_board_opencv = T_base_ee @ T_ee_cam_opencv @ T_cam_board
#        T_base_board_man = T_base_ee @ T_ee_cam_manual @ T_cam_board
        pos_opencv = T_base_board_opencv[:3, 3]
#        pos_manual = T_base_board_man[:3,3]
        positions_opencv.append(pos_opencv)
#        positions_manual.append(pos_manual)
        temp_matrix = R.from_matrix(T_base_board_opencv[:3,:3])
        rotations_opencv = temp_matrix.as_euler('xyz',degrees=True)
        
        print(f"Pose {i+1}: OpenCV=({pos_opencv[0]:.1f}, {pos_opencv[1]:.1f}, {pos_opencv[2]:.1f}) mm {rotations_opencv[0]} {rotations_opencv[1]} {rotations_opencv[2]}")
#        print(f"Pose {i+1}: Manual=({pos_manual[0]:.1f}, {pos_manual[1]:.1f}, {pos_manual[2]:.1f}) mm")
    
    # Compute standard deviation (should be low if calibration is good)
    positions_opencv = np.array(positions_opencv)
#    positions_manual = np.array(positions_manual)
    std_opencv = np.std(positions_opencv, axis=0)
    mean_opencv = np.mean(positions_opencv, axis=0)
#    std_manual = np.std(positions_manual, axis=0)
#    mean_manual = np.mean(positions_manual, axis=0)
    
    print("\n" + "="*60)
    print("CONSISTENCY ANALYSIS")
    print("="*60)
    print(f"OpenCV: Mean board position: ({mean_opencv[0]:.1f}, {mean_opencv[1]:.1f}, {mean_opencv[2]:.1f}) mm")
    print(f"OpenCV: Std deviation: ({std_opencv[0]:.1f}, {std_opencv[1]:.1f}, {std_opencv[2]:.1f}) mm")
#    print(f"Manual: Mean board position: ({mean_manual[0]:.1f}, {mean_manual[1]:.1f}, {mean_manual[2]:.1f}) mm")
#    print(f"Manual: Std deviation: ({std_manual[0]:.1f}, {std_manual[1]:.1f}, {std_manual[2]:.1f}) mm")
    
    # Quality assessment
    max_std = np.max(std_opencv)
    if max_std < 5:
        quality = "EXCELLENT (std < 5mm)"
    elif max_std < 10:
        quality = "GOOD (std < 10mm)"
    elif max_std < 20:
        quality = "ACCEPTABLE (std < 20mm)"
    else:
        quality = "POOR - Consider recapturing poses with more variation"
    
    print(f"\nCalibration quality: {quality}")
    
   # SAVE RESULTS
    # ============================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save OpenCV result as primary
    np.save('T_ee_cam.npy', T_ee_cam_opencv)
    print("✓ Saved: T_ee_cam.npy (OpenCV result)")
    
    # Also save as text file for easy reading
    with open('T_ee_cam.txt', 'w') as f:
        f.write("Hand-Eye Calibration Result: T_ee_cam (End Effector to Camera)\n")
        f.write("="*60 + "\n")
        f.write("This transform maps points from end effector frame to camera frame.\n\n")
        f.write("Transformation matrix (4x4):\n")
        for row in T_ee_cam_opencv:
            f.write(f"{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}\n")
        
        f.write(f"\nTranslation (mm): X={T_ee_cam_opencv[0,3]:.2f}, "
                f"Y={T_ee_cam_opencv[1,3]:.2f}, Z={T_ee_cam_opencv[2,3]:.2f}\n")
        
        # Extract Euler angles
        euler = R.from_matrix(T_ee_cam_opencv[:3, :3]).as_euler('xyz', degrees=True)
        f.write(f"Rotation (deg): roll={euler[0]:.2f}, pitch={euler[1]:.2f}, yaw={euler[2]:.2f}\n")
    
    print("✓ Saved: T_ee_cam.txt (human-readable)")
    
    # Save metadata about calibration
    np.savez('calibration_summary.npz',
             T_ee_cam=T_ee_cam_opencv,
             mean_board_position=mean_opencv,
             std_board_position=std_opencv,
             num_poses=len(calibration_data),
             quality=quality)
    print("✓ Saved: calibration_summary.npz")
    
    return T_ee_cam_opencv, std_opencv


def verify_with_live_camera(T_ee_cam):
    """
    Optional: Verify calibration by predicting board position from current robot pose
    """
    print("\n" + "="*60)
    print("LIVE VERIFICATION")
    print("="*60)
    print("This will use your current robot pose to predict where the board should be")
    
    # This would run in your main detection loop
    print("\nIn your main detection code, use:")
    print("""
    # Get current robot pose
    T_base_ee = robot.get_current_pose()  # Your robot API
    
    # Get current camera to board transform from ArUco
    success, T_cam_board = detect_board(frame)  # Your detection function
    
    # Compute board position in robot base frame
    T_base_board = T_base_ee @ T_ee_cam @ T_cam_board
    board_position = T_base_board[:3, 3]
    
    print(f"Board position in robot base: ({board_position[0]:.1f}, {board_position[1]:.1f}, {board_position[2]:.1f}) mm")
    """)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Solve calibration using your captured data
    T_ee_cam, std = solve_hand_eye_calibration('calibration_data.npy')
    
    if T_ee_cam is not None:
        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        print("T_ee_cam (End Effector to Camera):")
        print(T_ee_cam)
        
        print("\nTo use this in your robot code, load with:")
        print("  T_ee_cam = np.load('T_ee_cam.npy')")
        
        # Ask if user wants to run verification
        verify = input("\nRun live verification? (y/n): ").lower()
        if verify == 'y':
            verify_with_live_camera(T_ee_cam)
