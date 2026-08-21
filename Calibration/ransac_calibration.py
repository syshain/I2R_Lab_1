import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RobustHandEyeCalibrator:
    def __init__(self):
        self.calibration_data = []
        self.T_ee_cam = None
        self.inlier_mask = None
        
    # ============================================================
    # ROBOT INTERFACE - UFACTORY Lite 6
    # ============================================================
    def get_robot_end_effector_pose(self, arm):
        """
        Get T_base_ee from UFACTORY Lite 6.
        Uses extrinsic X-Y-Z fixed angles (base frame convention).
        """
        code, pose_data = arm.get_position()  # [x, y, z, roll, pitch, yaw]
        
        if code != 0:
            print(f"Error getting position: {code}")
            return None
        
        x, y, z, roll_deg, pitch_deg, yaw_deg = pose_data
        
        # Convert to radians
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        
        # Build rotation matrix: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        
        R_matrix = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,             cp*cr]
        ])
        
        # Build 4x4 transformation matrix
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = R_matrix
        T_base_ee[:3, 3] = [x, y, z]
        
        return T_base_ee
    
    # ============================================================
    # DATA COLLECTION
    # ============================================================
    def capture_calibration_pose(self, arm, detector, pose_id):
        """
        Capture one calibration pose.
        
        Parameters:
        - arm: UFACTORY arm object
        - detector: Your ArUco board detector
        - pose_id: Current pose number
        """
        # Get robot pose
        T_base_ee = self.get_robot_end_effector_pose(arm)
        if T_base_ee is None:
            return False
        
        # Get camera frame and detect board
        ret, frame = detector.cap.read()
        if not ret:
            print("Failed to capture frame")
            return False
        
        success, T_cam_board, rvec, tvec = detector.detect_board(frame)
        if not success:
            print("Failed to detect board")
            return False
        
        # Store data
        self.calibration_data.append({
            'pose_id': pose_id,
            'T_base_ee': T_base_ee,
            'T_cam_board': T_cam_board,
            'robot_position': T_base_ee[:3, 3].copy(),
            'board_distance': np.linalg.norm(T_cam_board[:3, 3])
        })
        
        print(f"✓ Pose {pose_id}: Robot=({T_base_ee[0,3]:6.1f}, {T_base_ee[1,3]:6.1f}, {T_base_ee[2,3]:6.1f}) mm, "
              f"Board dist={T_cam_board[2,3]:6.1f} mm")
        return True
    
    # ============================================================
    # OUTLIER REJECTION (RANSAC-style)
    # ============================================================
    def compute_pairwise_consistency(self):
        """
        Compute consistency between consecutive poses.
        Lower error = more consistent pair.
        """
        n = len(self.calibration_data)
        consistency_scores = []
        
        for i in range(n - 1):
            T_base1 = self.calibration_data[i]['T_base_ee']
            T_base2 = self.calibration_data[i+1]['T_base_ee']
            T_cam1 = self.calibration_data[i]['T_cam_board']
            T_cam2 = self.calibration_data[i+1]['T_cam_board']
            
            # Compute relative transformations
            T_base_rel = np.linalg.inv(T_base2) @ T_base1
            T_cam_rel = T_cam2 @ np.linalg.inv(T_cam1)
            
            # Rotation difference (angle in degrees)
            R_base = T_base_rel[:3, :3]
            R_cam = T_cam_rel[:3, :3]
            
            # Compute angular difference
            R_diff = R_base @ R_cam.T
            angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            angle_diff_deg = np.degrees(angle_diff)
            
            # Translation difference (normalized)
            t_base = T_base_rel[:3, 3]
            t_cam = T_cam_rel[:3, 3]
            t_diff = np.linalg.norm(t_base - t_cam) / (np.linalg.norm(t_base) + 1e-6)
            
            consistency_scores.append({
                'pair': (i, i+1),
                'angle_error_deg': angle_diff_deg,
                'translation_error': t_diff,
                'combined_score': angle_diff_deg + t_diff * 10
            })
        
        return consistency_scores
    
    def filter_outliers(self, max_angle_error_deg=45.0, keep_ratio=0.9):
        """
        Filter out inconsistent pose pairs.
        
        Parameters:
        - max_angle_error_deg: Maximum allowed angular error between consecutive poses
        - keep_ratio: Fraction of best poses to keep
        """
        if len(self.calibration_data) < 3:
            print("Not enough data for filtering")
            return
        
        # Compute consistency scores
        scores = self.compute_pairwise_consistency()
        
        # Score each pose by average consistency with neighbors
        pose_scores = {}
        for score in scores:
            for idx in score['pair']:
                if idx not in pose_scores:
                    pose_scores[idx] = []
                pose_scores[idx].append(score['angle_error_deg'])
        
        # Average score per pose
        avg_scores = []
        for pose_id, errors in pose_scores.items():
            avg_scores.append((pose_id, np.mean(errors)))
        
        # Sort by quality (lower error = better)
        avg_scores.sort(key=lambda x: x[1])
        
        # Keep best N poses
        n_keep = max(3, int(len(avg_scores) * keep_ratio))
        best_pose_indices = [idx for idx, _ in avg_scores[:n_keep]]
        
        # Also keep poses with low individual error
        self.inlier_mask = np.zeros(len(self.calibration_data), dtype=bool)
        for idx in best_pose_indices:
            self.inlier_mask[idx] = True
        
        print(f"\n Outlier Filtering Results:")
        print(f"  Total poses: {len(self.calibration_data)}")
        print(f"  Kept poses: {np.sum(self.inlier_mask)}")
        print(f"  Removed poses: {len(self.calibration_data) - np.sum(self.inlier_mask)}")
        
        return self.inlier_mask
    
    # ============================================================
    # HAND-EYE CALIBRATION SOLVERS
    # ============================================================
    def solve_hand_eye_tsai(self, use_inliers=True):
        """Solve using Tsai method (OpenCV)."""
        if use_inliers and self.inlier_mask is not None:
            data = [self.calibration_data[i] for i in range(len(self.calibration_data)) if self.inlier_mask[i]]
        else:
            data = self.calibration_data
        
        if len(data) < 3:
            print(f"Need at least 3 poses, have {len(data)}")
            return None
        
        # Extract rotation and translation
        R_base_ee = [d['T_base_ee'][:3, :3] for d in data]
        t_base_ee = [d['T_base_ee'][:3, 3] for d in data]
        
        R_cam_board = [d['T_cam_board'][:3, :3] for d in data]
        t_cam_board = [d['T_cam_board'][:3, 3] for d in data]
        
        # Try different methods
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis")
        ]
        
        best_result = None
        best_consistency = float('inf')
        
        for method, name in methods:
            try:
                R_cam_ee, t_cam_ee = cv2.calibrateHandEye(
                    R_base_ee, t_base_ee,
                    R_cam_board, t_cam_board,
                    method=method
                )
                
                # Build T_ee_cam (invert if needed)
                T_cam_ee = np.eye(4)
                T_cam_ee[:3, :3] = R_cam_ee
                T_cam_ee[:3, 3] = t_cam_ee.flatten()
                T_ee_cam = np.linalg.inv(T_cam_ee)
                
                # Check consistency
                consistency = self.evaluate_consistency(T_ee_cam, data)
                print(f"  {name}: consistency error = {consistency:.2f} mm")
                
                if consistency < best_consistency:
                    best_consistency = consistency
                    best_result = T_ee_cam
                    
            except Exception as e:
                print(f"  {name}: Failed - {e}")
        
        self.T_ee_cam = best_result
        return best_result
    
    def evaluate_consistency(self, T_ee_cam, data):
        """
        Evaluate how consistent the calibration is.
        Returns standard deviation of computed board positions.
        """
        board_positions = []
        
        for d in data:
            T_base_board = d['T_base_ee'] @ T_ee_cam @ d['T_cam_board']
            pos = T_base_board[:3, 3]
            board_positions.append(pos)
        
        board_positions = np.array(board_positions)
        std = np.std(board_positions, axis=0)
        return np.mean(std)
    
    # ============================================================
    # OPTIMIZATION-BASED REFINEMENT
    # ============================================================
    def refine_calibration(self, initial_T_ee_cam=None, use_inliers=True):
        """Refine calibration using nonlinear least squares."""
        
        if use_inliers and self.inlier_mask is not None:
            data = [self.calibration_data[i] for i in range(len(self.calibration_data)) if self.inlier_mask[i]]
        else:
            data = self.calibration_data
        
        if initial_T_ee_cam is None:
            initial_T_ee_cam = self.T_ee_cam
            if initial_T_ee_cam is None:
                print("No initial calibration available")
                return None
        
        # Parameterize T_ee_cam as 6 parameters (tx, ty, tz, rx, ry, rz)
        def param_to_transform(params):
            tx, ty, tz, rx, ry, rz = params
            T = np.eye(4)
            T[:3, 3] = [tx, ty, tz]
            r = R.from_euler('xyz', [rx, ry, rz])
            T[:3, :3] = r.as_matrix()
            return T
        
        def transform_to_param(T):
            tx, ty, tz = T[:3, 3]
            r = R.from_matrix(T[:3, :3])
            rx, ry, rz = r.as_euler('xyz')
            return [tx, ty, tz, rx, ry, rz]
        
        # Residual function: board positions should be consistent
        def residuals(params):
            T_ee_cam = param_to_transform(params)
            positions = []
            
            for d in data:
                T_base_board = d['T_base_ee'] @ T_ee_cam @ d['T_cam_board']
                positions.append(T_base_board[:3, 3])
            
            positions = np.array(positions)
            mean_pos = np.mean(positions, axis=0)
            
            # Residuals: deviation from mean position
            residuals_vec = (positions - mean_pos).flatten()
            return residuals_vec
        
        # Initial parameters
        initial_params = transform_to_param(initial_T_ee_cam)
        
        # Optimize
        print("\n Refining calibration with nonlinear optimization...")
        result = least_squares(residuals, initial_params, method='trf', verbose=0)
        
        # Convert back to transform
        self.T_ee_cam_refined = param_to_transform(result.x)
        
        # Evaluate improvement
        before_std = self.evaluate_consistency(initial_T_ee_cam, data)
        after_std = self.evaluate_consistency(self.T_ee_cam_refined, data)
        
        print(f"  Before refinement: {before_std:.2f} mm std")
        print(f"  After refinement:  {after_std:.2f} mm std")
        print(f"  Improvement: {before_std - after_std:.2f} mm")
        
        return self.T_ee_cam_refined
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    def visualize_results(self, T_ee_cam):
        """Visualize board position consistency."""
        
        if self.inlier_mask is not None:
            inlier_data = [self.calibration_data[i] for i in range(len(self.calibration_data)) if self.inlier_mask[i]]
            outlier_data = [self.calibration_data[i] for i in range(len(self.calibration_data)) if not self.inlier_mask[i]]
        else:
            inlier_data = self.calibration_data
            outlier_data = []
        
        # Compute board positions in base frame
        inlier_positions = []
        for d in inlier_data:
            T_base_board = d['T_base_ee'] @ T_ee_cam @ d['T_cam_board']
            inlier_positions.append(T_base_board[:3, 3])
        
        outlier_positions = []
        for d in outlier_data:
            T_base_board = d['T_base_ee'] @ T_ee_cam @ d['T_cam_board']
            outlier_positions.append(T_base_board[:3, 3])
        
        inlier_positions = np.array(inlier_positions)
        outlier_positions = np.array(outlier_positions) if outlier_positions else np.array([])
        
        # 3D scatter plot
        fig = plt.figure(figsize=(14, 5))
        
        # Plot 1: 3D view
        ax1 = fig.add_subplot(131, projection='3d')
        if len(inlier_positions) > 0:
            ax1.scatter(inlier_positions[:, 0], inlier_positions[:, 1], inlier_positions[:, 2], 
                       c='green', s=50, label='Inliers', alpha=0.7)
        if len(outlier_positions) > 0:
            ax1.scatter(outlier_positions[:, 0], outlier_positions[:, 1], outlier_positions[:, 2], 
                       c='red', s=30, label='Outliers', alpha=0.5)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('Board Position in Robot Base Frame')
        ax1.legend()
        
        # Plot 2: X/Y scatter
        ax2 = fig.add_subplot(132)
        if len(inlier_positions) > 0:
            ax2.scatter(inlier_positions[:, 0], inlier_positions[:, 1], c='green', s=50, alpha=0.7, label='Inliers')
        if len(outlier_positions) > 0:
            ax2.scatter(outlier_positions[:, 0], outlier_positions[:, 1], c='red', s=30, alpha=0.5, label='Outliers')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Board Position (Top View)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Consistency histogram
        ax3 = fig.add_subplot(133)
        all_positions = np.vstack([inlier_positions, outlier_positions]) if len(outlier_positions) > 0 else inlier_positions
        if len(all_positions) > 0:
            center = np.mean(inlier_positions, axis=0) if len(inlier_positions) > 0 else np.mean(all_positions, axis=0)
            distances = np.linalg.norm(all_positions - center, axis=1)
            ax3.hist(distances, bins=20, color='blue', alpha=0.7)
            ax3.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.1f}mm')
            ax3.set_xlabel('Distance from Center (mm)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Board Position Consistency')
            ax3.legend()
            ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        if len(inlier_positions) > 0:
            std = np.std(inlier_positions, axis=0)
            mean = np.mean(inlier_positions, axis=0)
            print("\n" + "="*60)
            print("FINAL CALIBRATION RESULTS")
            print("="*60)
            for i in range(len(inlier_positions)):
                print(inlier_positions[i])
            print("="*60)
            print(f"Board position in base frame (should be constant):")
            print(f"  Mean: ({mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f}) mm")
            print(f"  Std:  ({std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}) mm")
            print(f"  Max deviation from mean: {np.max(np.linalg.norm(inlier_positions - mean, axis=1)):.1f} mm")
            
            if np.max(std) < 5:
                print("\n✓ Calibration quality: EXCELLENT")
            elif np.max(std) < 10:
                print("\n✓ Calibration quality: GOOD")
            elif np.max(std) < 20:
                print("\n⚠ Calibration quality: ACCEPTABLE")
            else:
                print("\n✗ Calibration quality: POOR - Consider re-collecting data")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    def save_results(self, T_ee_cam, filename='T_ee_cam.npy'):
        """Save calibration results."""
        np.save(filename, T_ee_cam)
        
        # Save as text
        with open('T_ee_cam.txt', 'w') as f:
            f.write("Hand-Eye Calibration Result: T_ee_cam (End Effector to Camera)\n")
            f.write("="*60 + "\n\n")
            f.write("4x4 Transformation Matrix:\n")
            for row in T_ee_cam:
                f.write(f"  {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f}\n")
            
            f.write(f"\nTranslation (mm):\n")
            f.write(f"  X: {T_ee_cam[0,3]:.2f}\n")
            f.write(f"  Y: {T_ee_cam[1,3]:.2f}\n")
            f.write(f"  Z: {T_ee_cam[2,3]:.2f}\n")
            
            r = R.from_matrix(T_ee_cam[:3, :3])
            euler = r.as_euler('xyz', degrees=True)
            f.write(f"\nRotation (degrees):\n")
            f.write(f"  Roll:  {euler[0]:.2f}\n")
            f.write(f"  Pitch: {euler[1]:.2f}\n")
            f.write(f"  Yaw:   {euler[2]:.2f}\n")
        
        print(f"\n✓ Saved: {filename}")
        print(f"✓ Saved: T_ee_cam.txt")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Initialize calibrator
    calibrator = RobustHandEyeCalibrator()
    
    # Load your existing calibration data (if already captured)
    # Or capture new data with your ArUco detector
    
    print("="*60)
    print("ROBUST HAND-EYE CALIBRATION FOR UFACTORY LITE 6")
    print("="*60)
    
    # Option 1: Load existing data
    try:
        data = np.load('calibration_data.npy', allow_pickle=True)
        calibrator.calibration_data = data.tolist()
        print(f"\nLoaded {len(calibrator.calibration_data)} existing poses")
    except:
        print("\nNo existing data found. Please capture new poses.")
        # Here you would integrate with your ArUco detector and robot
        # Example:
        # calibrator.capture_calibration_pose(arm, detector, pose_id=1)
    
    # Step 1: Filter outliers
    print("\n" + "="*60)
    print("STEP 1: Outlier Rejection")
    print("="*60)
    calibrator.filter_outliers(max_angle_error_deg=45.0, keep_ratio=0.5)
    
    # Step 2: Solve hand-eye calibration
    print("\n" + "="*60)
    print("STEP 2: Hand-Eye Calibration")
    print("="*60)
    T_ee_cam = calibrator.solve_hand_eye_tsai(use_inliers=True)
    
    if T_ee_cam is not None:
        # Step 3: Refine with optimization
        print("\n" + "="*60)
        print("STEP 3: Nonlinear Refinement")
        print("="*60)
        T_ee_cam_refined = calibrator.refine_calibration(use_inliers=True)
        
        # Step 4: Visualize results
        print("\n" + "="*60)
        print("STEP 4: Visualization")
        print("="*60)
        calibrator.visualize_results(T_ee_cam_refined)
        
        # Step 5: Save results
        print("\n" + "="*60)
        print("STEP 5: Saving Results")
        print("="*60)
        calibrator.save_results(T_ee_cam_refined)
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
        print("\nTo use this calibration in your robot code:")
        print("  import numpy as np")
        print("  T_ee_cam = np.load('T_ee_cam.npy')")
        print("\nThen compute board position in base frame:")
        print("  T_base_board = T_base_ee @ T_ee_cam @ T_cam_board")
    else:
        print("\n✗ Calibration failed. Please check your data.")
