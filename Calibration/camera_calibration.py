import numpy as np
import cv2 as cv
import glob
import os

def calibrate_camera(images_path='*.jpg', chessboard_size=(10,7), square_size_mm=25.0):
    """
    Calibrate camera using chessboard images.
    
    Parameters:
    - images_path: pattern to match image files (default '*.jpg')
    - chessboard_size: (inner corners per row, inner corners per column)
    - square_size_mm: physical size of each chessboard square in mm
    """
    
    # Termination criteria for sub-pixel refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (3D coordinates of chessboard corners in real world)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size_mm  # Scale to real-world units (mm)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    good_images = []
    bad_images = []
    
    # Get list of images
    images = glob.glob(images_path)
    print(f"Found {len(images)} images")
    
    if len(images) == 0:
        print("ERROR: No images found! Check your path and file pattern.")
        return None
    
    # Process each image
    for idx, fname in enumerate(images):
        img = cv.imread(fname)
        if img is None:
            print(f"✗ Could not read: {fname}")
            bad_images.append(fname)
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corners to sub-pixel accuracy
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            good_images.append(fname)
            
            # Draw and display corners for verification
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow('Calibration Progress', img)
            cv.waitKey(100)  # Show each for 0.1 seconds
            
            print(f"✓ [{len(good_images)}] Processed: {os.path.basename(fname)}")
        else:
            bad_images.append(fname)
            print(f"✗ No chessboard found: {os.path.basename(fname)}")
    
    cv.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("IMAGE PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images: {len(images)}")
    print(f"Good images: {len(good_images)}")
    print(f"Bad images: {len(bad_images)}")
    
    if bad_images:
        print("\nBad images (remove or recapture these):")
        for img in bad_images:
            print(f"  - {os.path.basename(img)}")
    
    # Check if we have enough good images
    if len(good_images) < 5:
        print("\nERROR: Need at least 5 good images for calibration!")
        print(f"Only found {len(good_images)}. Please capture more images.")
        return None
    
    # Run calibration
    print("\n" + "="*60)
    print("RUNNING CALIBRATION...")
    print("="*60)
    
    # Get image size from first good image
    img = cv.imread(good_images[0])
    h, w = img.shape[:2]
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )
    
    # Calculate reprojection error for each image
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    # Display results
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"Image size: {w} x {h} pixels")
    print(f"Reprojection error (mean): {mean_error:.4f} pixels")
    print(f"Calibration quality: ", end="")
    
    if mean_error < 0.3:
        print("EXCELLENT ✓")
    elif mean_error < 0.5:
        print("GOOD ✓")
    elif mean_error < 1.0:
        print("ACCEPTABLE")
    else:
        print("POOR - Consider recapturing images")
    
    print("\nCamera Matrix (intrinsic parameters):")
    print("┌" + "─"*50 + "┐")
    for row in camera_matrix:
        print(f"│ {row[0]:12.4f} {row[1]:12.4f} {row[2]:12.4f} │")
    print("└" + "─"*50 + "┘")
    
    print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
    print(dist_coeffs.flatten())
    
    # Save results
    output_files = {
        'camera_matrix.npy': camera_matrix,
        'dist_coeffs.npy': dist_coeffs,
        'calibration_results.npz': {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'reprojection_error': mean_error,
            'image_size': (w, h),
            'chessboard_size': chessboard_size,
            'square_size_mm': square_size_mm
        }
    }
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save as .npy files (easy to load in OpenCV)
    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)
    print("✓ Saved: camera_matrix.npy")
    print("✓ Saved: dist_coeffs.npy")
    
    # Save as .npz (contains all info)
    np.savez('calibration_results.npz',
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             reprojection_error=mean_error,
             image_size=(w, h),
             chessboard_size=chessboard_size,
             square_size_mm=square_size_mm)
    print("✓ Saved: calibration_results.npz")
    
    # Also save as text file for reference
    with open('calibration_parameters.txt', 'w') as f:
        f.write("CAMERA CALIBRATION PARAMETERS\n")
        f.write("="*40 + "\n\n")
        f.write(f"Image size: {w} x {h}\n")
        f.write(f"Reprojection error: {mean_error:.4f} pixels\n\n")
        f.write("Camera Matrix:\n")
        f.write(str(camera_matrix) + "\n\n")
        f.write("Distortion Coefficients:\n")
        f.write(str(dist_coeffs.flatten()) + "\n")
    print("✓ Saved: calibration_parameters.txt")
    
    return camera_matrix, dist_coeffs, mean_error
# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CAMERA CALIBRATION TOOL")
    print("="*60)
    print("\nThis script will calibrate your camera using chessboard images.")
    
    # Configuration - MODIFY THESE TO MATCH YOUR SETUP!
    CHESSBOARD_SIZE = (10, 7)  # (inner corners per row, inner corners per column)
    SQUARE_SIZE_MM = 25.0     # Measure your printed chessboard square size in mm
    IMAGE_PATTERN = "*.jpg"   # Pattern to match your calibration images
    
    print(f"\nUsing configuration:")
    print(f"  - Chessboard pattern: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
    print(f"  - Square size: {SQUARE_SIZE_MM} mm")
    print(f"  - Image pattern: {IMAGE_PATTERN}")
    
    # Run calibration
    result = calibrate_camera(IMAGE_PATTERN, CHESSBOARD_SIZE, SQUARE_SIZE_MM)
    
    if result is not None:
        camera_matrix, dist_coeffs, error = result
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
    else:
        print("\nCalibration failed. Please check your images and try again.")
