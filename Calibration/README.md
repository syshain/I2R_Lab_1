## Camera calibration and hand eye solver - based transformations scripts
This repository contains camera calibration and hand-eye solver scripts for Robotics module labs involving the UFactory Lite 6. This document summarizes the programs and their roles

### Requirements are listed in the requirements.txt file
Simply run pip install requirements.txt if inside a virtual python environment, or its equivalent from a conda environment.

## Usage
- Connect the computer to the robot following the UFactory [Quick Start Guide](https://static.generation-robots.com/media/ufactory-lite6-quick-start.pdfhttps://static.generation-robots.com/media/ufactory-lite6-quick-start.pdf)   
- Camera calibration
    - Print out a chessboard pattern on a white sheet, several generators can be found online. [Here's](https://markhedleyjones.com/projects/calibration-checkerboard-collection) an example
    - Note the checker size and number of vertices (inner vertices) in the printed pattern
    - Identify the USB device index that the camera is connected to the computer through
        - On Linux, use v4l2-ctl --list-devices
        - On Windows, see connected devices
        - Change the index in cv2.VideoCapture(4) to match your index
    - Run the program image_capture.py 
        - Ensure the correct camera index is selected from before. The pop-up window should display a clear, RGB image if the correct index has been selected
        - Capture images by pressing the SPACE bar. Try to capture from various angles and orientations, always ensuring the full checkerboard can be seen in the image
        - Good practice is to capture 10-15 images
        - The script should save several .jpg files to your working directory
    - Modify the script camera_calibration.py to change the CHESSBOARD_SIZE and SQUARE_SIZE_MM parameters in the main function. Run this script. The script will use the .jpg files generated earlier to calibrate your camera. 
    - Find camera_matrix.npy, dist_coeffs.npy, calibration_results.npz and calibration_parameters.txt in the working directory. The .npy files will be loaded into programs that use the camera later

- Transformation matrix identification. The transformation matrix from the end effector to the camera frame is unknown. The following routine is recommended to identify an appropriate transformation matrix
    - Attach the camera to the robot using an appropriate mount
    - Run the file transformation_calibration.py 
    - The script in its current form expects a 2x1 arrangement of 100mm x 100mm arUco markers of indices 0 and 3 respectively. Ensure these are used, or change the script accordingly.
    - Move the robot manually with the camera attached. Capture the pose of the robot and the pose of the arUco marker by pressing the 'c' key on your keyboard. Ensure that both markers have been detected by the script every time the poses are captured. The program execution window should display the transformation matrices and the euler angles corresponding to the base->EE transformation and the camera->ArUco transformation. 
        - Good practice is to capture 10-15 images, but it's likely to be insufficient due to the presence of outliers. Capture images in various poses
        - Press 's' to save the captured data once sufficient poses have been captured (get 20-30), then 'q' to quit the program. Find calibration_data.npy in your working directory
    - Now run ransac_calibration.py. The script should attempt to trim the dataset provided in calibration_data.npy and identify an appropriate solution for the transformation matrix. This is likely to be inaccurate. The script will refine this matrix using a nonlinear optimization routine to reduce the error and judge the results of the calibration. If the calibration results are GOOD or EXCELLENT, you may proceed. Else, re-run the calibration. Tune the function parameters keep_ratio if the outlier detection is too aggressive.
    - Find T_ee_cam.npy in your working directory. This will be used to compute positions from the base frame in future scripts.

