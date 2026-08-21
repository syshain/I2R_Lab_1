# LAB 1 — Forward Kinematics, Camera Calibration & Hand-Eye Transformation

**Module:** Robotics for Mechanical Engineers (Undergraduate / MSc)
**Robot Platform:** UFACTORY Lite 6 (6-DOF articulated arm)
**Duration:** 3 hours
**Software:** MATLAB (Live Script), Python 3.12+, OpenCV 4.x

---

## Learning Objectives

By the end of this lab you will be able to:

1. Derive and implement forward kinematics for a 6-DOF serial manipulator using Standard Denavit-Hartenberg (DH) parameters in MATLAB.
2. Verify computed forward kinematics against the robot controller's reported end-effector pose using Python.
3. Calibrate an external camera using a chessboard pattern and compute intrinsic parameters (camera matrix and distortion coefficients).
4. Determine the rigid-body transformation between the robot end-effector frame and the camera frame (hand-eye calibration).

---

## Part A — Forward Kinematics in MATLAB

### Background

Forward kinematics maps joint angles **q** = [q₁, q₂, q₃, q₄, q₅, q₆]ᵀ to the end-effector pose expressed as a 4×4 homogeneous transformation matrix **T**₀⁶ that describes the position and orientation of the tool centre point relative to the robot base frame.

The Standard DH convention builds each inter-joint transform as:

$$A_i(\theta_i, d_i, a_i, \alpha_i) = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \phantom{-}\cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}$$

where θᵢ = qᵢ + θ_offset,i accounts for the mechanical zero-offset of each joint.

The full forward kinematics is the chain product:

$$^0_6T = A_1 \cdot A_2 \cdot A_3 \cdot A_4 \cdot A_5 \cdot A_6$$

### Standard DH Parameters — UFACTORY Lite 6

| Joint | θ offset (°) | d (mm) | α (°) | a (mm) |
|-------|-------------|--------|-------|--------|
| J1    | 0           | 243.3  | −90   | 0      |
| J2    | −90         | 0      | 180   | 200    |
| J3    | −90         | 0      | 90    | 87     |
| J4    | 0           | 227.6  | 90    | 0      |
| J5    | 0           | 0      | −90   | 0      |
| J6    | 0           | 61.5   | 0     | 0      |

*Source: UFACTORY official documentation — "Kinematic and Dynamic Parameters of UFACTORY Lite 6".*

### Task A1 — Run the Provided Live Script

1. Open `FK_test/position_ufactory_standardDH.mlx` in MATLAB.
2. Read through the script. Note how it:
   - Defines the six DH parameter sets (d, a, alpha in mm and degrees).
   - Generates random joint angles within safe limits (shoulder ±45°, elbow ±45° to avoid self-collision).
   - Applies the joint angle offsets (θ₂ = q₂ − 90°, θ₃ = q₃ − 90°).
   - Chains six DH transforms into the composite matrix **T**.
   - Extracts the end-effector position from columns 1–3, row 4 of **T**.
3. Run the script several times. Each run generates different random joint angles. Record one set of joint angles and the corresponding end-effector position (in mm).

### Task A2 — Manual Verification

Pick one configuration from Task A1. By hand (or in a separate MATLAB editor window), compute the first two transforms **A**₁ and **A**₂ symbolically or numerically, then multiply them. Compare your result with the intermediate transforms printed by the script. This exercise confirms you understand how each DH parameter enters the matrix.

### Task A3 — Explore Singular Configurations

Modify the random-angle ranges in the live script so that q₂ = 0° and q₃ = 0° (fully extended arm). Observe the end-effector position. What happens to the reachable workspace when joints 2 and 3 are aligned? Write 2–3 sentences explaining the geometric intuition.

---

## Part B — Forward Kinematics Verification in Python

### Background

The function `fk_lite6(q)` in `IK/fk_lite6.m` implements the same Standard DH forward kinematics in MATLAB. You will now cross-check the result using the robot controller itself via the xArm Python SDK.

### Task B1 — Connect to the Robot

1. Ensure the Lite 6 controller is powered on and connected to the same network as your workstation.
2. Confirm the IP address (default: `192.168.1.153`). Update if needed.
3. Activate the virtual environment and install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

4. Run `Programs/simulation.py` to verify connectivity. The script will:
   - Enable motion and clear faults.
   - Move the robot to its home position.
   - Print the current DH parameters stored in the controller.
   - Move to a test pose (x=250, y=100, z=250 mm), then return home.

### Task B2 — Cross-Check FK Output

1. In the simulation script, after moving to the test pose, call `arm.get_position()` to obtain the controller-reported end-effector pose **[x, y, z, roll, pitch, yaw]**.
2. Independently compute the forward kinematics for the same joint angles using the MATLAB function `fk_lite6.m` (you can call it from the MATLAB command window, or replicate the logic in Python — the `dh_matrix()` function in `DH_optimization/dh_optimization.py` uses the identical formula).
3. Compare the position components (x, y, z in mm). Document the discrepancy. Values within ±2 mm indicate your model matches the physical robot; larger errors suggest uncalibrated DH parameters (this is exactly what Lab 2 addresses).

---

## Part C — Camera Calibration

### Background

Before a camera can be used for visual servoing or pick-and-place, its internal parameters must be known. Camera calibration determines:

- **Intrinsic matrix K** — focal length (fx, fy) and principal point (cx, cy):

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

- **Distortion coefficients** — radial (k₁, k₂, k₃) and tangential (p₁, p₂) lens distortions.

OpenCV estimates these by detecting known 3D points (chessboard corners) across multiple images taken from different viewpoints.

### Task C1 — Capture Chessboard Images

1. Print the provided chessboard pattern (`Images/camera_calibration.pdf`) on stiff paper or cardstock. Measure the actual square size with a ruler — this value is `square_size_mm` (the default is 25.0 mm but verify!).
2. Mount the chessboard on a stable surface.
3. Run `Calibration/image_capture.py`:

```bash
cd Calibration
python image_capture.py
```

4. Hold the chessboard at various positions and orientations in front of the camera. Press **Spacebar** to capture each image. Aim for **15–20 images** covering:
   - Different distances from the camera
   - Tilts left/right, up/down
   - Positions near all four corners of the image
   - Centred positions
5. Press **Escape** to finish capturing.

### Task C2 — Run Calibration

1. Review `Calibration/camera_calibration.py`. Key parameters to check:
   - `CHESSBOARD_SIZE = (10, 7)` — inner corners per row/column. Count the squares on your printed board: if it has 11×8 squares, there are 10×7 inner intersections. Adjust if yours differs.
   - `SQUARE_SIZE_MM` — set to your measured value.
2. Run the calibration:

```bash
python camera_calibration.py
```

3. Inspect the output:
   - **Reprojection error**: below 0.5 px is good; below 0.3 px is excellent. If higher, recapture images ensuring sharper focus and more diverse viewpoints.
   - **Camera matrix**: fx and fy should be similar (within ~5%) for most cameras.
   - **Distortion coefficients**: typically k₁ dominates; values depend on lens quality.

4. The script saves `camera_matrix.npy`, `dist_coeffs.npy`, and `calibration_results.npz`. These files are loaded automatically by the hand-eye calibration scripts in the next part.

---

## Part D — Hand-Eye (Camera-to-End-Effector) Transformation

### Background

Hand-eye calibration solves for the fixed transformation **T**_ee_cam between the robot end-effector frame and the camera frame mounted on it. Given N measurements of robot poses **T**_base_ee⁽ⁱ⁾ and observed board poses **T**_cam_board⁽ⁱ⁾, the hand-eye equation is:

$$^{base}T_{ee}^{(i)} \cdot ^{ee}T_{cam} \cdot ^{cam}T_{board}^{(i)} = ^{base}T_{board}$$

Since the board is stationary, **T**_base_board should be the same for all i. OpenCV provides closed-form solvers (Tsai, Park, Horaud, Andreff, Daniilidis) that estimate **T**_ee_cam from multiple pose pairs.

### Task D1 — Collect Pose Pairs

1. Place the ArUco marker board (two vertically stacked markers, IDs 0 and 3) on a stable surface within the camera's field of view.
2. Run `Calibration/transformation_calibration.py` in calibrate mode:

```bash
python transformation_calibration.py
```

3. Move the robot arm to **at least 10 different positions** around the marker board, pressing **'c'** at each position to record a pose pair. Vary:
   - Distance to the board (near, mid, far)
   - Viewing angles (left, right, above, below)
   - Wrist orientations

4. Press **'s'** to save all captured data to `calibration_data.npy`. Press **'q'** to quit.

### Task D2 — Compute T_ee_cam

1. Run `Calibration/get_tranform_3D.py`:

```bash
python get_tranform_3D.py
```

This script:
- Loads the saved calibration data.
- Tests all 6 Euler angle conventions (xyz, xzy, yxz, yzx, zxy, zyx).
- Tests all 5 hand-eye calibration methods (Tsai, Park, Horaud, Andreff, Daniilidis).
- Selects the combination with the lowest positional standard deviation.

2. Examine the output. The best method and Euler convention will be reported along with:
   - Mean standard deviation in mm (below 5 mm = excellent, below 10 mm = good)
   - The 4×4 transformation matrix **T**_ee_cam
   - Translation and Euler angles

3. Results are saved as `T_ee_cam.npy` and `T_ee_cam.txt`.

### Task D3 — Robust RANSAC Calibration (Optional Extension)

Run `Calibration/ransac_calibration.py` for a more robust pipeline that includes outlier rejection and nonlinear refinement. Compare the final consistency metric with the result from Task D2. Which approach gave lower scatter? Why might one be preferable?

---

## Deliverables — Lab Report

Submit a report (PDF or Word) containing:

1. **Title page** — Name, student number, date, lab partner(s).
2. **Part A** — One table of joint angles vs. computed end-effector position. Your manual calculation of A₁·A₂ for one configuration. Answer to Task A3 (singular configuration discussion).
3. **Part B** — Comparison table: MATLAB FK prediction vs. robot controller reading. Comment on any discrepancies.
4. **Part C** — Reprojection error value, camera matrix, and distortion coefficients. Screenshot of the calibration progress showing detected corners. Brief explanation of what each distortion coefficient physically represents.
5. **Part D** — Best hand-eye method and Euler convention. The T_ee_cam matrix. Scatter plot or table showing board position consistency across all pose pairs. Discussion: why does collecting diverse poses improve calibration accuracy?
6. **Reflection** (½ page max) — What was the most challenging concept? How do forward kinematics and camera calibration relate to real-world robotic applications (e.g., visual pick-and-place)?

---

## Safety Notes

- Keep hands clear of the robot workspace during automated motions.
- Use the emergency stop button if the robot behaves unexpectedly.
- Do not exceed joint limits specified in the UFACTORY documentation.
- Ask the lab supervisor before modifying joint limit settings or increasing speeds.
