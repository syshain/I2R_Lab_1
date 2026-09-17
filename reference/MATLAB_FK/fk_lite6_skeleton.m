function T = fk_lite6(q)
    % Forward Kinematics for UFACTORY Lite 6 (modified / Craig D-H).
    %
    % Input:  q = [q1, q2, q3, q4, q5, q6] in radians
    % Output: T = 4x4 homogeneous transform (base to EE), translation in mm
    %
    % ------------------------------------------------------------------
    % STUDENT TASK
    %   Fill in the modified-DH (Craig) link transform Ti inside the loop.
    %   Everything else is already set up for you.
    %
    % Recall the modified (Craig) DH convention: each joint i contributes
    %   - theta_i  : rotation about z_{i-1}  (= joint angle + offset)
    %   - d_i      : translation along  z_{i-1}
    %   - alpha_i  : rotation about x_i
    %   - a_i      : translation along  x_i
    % expressed as a single 4x4 transform from frame i-1 to frame i.
    % ------------------------------------------------------------------

    % Modified-DH parameters (UFACTORY Lite 6 handbook)
    % Columns: [theta_offset_deg, d_mm, alpha_deg, a_mm]
    dh = [
        0,     243.3,   0,    0;
        -90,   0,      -90,   0;
        -90,   0,      180,   200;
        0,     227.6,  90,    87;
        0,     0,      90,    0;
        0,     61.5,   -90,   0
    ];

    d = dh(:,2);
    a = dh(:,4);
    alpha = deg2rad(dh(:,3));
    theta_offset = deg2rad(dh(:,1));

    T = eye(4);
    for i = 1:6
        theta = q(i) + theta_offset(i);
        ca = cos(alpha(i)); sa = sin(alpha(i));
        ct = cos(theta);    st = sin(theta);

        % TODO: build the modified (Craig) DH link transform Ti here.
        % It must be a 4x4 matrix using theta, d(i), alpha(i) and a(i).
        % Hint: the standard Craig form is
        %   Ti = [ ... ]   <- fill in all 16 entries
        Ti = [];   % <-- REPLACE THIS LINE WITH YOUR 4x4 TRANSFORM

        T = T * Ti;
    end
end
