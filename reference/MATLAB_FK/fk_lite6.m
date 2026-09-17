function T = fk_lite6(q)
    % Forward Kinematics for UFACTORY Lite 6 (modified / Craig D-H).
    % Input:  q = [q1, q2, q3, q4, q5, q6] in radians
    % Output: T = 4x4 homogeneous transform (base to EE), translation in mm

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

        % Modified (Craig) DH link transform
        Ti = [ct,  -st,       0,  a(i);
              st*ca, ct*ca, -sa, -sa*d(i);
              st*sa, ct*sa,  ca,  ca*d(i);
              0,    0,        0,  1];
        T = T * Ti;
    end
end
