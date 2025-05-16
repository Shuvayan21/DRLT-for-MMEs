function W = weight_W(A)
    [n, p] = size(A);
    np_total = n * p;

    % Parameters
    mu_1 = 2 * sqrt(2 * log(p) / n);
    mu_2 = 2 * sqrt(log((2 * p * n)) / (p * n)) + 1 / n;
    mu_3 = 2 * sqrt(2 * log(n) / p) / sqrt(1 - n / p);

    I_p = eye(p);
    I_n = eye(n);
    
    %% Objective: use residuals to represent constraints + objective
    penalty_scale = 100;  % adjust as needed

    % Residual function
    function r = residuals(x_vec)
        Wmat = reshape(x_vec, n, p);

        % === Objective part: minimize Frobenius norm
        r_obj = x_vec;

        % === Constraint 1: A'W/n ≈ I_p
        C1 = Wmat' * A / n - I_p;
        r_c1 = penalty_scale * max(0, abs(C1(:)) - mu_1);

        % === Constraint 2: W*Sig/p ≈ A/p
        C2 = A* Wmat' * A / (n*p) - A / p;
        r_c2 = penalty_scale * max(0, abs(C2(:)) - mu_2);

        % === Constraint 3: n/(p√(1 - n/p)) * (W*A'/n - I_n)
        scale = n / (p * sqrt(1 - n / p));
        C3 = scale * (A * Wmat' / n - p * I_n / n);
        r_c3 = penalty_scale * max(0, abs(C3(:)) - mu_3);

        % === Constraint 0: column norms ≤ 1 (ℓ2-norm)
        r_c4 = zeros(p, 1);
        for j = 1:p
            col_norm = norm(Wmat(:, j))^2 / n;
            r_c4(j) = penalty_scale * max(0, col_norm - 1);
        end

        % Combine all residuals
        r = [r_obj; r_c1; r_c2; r_c3; r_c4];
    end

    % Initial guess
    x0 = A(:);

    % Solve using lsqnonlin
    options = optimoptions('lsqnonlin', 'Display', 'final', ...
        'MaxIterations', 100, 'FunctionTolerance', 1e-6);
    x_opt = lsqnonlin(@residuals, x0, [], [], options);

    % Reshape back to W
    W = reshape(x_opt, n, p);
    W=W';
end
