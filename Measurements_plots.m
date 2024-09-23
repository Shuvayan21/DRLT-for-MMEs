clear 
rng(1)  % Set random seed for reproducibility

%% Initialize parameters
p = 500;  % Number of predictors
measurements = 200:50:500;  % Vector of different sample sizes (n)
f_adv = 0.01;  % Proportion of adversarial Model Mismatch Errors (MMEs)
f_sp = 0.01;  % Sparsity proportion
f_sig = 0.01;  % Noise scaling factor
run = 100;  % Number of simulations or runs
flag = 1;  % Set to 0 to recompute A, beta, delta, sigma, and weight matrix W; 1 to use preset A and W
z_alpha2 = 2.33;  % Critical value for hypothesis testing

% Preallocate matrices for results
RRMSE = zeros(7, size(measurements, 2));  % Relative Root Mean Square Error
Sens_beta = zeros(4, size(measurements, 2));  % Sensitivity for beta estimates
Sens_delta = zeros(3, size(measurements, 2));  % Sensitivity for delta estimates
Spec_beta = zeros(4, size(measurements, 2));  % Specificity for beta estimates
Spec_delta = zeros(3, size(measurements, 2));  % Specificity for delta estimates

% Loop over each measurement size
for l = 1:size(measurements, 2)
    n = measurements(l);  % Current sample size

    if flag == 0
        % Create synthetic data if flag is set to 0
        [A, A_tilde, beta, delta, sigma] = data_create(n, p, f_sig, f_adv, f_sp);
        % Generate the inverse weight matrix W
        W = weight_W(A);
    else 
        % If using preset data
        s = floor(p * f_sp);  % Determine sparsity level for beta
        beta = zeros(p, 1);  % Initialize beta vector with zeros
        S = randperm(p, s);  % Randomly select indices for non-zero elements in beta
        % Assign non-zero elements from a uniform distribution
        beta(S(1:floor(0.4 * s))) = 50 + 50 * rand(floor(0.4 * s), 1);  % First 40% from [50, 100]
        beta(S(floor(0.4 * s) + 1:s)) = 500 + 500 * rand(s - floor(0.4 * s), 1);  % Last 60% from [500, 1000]
        A = cell2mat(struct2cell(load(append("A_", num2str(n), ".mat"))));  % Load A matrix
        A_tilde = MME_create(n, p, f_adv, A, S);  % Create adversarial A_tilde
        W = cell2mat(struct2cell(load(append("W_", num2str(n), ".mat"))));  % Load weight matrix W
        delta = (A_tilde - A) * beta;  % Compute delta
        % Create standard deviation sigma
        sigma = mean(abs(A * beta)) * f_sig;
    end
    
    % Compute sensitivity, specificity, and RRMSE
    [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE(A_tilde, A, beta, delta, sigma, W);
    RRMSE(:, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(:, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(:, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(:, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(:, l) = mean(Sp_d);  % Store mean specificity for delta
end

%% Plot for Fig 2 (Sensitivity and Specificity for delta)
hold on
plot(measurements, Sens_delta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '+', 'Color', [0.07, 0.62, 1.00])
plot(measurements, Spec_delta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'o', 'Color', [0.07, 0.62, 1.00])
plot(measurements, Sens_delta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'x', 'Color', [0.85, 0.33, 0.10])
plot(measurements, Spec_delta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'square', 'Color', [0.85, 0.33, 0.10])
plot(measurements, Sens_delta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
plot(measurements, Spec_delta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
hold off
xlabel("measurements(n)")
ylabel("Sensitivity and Specificity values for \delta")
legend("Sensitivity-RL", "Specificity-RL", "Sensitivity-DRLT", "Specificity-DRLT", "Sensitivity-ODRLT", "Specificity-ODRLT")

%% Plot for Fig 3 (Sensitivity and Specificity for beta)
hold on
plot(measurements, Sens_beta(4, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'pentagram', 'Color', [0.47, 0.67, 0.19])
plot(measurements, Spec_beta(4, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'diamond', 'Color', [0.47, 0.67, 0.19])
plot(measurements, Sens_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '+', 'Color', [0.07, 0.62, 1.00])
plot(measurements, Spec_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'o', 'Color', [0.07, 0.62, 1.00])
plot(measurements, Sens_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'x', 'Color', [0.85, 0.33, 0.10])
plot(measurements, Spec_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'square', 'Color', [0.85, 0.33, 0.10])
plot(measurements, Sens_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
plot(measurements, Spec_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
hold off
xlabel("measurements(n)")
ylabel("Sensitivity and Specificity values for beta")
legend("Sensitivity-Baseline 3", "Specificity-Baseline 3", "Sensitivity-RL", "Specificity-RL", "Sensitivity-DRLT", "Specificity-DRLT", "Sensitivity-ODRLT", "Specificity-ODRLT")

%% Plot for Fig 4 (RRMSE)
hold on
plot(measurements, RRMSE(1, :))
plot(measurements, RRMSE(2, :))
plot(measurements, RRMSE(3, :))
plot(measurements, RRMSE(4, :))
plot(measurements, RRMSE(5, :))
plot(measurements, RRMSE(6, :))
plot(measurements, RRMSE(7, :))
hold off
xlabel("measurements(n)")
ylabel("RRMSE")
legend("L2", "L1", "RL2", "RL1", "Robust Lasso", "ODRLT", "DRLT")
