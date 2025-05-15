clear 
rng(1)  % Set random seed for reproducibility

%% Initialize parameters
p = 500;  % Number of predictors
measurements = 200:50:500;  % Vector of different sample sizes (n)
f_adv = 0.01;  % Proportion of adversarial Model Mismatch Errors (MMEs)
f_sp = 0.01;  % Sparsity proportion
f_sig = 0.01;  % Noise scaling factor
run = 25;  % Number of simulations or runs
z_alpha2 = 2.33;  % Critical value for hypothesis testing

% Preallocate matrices for results
RRMSE = zeros(3, size(measurements, 2));  % Relative Root Mean Square Error
Sens_beta = zeros(3, size(measurements, 2));  % Sensitivity for beta estimates
Sens_delta = zeros(3, size(measurements, 2));  % Sensitivity for delta estimates
Spec_beta = zeros(3, size(measurements, 2));  % Specificity for beta estimates
Spec_delta = zeros(3, size(measurements, 2));  % Specificity for delta estimates

% Loop over each measurement size
for l = 1:size(measurements, 2)
    n = measurements(l);  % Current sample size

  
        % Create synthetic data if flag is set to 0
        [A, A_tilde, beta, delta, sigma] = data_create_ct(n, p, f_sig, f_adv, f_sp);
        % Generate the inverse weight matrix W
        W = weight_W(A);
  
    % Compute sensitivity, specificity, and RRMSE
    [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE(A_tilde, A, beta, delta, sigma, W);
    RRMSE(:, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(:, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(:, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(:, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(:, l) = mean(Sp_d);  % Store mean specificity for delta
end

%% Plot for Fig 1(top-right) in supplemental (Sensitivity and Specificity for delta)
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

%% Plot for Fig 2(top-right) in supplemental (Sensitivity and Specificity for beta)
hold on
plot(measurements, Sens_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '+', 'Color', [0.07, 0.62, 1.00])
plot(measurements, Spec_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'o', 'Color', [0.07, 0.62, 1.00])
plot(measurements, Sens_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'x', 'Color', [0.85, 0.33, 0.10])
plot(measurements, Spec_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'square', 'Color', [0.85, 0.33, 0.10])
plot(measurements, Sens_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
plot(measurements, Spec_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
hold off
xlabel("measurements(n)")
ylabel("Sensitivity and Specificity values for beta")
legend("Sensitivity-RL", "Specificity-RL", "Sensitivity-DRLT", "Specificity-DRLT", "Sensitivity-ODRLT", "Specificity-ODRLT")

%% Plot for Fig 3(top-right) in supplemental(RRMSE)
hold on
plot(measurements, RRMSE(1, :))
plot(measurements, RRMSE(2, :))
plot(measurements, RRMSE(3, :))
hold off
xlabel("measurements(n)")
ylabel("RRMSE")
legend("Robust Lasso", "ODRLT", "DRLT")
