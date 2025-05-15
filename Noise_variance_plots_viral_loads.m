clear 
rng(1)  % Set random seed for reproducibility

%% Initialize parameters
p = 500;  % Number of predictors
n = 400;  % Sample size
f_adv = 0.01;  % Proportion of adversarial errors
f_sp = 0.01;  % Sparsity proportion
sigma_prop = 0:0.01:0.1;  % Range of noise scaling factors
run = 25;  % Number of simulations
flag = 1;  % Set to 0 to recompute data; 1 to use preset data
z_alpha2 = 2.33;  % Critical value for hypothesis testing

% Preallocate matrices for results
RRMSE = zeros(3, size(sigma_prop, 2));  % Relative Root Mean Square Error
Sens_beta = zeros(3, size(sigma_prop, 2));  % Sensitivity for beta estimates
Sens_delta = zeros(3, size(sigma_prop, 2));  % Sensitivity for delta estimates
Spec_beta = zeros(3, size(sigma_prop, 2));  % Specificity for beta estimates
Spec_delta = zeros(3, size(sigma_prop, 2));  % Specificity for delta estimates

% Loop over each noise level
for l = 1:size(sigma_prop, 2)
    f_sig = sigma_prop(l);  % Current noise level
    
  
        % Create synthetic data if flag is set to 0
        [A, A_tilde, beta, delta, sigma] = data_create_ct(n, p, f_sig, f_adv, f_sp);
        % Generate the inverse W matrix
        W = weight_W(A);
  
    % Compute sensitivity, specificity, and RRMSE
    [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE(A_tilde, A, beta, delta, sigma, W);
    RRMSE(:, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(:, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(:, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(:, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(:, l) = mean(Sp_d);  % Store mean specificity for delta
end

%% Plot for Sensitivity and Specificity-Fig 1(bottom-left) in supplemental (Delta)
hold on
plot(sigma_prop, Sens_delta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '+', 'Color', [0.07, 0.62, 1.00])
plot(sigma_prop, Spec_delta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'o', 'Color', [0.07, 0.62, 1.00])
plot(sigma_prop, Sens_delta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'x', 'Color', [0.85, 0.33, 0.10])
plot(sigma_prop, Spec_delta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'square', 'Color', [0.85, 0.33, 0.10])
plot(sigma_prop, Sens_delta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
plot(sigma_prop, Spec_delta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
hold off
xlabel("f_{\sigma}")
ylabel("Sensitivity and Specificity values for \delta")
legend("Sensitivity-RL", "Specificity-RL", "Sensitivity-DRLT", "Specificity-DRLT", "Sensitivity-ODRLT", "Specificity-ODRLT")

%% Plot for Sensitivity and Specificity-Fig 2(bottom-left) in supplemental (Beta)
hold on
plot(sigma_prop, Sens_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '+', 'Color', [0.07, 0.62, 1.00])
plot(sigma_prop, Spec_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'o', 'Color', [0.07, 0.62, 1.00])
plot(sigma_prop, Sens_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'x', 'Color', [0.85, 0.33, 0.10])
plot(sigma_prop, Spec_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'square', 'Color', [0.85, 0.33, 0.10])
plot(sigma_prop, Sens_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
plot(sigma_prop, Spec_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
hold off
xlabel("f_{\sigma}")
ylabel("Sensitivity and Specificity values for beta")
legend( "Sensitivity-RL", "Specificity-RL", "Sensitivity-DRLT", "Specificity-DRLT", "Sensitivity-ODRLT", "Specificity-ODRLT")

%% Plot for RRMSE-Fig 3(bottom-left) in supplemental
hold on
plot(sigma_prop, RRMSE(1, :))
plot(sigma_prop, RRMSE(2, :))
plot(sigma_prop, RRMSE(3, :))
hold off
xlabel("f_{\sigma}")
ylabel("RRMSE")
legend("Robust Lasso", "ODRLT", "DRLT")
