clear 
rng(1)  % Set random seed for reproducibility

%% Initialize parameters
p = 500;  % Number of predictors
n = 400;  % Sample size
MME = 0.01:0.01:0.1;  % Range of adversarial MME values
f_sp = 0.01;  % Sparsity proportion
f_sig = 0.01;  % Noise scaling factor
run = 25;  % Number of simulations or runs
z_alpha2 = 2.33;  % Critical value for hypothesis testing

% Preallocate matrices for results
RRMSE = zeros(3, size(MME, 2));  % Relative Root Mean Square Error
Sens_beta = zeros(3, size(MME, 2));  % Sensitivity for beta estimates
Sens_delta = zeros(3, size(MME, 2));  % Sensitivity for delta estimates
Spec_beta = zeros(3, size(MME, 2));  % Specificity for beta estimates
Spec_delta = zeros(3, size(MME, 2));  % Specificity for delta estimates

% Loop over each MME value
for l = 1:size(MME, 2)
    f_adv = MME(l);  % Current MME value
    
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

%% Plot for Fig 1(top-left) in supplemental (Sensitivity and Specificity for delta)
hold on
plot(MME, Sens_delta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '+', 'Color', [0.07, 0.62, 1.00])
plot(MME, Spec_delta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'o', 'Color', [0.07, 0.62, 1.00])
plot(MME, Sens_delta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'x', 'Color', [0.85, 0.33, 0.10])
plot(MME, Spec_delta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'square', 'Color', [0.85, 0.33, 0.10])
plot(MME, Sens_delta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
plot(MME, Spec_delta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
hold off
xlabel("f_{adv}")
ylabel("Sensitivity and Specificity values for \delta")
legend("Sensitivity-RL", "Specificity-RL", "Sensitivity-DRLT", "Specificity-DRLT", "Sensitivity-ODRLT", "Specificity-ODRLT")

%% Plot for Fig 2(top-left) in supplemental(Sensitivity and Specificity for beta)
hold on
plot(MME, Sens_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '+', 'Color', [0.07, 0.62, 1.00])
plot(MME, Spec_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'o', 'Color', [0.07, 0.62, 1.00])
plot(MME, Sens_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'x', 'Color', [0.85, 0.33, 0.10])
plot(MME, Spec_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'square', 'Color', [0.85, 0.33, 0.10])
plot(MME, Sens_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
plot(MME, Spec_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
hold off
xlabel("f_{adv}")
ylabel("Sensitivity and Specificity values for beta")
legend("Sensitivity-RL", "Specificity-RL", "Sensitivity-DRLT", "Specificity-DRLT", "Sensitivity-ODRLT", "Specificity-ODRLT")

%% Plot for Fig 3(top-left) in supplemental (RRMSE)
hold on
plot(MME, RRMSE(1, :))
plot(MME, RRMSE(2, :))
plot(MME, RRMSE(3, :))
hold off
xlabel("f_{adv}")
ylabel("RRMSE")
legend("Robust Lasso", "ODRLT", "DRLT")
