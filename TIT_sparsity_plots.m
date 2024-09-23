clear 
rng(1)  % Set random seed for reproducibility

%% Initialize parameters
p = 500;  % Number of predictors
n = 400;  % Sample size
f_adv = 0.01;  % Proportion of adversarial errors
sparsity = 0.01:0.01:0.1;  % Range of sparsity levels
f_sig = 0.01;  % Noise level
run = 100;  % Number of simulations
flag = 1;  % Set to 0 to recompute data; 1 to use preset data
z_alpha2 = 2.33;  % Critical value for hypothesis testing

% Preallocate matrices for results
RRMSE = zeros(7, size(sparsity, 2));  % Relative Root Mean Square Error
Sens_beta = zeros(4, size(sparsity, 2));  % Sensitivity for beta estimates
Sens_delta = zeros(3, size(sparsity, 2));  % Sensitivity for delta estimates
Spec_beta = zeros(4, size(sparsity, 2));  % Specificity for beta estimates
Spec_delta = zeros(3, size(sparsity, 2));  % Specificity for delta estimates

% Loop over each sparsity level
for l = 1:size(sparsity, 2)
    f_sp = sparsity(l);  % Current sparsity level
    
    if flag == 0
        % Create synthetic data if flag is set to 0
        [A, A_tilde, beta, delta, sigma] = data_create(n, p, f_sig, f_adv, f_sp);
        % Generate the inverse W matrix
        W = weight_W(A);
    else 
        % If using preset data
        s = floor(p * f_sp);  % Determine sparsity level for beta
        beta = zeros(p, 1);  % Initialize beta vector
        S = randperm(p, s);  % Randomly select indices for non-zero elements
        
        % Assign non-zero elements from uniform distribution
        beta(S(1:floor(0.4 * s))) = 50 + 50 * rand(floor(0.4 * s), 1);  % First 40%
        beta(S(floor(0.4 * s) + 1:s)) = 500 + 500 * rand(s - floor(0.4 * s), 1);  % Last 60%
        
        % Load preset A matrix and create A_tilde
        A = cell2mat(struct2cell(load(append("A_", num2str(n), ".mat"))));
        A_tilde = MME_create(n, p, f_adv, A, S);  % Create adversarial A_tilde
        
        % Load preset weight matrix W
        W = cell2mat(struct2cell(load(append("W_", num2str(n), ".mat"))));
        
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

%% Plot for Sensitivity and Specificity (Delta)
hold on
plot(sparsity, Sens_delta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '+', 'Color', [0.07, 0.62, 1.00])
plot(sparsity, Spec_delta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'o', 'Color', [0.07, 0.62, 1.00])
plot(sparsity, Sens_delta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'x', 'Color', [0.85, 0.33, 0.10])
plot(sparsity, Spec_delta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'square', 'Color', [0.85, 0.33, 0.10])
plot(sparsity, Sens_delta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
plot(sparsity, Spec_delta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
hold off
xlabel("f_{sp}")
ylabel("Sensitivity and Specificity values for \delta")
legend("Sensitivity-RL", "Specificity-RL", "Sensitivity-DRLT", "Specificity-DRLT", "Sensitivity-ODRLT", "Specificity-ODRLT")

%% Plot for Sensitivity and Specificity (Beta)
hold on
plot(sparsity, Sens_beta(4, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'pentagram', 'Color', [0.47, 0.67, 0.19])
plot(sparsity, Spec_beta(4, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'diamond', 'Color', [0.47, 0.67, 0.19])
plot(sparsity, Sens_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '+', 'Color', [0.07, 0.62, 1.00])
plot(sparsity, Spec_beta(1, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'o', 'Color', [0.07, 0.62, 1.00])
plot(sparsity, Sens_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', 'x', 'Color', [0.85, 0.33, 0.10])
plot(sparsity, Spec_beta(2, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', 'square', 'Color', [0.85, 0.33, 0.10])
plot(sparsity, Sens_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'LineStyle', '--', 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
plot(sparsity, Spec_beta(3, :), 'MarkerSize', 20, 'LineWidth', 6, 'Marker', '>', 'Color', [0.88, 0.88, 0.12])
hold off
xlabel("f_{sp}")
ylabel("Sensitivity and Specificity values for beta")
legend("Sensitivity-Baseline 3", "Specificity-Baseline 3", "Sensitivity-RL", "Specificity-RL", "Sensitivity-DRLT", "Specificity-DRLT", "Sensitivity-ODRLT", "Specificity-ODRLT")

%% Plot for RRMSE
hold on
plot(sparsity, RRMSE(1, :))
plot(sparsity, RRMSE(2, :))
plot(sparsity, RRMSE(3, :))
plot(sparsity, RRMSE(4, :))
plot(sparsity, RRMSE(5, :))
plot(sparsity, RRMSE(6, :))
plot(sparsity, RRMSE(7, :))
hold off
xlabel("f_{sp}")
ylabel("RRMSE")
legend("L2", "L1", "RL2", "RL1", "Robust Lasso", "ODRLT", "DRLT")
