function [Sens_delta, Spec_delta, Sens_beta, Spec_beta, RRMSE] = results_Sens_Spec_RRMSE(A_tilde, A, beta, delta, sigma, W)
    % This function evaluates the performance of different regression methods
    % based on sensitivity, specificity, and relative root mean square error (RRMSE).
    % 
    % Inputs:
    % A_tilde - Design matrix for the response variable (n x p).
    % A       - Design matrix for predictors (n x p).
    % beta    - True coefficients for the predictors (p x 1).
    % delta   - True effects of interest (n x 1).
    % sigma   - Standard deviation of noise (scalar).
    % W       - Weight matrix used in calculations (n x n).
    %
    % Outputs:
    % Sens_delta - Sensitivity estimates for delta (3 x run).
    % Spec_delta - Specificity estimates for delta (3 x run).
    % Sens_beta  - Sensitivity estimates for beta (4 x run).
    % Spec_beta  - Specificity estimates for beta (4 x run).
    % RRMSE      - Relative root mean square errors (7 x run).

    rng(1) % Set random seed for reproducibility.
    [n, p] = size(A); % Get dimensions of the design matrix A.
    run = 100; % Number of runs for the simulation.
    
    % Generate noise and compute the response variable.
    eta = random("normal", 0, sigma, [n 1]); % Generate normal noise.
    y = A_tilde * beta + eta; % Response variable with noise.
    
    % Cross-validation to determine optimal regularization parameters.
    [lambda_1, lambda_2] = CV_Drlt(y, A, W, sigma); % Cross-validation for ODRL.
    l1 = CV_l1(y, A); % Cross-validation for L1 (lasso).
    l2 = CV_l2(y, A); % Cross-validation for L2 (ridge).
    [lamb_1, lamb_2] = CV_RL(y, A); % Cross-validation for robust lasso.
    
    % Compute covariance matrices.
    Sig = (A' * A) / n; % Empirical covariance matrix of A.
    Sigma_beta_W = sigma^2 / n * (W' * W); % Covariance for beta with W.
    Sigma_delta_W = sigma^2 * (eye(n) - 2/n * W * A' + 1/n * W * Sig * W'); % Covariance for delta with W.
    Sigma_beta_A = sigma^2 / n * (A' * A); % Covariance for beta with A.
    Sigma_delta_A = sigma^2 * (eye(n) - 2/n * A * A' + 1/n * A * Sig * A'); % Covariance for delta with A.
    
    z_alpha2 = 2.33; % Threshold value for statistical significance (corresponds to alpha = 0.01).
    I = eye(n); % Identity matrix of size n.

    % Initialize matrices to store results.
    RRMSE = zeros(7, run); % RRMSE results for different methods.
    Sens_beta = zeros(4, run); % Sensitivity for beta.
    Spec_beta = zeros(4, run); % Specificity for beta.
    Sens_delta = zeros(3, run); % Sensitivity for delta.
    Spec_delta = zeros(3, run); % Specificity for delta.
    beta_l = zeros(p, run); % Estimated beta from lasso.
    beta_d_W = zeros(p, run); % Debiased beta with W.
    beta_d_A = zeros(p, run); % Debiased beta with A.
    delta_l = zeros(n, run); % Estimated delta from lasso.
    delta_d_W = zeros(n, run); % Debiased delta with W.
    delta_d_A = zeros(n, run); % Debiased delta with A.
    TG = zeros(p, run); % Test statistics for beta using W.
    TG_A = zeros(p, run); % Test statistics for beta using A.
    TH = zeros(n, run); % Test statistics for delta using W.
    TH_A = zeros(n, run); % Test statistics for delta using A.
    confusion_matrix_delta_odrl = zeros(n, run); % Confusion matrix for delta (ODRL).
    confusion_matrix_beta_odrl = zeros(p, run); % Confusion matrix for beta (ODRL).
    confusion_matrix_delta_drl = zeros(n, run); % Confusion matrix for delta (DRL).
    confusion_matrix_beta_drl = zeros(p, run); % Confusion matrix for beta (DRL).

    % Main loop to perform simulations.
    for k = 1:run
        eta = random("normal", 0, sigma, [n 1]); % Generate new noise for each run.
        y = A_tilde * beta + eta; % Update response variable with new noise.

        % L2 Lasso optimization.
        cvx_begin quiet
            variable beta_l2(p) % Declare variable for L2 estimates.
            minimise (pow_pos(norm(y - A * beta_l2), 2) + l2 * norm(beta_l2, 1)); % Minimize objective function.
        cvx_end
        RRMSE(1, k) = norm(beta - beta_l2) / norm(beta); % Store RRMSE for L2.

        % L1 Lasso optimization.
        cvx_begin quiet
            variable beta_l1(p) % Declare variable for L1 estimates.
            minimise (norm(y - A * beta_l1, 1) + l1 * norm(beta_l1, 1)); % Minimize objective function.
        cvx_end
        RRMSE(2, k) = norm(beta - beta_l1) / norm(beta); % Store RRMSE for L1.

        % Robust Lasso optimization.
        cvx_begin quiet
            variable x_l(n + p) % Declare combined variable for robust lasso.
            minimise (0.5 * pow_pos(norm(y - [A I] * x_l), 2) + lamb_1 * norm(x_l(1:p), 1) + lamb_2 * norm(x_l(p + 1:p + n), 1)); % Minimize objective function.
        cvx_end
        RRMSE(5, k) = norm(beta - x_l(1:p)) / norm(beta); % Store RRMSE for robust lasso.

        % Calculate sensitivity and specificity for beta and delta.
        [Sens_beta(1, k), Spec_beta(1, k), ~, ~] = calculateSensitivitySpecificity(beta, x_l(1:p));
        [Sens_delta(1, k), Spec_delta(1, k), ~, ~] = calculateSensitivitySpecificity(delta, x_l(p + 1:n + p));

        % RANSAC L1 Lasso.
        [RRMSE(4, k)] = ransac_l1(y, A, beta, l1, 100);
        % RANSAC L2 Lasso.
        [RRMSE(3, k)] = ransac_l2(y, A, beta, l2, 100);

        % ODRL optimization.
        cvx_begin quiet
            variable x_l(n + p) % Declare variable for ODRL.
            minimise (0.5 * pow_pos(norm(y - [A eye(n)] * x_l), 2) + lambda_1 * norm(x_l(1:p), 1) + lambda_2 * norm(x_l(p + 1:p + n), 1)); % Minimize objective function.
        cvx_end

        % Extract estimated parameters.
        beta_l(:, k) = x_l(1:p); % Estimated beta.
        delta_l(:, k) = x_l((p + 1):(p + n)); % Estimated delta.

        % Debiasing step using W.
        beta_d_W(:, k) = beta_l(:, k) + 1/n * W' * (y - A * beta_l(:, k) - delta_l(:, k));
        delta_d_W(:, k) = delta_l(:, k) + (eye(n) - 1/n * A * W') * (y - A * beta_l(:, k) - delta_l(:, k));

        % Calculate test statistics for delta and beta using W.
        for j = 1:n
            TH(j, k) = (delta_d_W(j, k)) / sqrt(Sigma_delta_W(j, j)); % Test statistic for delta.
        end
        for i = 1:p
            TG(i, k) = (sqrt(n) * (beta_d_W(i, k))) / sqrt(Sigma_beta_W(i, i)); % Test statistic for beta.
        end

        % Update confusion matrix and compute sensitivity and specificity for delta.
        neg_neg = sum(and(delta == 0, TH(:, k) <= z_alpha2)); % True negatives for delta.
        neg_pos = sum(and(delta == 0, TH(:, k) > z_alpha2)); % False positives for delta.
        pos_neg = sum(and(delta ~= 0, TH(:, k) <= z_alpha2)); % False negatives for delta.
        pos_pos = sum(and(delta ~= 0, TH(:, k) > z_alpha2)); % True positives for delta.
        confusion_matrix_delta_odrl(:, k) = confusion_matrix_delta_odrl(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]'; % Update confusion matrix.
        Sens_delta(2, k) = confusion_matrix_delta_odrl(4, k) / (confusion_matrix_delta_odrl(4, k) + confusion_matrix_delta_odrl(3, k)); % Sensitivity for delta.
        Spec_delta(2, k) = confusion_matrix_delta_odrl(1, k) / (confusion_matrix_delta_odrl(1, k) + confusion_matrix_delta_odrl(2, k)); % Specificity for delta.

        % Update confusion matrix and compute sensitivity and specificity for beta.
        neg_neg = sum(and(beta == 0, TG(:, k) <= z_alpha2)); % True negatives for beta.
        neg_pos = sum(and(beta == 0, TG(:, k) > z_alpha2)); % False positives for beta.
        pos_neg = sum(and(beta ~= 0, TG(:, k) <= z_alpha2)); % False negatives for beta.
        pos_pos = sum(and(beta ~= 0, TG(:, k) > z_alpha2)); % True positives for beta.
        confusion_matrix_beta_odrl(:, k) = confusion_matrix_beta_odrl(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]'; % Update confusion matrix.
        Sens_beta(2, k) = confusion_matrix_beta_odrl(4, k) / (confusion_matrix_beta_odrl(4, k) + confusion_matrix_beta_odrl(3, k)); % Sensitivity for beta.
        Spec_beta(2, k) = confusion_matrix_beta_odrl(1, k) / (confusion_matrix_beta_odrl(1, k) + confusion_matrix_beta_odrl(2, k)); % Specificity for beta.

        % Data cleaning for the next iteration.
        y_new = y(~(abs(TH(:, k)) > z_alpha2)); % Filter out outliers based on TH.
        A_new = A(~(abs(TH(:, k)) > z_alpha2), :); % Filter corresponding rows in A.
        n_new = size(y_new, 2); % New sample size after filtering.

        % Cross-validation for new data to determine new lambda values.
        [lamb_1_odrl, lamb_2_odrl] = CV_RL(y_new, A_new);
        
        % ODRL optimization for new data.
        cvx_begin quiet
            variable x_odrl(n_new + p) % Declare variable for ODRL on cleaned data.
            minimise (0.5 * pow_pos(norm(y_new - [A_new eye(n_new)] * x_odrl), 2) + lamb_1_odrl * norm(x_odrl(1:p), 1) + lamb_2_odrl * norm(x_odrl(p + 1:p + n_new), 1)); % Minimize objective function.
        cvx_end
        RRMSE(6, k) = norm(beta - x_odrl(1:p)) / norm(beta); % Store RRMSE for ODRL.
        
        % DRL optimization.
        beta_d_A(:, k) = beta_l(:, k) + 1/n * A' * (y - A * beta_l(:, k) - delta_l(:, k));
        delta_d_A(:, k) = delta_l(:, k) + (eye(n) - 1/n * (A * A')) * (y - A * beta_l(:, k) - delta_l(:, k));

        % Calculate test statistics for delta and beta using A.
        for j = 1:n
            TH_A(j, k) = (delta_d_A(j, k)) / sqrt(Sigma_delta_A(j, j)); % Test statistic for delta.
        end
        for i = 1:p
            TG_A(i, k) = (sqrt(n) * (beta_d_W(i, k))) / sqrt(Sigma_beta_A(i, i)); % Test statistic for beta.
        end

        % Update confusion matrix and compute sensitivity and specificity for delta (DRL).
        neg_neg = sum(and(delta == 0, TH_A(:, k) <= z_alpha2)); % True negatives for delta.
        neg_pos = sum(and(delta == 0, TH_A(:, k) > z_alpha2)); % False positives for delta.
        pos_neg = sum(and(delta ~= 0, TH_A(:, k) <= z_alpha2)); % False negatives for delta.
        pos_pos = sum(and(delta ~= 0, TH_A(:, k) > z_alpha2)); % True positives for delta.
        confusion_matrix_delta_drl(:, k) = confusion_matrix_delta_drl(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]'; % Update confusion matrix.
        Sens_delta(3, k) = confusion_matrix_delta_drl(4, k) / (confusion_matrix_delta_drl(4, k) + confusion_matrix_delta_drl(3, k)); % Sensitivity for delta.
        Spec_delta(3, k) = confusion_matrix_delta_drl(1, k) / (confusion_matrix_delta_drl(1, k) + confusion_matrix_delta_drl(2, k)); % Specificity for delta.

        % Update confusion matrix and compute sensitivity and specificity for beta (DRL).
        neg_neg = sum(and(beta == 0, TG_A(:, k) <= z_alpha2)); % True negatives for beta.
        neg_pos = sum(and(beta == 0, TG_A(:, k) > z_alpha2)); % False positives for beta.
        pos_neg = sum(and(beta ~= 0, TG_A(:, k) <= z_alpha2)); % False negatives for beta.
        pos_pos = sum(and(beta ~= 0, TG_A(:, k) > z_alpha2)); % True positives for beta.
        confusion_matrix_beta_drl(:, k) = confusion_matrix_beta_drl(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]'; % Update confusion matrix.
        Sens_beta(3, k) = confusion_matrix_beta_drl(4, k) / (confusion_matrix_beta_drl(4, k) + confusion_matrix_beta_drl(3, k)); % Sensitivity for beta.
        Spec_beta(3, k) = confusion_matrix_beta_drl(1, k) / (confusion_matrix_beta_drl(1, k) + confusion_matrix_beta_drl(2, k)); % Specificity for beta.

        % Data cleaning for the next iteration for DRL.
        y_new = y(~(abs(TH_A(:, k)) > z_alpha2)); % Filter out outliers based on TH_A.
        A_new = A(~(abs(TH_A(:, k)) > z_alpha2), :); % Filter corresponding rows in A.
        n_new = size(y_new, 2); % New sample size after filtering.

        % Cross-validation for new data to determine new lambda values for DRL.
        [lamb_1_drl, lamb_2_drl] = CV_RL(y_new, A_new);
        
        % DRL optimization for new data.
        cvx_begin quiet
            variable x_drl(n_new + p) % Declare variable for DRL on cleaned data.
            minimise (0.5 * pow_pos(norm(y_new - [A_new eye(n_new)] * x_drl), 2) + lamb_1_drl * norm(x_drl(1:p), 1) + lamb_2_drl * norm(x_drl(p + 1:p + n_new), 1)); % Minimize objective function.
        cvx_end
        RRMSE(7, k) = norm(beta - x_drl(1:p)) / norm(beta); % Store RRMSE for DRL.

        % Baseline 3: Perform sensitivity and specificity calculations without MMEs.
        y = A * beta + eta; % Generate response without MME.
        cvx_begin quiet
            variable beta_base3(p) % Declare variable for baseline estimates.
            minimise (pow_pos(norm(y - A * beta_base3), 2) + l2 * norm(beta_base3, 1)); % Minimize objective function.
        cvx_end
        [Sens_beta(4, k), Spec_beta(4, k)] = calculateSensitivitySpecificity(beta, beta_base3); % Calculate sensitivity and specificity for baseline.
    end
end
