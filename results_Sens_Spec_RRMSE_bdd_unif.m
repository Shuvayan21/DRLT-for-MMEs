function [Sens_delta, Spec_delta, Sens_beta, Spec_beta, RRMSE] = results_Sens_Spec_RRMSE_bdd_unif(A_tilde, A, beta, delta, sigma, W)
    % This function evaluates the performance of different regression methods
    % based on sensitivity, specificity, and relative root mean square error (RRMSE).
    % 
    % Inputs:
    % A_tilde - Design matrix for the response variable (n x p).
    % A       - Design matrix for predictors (n x p).
    % beta    - Viral loads for the predictors (p x 1) obtained from CT-cycle distribution.
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
    run = 25; % Number of runs for the simulation.
    a=sqrt(3)*sigma;
    % Generate noise and compute the response variable.
    eta=-a+2*a*rand(n,1); % Generate uniform noise.
    y = A_tilde * beta + eta; % Response variable with noise.
    
    % Cross-validation to determine optimal regularization parameters.
    [lambda_1, lambda_2] = CV_Drlt(y, A, W, sigma); % Cross-validation for ODRL.
    
    % Compute covariance matrices.
    
    Sigma_beta_W = sigma^2 / n * (W' * W); % Covariance for beta with W.
    Sigma_delta_W = sigma^2 * (eye(n) - 2/n * A * W' + 1/(n^2) * A * (W'*W) * A'); % Covariance for delta with W.
        
    z_alpha2 = 2.33; % Threshold value for statistical significance (corresponds to alpha = 0.01).
    I = eye(n); % Identity matrix of size n.

    % Initialize matrices to store results.
    RRMSE = zeros(1, run); % RRMSE results for different methods.
    Sens_beta = zeros(1, run); % Sensitivity for beta.
    Spec_beta = zeros(1, run); % Specificity for beta.
    Sens_delta = zeros(1, run); % Sensitivity for delta.
    Spec_delta = zeros(1, run); % Specificity for delta.
    beta_l = zeros(p, run); % Estimated beta from lasso.
    beta_d_W = zeros(p, run); % Debiased beta with W.
    delta_l = zeros(n, run); % Estimated delta from lasso.
    delta_d_W = zeros(n, run); % Debiased delta with W.
    TG = zeros(p, run); % Test statistics for beta using W
    TH = zeros(n, run); % Test statistics for delta using W.
    confusion_matrix_delta_odrl = zeros(n, run); % Confusion matrix for delta (ODRL).
    confusion_matrix_beta_odrl = zeros(p, run); % Confusion matrix for beta (ODRL).
 
    % Main loop to perform simulations.
    for k = 1:run
        eta=-a+2*a*rand(n,1); % Generate new noise for each run.
        y = A_tilde * beta + eta; % Update response variable with new noise.

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
        Sens_delta(1, k) = confusion_matrix_delta_odrl(4, k) / (confusion_matrix_delta_odrl(4, k) + confusion_matrix_delta_odrl(3, k)); % Sensitivity for delta.
        Spec_delta(1, k) = confusion_matrix_delta_odrl(1, k) / (confusion_matrix_delta_odrl(1, k) + confusion_matrix_delta_odrl(2, k)); % Specificity for delta.

        % Update confusion matrix and compute sensitivity and specificity for beta.
        neg_neg = sum(and(beta == 0, TG(:, k) <= z_alpha2)); % True negatives for beta.
        neg_pos = sum(and(beta == 0, TG(:, k) > z_alpha2)); % False positives for beta.
        pos_neg = sum(and(beta ~= 0, TG(:, k) <= z_alpha2)); % False negatives for beta.
        pos_pos = sum(and(beta ~= 0, TG(:, k) > z_alpha2)); % True positives for beta.
        confusion_matrix_beta_odrl(:, k) = confusion_matrix_beta_odrl(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]'; % Update confusion matrix.
        Sens_beta(1, k) = confusion_matrix_beta_odrl(4, k) / (confusion_matrix_beta_odrl(4, k) + confusion_matrix_beta_odrl(3, k)); % Sensitivity for beta.
        Spec_beta(1, k) = confusion_matrix_beta_odrl(1, k) / (confusion_matrix_beta_odrl(1, k) + confusion_matrix_beta_odrl(2, k)); % Specificity for beta.

        % Dropping measurements with bitflips.
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
        RRMSE(1, k) = norm(beta - x_odrl(1:p)) / norm(beta); % Store RRMSE for ODRL.
        
        
    end
end
