clear  % Clear workspace variables
rng(1)  % Set the random seed for reproducibility

%% Table for comparing Baseline 1, Baseline 2 and Odrlt
p = 500;  % Number of features
measurements = 200:50:500;  % Range of sample sizes to test
f_adv = 0.01;  % adversarial proportion for MMEs
f_sp = 0.01;  % Sparsity level
f_sig = 0.01;  % Signal strength
run = 100;  % Number of runs for averaging results
flag = 1;  % Flag to determine whether to recompute A, beta, delta, sigma, and weight matrix W
z_alpha2 = 2.33;  % Critical value for hypothesis testing

%% Creating blank vectors to store results
% Sensitivity and specificity for ODrlt method
sens_delta_odrlt = zeros(run, size(measurements, 2));
spec_delta_odrlt = zeros(run, size(measurements, 2));
sens_beta_odrlt = zeros(run, size(measurements, 2));
spec_beta_odrlt = zeros(run, size(measurements, 2));

% Sensitivity and specificity for Baseline methods
sens_beta_base1 = zeros(run, size(measurements, 2));
spec_beta_base1 = zeros(run, size(measurements, 2));
sens_beta_base2 = zeros(run, size(measurements, 2));
spec_beta_base2 = zeros(run, size(measurements, 2));

%% Generating the sensitivity and specificity of Baseline 1, Baseline 2, and ODrlt
for l = 1:size(measurements, 2)  % Loop over different sample sizes
    n = measurements(l);  % Current sample size
    % Initialize matrices for storing estimates and results
    beta_l = zeros(p, run);  % Estimates for beta in ODrlt
    beta_d_W = zeros(p, run);  % Debiased estimates for beta with W adjustment
    delta_l = zeros(n, run);  % Estimates for delta in ODrlt
    delta_d_W = zeros(n, run);  % Debiased estimates for delta with W adjustment
    TG = zeros(p, run);  % Test statistics for beta in ODrlt
    TH = zeros(n, run);  % Test statistics for delta in ODrlt
    confusion_matrix_delta = zeros(n, run);  % Confusion matrix for delta
    confusion_matrix_beta = zeros(p, run);  % Confusion matrix for beta
    beta_d_base1 = zeros(p, run);  % Debiased estimates for beta in Baseline 1
    T_base1 = zeros(p, run);  % Test statistics for Baseline 1
    confusion_matrix_beta_base1 = zeros(p, run);  % Confusion matrix for Baseline 1
    beta_d_base2 = zeros(p, run);  % Debiased estimates for beta in Baseline 2
    T_base2 = zeros(p, run);  % Test statistics for Baseline 2
    confusion_matrix_beta_base2 = zeros(p, run);  % Confusion matrix for Baseline 2

    if flag == 0
        % Creating the data if flag is set to 0
        [A, A_tilde, beta, delta, sigma] = data_create(n, p, f_sig, f_adv, f_sp);
        % Generating the inverse W matrix
        W = weight_W(A);
    else 
        % Initialize beta and W using predefined parameters if flag is 1
        s = floor(p * f_sp);  % Sparsity level of beta
        beta = zeros(p, 1);  % Initialize beta vector
        S = randperm(p, s);  % Randomly select indices for non-zero elements of beta
        % Assign values to non-zero elements of beta from a uniform distribution
        beta(S(1:floor(0.4 * s))) = 50 + 50 * rand(floor(0.4 * s), 1);
        beta(S(floor(0.4 * s) + 1:s)) = 500 + 500 * rand(s - floor(0.4 * s), 1);
        
        % Load matrix A from a pre-saved .mat file
        A = cell2mat(struct2cell(load(append("A_", num2str(n), ".mat"))));
        % Create a modified version of A based on the model mismatch error
        A_tilde = MME_create(n, p, f_adv, A, S);
        % Load the weight matrix W from a pre-saved .mat file
        W = cell2mat(struct2cell(load(append("W_", num2str(n), ".mat"))));
        delta = (A_tilde - A) * beta;  % Calculate delta based on the difference
        % Creating the standard deviation sigma
        sigma = mean(abs(A * beta)) * f_sig;  % Scale mean absolute value of A*beta
    end

    % Compute the inverse of the covariance matrix for ODrlt
    M = inverse3(A, 2 * sqrt(log(p) / n));
    M_tilde = inverse3([A eye(n)], 2 * sqrt(log(p + n) / n));
    eta = random("normal", 0, sigma, [n, 1]);  % Generate noise
    y = A_tilde * beta + eta;  % Generate response variable y

    % Cross-validation for ODrlt and Baselines
    [lambda_1, lambda_2] = CV_Drlt(y, A, W, sigma);  % CV for ODrlt
    [lambda_Base1] = CV_l2(y, A);  % CV for Baseline 1
    [lambda_Base2] = CV_l2(y, [A eye(n)]);  % CV for Baseline 2

    % Define covariance matrices for beta and delta
    Sig = (A' * A) / n;
    Sigma_beta_W = sigma^2 / n * (W' * W);
    Sigma_delta_W = sigma^2 * (eye(n, n) - 2 / n * W * A' + 1 / n * W * Sig * W');
    Sigma_M = sigma^2 / n * M * (A' * A) * M';
    Sigma_M_tilde = sigma^2 / n * M_tilde * ([A eye(n)]' * [A eye(n)]) * M_tilde';

    for k = 1:run  % Loop over number of runs
        eta = random("normal", 0, sigma, [n, 1]);  % Generate new noise for each run
        y = A_tilde * beta + eta;  % Generate response variable y

        %% ODrlt Sensitivity and Specificity
        cvx_begin quiet  % CVX optimization for ODrlt
            variable x_l(n + p)  % Combined variable for optimization
            minimise (0.5 * pow_pos(norm(y - [A eye(n)] * x_l), 2) + ...
                       lambda_1 * norm(x_l(1:p), 1) + ...
                       lambda_2 * norm(x_l(p + 1:p + n), 1))  % Objective function
        cvx_end

        % Extract estimates from the optimization result
        beta_l(:, k) = x_l(1:p);  % Estimates for beta
        delta_l(:, k) = x_l((p + 1):(p + n));  % Estimates for delta
        
        % Compute debiased estimates for beta and delta
        beta_d_W(:, k) = beta_l(:, k) + 1 / n * W' * (y - A * beta_l(:, k) - delta_l(:, k));
        delta_d_W(:, k) = delta_l(:, k) + (eye(n) - 1 / n * A * W') * (y - A * beta_l(:, k) - delta_l(:, k));
        
        % Calculate test statistics for beta
        for i = 1:p
            TG(i, k) = (sqrt(n) * (beta_d_W(i, k))) / sqrt(Sigma_beta_W(i, i));
        end
        
        % Calculate test statistics for delta
        for j = 1:n
            TH(j, k) = (delta_d_W(j, k)) / sqrt(Sigma_delta_W(j, j));
        end

        % Compute confusion matrix and sensitivity/specificity for delta
        reject_H = find(TH(:, k) >= z_alpha2); 
        accept_H = find(TH(:, k) < z_alpha2);
        neg_neg = sum(and(delta == 0, TH(:, k) <= z_alpha2));  % True negatives
        neg_pos = sum(and(delta == 0, TH(:, k) > z_alpha2));  % False positives
        pos_neg = sum(and(delta ~= 0, TH(:, k) <= z_alpha2));  % False negatives
        pos_pos = sum(and(delta ~= 0, TH(:, k) > z_alpha2));  % True positives
        confusion_matrix_delta(:, k) = confusion_matrix_delta(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]';
        
        % Calculate sensitivity and specificity for ODrlt
        sens_delta_odrlt(k, l) = confusion_matrix_delta(4, k) / ...
                                  (confusion_matrix_delta(4, k) + confusion_matrix_delta(3, k));
        spec_delta_odrlt(k, l) = confusion_matrix_delta(1, k) / ...
                                  (confusion_matrix_delta(1, k) + confusion_matrix_delta(2, k));

        % Compute confusion matrix and sensitivity/specificity for beta
        reject_G = find(TG(:, k) >= z_alpha2); 
        accept_G = find(TG(:, k) < z_alpha2);
        neg_neg = sum(and(beta == 0, TG(:, k) <= z_alpha2));
        neg_pos = sum(and(beta == 0, TG(:, k) > z_alpha2));
        pos_neg = sum(and(beta ~= 0, TG(:, k) <= z_alpha2));
        pos_pos = sum(and(beta ~= 0, TG(:, k) > z_alpha2));
        confusion_matrix_beta(:, k) = confusion_matrix_beta(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]';
        
        % Calculate sensitivity and specificity for ODrlt beta estimates
        sens_beta_odrlt(k, l) = confusion_matrix_beta(4, k) / ...
                                 (confusion_matrix_beta(4, k) + confusion_matrix_beta(3, k));
        spec_beta_odrlt(k, l) = confusion_matrix_beta(1, k) / ...
                                 (confusion_matrix_beta(1, k) + confusion_matrix_beta(2, k));
        
        %% Baseline 1 Sensitivity and Specificity
        cvx_begin quiet  % CVX optimization for Baseline 1
            variable beta_base1(p)  % Variable for beta
            minimise (0.5 * pow_pos(norm(y - A * beta_base1), 2) + ...
                       lambda_Base1 * norm(beta_base1, 1))  % Objective function
        cvx_end
        
        % Compute debiased estimates for Baseline 1
        beta_d_base1(:, k) = beta_base1 + 1 / n * M * A' * (y - A * beta_base1);
        for i = 1:p
            T_base1(i, k) = (sqrt(n) * (beta_d_base1(i, k))) / sqrt(Sigma_M(i, i));  % Test statistics
        end
        
        % Compute confusion matrix and sensitivity/specificity for Baseline 1
        neg_neg = sum(and(beta == 0, T_base1(:, k) <= z_alpha2));
        neg_pos = sum(and(beta == 0, T_base1(:, k) > z_alpha2));
        pos_neg = sum(and(beta ~= 0, T_base1(:, k) <= z_alpha2));
        pos_pos = sum(and(beta ~= 0, T_base1(:, k) > z_alpha2));
        confusion_matrix_beta_base1(:, k) = confusion_matrix_beta_base1(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]';
        
        sens_beta_base1(k, l) = confusion_matrix_beta_base1(4, k) / ...
                                 (confusion_matrix_beta_base1(4, k) + confusion_matrix_beta_base1(3, k));
        spec_beta_base1(k, l) = confusion_matrix_beta_base1(1, k) / ...
                                 (confusion_matrix_beta_base1(1, k) + confusion_matrix_beta_base1(2, k));
        
        %% Baseline 2 Sensitivity and Specificity
        cvx_begin quiet  % CVX optimization for Baseline 2
            variable x_base2(n + p)  % Combined variable for optimization
            minimise (0.5 * pow_pos(norm(y - [A eye(n)] * x_base2), 2) + ...
                       lambda_Base2 * norm(x_base2, 1))  % Objective function
        cvx_end
        
        % Extract estimates for Baseline 2
        beta_l_base2 = x_base2(1:p);
        beta_d_base2(:, k) = beta_l_base2 + 1 / n * M_tilde * [A eye(n)]' * ...
                              (y - [A eye(n)] * x_base2);  % Debiased estimate for Baseline 2
        
        for i = 1:p
            T_base2(i, k) = (sqrt(n) * (beta_d_base2(i, k))) / sqrt(Sigma_M_tilde(i, i));  % Test statistics
        end
        
        % Compute confusion matrix and sensitivity/specificity for Baseline 2
        neg_neg = sum(and(beta == 0, T_base2(:, k) <= z_alpha2));
        neg_pos = sum(and(beta == 0, T_base2(:, k) > z_alpha2));
        pos_neg = sum(and(beta ~= 0, T_base2(:, k) <= z_alpha2));
        pos_pos = sum(and(beta ~= 0, T_base2(:, k) > z_alpha2));
        confusion_matrix_beta_base2(:, k) = confusion_matrix_beta_base2(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]';
        
        sens_beta_base2(k, l) = confusion_matrix_beta_base2(4, k) / ...
                                 (confusion_matrix_beta_base2(4, k) + confusion_matrix_beta_base2(3, k));
        spec_beta_base2(k, l) = confusion_matrix_beta_base2(1, k) / ...
                                 (confusion_matrix_beta_base2(1, k) + confusion_matrix_beta_base2(2, k));
    end
end

%% Summary Table
Table_1 = [measurements' mean(sens_beta_odrlt)' mean(sens_beta_base1)' ...
            mean(sens_beta_base2)' mean(spec_beta_odrlt)' ...
            mean(spec_beta_base1)' mean(spec_beta_base2)'];  % Create summary table
save("Table 1 TIT", "Table_1");  % Save the summary table to a .mat file
