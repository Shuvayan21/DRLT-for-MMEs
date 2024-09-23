function [lambda_1, lambda_2] = CV_Drlt(y, A, W, sigma)
    % This function performs cross-validation to find the optimal regularization
    % parameters (lambda_1 and lambda_2) for a regression model using a debiased
    % robust lasso with a debiasing matrix W. It minimizes prediction error on
    % a test set and maximizes p-value acceptances based on Lilliefors test.
    
    % Initialize lambda values as 1 (to indicate no value chosen yet)
    lambda_1 = 1;
    lambda_2 = 1;
    
    % Define ranges of possible lambda_1 and lambda_2 values, spaced exponentially
    lamb_1 = exp(1:0.25:7);  % Range for lambda_1
    lamb_2 = exp(1:0.25:7);  % Range for lambda_2
    
    [n, p] = size(A);  % Get dimensions of matrix A (n: rows, p: columns)
    run = 10;  % Number of iterations/runs for cross-validation
    
    % Initialize arrays to store p-values, residual errors, and RMSE values
    p_value_beta = zeros(length(lamb_1), length(lamb_2), p);
    p_value_delta = zeros(length(lamb_1), length(lamb_2), n);
    err_y = zeros(length(lamb_1), length(lamb_2));  % Test residual error
    rmse = zeros(length(lamb_1), length(lamb_2));  % Root Mean Squared Error (RMSE)
    min_err = inf;  % Initialize minimum error with a high value for comparison
    
    l = ceil(0.9 * n);  % Take 90% of data as training set size
    % Initialize storage for beta, delta, and test statistics for each run
    beta_l = zeros(p, run, length(lamb_1), length(lamb_2));
    beta_d_W = zeros(p, run, length(lamb_1), length(lamb_2));
    delta_l = zeros(n, run, length(lamb_1), length(lamb_2));
    delta_d_W = zeros(n, run, length(lamb_1), length(lamb_2));
    TG = zeros(p, run, length(lamb_1), length(lamb_2));  % T-statistics for beta
    TH = zeros(n, run, length(lamb_1), length(lamb_2));  % T-statistics for delta
    
    I_n = eye(n);  % Identity matrix of size n
    Sig = (A' * A) / n;  % Covariance matrix for matrix A
    Sigma_beta_W = sigma^2 / n * (W' * W);  % Covariance for beta estimates using W
    Sigma_delta_W = sigma^2 * (eye(n) - 2 / n * W * A' + 1 / n * W * Sig * W');  % Covariance for delta estimates using W
    
    % Iterate over all combinations of lamb_1 and lamb_2 to find the best parameters
    for i = 1:length(lamb_1)
        for j = 1:length(lamb_2)
            for k = 1:run
                % Use convex optimization (CVX) to solve the regularized regression problem
                cvx_begin quiet
                    variable x_l(n + p)  % Define optimization variable (length n+p)
                    % Minimize the loss function with L1-regularized beta and delta
                    minimise (0.5 * pow_pos(norm(y - [A I_n] * x_l), 2) + lamb_1(i) * norm(x_l(1:p), 1) + lamb_2(j) * norm(x_l(p+1:p+n), 1))
                cvx_end
                
                % Extract beta and delta parts from the solution vector x_l
                beta_l(:, k, i, j) = x_l(1:p);
                delta_l(:, k, i, j) = x_l(p+1:p+n);
                
                % Debias beta and delta using W matrix
                beta_d_W(:, k, i, j) = beta_l(:, k, i, j) + 1 / n * W' * (y - A * beta_l(:, k, i, j) - delta_l(:, k, i, j));
                delta_d_W(:, k, i, j) = delta_l(:, k, i, j) + (I_n - 1 / n * A * W') * (y - A * beta_l(:, k, i, j) - delta_l(:, k, i, j));
                
                % Calculate T-statistics for beta and delta
                for i1 = 1:p
                    TG(i1, k, i, j) = sqrt(n) * beta_d_W(i1, k, i, j) / sqrt(Sigma_beta_W(i1, i1));
                end
                for j1 = 1:n
                    TH(j1, k, i, j) = delta_d_W(j1, k, i, j) / sqrt(Sigma_delta_W(j1, j1));
                end
            end
            
            % Perform Lilliefors test (normality test) on T-statistics for beta and delta
            for i1 = 1:p
                [~, p_val] = lillietest(TG(i1, :, i, j));
                p_value_beta(i, j, i1) = p_val;  % Store p-value for beta
            end
            for j1 = 1:n
                [~, p_val] = lillietest(TH(j1, :, i, j));
                p_value_delta(i, j, j1) = p_val;  % Store p-value for delta
            end
        end
    end
    
    % Compute proportion of significant p-values (p > 0.01) for beta and delta
    p_value_beta_cutoff = p_value_beta > 0.01;
    L_test_beta_ratio = mean(p_value_beta_cutoff, 3);  % Ratio of accepted p-values for beta
    p_value_delta_cutoff = p_value_delta > 0.01;
    L_test_delta_ratio = mean(p_value_delta_cutoff, 3);  % Ratio of accepted p-values for delta
    
    % Perform cross-validation on lambdas that pass Lilliefors test with > 80% acceptance
    A1 = [A eye(n)];  % Augmented matrix for beta and delta estimates
    for i = 1:length(lamb_1)
        for j = 1:length(lamb_2)
            % Only consider combinations where the test ratio for both beta and delta is > 80%
            if (L_test_beta_ratio(i, j) > 0.7) && (L_test_delta_ratio(i, j) > 0.7)
                for k = 1:run
                    idx = randperm(n);  % Randomly permute the indices to split the data into training and test sets
                    
                    % Split the data into training and test sets
                    y_train = y(idx(1:l));
                    y_test = y(idx(l+1:end));
                    A_train = A1(idx(1:l), :);
                    A_test = A1(idx(l+1:end), :);
                    
                    % Solve the regularized regression problem on the training set
                    cvx_begin quiet
                        variable x_l(n + p)
                        minimise (0.5 * pow_pos(norm(y_train - A_train * x_l), 2) + lamb_1(i) * norm(x_l(1:p), 1) + lamb_2(j) * norm(x_l(p+1:n+p), 1))
                    cvx_end
                    
                    % Compute prediction error on the test set
                    y_new = A_test * x_l;
                    err_y(i, j) = err_y(i, j) + pow_pos(norm(y_new - y_test), 2);  % Test set prediction error
                    
                    % Compute RMSE for beta
                    rmse(i, j) = rmse(i, j) + norm(beta - x_l(1:n)) / norm(beta);
                end
                
                % Average error and RMSE over the number of runs
                err_y(i, j) = err_y(i, j) / run;
                rmse(i, j) = rmse(i, j) / run;
                
                % Track the lambda values that minimize the test set error
                if min_err > err_y(i, j)
                    min_err = err_y(i, j);
                    lambda_1 = lamb_1(i);  % Optimal lambda_1
                    lambda_2 = lamb_2(j);  % Optimal lambda_2
                end
            end
        end
    end
end
