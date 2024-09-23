function [lambda_1, lambda_2] = CV_RL(y, A)
    % CV_RL: Cross-Validation for Robust Lasso
    % This function performs cross-validation to find the optimal regularization parameters
    % lambda_1 and lambda_2 for the Robust Lasso algorithm, which includes both L1 penalties 
    % on the model parameters and an additional penalty for errors.
    
    % Input:
    % y - response vector (observations)
    % A - design matrix (features matrix)
    
    % Output:
    % lambda_1 - optimal regularization parameter for the model coefficients (L1 penalty)
    % lambda_2 - optimal regularization parameter for the error (L1 penalty)

    lambda_1 = -1;  % Initialize lambda_1 to -1, indicating no value has been selected yet
    lambda_2 = -1;  % Initialize lambda_2 to -1, indicating no value has been selected yet
    
    % Define the range of candidate lambda_1 and lambda_2 values (exponentially spaced)
    lamb_1 = exp(1:0.25:7);  % Lambda range for L1 regularization on model coefficients
    lamb_2 = exp(1:0.25:7);  % Lambda range for L1 regularization on errors
    
    % Get the dimensions of the design matrix A
    [m, n] = size(A);  % m: number of data points, n: number of features
    
    % Initialize a matrix to store the test set errors for each pair of (lambda_1, lambda_2)
    err_y = zeros(size(lamb_1, 2), size(lamb_2, 2));  
    
    % Set an initial high value for the minimum error
    min_err = inf;  % 'inf' is used as a placeholder for a very large value
    
    itr = 5;  % Number of cross-validation iterations for each lambda pair
    
    % Define the size of the training set (70% of the total data)
    k = ceil(0.7 * m);  % Training set size
    
    % Loop over all candidate lambda_1 values
    for i = 1:size(lamb_1, 2)
        % Loop over all candidate lambda_2 values
        for l = 1:size(lamb_2, 2)
            % Perform cross-validation for each pair of (lambda_1, lambda_2)
            for j = 1:itr
                % Augment the design matrix A by adding an identity matrix to account
                % for errors as additional variables (this is a Robust Lasso approach)
                A1 = [A, eye(m)];
                
                % Randomly shuffle the data indices to create train/test splits
                idx = randperm(m);
                
                % Split data into training and test sets based on the shuffled indices
                y_train = y(idx(1:k));  % Training set for the response variable
                y_test = y(idx(k+1:end));  % Test set for the response variable
                A_train = A1(idx(1:k), :);  % Training set for the augmented design matrix
                A_test = A1(idx(k+1:end), :);  % Test set for the augmented design matrix
                
                % Optimization using CVX (Convex Optimization Toolbox)
                cvx_begin quiet
                    variable x_l(n + m)  % Define the variable 'x_l' (includes model coefficients and error variables)
                    
                    % Objective function: L2-norm of the prediction error + L1 regularization terms
                    minimise (0.5 * pow_pos(norm(y_train - A_train * x_l), 2) + ...
                              lamb_1(i) * norm(x_l(1:n), 1) + ...  % L1 penalty on model coefficients
                              lamb_2(l) * norm(x_l(n+1:n+m), 1))  % L1 penalty on error variables
                cvx_end
                
                % Predict the response on the test set using the trained model
                y_new = A_test * x_l;
                
                % Calculate the test set prediction error (squared error)
                err_y(i, l) = err_y(i, l) + norm(y_new - y_test)^2;
            end
            
            % Average the test set error over iterations (itr times)
            err_y(i, l) = err_y(i, l) / itr;
            
            % Update the optimal lambda values if the current pair yields a lower error
            if min_err > err_y(i, l)
                min_err = err_y(i, l);  % Update the minimum error encountered
                lambda_1 = lamb_1(i);  % Set the optimal lambda_1 to the current value
                lambda_2 = lamb_2(l);  % Set the optimal lambda_2 to the current value
            end
        end
    end
end
