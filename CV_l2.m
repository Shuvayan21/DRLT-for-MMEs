function lamb = CV_l2(y, A)
    % This function performs cross-validation to find the optimal L2 regularization
    % parameter (lambda) for a regression model. The L2 regularization is also known
    % as Ridge Regression, where the goal is to minimize the prediction error on a test set.

    % Input:
    % y - response vector (observations)
    % A - design matrix (features matrix)
    %
    % Output:
    % lamb - the optimal regularization parameter (lambda)

    lamb = -1;  % Initialize lambda to -1, indicating no value has been selected yet
    lam = exp(1:0.25:7);  % Range of candidate lambda values (exponentially spaced)
    s = size(lam, 2);  % Number of candidate lambda values
    err_y = zeros(s, 1);  % Array to store test set errors for each lambda
    min_err = 100;  % Set an initial high value for minimum error
    itr = 1;  % Number of cross-validation iterations (set to 1 here)
    
    % Get dimensions of design matrix A
    [m, n] = size(A);  % m: number of data points, n: number of features
    
    % Determine the size of the training set (90% of the total data)
    k = ceil(0.9 * m);  % Taking 90% of the data for training
    
    % Loop over all lambda values
    for i = 1:s
        % Perform cross-validation for each lambda value
        for j = 1:itr
            % Randomly shuffle the data indices to create train/test splits
            idx = randperm(m);  
            
            % Split data into training and test sets based on the shuffled indices
            y_train = y(idx(1:k));  % Training set for the response variable
            y_test = y(idx(k+1:end));  % Test set for the response variable
            A_train = A(idx(1:k), :);  % Training set for the design matrix
            A_test = A(idx(k+1:end), :);  % Test set for the design matrix
            
            % Optimization using CVX (Convex Optimization Toolbox)
            cvx_begin quiet
                variable the(n)  % Define the variable 'the' (theta) representing model coefficients
                
                % Minimize the sum of the squared error (L2-norm) and L1 regularization term
                minimise(pow_pos(norm(y_train - A_train * the), 2) + lam(i) * norm(the, 1))
            cvx_end
            
            % Predict the output on the test set using the trained model
            y_new = A_test * the;
            
            % Compute the test set prediction error (normalized by the norm of predictions)
            err_y(i) = err_y(i) + norm(y_new - y_test, 2) / norm(y_new, 2);
        end
        
        % Average the test set error over iterations (10 iterations here)
        err_y(i) = err_y(i) / itr;
        
        % Update the optimal lambda if the current lambda has a lower test set error
        if min_err > err_y(i)
            min_err = err_y(i);  % Update the minimum error encountered
            lamb = lam(i);  % Set the optimal lambda to the current value
        end
    end
    
end
