function lamb = CV_l1(y, A)
    % This function performs cross-validation to find the optimal L1 regularization
    % parameter (lambda) for a lasso regression model. The goal is to minimize
    % the prediction error on a test set.

    % Define a range of lambda values (exponentially spaced) to be tested
    lam = exp(1:0.25:7);  % Range of lambda values
    s = length(lam);  % Number of lambda values to test
    err_y = zeros(s, 1);  % Initialize error array for each lambda
    min_err = 100;  % Set a high initial value for minimum error
    itr = 5;  % Number of iterations for cross-validation
    
    % Get the dimensions of matrix A
    [m, n] = size(A);  % m: number of rows (data points), n: number of columns (features)
    
    % Take 90% of the data for training (cross-validation)
    k = ceil(0.9 * m);  % Calculate the size of the training set (90% of the total data)
    
    % Loop over all lambda values
    for i = 1:s
        % Perform cross-validation over several iterations
        for j = 1:itr
            % Randomly shuffle the indices of the data points
            idx = randperm(m);  
            
            % Split data into training and test sets using the shuffled indices
            y_train = y(idx(1:k));  % Training response variable (first 90% of data)
            y_test = y(idx(k+1:end));  % Test response variable (remaining 10% of data)
            A_train = A(idx(1:k), :);  % Training feature matrix (first 90% of data)
            A_test = A(idx(k+1:end), :);  % Test feature matrix (remaining 10% of data)
            
            % Solve the L1-regularized lasso problem using CVX
            cvx_begin quiet
                variable the(n)  % Define the variable 'the' (theta) representing model coefficients
                % Minimize the L1-norm loss function with L1-regularization
                minimise(norm(y_train - A_train * the, 1) + lam(i) * norm(the, 1))
            cvx_end
            
            % Predict the output for the test set using the trained coefficients
            y_new = A_test * the;
            
            % Calculate the error between predicted and actual values (normalized)
            err_y(i) = err_y(i) + norm(y_new - y_test, 2) / norm(y_new, 2);
        end
        
        % Average the error over all iterations
        err_y(i) = err_y(i) / itr;
        
        % Keep track of the lambda that minimizes the prediction error
        if min_err > err_y(i)
            min_err = err_y(i);  % Update minimum error
            lamb = lam(i);  % Store the optimal lambda value
        end
    end
end