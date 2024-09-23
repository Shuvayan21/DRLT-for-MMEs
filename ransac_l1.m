function [rmse] = ransac_l1(y, A, x, lam, itr)  
    % RANSAC_L1_CONFUSION performs the RANSAC algorithm to fit a model using L1 regularization.
    % 
    % Inputs:
    %   y    - Response variable (observations).
    %   A    - Design matrix (features).
    %   x    - True parameter vector (for RMSE calculation).
    %   lam  - Regularization parameter for L1 norm.
    %   itr  - Number of iterations for RANSAC.
    %
    % Output:
    %   rmse - Root Mean Square Error of the estimated parameters compared to the true parameters.

    [m, n] = size(A); % Get the number of samples (m) and features (n)
    
    % Set the number of samples to use for training (90% of m)
    k = ceil(0.9 * m); 
    good_test = 100; % Initialize the threshold for good test error
    lamb = lam; % Store the regularization parameter

    % RANSAC iterations
    for j = 1:1:itr
        % Randomly permute the indices for train-test splitting
        idx = randperm(m);
        
        % Create training and testing sets
        y_train = y(idx(1:k));          % Training responses
        y_test = y(idx(k+1:end));      % Testing responses
        A_train = A(idx(1:k), :);      % Training features
        A_test = A(idx(k+1:end), :);   % Testing features
        
        % Optimization step: Fit the model to the training set
        cvx_begin
            variable the(n) % Coefficients to be estimated
            minimise (norm(y_train - A_train * the, 1) + lamb * norm(the, 1)) 
        cvx_end
        
        % Test the model on the testing set
        test = norm(y_test - A_test * the, 1) / norm(y_test, 1); % L1 norm error normalized by y_test
        if (test < good_test) % Update the best model if the current test error is lower
            good_test = test; % Update the threshold for a good test error
            then = the;       % Store the coefficients for the best model
        end
    end
    
    % Compute new predictions using the best coefficients found
    y_new = A * then; 
    
    % Fitting again using the new y values
    cvx_begin
        variable thez(n) % Coefficients for the new fitting
        minimise (norm(y_new - A * thez, 1) + lamb * norm(thez, 1)) 
    cvx_end
    
    % Calculate the root mean square error (RMSE) compared to the true parameter x
    rmse = norm(x - thez) / norm(x); 
end
