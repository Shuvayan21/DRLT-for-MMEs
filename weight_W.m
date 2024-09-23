function W = weight_W(A)
    % This function computes a weight matrix W based on the input design matrix A
    % using convex optimization techniques to enforce specific constraints on W.
    %
    % Input:
    % A - Design matrix of size (n x p), where n is the number of observations and
    %     p is the number of predictors.
    %
    % Output:
    % W - Weight matrix of the same size as A (n x p) that satisfies the constraints
    %     imposed by the optimization problem.

    [n, p] = size(A); % Get the dimensions of the design matrix A.

    % Define parameters for the constraints based on the dimensions of A.
    mu_1 = 2 * sqrt(2 * log(p) / n); % Upper bound for the first constraint.
    mu_2 = 2 * sqrt(log((2 * p * n)) / (p * n)) + 1 / n; % Upper bound for the second constraint.
    mu_3 = 2 * sqrt(2 * log(n) / p) / sqrt(1 - n / p); % Upper bound for the third constraint.

    % Identity matrices and matrices of ones for constraints.
    I_n = speye(n, n); % n x n identity matrix.
    I_p = speye(p, p); % p x p identity matrix.
    Ones_pp = ones(p, p); % p x p matrix of ones.
    Ones_np = ones(n, p); % n x p matrix of ones.
    Ones_nn = ones(n, n); % n x n matrix of ones.
    
    % Compute the empirical covariance matrix of A.
    Sig = (A' * A) / n; % p x p covariance matrix.
    
    % Initialize W as a variable to optimize.
    W = A; % Start with W set to A.
    
    % Begin convex optimization problem definition.
    cvx_begin 
        variable W(n, p) % Declare W as a variable of size (n x p).
        
        % Objective function: Minimize the Frobenius norm of W.
        minimize((W(:)' * W(:))) % Equivalent to minimizing the sum of squares of W's elements.
        
        % Constraints
        subject to
            % First constraint: ensures that the average of W's columns aligns with I_p.
            abs(A' * W / n - I_p) <= mu_1 * Ones_pp;
            
            % Second constraint: ensures that the average of W's rows aligns with A.
            abs(W * Sig / p - A / p) <= mu_2 * Ones_np;
            
            % Third constraint: bounds the scaled difference of W and A.
            -mu_3 * Ones_nn <= n / (p * sqrt(1 - n / p)) * (W * A' / n - p * I_n / n) <= mu_3 * Ones_nn;
            
            % Loop constraint: ensures each column of W has a unit norm.
            for j = 1:p
                W(:, j)' * W(:, j) / n <= 1; % Normalize each column of W.
            end
    cvx_end % End of the convex optimization problem.
end
