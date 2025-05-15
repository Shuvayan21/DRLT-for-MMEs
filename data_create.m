function [A, A_hat, beta, delta, sigma] = data_create(n, p, f_sig, f_adv, f_sp)
    % data_create: Generates data for a robust regression problem.
    % The function creates a design matrix, sparse coefficient vector, and
    % introduces adversarial modifications to simulate malicious errors.
    
    % Input:
    % n     - Number of observations (rows of the design matrix A)
    % p     - Number of features (columns of the design matrix A)
    % f_sig - Noise scaling factor to calculate sigma (controls noise level)
    % f_adv - Fraction of adversarial MMEs applied to the design matrix
    % f_sp  - Fraction of non-zero (sparse) entries in the coefficient vector beta
    
    % Output:
    % A      - Original design matrix (n x p) with centered random binary entries
    % A_hat  - Adversarially modified design matrix
    % beta   - Sparse coefficient vector (p x 1)
    % delta  - Difference between the adversarial and original design matrices, applied to beta
    % sigma  - Standard deviation of the noise, calculated based on A and beta
    
    rng(1)  % Set the random seed for reproducibility (ensures consistent results)

    % Step 1: Generate the sparse coefficient vector beta
    s = floor(p * f_sp);  % Calculate the sparsity level (number of non-zero elements in beta)
    beta = zeros(p, 1);  % Initialize beta as a zero vector of length p

    % Select the support (non-zero positions) of beta randomly
    S = randperm(p, s);  % Randomly select 's' indices from 1 to p (the support)

    % Assign non-zero values to the selected indices of beta
    % 40% of non-zero elements are chosen from the range [50, 100]
    beta(S(1:floor(0.4 * s))) = 50 + 50 * rand(floor(0.4 * s), 1);
    
    % The remaining 60% of non-zero elements are chosen from the range [500, 1000]
    beta(S(floor(0.4 * s) + 1:s)) = 500 + 500 * rand(s - floor(0.4 * s), 1);

    % Step 2: Generate the original design matrix A
    % Create a binary matrix with random entries (0/1)
    mat = (rand(n, p) < 0.5);  % Generate an n x p matrix where each entry is 0 or 1 with probability 0.5
    
    % Center the binary matrix to have entries -1 and 1
    A = 2 * (mat - 0.5 * ones(n, p));  % Convert the binary matrix to a matrix with entries in {-1, +1}

    % Step 3: Introduce adversarial modifications to the design matrix A
    A_hat = MME_create(n, p, f_adv, A, S);  % A_hat is the modified version of A, where adversarial errors are introduced

    % Step 4: Calculate the adversarial effect delta
    delta = (A_hat - A) * beta;  % The difference between the adversarial matrix A_hat and the original matrix A, applied to beta

    % Step 5: Calculate the standard deviation sigma of the noise
    sigma = mean(abs(A * beta)) * f_sig;  % sigma is proportional to the mean absolute value of A * beta, scaled by f_sig
end
