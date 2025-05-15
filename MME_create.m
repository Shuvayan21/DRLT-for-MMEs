function [A_tilde]= MME_create(n,p,f_adv,A,S) 
    % MME_CREATE generates a modified version of the matrix A by randomly introducing 
    % model mismatch errors (bitflips) based on a specified probability.
    % 
    % Inputs:
    %   n      - Number of rows (samples) in matrix A.
    %   p      - Number of columns (features) in matrix A.
    %   f_adv   - Probability of inducing bitflips in the rows of A.
    %   A      - Original data matrix of size n by p.
    %   S      - Indices of the non-zero elements in the original coefficient vector beta.
    % 
    % Outputs:
    %   A_tilde  - Modified matrix A with induced model mismatch errors.

    rng(1) % Set the random seed for reproducibility

    % Create a sparse random vector indicating which rows will have bitflips
    B_set = sprand(n, 1, f_adv); 

    % Create an identity matrix of size p for bitflip operations
    I = eye(p); 

    % Initialize A_hat as a copy of A to store the modified matrix
    A_tilde = A;

    % Loop through each row of the matrix A
    for i = 1:1:n
        % Check if this row will have a bitflip based on B_set
        if (B_set(i) > 0)
            % Randomly select an index from S for the bitflip
            k = S(randi(length(S)));

            % Create a modification matrix R that flips the k-th column
            R = I; 
            R(k, k) = -1; % Flip the sign of the k-th feature

            % Update the i-th row of A_hat by multiplying with the modification matrix R
            A_tilde(i, :) = A(i, :) * R; 
        end
    end
end
