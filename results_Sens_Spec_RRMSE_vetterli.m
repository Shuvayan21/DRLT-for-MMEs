function [ Sens_beta, Spec_beta] = results_Sens_Spec_RRMSE_vetterli(A,A_tilde,x,theta,q)
    
    rng(1) % Set random seed for reproducibility.
    %[n, p] = size(A); % Get dimensions of the design matrix A.
    run = 25; % Number of runs for the simulation.
    
    Sens_beta = zeros(1, run); % Sensitivity for beta.
    Spec_beta = zeros(1, run); % Specificity for beta.
    confusion_matrix_vetterli = zeros(4, run); % Confusion matrix for Vetterli.

    % Main loop to perform simulations.
    for k = 1:run
        
        x_hat=Distance_Decoder(A,A_tilde,x,theta,q);

        % Update confusion matrix and compute sensitivity and specificity for beta.
        neg_neg = sum(and(x == 0, x_hat == 0)); % True negatives for beta.
        neg_pos = sum(and(x == 0, x_hat ~= 0)); % False positives for beta.
        pos_neg = sum(and(x ~= 0, x_hat == 0)); % False negatives for beta.
        pos_pos = sum(and(x ~= 0, x_hat ~= 0)); % True positives for beta.
        confusion_matrix_vetterli(:, k) = confusion_matrix_vetterli(:, k) + [neg_neg, neg_pos, pos_neg, pos_pos]'; % Update confusion matrix.
        Sens_beta(1, k) = confusion_matrix_vetterli(4, k) / (confusion_matrix_vetterli(4, k) + confusion_matrix_vetterli(3, k)); % Sensitivity for beta.
        Spec_beta(1, k) = confusion_matrix_vetterli(1, k) / (confusion_matrix_vetterli(1, k) + confusion_matrix_vetterli(2, k)); % Specificity for beta.

                
        
    end
end
