clear 
rng(1)  % Set random seed for reproducibility

%% Initialize parameters
p = 500;  % Number of predictors
n = 400;  % Sample size
MME = 0.01:0.01:0.1;  % Range of adversarial MME values
f_sp = 0.01;  % Sparsity proportion
f_sig = 0.01;  % Noise scaling factor
run = 25;  % Number of simulations or runs
z_alpha2 = 2.33;  % Critical value for hypothesis testing

% Preallocate matrices for results
RRMSE = zeros(4, size(MME, 2));  % Relative Root Mean Square Error
Sens_beta = zeros(4, size(MME, 2));  % Sensitivity for beta estimates
Sens_delta = zeros(4, size(MME, 2));  % Sensitivity for delta estimates
Spec_beta = zeros(4, size(MME, 2));  % Specificity for beta estimates
Spec_delta = zeros(4, size(MME, 2));  % Specificity for delta estimates

% Loop over each MME value
for l = 1:size(MME, 2)
    f_adv = MME(l);  % Current MME value
    
        % Create data for Centered Bernoulli 0.1
        [A_0_1, A_tilde_0_1, beta, delta, sigma] = data_create_Rad_p(n,p,f_sig,f_adv,f_sp,0.1);
        % Generate the inverse weight matrix W
        W_0_1 = weight_hW(A_0_1); 
    % Compute sensitivity, specificity, and RRMSE for Centered Bernoulli 0.1
    [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE_gen(A_tilde_0_1, A_0_1, beta, delta, sigma, W_0_1);
    RRMSE(1, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(1, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(1, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(1, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(1, l) = mean(Sp_d);  % Store mean specificity for delta

    % Create data for Centered Bernoulli 0.3
        [A_0_3, A_tilde_0_3, beta, delta, sigma] = data_create_Rad_p(n,p,f_sig,f_adv,f_sp,0.3);
        % Generate the inverse weight matrix W
        W_0_3 = weight_hW(A_0_3); 
    % Compute sensitivity, specificity, and RRMSE for Centered Bernoulli
    % 0.3
    [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE_gen(A_tilde_0_3, A_0_3, beta, delta, sigma, W_0_3);
    RRMSE(2, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(2, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(2, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(2, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(2, l) = mean(Sp_d);  % Store mean specificity for delta

    % Create data for Centered Bernoulli 0.5
        [A_0_5, A_tilde_0_5, beta, delta, sigma] = data_create_Rad_p(n,p,f_sig,f_adv,f_sp,0.5);
        % Generate the inverse weight matrix W
        W_0_5 = weight_hW(A_0_5); 
    % Compute sensitivity, specificity, and RRMSE for Centered Bernoulli
    % 0.5
    [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE_gen(A_tilde_0_3, A_0_3, beta, delta, sigma, W_0_5);
    RRMSE(3, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(3, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(3, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(3, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(3, l) = mean(Sp_d);  % Store mean specificity for delta

    %Create data for Doubly Regular
    [A_Bal, A_tilde_Bal, beta, delta, sigma] =data_create_Balanced(n,p,f_sig,f_adv,f_sp);
    % Generate the inverse weight matrix W
        W_Bal = weight_hW(A_Bal); 
    % Compute sensitivity, specificity, and RRMSE for Doubly Regular
    [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE_gen(A_tilde_Bal, A_Bal, beta, delta, sigma, W_Bal);
    RRMSE(4, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(4, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(4, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(4, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(4, l) = mean(Sp_d);  % Store mean specificity for delta
end
% Colors
red_1 = [0.8500, 0.3250, 0.0980];
red_2 = [0.9290, 0.1500, 0.1410];
red_3 = [0.6350, 0.0780, 0.1840];
red_4 = [0.88,0.58,0.12];

blue_1 = [0, 0.4470, 0.7410];
blue_2 = [0.3010, 0.7450, 0.9330];
blue_3 = [0.0000, 0.4470, 0.7410];
blue_4 = [0.2500, 0.2500, 1.0000];

%% Plot for Fig 4(top-left) in supplemental (Sensitivity and Specificity for delta)
% Plotting
figure;
hold on;

plot(MME, Sens_delta(1,:), '-o', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_1); % circle
plot(MME, Spec_delta(1,:), '--v', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_1); % down triangle
plot(MME, Sens_delta(2,:), '-s', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_2); % square
plot(MME, Spec_delta(2,:), '--p', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_2); % pentagram
plot(MME, Sens_delta(3,:), '-^', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_3); % triangle
plot(MME, Spec_delta(3,:), '--h', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_3); % hexagram
plot(MME, Sens_delta(4,:), '-d', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_4); % diamond
plot(MME, Spec_delta(4,:), '--x', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_4); % x
% Labels and legend
xlabel('f_{adv}', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Sensitivity and Specificity for \delta', 'FontSize', 25, 'FontWeight', 'bold');

legend({'Sensitivity CB (\theta = 0.1)', ...
        'Specificity CB (\theta = 0.1)', 'Sensitivity CB (\theta = 0.3)', 'Specificity CB (\theta = 0.3)', 'Sensitivity CB (\theta = 0.5)', 'Specificity CB (\theta = 0.5)', 'Sensitivity DR', 'Specificity DR'}, ...
        'Location', 'southeast', 'FontSize', 25, 'FontWeight', 'bold');

set(gca, 'FontSize', 25, 'FontWeight', 'bold');
grid on;
hold off;
%% Plot for Fig 5(top-left) in supplemental (Sensitivity and Specificity for beta)
figure;
hold on;

plot(MME, Sens_beta(1,:), '-o', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_1); % circle
plot(MME, Spec_beta(1,:), '--v', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_1); % down triangle
plot(MME, Sens_beta(2,:), '-s', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_2); % square
plot(MME, Spec_beta(2,:), '--p', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_2); % pentagram
plot(MME, Sens_beta(3,:), '-^', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_3); % triangle
plot(MME, Spec_beta(3,:), '--h', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_3); % hexagram
plot(MME, Sens_beta(4,:), '-d', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_4); % diamond
plot(MME, Spec_beta(4,:), '--x', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_4); % x

% Labels and legend
xlabel('f_{adv}', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Sensitivity and Specificity for \beta', 'FontSize', 25, 'FontWeight', 'bold');

legend({'Sensitivity CB (\theta = 0.1)', ...
        'Specificity CB (\theta = 0.1)', 'Sensitivity CB (\theta = 0.3)', 'Specificity CB (\theta = 0.3)', 'Sensitivity CB (\theta = 0.5)', 'Specificity CB (\theta = 0.5)', 'Sensitivity DR', 'Specificity DR'}, ...
        'Location', 'southeast', 'FontSize', 25, 'FontWeight', 'bold');

set(gca, 'FontSize', 25, 'FontWeight', 'bold');
grid on;
hold off;

%% Plot for Fig 6(top-left) in supplemental (RRMSE)
figure;
hold on;
plot(MME, RRMSE(1,:), '-s', 'LineWidth', 6, 'MarkerSize', 20);
plot(MME, RRMSE(2,:), '-o', 'LineWidth', 6, 'MarkerSize', 20);

plot(MME, RRMSE(3,:), '-^', 'LineWidth', 6, 'MarkerSize', 20);
plot(MME, RRMSE(4,:), '-d', 'LineWidth', 6, 'MarkerSize', 20);

% Labels, title, and legend with font size 20
xlabel('f_{adv}', 'FontSize', 25);
ylabel('RMSE', 'FontSize', 25);
legend({'CB(\theta = 0.1) ', 'CB(\theta = 0.3) ', 'CB(\theta = 0.5)', 'DR'}, ...
       'Location', 'northeast', 'FontSize', 25);
set(gca, 'FontSize', 25, 'FontWeight', 'bold');
grid on;
hold off;