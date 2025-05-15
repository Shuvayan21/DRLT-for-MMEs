clear 
rng(1)  % Set random seed for reproducibility

%% Initialize parameters
p = 500;  % Number of predictors
measurements = 200:50:500;  % Vector of different sample sizes (n)
f_adv = 0.01;  % Proportion of adversarial Model Mismatch Errors (MMEs)
f_sp = 0.01;  % Sparsity proportion
f_sig = 0.01;  % Noise scaling factor
run = 25;  % Number of simulations or runs
z_alpha2 = 2.33;  % Critical value for hypothesis testing

% Preallocate matrices for results
RRMSE = zeros(3, size(measurements, 2));  % Relative Root Mean Square Error
Sens_beta = zeros(3, size(measurements, 2));  % Sensitivity for beta estimates
Sens_delta = zeros(3, size(measurements, 2));  % Sensitivity for delta estimates
Spec_beta = zeros(3, size(measurements, 2));  % Specificity for beta estimates
Spec_delta = zeros(3, size(measurements, 2));  % Specificity for delta estimates

% Loop over each measurement size
for l = 1:size(measurements, 2)
    n = measurements(l);  % Current sample size
% Create synthetic data if flag is set to 0
        [A, A_tilde, beta, delta, sigma] = data_create(n, p, f_sig, f_adv, f_sp);
        % Generate the inverse weight matrix W
        W = weight_W(A);
% Sensitivity Specificity and RRMSE for Gaussian noise
        [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE_gauss(A_tilde, A, beta, delta, sigma, W);
        RRMSE(1, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(1, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(1, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(1, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(1, l) = mean(Sp_d);  % Store mean specificity for delta
% Sensitivity Specificity and RRMSE for Bounded Uniform
    [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE_bdd_unif(A_tilde, A, beta, delta, sigma, W);
        RRMSE(2, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(2, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(2, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(2, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(2, l) = mean(Sp_d);  % Store mean specificity for delta
    % Sensitivity Specificity and RRMSE for Genenralised Gaussian noise
    % with shape 3/2
    [Se_d, Sp_d, Se_b, Sp_b, R] = results_Sens_Spec_RRMSE_gen_gauss(A_tilde, A, beta, delta, sigma, W);
        RRMSE(3, l) = mean(R, 2);  % Store mean RRMSE values
    Sens_beta(3, l) = mean(Se_b);  % Store mean sensitivity for beta
    Sens_delta(3, l) = mean(Se_d);  % Store mean sensitivity for delta
    Spec_beta(3, l) = mean(Sp_b);  % Store mean specificity for beta
    Spec_delta(3, l) = mean(Sp_d);  % Store mean specificity for delta
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

%% Plot for Fig 7 (top-right) in supplemental and Fig 6(left) in main (Sensitivity and Specificity for delta)
% Plotting
figure;
hold on;

plot(measurements, Sens_delta(1,:), '-o', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_1); % circle
plot(measurements, Sens_delta(2,:), '-s', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_2); % square
plot(measurements, Sens_delta(3,:), '-^', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_3); % triangle
plot(measurements, Spec_delta(1,:), '--v', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_1); % down triangle
plot(measurements, Spec_delta(2,:), '--p', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_2); % pentagram
plot(measurements, Spec_delta(3,:), '--h', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_3); % hexagram
% Labels and legend
xlabel('n', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Sensitivity and Specificity for \delta', 'FontSize', 25, 'FontWeight', 'bold');

legend({'Sens. Gaussian', ...
        'Sens. Bounded Uniform', ...
        'Sens. General Gaussian', ...
        'Spec. Gaussian', ...
        'Spec. Bounded Uniform', ...
        'Spec. General Gaussian'}, ...
        'Location', 'southeast', 'FontSize', 25, 'FontWeight', 'bold');

set(gca, 'FontSize', 25, 'FontWeight', 'bold');
grid on;
hold off;
%% Plot for Fig 8(top-right) in supplemental and Fig 6(middle) in main (Sensitivity and Specificity for beta)
figure;
hold on;

plot(measurements, Sens_beta(1,:), '-o', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_1); % circle
plot(measurements, Sens_beta(2,:), '-s', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_2); % square
plot(measurements, Sens_beta(3,:), '-^', 'LineWidth', 6, 'MarkerSize', 20, 'Color', red_3); % triangle
plot(measurements, Spec_beta(1,:), '--v', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_1); % down triangle
plot(measurements, Spec_beta(2,:), '--p', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_2); % pentagram
plot(measurements, Spec_beta(3,:), '--h', 'LineWidth', 6, 'MarkerSize', 20, 'Color', blue_3); % hexagram


% Labels and legend
xlabel('n', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Sensitivity and Specificity for \beta', 'FontSize', 25, 'FontWeight', 'bold');

legend({'Sens. Gaussian', ...
        'Sens. Bounded Uniform', ...
        'Sens. General Gaussian', ...
        'Spec. Gaussian', ...
        'Spec. Bounded Uniform', ...
        'Spec. General Gaussian'}, ...
        'Location', 'southeast', 'FontSize', 25, 'FontWeight', 'bold');


set(gca, 'FontSize', 25, 'FontWeight', 'bold');
grid on;
hold off;

%% Plot for Fig 9(top-right) in supplemental and Fig 6(right) in main (RRMSE)
figure;
hold on;
plot(measurements, RRMSE(1,:), '-s', 'LineWidth', 6, 'MarkerSize', 20);
plot(measurements, RRMSE(2,:), '-o', 'LineWidth', 6, 'MarkerSize', 20);

plot(measurements, RRMSE(3,:), '-^', 'LineWidth', 6, 'MarkerSize', 20);

% Labels, title, and legend with font size 20
xlabel('n', 'FontSize', 25);
ylabel('RMSE', 'FontSize', 25);
legend({'Gaussian', 'Bounded Uniform', 'Generalised. Gaussian'}, ...
       'Location', 'northeast', 'FontSize', 25);
set(gca, 'FontSize', 25, 'FontWeight', 'bold');
grid on;
hold off;