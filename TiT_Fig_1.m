clear % Clear the workspace

rng(1) % Set random seed for reproducibility

%% Generating the values of the test statistics T_{G,j} and T_{H,i} for QQ plots of Fig 1
% Setting the values of the parameters
p = 500; % Number of features (variables)
n = 400; % Number of observations (samples)
f_adv = 0.01; % Fraction of adversarial flips
f_sp = 0.01; % Sparsity level (percentage of non-zero coefficients)
f_sig = 0.01; % Factor for noise level
run = 100; % Number of runs for statistical estimates
flag = 1; % Set to 0 to recompute data; set to 1 to use pre-set data

if flag == 0
    % Creating the data if flag is set to 0
    [A, A_tilde, beta, delta, sigma] = data_create(n, p, f_sig, f_adv, f_sp);
    % Generating the inverse weight matrix
    W = weight_W(A);
else 
    % If using pre-set data, generate beta with sparsity
    s = floor(p * f_sp); % Calculate sparsity level
    beta = zeros(p, 1); % Initialize beta
    S = randperm(p, s); % Support of the non-zero elements of beta

    % Assign non-zero elements from uniform distribution
    beta(S(1:floor(0.4*s))) = 50 + 50 * rand(floor(0.4*s), 1); % First 40% of non-zeros
    beta(S(floor(0.4*s)+1:s)) = 500 + 500 * rand(s - floor(0.4*s), 1); % Remaining non-zeros
    
    % Load pre-saved matrix A and create the perturbed matrix A_tilde
    A = cell2mat(struct2cell(load(append("A_", num2str(n), ".mat"))));
    A_tilde = MME_create(n, p, f_adv, A, S); % Create A_tilde with bitflips
    W = cell2mat(struct2cell(load(append("W_", num2str(n), ".mat")))); % Load weight matrix
    delta = (A_tilde - A) * beta; % Calculate the delta
    % Create the standard deviation sigma
    sigma = mean(abs(A * beta)) * f_sig; % Estimate noise level
end

% Creating blank vectors to store values for runs
x_d = zeros(n + p, run);
beta_l = zeros(p, run);
beta_d_W = zeros(p, run);
delta_l = zeros(n, run);
delta_d_W = zeros(n, run);
TG = zeros(p, run); % Store T_{G,j} values
TH = zeros(n, run); % Store T_{H,i} values

%% Obtaining lambda_1 and lambda_2 for cross-validation
% Create measurements with added noise for obtaining regularization parameters
eta = random("normal", 0, sigma, [n 1]); % Generate noise
y = A_tilde * beta + eta; % Measurements
[lambda_1, lambda_2] = CV_Drlt(y, A, W, sigma); % Cross-validation to find optimal lambdas

% Prepare covariance structures for testing statistics
Sig = (A' * A) / n; % Covariance of A
Sigma_beta_W = sigma^2 / n * (W' * W); % Covariance for beta
Sigma_delta_W = sigma^2 * (eye(n, n) - 2 / n * W * A' + 1 / n * W * Sig * W'); % Covariance for delta

% Loop through specified number of runs
for k = 1:1:run
    eta = random("normal", 0, sigma, [n 1]); % Generate new noise for each run
    y = A_tilde * beta + eta; % Measurements with noise
    cvx_begin quiet
        variable x_l(n + p) % Variable for optimization
        % Objective function to minimize (loss + regularization)
        minimise (0.5 * pow_pos(norm(y - [A eye(n)] * x_l), 2) + lambda_1 * norm(x_l(1:p), 1) + lambda_2 * norm(x_l(p + 1:p + n), 1))
    cvx_end
    
    % Extracting debiased estimates
    beta_l(:, k) = x_l(1:p); % Estimated beta
    delta_l(:, k) = x_l((p + 1):(p + n)); % Estimated delta
    % Compute debiased estimates
    beta_d_W(:, k) = beta_l(:, k) + 1 / n * W' * (y - A * beta_l(:, k) - delta_l(:, k));
    delta_d_W(:, k) = delta_l(:, k) + (eye(n) - 1 / n * A * W') * (y - A * beta_l(:, k) - delta_l(:, k));
    
    % Calculate test statistics T_{G,j}
    for i = 1:1:p
        TG(i, k) = (sqrt(n) * (beta_d_W(i, k) - beta(i))) / sqrt(Sigma_beta_W(i, i));
    end
    % Calculate test statistics T_{H,i}
    for j = 1:1:n
        TH(j, k) = (delta_d_W(j, k) - delta(j)) / sqrt(Sigma_delta_W(j, j));
    end
end

%% QQ Plot of T_{G,j} for all j=1,2,...,p
qqplot(TG) % Create QQ plot for T_{G,j}
hold on
plot(-3:0.1:3, -3:0.1:3, 'k') % Add reference line
hold off
axis equal
ylim([-4 4]) % Set y-limits for the plot
xlim([-4 4]) % Set x-limits for the plot

%% QQ Plot of T_{H,i} for all i=1,2,...,n
qqplot(TH) % Create QQ plot for T_{H,i}
hold on
plot(-3:0.1:3, -3:0.1:3, 'k') % Add reference line
hold off
axis equal
ylim([-4 4]) % Set y-limits for the plot
xlim([-4 4]) % Set x-limits for the plot
