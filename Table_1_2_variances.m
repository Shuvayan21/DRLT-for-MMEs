
rng(1);

% Parameters
p = 500;
f_sp = 0.01;
f_sig = 0.01;
f_adv=0.01;
n_vals = 200:50:450;
run = 25;

% Containers
ETV_beta_W = zeros(size(n_vals));
ETV_delta_W = zeros(size(n_vals));
ETV_beta_A = zeros(size(n_vals));
ETV_delta_A = zeros(size(n_vals));
ATV_beta_W = zeros(size(n_vals));
ATV_delta_W = zeros(size(n_vals));
ATV_beta_A = zeros(size(n_vals));
ATV_delta_A = zeros(size(n_vals));

for idx = 1:length(n_vals)
    n = n_vals(idx);
    [A,A_tilde,beta,delta,sigma]=data_create(n,p,f_sig,f_adv,f_sp);
    W=weight_W(A);
    beta_d_W = zeros(p, run);
    delta_d_W = zeros(n, run);
    beta_d_A = zeros(p, run);
    delta_d_A = zeros(n, run);
    % Generate noise and compute the response variable.
    eta = random("normal", 0, sigma, [n 1]); % Generate normal noise.
    y = A_tilde * beta + eta; % Response variable with noise.
    
    % Cross-validation to determine optimal regularization parameters.
    [lambda_1, lambda_2] = CV_Drlt(y, A, W, sigma); % Cross-validation for ODRL.


    for k = 1:run
        
        eta = random("normal", 0, sigma, [n 1]); % Generate normal noise.
        y = A_tilde * beta + eta;
        
        % Optimization problem
        cvx_begin quiet
            variable x_l(n + p)
            minimize (0.5 * sum_square(y - [A eye(n)] * x_l) + lambda_1 * norm(x_l(1:p),1) + lambda_2 * norm(x_l(p+1:end),1))
        cvx_end

        beta_l = x_l(1:p);
        delta_l = x_l(p+1:end);

        % Debiased estimates
        beta_d_W(:,k) = beta_l + 1/n * W' * (y - A * beta_l - delta_l);
        delta_d_W(:,k) = delta_l + (eye(n) - 1/n * A * W') * (y - A * beta_l - delta_l);

        beta_d_A(:,k) = beta_l + 1/n * A' * (y - A * beta_l - delta_l);
        delta_d_A(:,k) = delta_l + (eye(n) - 1/n * A * A') * (y - A * beta_l - delta_l);
    end

    % ETV: Empirical average of variances across coordinates
    ETV_beta_W(idx) = mean(var(beta_d_W, 0, 2));
    ETV_delta_W(idx) = mean(var(delta_d_W, 0, 2));
    ETV_beta_A(idx) = mean(var(beta_d_A, 0, 2));
    ETV_delta_A(idx) = mean(var(delta_d_A, 0, 2));

    % ATV: Theoretical average diagonal entries of covariance matrices
    Sigma_beta_W = sigma^2 / n * (W' * W);
    Sigma_delta_W = sigma^2 * (eye(n) - 2/n * A * W + 1/n^2 * A * (W * W') * A');
    ATV_beta_W(idx) = mean(diag(Sigma_beta_W));
    ATV_delta_W(idx) = mean(diag(Sigma_delta_W));

    % Simplified expressions
    ATV_beta_A(idx) = (p / n) * sigma^2;
    sum_term = 0;
    AtA = A' * A;
    for i = 1:n
        ai = A(i,:)';
        sum_term = sum_term + (1 - 2*p/n + (1/n^2) * ai' * AtA * ai);
    end
    ATV_delta_A(idx) = sigma^2 * sum_term;
end

% Show table
table(n_vals', ETV_beta_W', ETV_delta_W', ETV_beta_A', ETV_delta_A', ATV_beta_W', ATV_delta_W', ATV_beta_A', ATV_delta_A', ...
    'VariableNames', {'n', 'ETV_beta_W', 'ETV_delta_W', 'ETV_beta_A', 'ETV_delta_A', ...
                      'ATV_beta', 'ATV_delta', 'AsympVar_beta', 'AsympVar_delta'})
