function x = gen_ggd_inv(n, sigma2, shape)

    % Compute scale parameter alpha from variance
    gamma_ratio = gamma(3/shape) / gamma(1/shape);
    alpha = sqrt(sigma2 / gamma_ratio);

    % Create grid for CDF
    x_grid = linspace(-10*alpha, 10*alpha, 1e5);
    cdf_vals = generalized_gaussian_cdf(x_grid, alpha, shape);

    % Ensure strictly increasing CDF for interp1
    [cdf_vals_unique, ia] = unique(cdf_vals);
    x_grid_unique = x_grid(ia);

    % Generate uniform samples
    u = rand(n, 1);

    % Invert CDF using interpolation
    x = interp1(cdf_vals_unique, x_grid_unique, u, 'linear', 'extrap');
end