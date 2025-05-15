function F = generalized_gaussian_cdf(x, sigma, beta)
    % Vectorized computation of the GGD CDF
    % CDF for x < 0
    F = zeros(size(x));
    neg = x < 0;
    pos = ~neg;

    % Integrate PDF numerically for CDF estimation
    F(neg) = 0.5 * gammainc((abs(x(neg))/sigma).^beta, 1/beta, 'upper');
    F(pos) = 1 - 0.5 * gammainc((abs(x(pos))/sigma).^beta, 1/beta, 'upper');
end