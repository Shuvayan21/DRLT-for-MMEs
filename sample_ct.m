function samples = sample_ct(n)
    % sample_from_custom_pmf - Generates 'n' samples from a custom PMF
    
    % Support values
    ct = 15:1:40;

    % Corresponding PMF values
    pmf = [0.009661836;
           0.032206119;
           0.032206119;
           0.040257649;
           0.048309179;
           0.048309179;
           0.048309179;
           0.052334944;
           0.052334944;
           0.048309179;
           0.052334944;
           0.052334944;
           0.048309179;
           0.048309179;
           0.032206119;
           0.040257649;
           0.040257649;
           0.056360709;
           0.044283414;
           0.052334944;
           0.040257649;
           0.032206119;
           0.024154589;
           0.012077295;
           0.00805153;
           0.004025765];

    % Sanity check
    if abs(sum(pmf) - 1) > 1e-6
        error('PMF does not sum to 1.');
    end

    % Generate samples
    samples = randsample(ct, n, true, pmf);
end
