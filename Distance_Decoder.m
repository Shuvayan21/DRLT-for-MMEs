function x_hat=Distance_Decoder(A, A_tilde, x, p, q)
    [m,n]=size(A);
    M_c=(A+1)/2;
    M_s=(A_tilde+1)/2;
    y = any(M_s(:, x==1), 2);  % vector of test results

    % 5. Distance decoder
    e = round((1 - p) * q * m);  % max number of flips per column tolerated
    x_hat = zeros(n, 1);

    for i = 1:n
        c_i = M_c(:, i);
        % Count number of test positions where c_i == 1 but y == 0
        test_mismatch = sum((c_i == 1) & (y == 0));
        if test_mismatch <= e
            x_hat(i) = 1;
        end
    end

end