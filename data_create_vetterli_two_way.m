function [A,A_hat,beta,delta,sigma]=data_create_vetterli_two_way(n,p,f_sig,f_sp,alpha,theta)
    %rng(1) %setting seed
    s=floor(p*f_sp); % sparsity level of \beta
    beta = zeros(p, 1);
    S = randperm(p, s); %support of the non-zero elements of \beta
    % Choosing non-zero elements from uniform distribution
    beta(S(1:floor(0.4*s))) =500+500*rand(floor(0.4*s),1);
    beta(S(floor(0.4*s)+1:s)) =500+500*rand(s-floor(0.4*s),1);
    % 0/1 pooling matrix
    q = alpha / s;
    A = double(rand(n, p) < q);

    % Generate sampling matrix A-hat by flipping 1s in A with probability (1-theta)
    A_hat = A;
    flip_mask = (rand(n, p) > theta) & (A == 1);
    A_hat(flip_mask) = 0;
    flip_mask_2= (rand(n, p) > theta) & (A == 0);
    A_hat(flip_mask_2) = 1;
    delta=(A_hat-A)*beta;
    %creating the standard deviation sigma
    sigma=mean(abs(A*beta))*f_sig;
end