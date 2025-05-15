function [A,A_hat,beta,delta,sigma]=data_create_ct(n,p,f_sig,f_adv,f_sp)
    %rng(1) %setting seed
    s=floor(p*f_sp); % sparsity level of \beta
    beta = zeros(p, 1);
    S = randperm(p, s); %support of the non-zero elements of \beta
    % Choosing non-zero elements from uniform distribution
    ct=sample_ct(s);
    beta(S)=2.^(40-ct);
    % 0/1 pooling matrix
    mat=(rand(n,p)<0.5);
    % Centering the measurements
    A=2*(mat-0.5*ones(n,p));%original -1/+1 matrix
    %inducing MMEs in the matrix A
    A_hat=MME_create(n,p,f_adv,A,S);
    delta=(A_hat-A)*beta;
    %creating the standard deviation sigma
    sigma=mean(abs(A*beta))*f_sig;
end