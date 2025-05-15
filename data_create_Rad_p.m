function [A,A_hat,beta,delta,sigma]=data_create_Rad_p(n,p,f_sig,f_adv,f_sp,prob)
    rng(1) %setting seed
    s=floor(p*f_sp); % sparsity level of \beta
    beta = zeros(p, 1);
    S = randperm(p, s); %support of the non-zero elements of \beta
    % Choosing non-zero elements from uniform distribution
    beta(S(1:floor(0.4*s))) =500+500*rand(floor(0.4*s),1);
    beta(S(floor(0.4*s)+1:s)) =500+500*rand(s-floor(0.4*s),1);
    % 0/1 pooling matrix
    mat=(rand(2*n,p)<prob);
    % Centering the measurements
    A=(mat(1:n,:)-mat((n+1:2*n),:))/(2*prob*(1-prob));%original -1/+1 matrix
    %inducing MMEs in the matrix A
    A_hat=MME_create(n,p,f_adv,A,S);
    delta=(A_hat-A)*beta;
    %creating the standard deviation sigma
    sigma=mean(abs(A*beta))*f_sig;
end
