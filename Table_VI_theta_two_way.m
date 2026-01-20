clear
rng(1)
%% setting the values of the parameters
p=500;
n=400;
alpha=0.1;
actic_prob=[0.7:0.05:0.95];
f_sp=0.05;
f_sig=0.01;
run=1;
Sens_Odrlt=zeros(size(actic_prob,2),1);
Spec_Odrlt=zeros(size(actic_prob,2),1);
Sens_DistD=zeros(size(actic_prob,2),1);
Spec_DistD=zeros(size(actic_prob,2),1);
for l=1:size(actic_prob,2)
    theta=actic_prob(l);
    %creating the data
    [B,B_tilde,A,A_tilde,beta,delta,sigma]=data_create_vetterli_two_way(n,p,f_sig,f_sp,alpha,theta);
%% vetterli application
x=abs(beta)>0;
    [ Sens_beta, Spec_beta] = results_Sens_Spec_RRMSE_vetterli_two_way(B,B_tilde,x,theta,0.5);
    Sens_DistD(l,1)=mean(Sens_beta);
    Spec_DistD(l,1)=mean(Spec_beta);
    % generating the inverse W matrix
    W=weight_W(A);
[~, ~, Sens_beta, Spec_beta, ~] = results_Sens_Spec_RRMSE_gauss(B_tilde, B, A_tilde, A, beta, delta, sigma, W);
Sens_Odrlt(l,1)=mean(Sens_beta);
Spec_Odrlt(l,1)=mean(Spec_beta);

end
