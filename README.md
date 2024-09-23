# DRLT-for-MMEs
This folder contains the matlab codes for the simulations shown in the paper "Robust Non-adaptive Group Testing under Errors in
Group Membership Specifications" to be submitted to IEEE Transactions in Information Theory journal. Current version of this paper is submitted it arxiv in the following link: http://arxiv.org/abs/2409.05345 .

This folder contains files than produce the figures and tables given in the paper. It also contains functions that are necessary to create the figures and tables. This folder also contains .mat files for the matrices used for simulations and the weight matrix pbtained from the aforementioned matrices using the optimisation algorithm Alg.2 of the paper.

Note that detailed description of all the functions and simulation files are provided in Sec.V of the paper.
We will first give a brief describe the following functions:

1) calculateSensitivitySpecificity.m - Given a true vector x and an estimated vector x_l, this function evaluates the sensitivity and specificity of x_l by choosing a threshold that maximised Youden's Index (Sensitivity+Specificity-1).

2) CV_Drlt.m - Cross validation function to generate regularisation parameters lambda_1 and lambda_2 for the Drlt and Odrlt methods using the same procedure as described in Sec.V of the paper.

3) CV_l1.m - Cross validation function to generate regularisation parameter l1 for L1-Lasso as described in Sec.V-E of the paper.

4) CV_l2.m - Cross validation function to generate regularisation parameter l2 for L2-Lasso as described in Sec.V-E of the paper.

5) CV_RL.m - Cross validation function to generate regularisation parameters lamb_1 and lamb_2 for Robust Lasso algorithm given in Eqn.(6) of the paper.

6) data_create.m - function to generate the  true sensing matrix A, the MME-induced sensing matrix A_tilde, the signal \beta, the MME signal \delta and the noise standard deviation \sigma given the paramaters n,p,f_adv,f_sp and f_sig.

7) MME_create.m - function to generate the MME induced sensing matrix A_tilde.

8) ransac_l1.m - function to obtain the rmse of the RANSAC L1 algorithm given in Sec. V-E of the paper.

9) ransac_l2.m - function to obtain the rmse of the RANSAC L2 algorithm given in Sec. V-E of the paper.

10) weight_W.m - function to obtain the weight matrix W given A using the optimisation algorithm given in  Alg. 2 of the paper.

11) results_Sens_Spec_RRMSE.m - Given a MME-induced matrix A_tilde, true sensing matrix A, beta, delta and sigma, this is a function to generate the following:
	I) RRMSE of estimating beta for the algorithms L1 Lasso, L2 Lasso, Ransac L1 Lasso, Ransac L2 Lasso, Robust Lasso, Drl and Odrl as described in Sec V-E.
	II) Sensitivity and Specificity of estimating delta for Robust Lasso, Drlt and Odrlt as described in Sec V-C.
	III) Sensitivity and Specificity of estimating beta for Baseline 3, Robust Lasso, Drlt and Odrlt as described in Sec V-D.

Now we describe the files that is used to obtain the figures and tables of the simulations given in the paper. Note that for all these figures the pre-set true sensing matrix A and the weight matrix W is already provided as .mat files for different n. The nomenclature of the .mat files W_<n>.mat and A_<n>.mat respectively where n is the corresponding measurements. Furthermore, while running one can choose to use the pre-set matrices given or they can generate their own matrices and re-run the optimisation algorithm for W again using the 'flag' variable given in all of the upcoming codes.

12) TiT_Fig_1.m - code to generate the QQPlots of T_{Gj} and T_{Hi} of Figure 1 as described in Section V-B of the paper.

13) TIT_Table1.m - code to generate the matrix given in Table 1 of Sec. V-A of the paper.

14) Measurements_plots.m - This is a joint code to generate the plots for :
	I) measurements vs Sensitivity and Specificity of delta (Fig 2 (top-right)) 
	II) measurements vs Sensitivity and Specificity of beta (Fig 3 (top-right)) 
	III) measurements vs RRMSE (Fig 4 (top-right)) 
15) MMEs_plots.m - This is a joint code to generate the plots for :
	I) f_adv vs Sensitivity and Specificity of delta (Fig 2 (top-left)) 
	II) f_adv vs Sensitivity and Specificity of beta (Fig 3 (top-left)) 
	III) f_adv vs RRMSE (Fig 4 (top-left)) 
16) Noise_variance_plots.m - This is a joint code to generate the plots for :
	I) f_sig vs Sensitivity and Specificity of delta (Fig 2 (bottom-left)) 
	II) f_sig vs Sensitivity and Specificity of beta (Fig 3 (bottom-left)) 
	III) f_sig vs RRMSE (Fig 4 (bottom-left))
17) Sparsity_plots.m - This is a joint code to generate the plots for :
	I) f_sp vs Sensitivity and Specificity of delta (Fig 2 (bottom-right)) 
	II) f_sp vs Sensitivity and Specificity of beta (Fig 3 (bottom-right)) 
	III) f_sp vs RRMSE (Fig 4 (bottom-right))

This concludes the readme.txt
