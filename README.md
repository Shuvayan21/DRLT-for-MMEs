This folder contains the matlab codes for the simulations shown in the paper "Robust Non-adaptive Group Testing under Errors in Group Membership Specifications" submitted to IEEE Transactions in Information Theory journal.

This folder contains files than produce the figures and tables given in the paper. It also contains functions that are necessary to create the figures and tables. This folder also contains .mat files for the matrices used for simulations and the weight matrix obtained from the aforementioned matrices using the optimisation algorithm Alg.1 of the paper.

Note that detailed description of all the functions and simulation files are provided in Sec.IV of the paper. We will first give a brief describe the following functions:

calculateSensitivitySpecificity.m - Given a true vector x and an estimated vector x_l, this function evaluates the sensitivity and specificity of x_l by choosing a threshold that maximised Youden's Index (Sensitivity+Specificity-1).

CV_Drlt.m - Cross validation function to generate regularisation parameters lambda_1 and lambda_2 for the Drlt and Odrlt methods using the same procedure as described in Sec.V of the paper.

CV_l1.m - Cross validation function to generate regularisation parameter l1 for L1-Lasso as described in Sec.V-E of the paper.

CV_l2.m - Cross validation function to generate regularisation parameter l2 for L2-Lasso as described in Sec.V-E of the paper.

CV_RL.m - Cross validation function to generate regularisation parameters lamb_1 and lamb_2 for Robust Lasso algorithm given in Eqn.(6) of the paper.

Distance_Decoder.m- Implements the distance-based decoding algorithm to solve for probabilistic errors in the pooling matrix.

Distance_Decoder_two_ways.m- Implements the distance-based decoding algorithm to solve for probabilistic errors in the pooling matrix.

data_create.m - function to generate the true Rademacher sensing matrix A, the MME-induced sensing matrix A_tilde, the signal \beta, the MME signal \delta and the noise standard deviation \sigma given the paramaters n,p,f_adv,f_sp and f_sig.

MME_create.m - function to generate the MME induced sensing matrix A_tilde for Rademacher A.

data_create_Balanced.m - function to generate the true centered Doubly Regular sensing matrix A, the MME-induced sensing matrix A_tilde, the signal \beta, the MME signal \delta and the noise standard deviation \sigma given the paramaters n,p,f_adv,f_sp and f_sig.

data_create_Rad_p.m - function to generate the true Centered Bernoulli(\theta) sensing matrix A, the MME-induced sensing matrix A_tilde, the signal \beta, the MME signal \delta and the noise standard deviation \sigma given the paramaters n,p,f_adv,f_sp and f_sig.

data_create_ct.m - function to generate the true Rademacher sensing matrix A, the MME-induced sensing matrix A_tilde, the viral loads signal \beta obtained from ct-cycle distribution, the MME signal \delta and the noise standard deviation \sigma given the paramaters n,p,f_adv,f_sp and f_sig.

data_create_vetterli.m - function to generate the true Rademacher sensing matrix A, the one way probabilistic error induced sensing matrix A_tilde, the signal \beta, the MME signal \delta and the noise standard deviation \sigma given the paramaters n,p,f_adv,f_sp and f_sig.

data_create_vetterli_two_way.m - function to generate the true Rademacher sensing matrix A, the two way probabilistic error induced sensing matrix A_tilde, the signal \beta, the MME signal \delta and the noise standard deviation \sigma given the paramaters n,p,f_adv,f_sp and f_sig.

ransac_l1.m - function to obtain the rmse of the RANSAC L1 algorithm given in Sec. IV-F of the paper.

ransac_l2.m - function to obtain the rmse of the RANSAC L2 algorithm given in Sec. IV-F of the paper.

weight_W.m - function to obtain the weight matrix W given A using the optimisation algorithm given in Alg. 1 of the paper.

weight_hW.m - function to obtain the weight matrix W given A using the optimisation algorithm given in Alg. 2 of the paper.

results_Sens_Spec_RRMSE.m - Given a MME-induced matrix A_tilde, true sensing matrix A, beta, delta and sigma, this is a function to generate the following: I) RRMSE of estimating beta for the algorithms L1 Lasso, L2 Lasso, Ransac L1 Lasso, Ransac L2 Lasso, Robust Lasso, Drl and Odrl as described in Sec V-E. II) Sensitivity and Specificity of estimating delta for Robust Lasso, Drlt and Odrlt as described in Sec V-C. III) Sensitivity and Specificity of estimating beta for Baseline 3, Robust Lasso, Drlt and Odrlt as described in Sec IV-E.

results_Sens_Spec_RRMSE_bdd_unif.m - Given a MME-induced matrix A_tilde, true sensing matrix A, beta, delta and sigma, this is a function to generate the following: I) RRMSE of estimating beta for the algorithms Robust Lasso, DrlT and OdrlT. II) Sensitivity and Specificity of estimating delta for Robust Lasso, Drlt and Odrlt as described in Sec V-C. III) Sensitivity and Specificity of estimating beta for Robust Lasso, Drlt and Odrlt under the Bounded Uniform noise model.

results_Sens_Spec_RRMSE_gauss.m - Given a MME-induced matrix A_tilde, true sensing matrix A, beta, delta and sigma, this is a function to generate the following: I) RRMSE of estimating beta for the algorithms Robust Lasso, DrlT and OdrlT. II) Sensitivity and Specificity of estimating delta for Robust Lasso, Drlt and Odrlt as described in Sec V-C. III) Sensitivity and Specificity of estimating beta for Robust Lasso, Drlt and Odrlt under the Gaussian noise model.

results_Sens_Spec_RRMSE_gen_gauss.m - Given a MME-induced matrix A_tilde, true sensing matrix A, beta, delta and sigma, this is a function to generate the following: I) RRMSE of estimating beta for the algorithms Robust Lasso, DrlT and OdrlT. II) Sensitivity and Specificity of estimating delta for Robust Lasso, Drlt and Odrlt as described in Sec V-C. III) Sensitivity and Specificity of estimating beta for Robust Lasso, Drlt and Odrlt under the Generalised Gaussian noise model.

results_Sens_Spec_RRMSE_vetterli.m - Given a MME-induced matrix A_tilde, true sensing matrix A, beta, delta and sigma, this is a function to generate the following: I) RRMSE of estimating beta for the algorithms Robust Lasso, DrlT and OdrlT. II) Sensitivity and Specificity of estimating delta for Robust Lasso, Drlt and Odrlt as described in Sec V-C. III) Sensitivity and Specificity of estimating beta for Robust Lasso, Drlt and Odrlt under the One way probabilistic error induced sensing matrix.

results_Sens_Spec_RRMSE_vetterli_two_way.m - Given a MME-induced matrix A_tilde, true sensing matrix A, beta, delta and sigma, this is a function to generate the following: I) RRMSE of estimating beta for the algorithms Robust Lasso, DrlT and OdrlT. II) Sensitivity and Specificity of estimating delta for Robust Lasso, Drlt and Odrlt as described in Sec V-C. III) Sensitivity and Specificity of estimating beta for Robust Lasso, Drlt and Odrlt under the Two way probabilistic error induced sensing matrix.

results_Sens_Spec_RRMSE_viral_loads.m - Given a MME-induced matrix A_tilde, true sensing matrix A, beta, delta and sigma, this is a function to generate the following: I) RRMSE of estimating beta for the algorithms Robust Lasso, DrlT and OdrlT. II) Sensitivity and Specificity of estimating delta for Robust Lasso, Drlt and Odrlt as described in Sec V-C. III) Sensitivity and Specificity of estimating beta for Robust Lasso, Drlt and Odrlt under the viral loads \beta obtained from the ct-cycle distribution.

gen_ggd_inv.m - function to obtain a sample from the generalise gaussian distribution.

generalised_gaussian_cdf.m - function to obtain a generalised gaussian cdf with given parameters.

Now we describe the files that is used to obtain the figures and tables of the simulations given in the paper. Note that for all these figures the pre-set true sensing matrix A and the weight matrix W is already provided as .mat files for different n. The nomenclature of the .mat files W_.mat and A_.mat respectively where n is the corresponding measurements. Furthermore, while running one can choose to use the pre-set matrices given or they can generate their own matrices and re-run the optimisation algorithm for W again using the 'flag' variable given in all of the upcoming codes.

Figure_1_QQPlots.m - code to generate the QQPlots of T_{Gj} and T_{Hi} of Figure 1 as described in Section IV-C of the paper.

Table_1_2_variances.m - code to generate the ratios of empirical and asymptotic variances of the debiased lasso estimators in Table 1 and 2 of Sec IV-B of the paper. 

Table3_Sens_Spec.m - code to generate the sensitivity and specificity of Baseline 1 and Baseline 2 given in Table 1 of Sec. IV-B of the paper.

Measurements_plots.m - This is a joint code to generate the plots for : I) measurements vs Sensitivity and Specificity of delta (Fig 2 (top-right)) II) measurements vs Sensitivity and Specificity of beta (Fig 3 (top-right)) III) measurements vs RRMSE (Fig 4 (top-right))

MMEs_plots.m - This is a joint code to generate the plots for : I) f_adv vs Sensitivity and Specificity of delta (Fig 2 (top-left)) II) f_adv vs Sensitivity and Specificity of beta (Fig 3 (top-left)) III) f_adv vs RRMSE (Fig 4 (top-left))

Noise_variance_plots.m - This is a joint code to generate the plots for : I) f_sig vs Sensitivity and Specificity of delta (Fig 2 (bottom-left)) II) f_sig vs Sensitivity and Specificity of beta (Fig 3 (bottom-left)) III) f_sig vs RRMSE (Fig 4 (bottom-left))

Sparsity_plots.m - This is a joint code to generate the plots for : I) f_sp vs Sensitivity and Specificity of delta (Fig 2 (bottom-right)) II) f_sp vs Sensitivity and Specificity of beta (Fig 3 (bottom-right)) III) f_sp vs RRMSE (Fig 4 (bottom-right))

These are the codes to obtain the plots for the comparison of different pooling matrices as described in Sec IV-G.

Measurements_plots_matrix.m - This is a joint code to generate the plots for : I) measurements vs Sensitivity and Specificity of delta (Fig 4 (top-right)) II) measurements vs Sensitivity and Specificity of beta (Fig 5 (top-right)) III) measurements vs RRMSE (Fig 6 (top-right)) in the supplemental.

MMEs_plots_matrix.m - This is a joint code to generate the plots for : I) f_adv vs Sensitivity and Specificity of delta (Fig 4 (top-left)) II) f_adv vs Sensitivity and Specificity of beta (Fig 5 (top-left)) III) f_adv vs RRMSE (Fig 6 (top-left)) in the supplemental. 

Noise_variance_plots_matrix.m - This is a joint code to generate the plots for : I) f_sig vs Sensitivity and Specificity of delta (Fig 4 (bottom-left)) II) f_sig vs Sensitivity and Specificity of beta (Fig 5 (bottom-left)) III) f_sig vs RRMSE (Fig 6 (bottom-left)) in the supplemental.

Sparsity_plots_matrix.m - This is a joint code to generate the plots for : I) f_sp vs Sensitivity and Specificity of delta (Fig 4 (bottom-right)) II) f_sp vs Sensitivity and Specificity of beta (Fig 5 (bottom-right)) III) f_sp vs RRMSE (Fig 6 (bottom-right)) in the supplemental.

These are the codes to obtain the plots for the comparison of different noise models as described in Sec IV-H.

Measurements_plots_noise_models.m - This is a joint code to generate the plots for : I) measurements vs Sensitivity and Specificity of delta (Fig 7 (top-right)) II) measurements vs Sensitivity and Specificity of beta (Fig 8 (top-right)) III) measurements vs RRMSE (Fig 9 (top-right)) in the supplemental.

MMEs_plots_noise_models.m - This is a joint code to generate the plots for : I) f_adv vs Sensitivity and Specificity of delta (Fig 7 (top-left)) II) f_adv vs Sensitivity and Specificity of beta (Fig 8 (top-left)) III) f_adv vs RRMSE (Fig 9 (top-left)) in the supplemental. 

Noise_variance_plots_noise_models.m - This is a joint code to generate the plots for : I) f_sig vs Sensitivity and Specificity of delta (Fig 7 (bottom-left)) II) f_sig vs Sensitivity and Specificity of beta (Fig 8 (bottom-left)) III) f_sig vs RRMSE (Fig 9 (bottom-left)) in the supplemental.

Sparsity_plots_noise_models.m - This is a joint code to generate the plots for : I) f_sp vs Sensitivity and Specificity of delta (Fig 7 (bottom-right)) II) f_sp vs Sensitivity and Specificity of beta (Fig 8 (bottom-right)) III) f_sp vs RRMSE (Fig 9 (bottom-right)) in the supplemental.

These are the codes to obtain the comparison of ODRLT, DRLT and Robust LASSO for beta obtained as real viral loads from CT-cycle distribution given in the supplemental.

Measurements_plots_viral_loads.m - This is a joint code to generate the plots for : I) measurements vs Sensitivity and Specificity of delta (Fig 1 (top-right)) II) measurements vs Sensitivity and Specificity of beta (Fig 2 (top-right)) III) measurements vs RRMSE (Fig 3 (top-right)) in the supplemental.

MMEs_plots_viral_loads.m - This is a joint code to generate the plots for : I) f_adv vs Sensitivity and Specificity of delta (Fig 1 (top-left)) II) f_adv vs Sensitivity and Specificity of beta (Fig 2 (top-left)) III) f_adv vs RRMSE (Fig 3 (top-left)) in the supplemental. 

Noise_variance_plots_viral_loads.m - This is a joint code to generate the plots for : I) f_sig vs Sensitivity and Specificity of delta (Fig 1 (bottom-left)) II) f_sig vs Sensitivity and Specificity of beta (Fig 2 (bottom-left)) III) f_sig vs RRMSE (Fig 3 (bottom-left)) in the supplemental.

Sparsity_plots_viral_loads.m - This is a joint code to generate the plots for : I) f_sp vs Sensitivity and Specificity of delta (Fig 1 (bottom-right)) II) f_sp vs Sensitivity and Specificity of beta (Fig 2 (bottom-right)) III) f_sp vs RRMSE (Fig 3 (bottom-right)) in the supplemental.

These are the codes to obtain the tables in Sec IV-I.

Table_IV_theta_one_way.m- Code to obtain Table IV of the main paper.

Table_V_n_one_way.m- Code to obtain Table V of the main paper.

Table_VI_theta_two_way.m- Code to obtain Table VI of the main paper.

Table_VII_n_two_way.m- Code to obtain Table VII of the main paper.
