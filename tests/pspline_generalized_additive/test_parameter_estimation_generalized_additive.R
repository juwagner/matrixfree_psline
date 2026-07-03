# ------------------------------------------------------------------------------
# Test suite: Regularization parameter estimation for generalized additive
# P-splines
# ------------------------------------------------------------------------------

rm(list=ls())
gc()

library(rTensor)
library(Matrix)

source("src/pspline/pspline_matrices.R")
source("src/pspline_generalized_additive/parameter_estimation_generalized_additive.R")

# ------------------------------------------------------------------------------
# Generate test data
# ------------------------------------------------------------------------------

source("tests/pspline_generalized_additive/generate_test_data_generalized_additive.R")

# ------------------------------------------------------------------------------
# Correctness tests
# ------------------------------------------------------------------------------

cat("======= Correctness Test =======\n")

# ------------------------------------------------------------------------------
# Estimate alpha_terms for fixed lambda_vec
n_iter <- 5

# With matrix-free algorithm
alpha_terms_mf <- estimate_alpha_generalized_terms(
  n_iter,
  y,
  PhiT_terms,
  L_terms,
  lambda_vec,
  alpha_init=alpha_terms,
  pcg_tol=10^(-10),
  pcg_verbose=FALSE
)

# Full computation for comparison
alpha_full <- alpha
for (i in 1:n_iter){
  Phi_alpha <- Phi %*% alpha_full
  W1 <- as.vector(exp(Phi_alpha))
  W2 <- as.vector(exp(2*Phi_alpha))
  PhiT_W_Phi <- PhiT_full %*% (W2 * Phi)
  A_W_lambda <- PhiT_W_Phi + lambda_Lambda_full
  s <- as.vector(PhiT_full %*% (W1*(y-W1))) - as.vector(lambda_Lambda_full %*% alpha_full)
  alpha_full <- as.vector(alpha_full + solve(A_W_lambda) %*% s)
}

# Individual per-term coefficients are not uniquely identified in an
# unconstrained additive model, so compare fitted values instead
y_hat_mf   <- as.vector(Phi %*% unlist(alpha_terms_mf))
y_hat_full <- as.vector(Phi %*% alpha_full)

rel_err <- sqrt(mean((y_hat_mf - y_hat_full)^2)) / sqrt(mean(y_hat_full^2))
cat("Estimation of alpha correct (fitted values) <= 1% relative error: ",
    rel_err <= 0.01, "\n")

# ------------------------------------------------------------------------------
# Estimate joint df of the full (coupled) additive generalized system

V_rad_terms <- rademacher_matrix_terms(K_terms=unlist(K_terms), M=20, seed=42)
W2 <- as.vector(exp(2 * mvp_Phi_terms(PhiT_terms, alpha_terms)))

df_est <- estimate_df_generalized_terms(
  PhiT_terms = PhiT_terms,
  L_terms    = L_terms,
  W          = W2,
  lambda_vec = lambda_vec,
  V_rad_terms = V_rad_terms,
  pcg_tol    = 1e-10
)

# Full computation for comparison
PhiT_W_Phi_true <- PhiT_full %*% (W2 * Phi)
A_W_lambda_true <- PhiT_W_Phi_true + lambda_Lambda_full
S_lambda <- solve(A_W_lambda_true) %*% PhiT_W_Phi_true
df_ref <- sum(diag(S_lambda))

cat("df estimation <= 5% relative error: ",
    abs(df_est - df_ref) / df_ref <= 0.05, "\n")

# ------------------------------------------------------------------------------
# Estimate lambda_vec for fixed alpha_terms

lambda_vec_est <- estimate_lambda_vec_generalized(
  PhiT_terms = PhiT_terms,
  L_terms    = L_terms,
  alpha_terms = alpha_terms,
  y          = y,
  lambda_vec = lambda_vec,
  V_rad_terms = V_rad_terms,
  pcg_tol    = 1e-10
)

cat("lambda_vec estimate: ", lambda_vec_est, "\n")

# ------------------------------------------------------------------------------
# Estimate alpha_terms and lambda_vec in parallel

result <- fit_pspline_generalized_terms(
  n_iter        = 3,
  n_iter_alpha  = 3,
  y             = y,
  PhiT_terms    = PhiT_terms,
  L_terms       = L_terms,
  lambda_vec    = lambda_vec,
  V_rad_terms   = V_rad_terms,
  pcg_tol       = 1e-4
)

alpha_terms_hat <- result$alpha_terms
y_hat <- exp(mvp_Phi_terms(PhiT_terms = PhiT_terms, alpha_terms = alpha_terms_hat))

cat("RMSE of y and y_hat: ", sqrt(mean((y-y_hat)^2)), "\n")
cat("RRMSE of y and y_hat: ", sqrt(mean((y-y_hat)^2)) / mean(y), "\n")
