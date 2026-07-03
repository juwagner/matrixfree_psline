# ------------------------------------------------------------------------------
# P-spline models for the LUCAS dataset
# ------------------------------------------------------------------------------

rm(list=ls())

library(Rcpp)
library(tictoc)

sourceCpp("src/base/matrix_free_operations.cpp")
source("src/pspline/pspline_matrices.R")
source("src/pspline_generalized/pspline_operations_generalized.R")
source("src/pspline_generalized/parameter_estimation_generalized.R")

# ------------------------------------------------------------------------------
# load data
source("src/utils/load_lucas_data.R")

X_input <- X           # U

# ------------------------------------------------------------------------------
# P-spline setup

P     <- ncol(X_input)        # dimension of tensor product
m     <- rep(36, P)           # interior knots per dimension
q     <- rep(3,  P)           # spline degree per dimension
l     <- rep(2,  P)           # difference penalty order
J_vec <- m + q + 1            # basis sizes
K     <- prod(J_vec)          # total number of coefficients


PhiT_list <- lapply(
  1:P, function(p) build_univarate_bspline_basis_T(X_input[,p], m[p], q[p])
)

L_list <- lapply(1:P, function(p) build_penalty_difference(J=J_vec[p], l=l[p]))

# ------------------------------------------------------------------------------
# Estimate α using a fixed λ

lambda <- 0.01
n_iter <- 3

tic("Iteration to estimate alpha for generalized P-spline (fixed λ)")

alpha <- estimate_alpha_generalized(
  n_iter=n_iter,
  y=y,
  PhiT_list=PhiT_list,
  L_list=L_list,
  lambda=lambda,
  pcg_tol=1e-3,
  pcg_verbose=TRUE
)

toc()

# ------------------------------------------------------------------------------
# Solve for α and λ

V_rad <- rademacher_matrix(K, M=3, seed=42)
lambda_init <- 0.1

tic("Iteration for generalized P-spline (α and λ)")

result <- fit_pspline_generalized(
    n_iter=2,
    n_iter_alpha=2,
    y=y,
    PhiT_list=PhiT_list,
    L_list=L_list,
    lambda=lambda_init,
    V_rad=V_rad,
    pcg_tol=1e-2
)

alpha <- result$alpha
lambda <- result$lambda

toc()

# ------------------------------------------------------------------------------
# Validation metrics

y_hat <- exp(mvp_Phi(PhiT_list, alpha))

res <- y - y_hat
RSS <- sum(res^2)

W2 <- as.vector(exp(2*mvp_Phi(PhiT_list, alpha)))
df <- estimate_df_generalized(
  PhiT_list=PhiT_list, L_list=L_list, W=W2, lambda=lambda, V_rad=V_rad, pcg_tol=1e-3
)
AIC <- 2*n*log(RSS) + 2*df

cat(
  "Generalized P-Spline Model Validation | ",
  "RSS:", RSS,
  "DF:", df,
  "AIC:", AIC,
  "Min fitted:", min(y_hat),
  "\n"
)
