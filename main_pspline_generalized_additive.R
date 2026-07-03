# ------------------------------------------------------------------------------
# Generalized additive P-spline models for the LUCAS dataset
# ------------------------------------------------------------------------------

rm(list=ls())
gc()

library(Rcpp)
library(tictoc)

sourceCpp("src/base/matrix_free_operations.cpp")
source("src/pspline/pspline_matrices.R")
source("src/pspline_generalized_additive/pspline_operations_generalized_additive.R")
source("src/pspline_generalized_additive/parameter_estimation_generalized_additive.R")

# ------------------------------------------------------------------------------
# load data

source("src/utils/load_lucas_data.R")

X_terms <- list(X, U)

# ------------------------------------------------------------------------------
# Generalized additive P-spline setup

n_terms <- length(X_terms)
P <- sapply(1:n_terms, function(s) dim(X_terms[[s]])[2])
m <- lapply(1:n_terms, function(s) rep(36, P[s]))
q <- lapply(1:n_terms, function(s) rep(3, P[s]))
l <- lapply(1:n_terms, function(s) rep(2, P[s]))
J <- lapply(1:n_terms, function(s) m[[s]]+q[[s]]+1)
K <- sapply(1:n_terms, function(s) prod(J[[s]]))

PhiT_terms <- lapply(1:n_terms, function(s) lapply(1:P[[s]], function(p) {
  build_univarate_bspline_basis_T(X_terms[[s]][,p], m[[s]][p], q[[s]][p])
}))

L_terms <- lapply(1:n_terms, function(s) lapply(1:P[[s]], function(p){
  build_penalty_difference(J[[s]][p], l[[s]][p])
}))

# ------------------------------------------------------------------------------
# Estimate alpha_terms using a fixed lambda_vec

lambda_vec <- c(0.015, 0.2)
n_iter <- 3

tic("Iteration to estimate alpha for generalized additive P-spline (fixed lambda_vec)")

alpha_terms <- estimate_alpha_generalized_terms(
  n_iter=n_iter,
  y=y,
  PhiT_terms=PhiT_terms,
  L_terms=L_terms,
  lambda_vec=lambda_vec,
  pcg_tol=1e-3,
  pcg_verbose=TRUE
)

toc()

# ------------------------------------------------------------------------------
# Solve for alpha_terms and lambda_vec

V_rad_terms <- rademacher_matrix_terms(K_terms=K, M=3, seed=42)
lambda_vec_init <- c(0.176, 0.14)

tic("Iteration for generalized additive P-spline (alpha_terms and lambda_vec)")

result <- fit_pspline_generalized_terms(
    n_iter=2,
    n_iter_alpha=2,
    y=y,
    PhiT_terms=PhiT_terms,
    L_terms=L_terms,
    lambda_vec=lambda_vec_init,
    V_rad_terms=V_rad_terms,
    pcg_tol=1e-2
)

alpha_terms <- result$alpha_terms
lambda_vec <- result$lambda_vec

toc()

# ------------------------------------------------------------------------------
# Validation metrics

y_hat <- exp(mvp_Phi_terms(PhiT_terms, alpha_terms))

res <- y - y_hat
RSS <- sum(res^2)

W2 <- as.vector(exp(2*mvp_Phi_terms(PhiT_terms, alpha_terms)))
df <- estimate_df_generalized_terms(
  PhiT_terms=PhiT_terms, L_terms=L_terms, W=W2, lambda_vec=lambda_vec,
  V_rad_terms=V_rad_terms, pcg_tol=1e-3
)
AIC <- 2*n*log(RSS) + 2*df

cat(
  "Generalized Additive P-Spline Model Validation | ",
  "RSS:", RSS,
  "DF:", df,
  "AIC:", AIC,
  "Min fitted:", min(y_hat),
  "\n"
)
