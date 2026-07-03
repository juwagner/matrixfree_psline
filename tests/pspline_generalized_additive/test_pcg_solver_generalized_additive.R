# ------------------------------------------------------------------------------
# Test suite for PCG solver + diagonal matrix-free operators for generalized
# additive P-splines
# ------------------------------------------------------------------------------

rm(list=ls())

library(rTensor)
library(Matrix)

source("src/pspline/pspline_matrices.R")
source("src/pspline_generalized_additive/pcg_solver_generalized_additive.R")

# ------------------------------------------------------------------------------
# Generate test data
# ------------------------------------------------------------------------------

source("tests/pspline_generalized_additive/generate_test_data_generalized_additive.R")

# ------------------------------------------------------------------------------
# Correctness tests
# ------------------------------------------------------------------------------

cat("======= Correctness Test =======\n")

# Weighted diagonal preconditioner term diag(Phi_s^T W Phi_s) per term
diag_PhiTWPhi_mf   <- get_diag_PhiTWPhi_terms(PhiT_terms, W2)
diag_PhiTWPhi_full <- lapply(1:n_terms, function(s) {
  PhiT_s <- Reduce(rTensor::khatri_rao, PhiT_terms[[s]])
  diag(PhiT_s %*% (W2 * t(PhiT_s)))
})
cat("Weighted diag(PhiT_s W Phi_s) correct: ",
    all.equal(diag_PhiTWPhi_mf, diag_PhiTWPhi_full, tol=1e-10), "\n")

# PCG solver
b_terms <- lapply(PhiT_terms, function(PhiT_s) {
  as.vector(Reduce(rTensor::khatri_rao, PhiT_s) %*% y)
})

alpha_pcg <- solve_pcg_generalized_terms(
  PhiT_terms = PhiT_terms,
  L_terms = L_terms,
  W = W2,
  lambda_vec = lambda_vec,
  b_terms = b_terms,
  tol = 1e-12,
  verbose = TRUE
)

b <- PhiT_full %*% y
alpha_full <- as.vector(solve(A_W_lambda_full, b))

y_hat_pcg <- Phi %*% unlist(alpha_pcg)
y_hat_full <- Phi %*% alpha_full

cat("PCG solver predictions correct: ",
    all.equal(y_hat_pcg, y_hat_full, tol=1e-8), "\n")
