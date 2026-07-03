# ------------------------------------------------------------------------------
# Test suite for high-level generalized additive P-spline matrix-free operators
# ------------------------------------------------------------------------------

rm(list=ls())

library(rTensor)
library(Matrix)

source("src/pspline/pspline_matrices.R")
source("src/pspline_generalized_additive/pspline_operations_generalized_additive.R")

# ------------------------------------------------------------------------------
# Generate test data
# ------------------------------------------------------------------------------

source("tests/pspline_generalized_additive/generate_test_data_generalized_additive.R")

mvp_ref <- function(A, x) {
  as.vector(A %*% x)
}

# ------------------------------------------------------------------------------
# Correctness tests
# ------------------------------------------------------------------------------

cat("======= Correctness Test =======\n")

# mvp_PhiT_W_Phi_terms
mvp_PhiT_W_Phi_mf <- unlist(mvp_PhiT_W_Phi_terms(PhiT_terms, W2, alpha_terms))
mvp_PhiT_W_Phi_ref <- mvp_ref(PhiT_W_Phi_full, alpha)
cat("MVP with Phi^T %*% W %*% Phi correct:",
    all.equal(mvp_PhiT_W_Phi_mf, mvp_PhiT_W_Phi_ref, tol=1e-10), "\n")

# mvp_A_W_lambda_terms
mvp_A_W_lambda_mf <- unlist(
  mvp_A_W_lambda_terms(PhiT_terms, L_terms, W2, lambda_vec, alpha_terms)
)
mvp_A_W_lambda_ref <- mvp_ref(A_W_lambda_full, alpha)
cat("MVP with weighted system matrix correct:",
    all.equal(mvp_A_W_lambda_mf, mvp_A_W_lambda_ref, tol=1e-10), "\n")
