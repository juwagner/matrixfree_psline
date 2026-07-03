# ------------------------------------------------------------------------------
# Test suite for high-level P-spline matrix-free operators
# ------------------------------------------------------------------------------

library(microbenchmark)
library(rTensor)

source("src/pspline/pspline_matrices.R")
source("src/pspline_generalized/pspline_operations_generalized.R")

# ------------------------------------------------------------------------------
# Generate test data
# ------------------------------------------------------------------------------

source("tests/pspline_generalized/generate_test_data_generalized.R")

mvp_ref <- function(A, x) {
  as.vector(A %*% x)
}

# ------------------------------------------------------------------------------
# Correctness tests
# ------------------------------------------------------------------------------

cat("======= Correctness Test =======\n")

# mvp_PhiT_W_Phix
mvp_PhiT_W_Phi_mf <- mvp_PhiT_W_Phi(PhiT_list, W2, alpha)
mvp_PhiT_W_Phi_ref <- mvp_ref(PhiT_W_Phi, alpha)
cat("MVP with Phi^T %*% W %*% Phi correct:", 
    all.equal(mvp_PhiT_W_Phi_mf, mvp_PhiT_W_Phi_ref, tol=1e-10), "\n")

# mvp_A_W_lambda
mvp_system_mf <- mvp_A_W_lambda(PhiT_list, L_list, W2, lambda, alpha)
mvp_system_ref <- mvp_ref(A_W_lambda, alpha)
cat("MVP with weigthed system matrix correct:", 
    all.equal(mvp_system_mf, mvp_system_ref, tol=1e-10), "\n")
