# ------------------------------------------------------------------------------
# Test suite for PCG solver + diagonal matrix-free operators for P-splines
# ------------------------------------------------------------------------------

library(microbenchmark)

source("src/pspline/pspline_matrices.R")
source("src/pspline_generalized/pcg_solver_generalized.R")

# ------------------------------------------------------------------------------
# Generate test data
# ------------------------------------------------------------------------------

source("tests/pspline_generalized/generate_test_data_generalized.R")

# ------------------------------------------------------------------------------
# Correctness tests
# ------------------------------------------------------------------------------

cat("======= Correctness Test =======\n")

# Weighted diagonal preconditioner term diag(Φᵀ W Φ)
diag_PhiTWPhi_mf   <- get_diag_PhiTWPhi(PhiT_list, W2)
diag_PhiTWPhi_full <- diag(PhiT %*% (W2 * Phi))
cat("Weighted diag(PhiT W Phi) correct: ",
    all.equal(diag_PhiTWPhi_mf, diag_PhiTWPhi_full, tol=1e-10), "\n")

# PCG solver
b <- as.vector(PhiT %*% y)
alpha_pcg  <- solve_pcg_generalized(PhiT_list, L_list, W2, lambda, b, tol = 1e-12, verbose=TRUE)
alpha_full <- solve(A_W_lambda, b)
cat("PCG solver correct: ", 
    all.equal(alpha_pcg, alpha_full, tol=1e-10),  "\n")
