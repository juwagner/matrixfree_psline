# ------------------------------------------------------------------------------
# Generate data for generalized P-spline tests
# ------------------------------------------------------------------------------

set.seed(42)

# P-Spline setup
P  <- 3                         # Dimension of tensor-product spline
m  <- c(15, 17, 21)             # interior knots per dimension
q  <- c(3, 2, 3)                # spline degrees
l  <- c(2, 2, 2)                # penalty difference orders
J_vec  <- m + q + 1                 # basis sizes per dimension
K <- prod(J_vec) 
n  <- 1000                      # number of data points

# Random input data
X <- matrix(runif(n * P), nrow = n, ncol = P)
y <- as.vector(exp(sin(2*pi*X[,1]*X[,2])*cos(2*pi*X[,3])) + rnorm(n, sd=0.1))
alpha <- rnorm(K)

# P-Spline bases and penalties
PhiT_list <- lapply(1:P, function(p) {
  build_univarate_bspline_basis_T(X[,p], m[p], q[p])
})

L_list <- lapply(1:P, function(p){
  build_penalty_difference(J_vec[p], l[p])
})

Phi_alpha <- mvp_Phi(PhiT_list, alpha)
W1 <- exp(Phi_alpha)
W2 <- exp(2*mvp_Phi(PhiT_list, alpha))

# Full matrices for reference
PhiT <- Reduce(rTensor::khatri_rao, PhiT_list)
Phi <- t(PhiT)
PhiT_W_Phi <- PhiT %*% diag(W2) %*% Phi

Lambda <- Reduce(`+`, lapply(1:P, function(p) {
  left  <- if(p>1) diag(prod(J_vec[1:(p-1)])) else 1
  right <- if(p<P) diag(prod(J_vec[(p+1):P])) else 1
  kronecker(left, kronecker(L_list[[p]], right))
}))

lambda <- 0.1
A_W_lambda <- PhiT_W_Phi + lambda*Lambda