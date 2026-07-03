# ------------------------------------------------------------------------------
# Diagonal-preconditioned Conjugate Gradient (PCG) solver for generalized P-splines
# ------------------------------------------------------------------------------

source("src/pspline/pcg_solver.R")
source("src/pspline_generalized/pspline_operations_generalized.R")

# ------------------------------------------------------------------------------
# Compute diag(Φᵀ W Φ) matrix-free, W a per-observation weight vector
# PhiT_list List of transposed marginal B-spline bases Φ_pᵀ.
get_diag_PhiTWPhi <- function(PhiT_list, W) {
  diag_gram_khatrirao_weighted(A_list = PhiT_list, W = W)
}

# ------------------------------------------------------------------------------
# Solve (Φᵀ W Φ + λΛ) α = b using diagonal-preconditioned CG
solve_pcg_generalized = function(
    PhiT_list, 
    L_list,
    W,
    lambda, 
    b,
    alpha_init=NULL, 
    it_max=length(b),
    tol=10^(-4), 
    verbose=FALSE
){
  
  P <- length(L_list)
  J_vec <- sapply(1:P, function(p) dim(L_list[[p]])[2] )
  K <- prod(J_vec)
  
  diag_PhiTWPhi <- get_diag_PhiTWPhi(PhiT_list = PhiT_list, W = W)
  diag_Lambda <- get_diag_Lambda(L_list = L_list)
  preconditioner <- 1 / (diag_PhiTWPhi + lambda*diag_Lambda)
  
  norm_b <- sqrt(drop(crossprod(b)))
  
  if(is.null(alpha_init)){
    r <- b
    alpha <- rep(0,K)
  } else{
    alpha <- alpha_init
    r <- b - mvp_A_W_lambda(PhiT_list, L_list, W, lambda, alpha)
  }
  z <- preconditioner*r
  d <- z
  rz <- as.numeric( crossprod(r,z) )
  
  for(k in 1:it_max){
    Ad <- mvp_A_W_lambda(PhiT_list, L_list, W, lambda, d)
    step_len <- as.numeric(rz / crossprod(d,Ad))
    alpha <- alpha + step_len*d
    r <- r - step_len*Ad
    z <- preconditioner*r
    rz_old <- rz
    rz <- as.numeric(crossprod(r,z))
    
    relres <- sqrt(drop(crossprod(r))) / norm_b
    if(verbose == TRUE) {
      cat("PCG iteration: ", k, " relres: ", relres , "\n" )
    }
      
    if(relres < tol){
      break
    }
    
    beta <- rz / rz_old
    d <- z + beta*d
    
  }
  
  return(as.vector(alpha))
  
}
