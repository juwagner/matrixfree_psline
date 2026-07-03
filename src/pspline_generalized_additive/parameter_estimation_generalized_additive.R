# ------------------------------------------------------------------------------
# Estimation of α_terms and λ_vec for generalized additive P-splines
# ------------------------------------------------------------------------------

source("src/utils/rademacher.R")
source("src/pspline_generalized/parameter_estimation_generalized.R")
source("src/pspline_generalized_additive/pspline_operations_generalized_additive.R")
source("src/pspline_generalized_additive/pcg_solver_generalized_additive.R")

# ------------------------------------------------------------------------------
# Iteration to estimate α_terms (with fixed λ_vec) in generalized additive
# p-spline model
estimate_alpha_generalized_terms <- function(
    n_iter,
    y,
    PhiT_terms,
    L_terms,
    lambda_vec,
    alpha_init = NULL,
    pcg_tol = 1e-4,
    pcg_verbose = FALSE
){
  n_terms <- length(PhiT_terms)
  K_terms <- vapply(
    L_terms, function(Ls) prod(vapply(Ls, ncol, numeric(1))), numeric(1)
  )

  if(is.null(alpha_init)){
    alpha_terms <- lapply(seq_len(n_terms), function(s) rep(0, K_terms[s]))
  } else{
    alpha_terms <- alpha_init
  }

  for(i in 1:n_iter){
    cat("Iteration for alpha: ", i, "/", n_iter, "\n")
    Phi_alpha <- mvp_Phi_terms(PhiT_terms, alpha_terms)
    W1 <- as.vector(exp(Phi_alpha))
    W2 <- as.vector(exp(2*Phi_alpha))

    score_terms <- mvp_PhiT_terms(PhiT_terms, W1*(y-W1))
    penalty_terms <- mvp_lambda_Lambda_terms(L_terms, lambda_vec, alpha_terms)
    rhs_terms <- lapply(
      seq_len(n_terms), function(s) score_terms[[s]] - penalty_terms[[s]]
    )

    v_terms <- solve_pcg_generalized_terms(
      PhiT_terms = PhiT_terms,
      L_terms = L_terms,
      W = W2,
      lambda_vec = lambda_vec,
      b_terms = rhs_terms,
      alpha_init = alpha_terms,
      verbose = pcg_verbose,
      tol = pcg_tol
    )

    alpha_terms_new <- lapply(
      seq_len(n_terms), function(s) alpha_terms[[s]] + v_terms[[s]]
    )
    rel <- mean((unlist(alpha_terms) - unlist(alpha_terms_new))^2)
    cat("Relative change of alpha: ", rel, "\n")
    alpha_terms <- alpha_terms_new
  }

  cat("Solved for alpha \n")
  return(alpha_terms)

}

# ------------------------------------------------------------------------------
# Estimate df(S_λ_s) = trace((A_s)^{-1} Φ_sᵀ W Φ_s) per term, treating each
# term in isolation (ignores cross-term correlation), reusing estimate_df_generalized
estimate_df_generalized_terms_marginal <- function(
    PhiT_terms, L_terms, W, lambda_vec, V_rad_terms, pcg_tol = 1e-4, pcg_verbose = FALSE
) {
  n_terms <- length(PhiT_terms)
  df_terms <- lapply(
    1:n_terms,
    function(s) estimate_df_generalized(
      PhiT_list = PhiT_terms[[s]],
      L_list = L_terms[[s]],
      W = W,
      lambda = lambda_vec[s],
      V_rad = V_rad_terms[[s]],
      pcg_tol = pcg_tol,
      pcg_verbose = pcg_verbose
    )
  )
  return(df_terms)
}

# ------------------------------------------------------------------------------
# Estimate df(λ) = trace(S_λ), with S_λ = A(λ)^{-1} Φᵀ W Φ, for the full
# (coupled) additive generalized system
estimate_df_generalized_terms <- function(
    PhiT_terms, L_terms, W, lambda_vec, V_rad_terms, pcg_tol = 1e-4, pcg_verbose = FALSE
) {
  n_terms <- length(PhiT_terms)
  K_terms <- vapply(V_rad_terms, function(Ms) nrow(Ms), numeric(1))
  M <- ncol(V_rad_terms[[1]])
  trace_terms <- numeric(M)
  for (m in seq_len(M)) {
    v_terms <- lapply(seq_len(n_terms), function(s) V_rad_terms[[s]][, m])
    w_terms <- mvp_lambda_Lambda_terms(L_terms, lambda_vec, v_terms)
    u_terms <- solve_pcg_generalized_terms(
      PhiT_terms = PhiT_terms,
      L_terms = L_terms,
      W = W,
      lambda_vec = lambda_vec,
      b_terms = w_terms,
      tol = pcg_tol,
      verbose = pcg_verbose
    )
    trace_terms[m] <- sum(unlist(v_terms) * unlist(u_terms))
  }
  df_total <- sum(K_terms) - mean(trace_terms)
  return(as.numeric(df_total))
}

# ------------------------------------------------------------------------------
# Iteration to estimate λ_vec (for fixed α_terms) in generalized additive
# p-spline model
estimate_lambda_vec_generalized <- function(
    PhiT_terms,
    L_terms,
    alpha_terms,
    y,
    lambda_vec = rep(0.1, length(PhiT_terms)),
    V_rad_terms,
    pcg_tol = 1e-4,
    pcg_verbose = FALSE
){
  n_terms <- length(PhiT_terms)
  Phi_alpha <- mvp_Phi_terms(PhiT_terms, alpha_terms)
  W1 <- as.vector(exp(Phi_alpha))
  W2 <- as.vector(exp(2*Phi_alpha))

  sigma_eps <- mean((y-W1)^2)

  df_terms <- estimate_df_generalized_terms_marginal(
    PhiT_terms = PhiT_terms,
    L_terms = L_terms,
    W = W2,
    lambda_vec = lambda_vec,
    V_rad_terms = V_rad_terms,
    pcg_tol = pcg_tol,
    pcg_verbose = pcg_verbose
  )

  sigma_alpha_terms <- vapply(
    seq_len(n_terms),
    function(s) {
      drop(crossprod(
        alpha_terms[[s]], mvp_Lambda(L_terms[[s]], alpha_terms[[s]])
      )) / df_terms[[s]]
    },
    numeric(1)
  )

  lambda_vec_new <- as.numeric(sigma_eps / sigma_alpha_terms)

  cat("Estimated lambda_vec \n")
  return(lambda_vec_new)
}


# ------------------------------------------------------------------------------
# Iteration to estimate α_terms and λ_vec in parallel in generalized additive
# p-spline model
fit_pspline_generalized_terms <- function(
    n_iter=2,
    n_iter_alpha=2,
    y,
    PhiT_terms,
    L_terms,
    lambda_vec,
    V_rad_terms,
    pcg_tol=1e-2
    ){

  alpha_terms <- NULL

  for (i in 1:n_iter) {
    cat("--- Outer iteration: ", i ,"/", n_iter, "\n")
    alpha_terms <- estimate_alpha_generalized_terms(
      n_iter=n_iter_alpha,
      y=y,
      PhiT_terms=PhiT_terms,
      L_terms=L_terms,
      lambda_vec=lambda_vec,
      alpha_init=alpha_terms,
      pcg_tol=pcg_tol
    )

    if(i != n_iter){
      lambda_vec <- estimate_lambda_vec_generalized(
        PhiT_terms=PhiT_terms,
        L_terms=L_terms,
        alpha_terms=alpha_terms,
        y=y,
        lambda_vec=lambda_vec,
        V_rad_terms=V_rad_terms,
        pcg_tol=pcg_tol
      )
      cat("Current lambda_vec: ", lambda_vec, "\n")
    }
  }

  return(list(
    lambda_vec = lambda_vec,
    alpha_terms = alpha_terms
  ))

}
