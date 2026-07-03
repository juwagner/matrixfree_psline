# Function reference

Map of what each function computes and where it lives.
Kept intentionally short.

## Module structure

Four parallel model variants, each following the same three-file layout
(`pspline_operations*`, `pcg_solver*`, `parameter_estimation*`):

| Module | Response model | Coefficient structure |
|---|---|---|
| `src/pspline/` | Gaussian, identity link: `y ≈ Φα + ε` | single coefficient vector `α` |
| `src/pspline_additive/` | Gaussian, identity link, additive: `y ≈ Σₛ Φₛαₛ + ε` | list of per-term vectors `α_terms` |
| `src/pspline_generalized/` | Gaussian, log link (Gauss-Newton): `y ≈ exp(Φα) + ε` | single coefficient vector `α` |
| `src/pspline_generalized_additive/` | Gaussian, log link, additive: `y ≈ exp(Σₛ Φₛαₛ) + ε` | list of per-term vectors `α_terms` |

`_terms`/`_additive` suffixes mark the multi-term (additive) variant;
`_generalized` marks the log-link/Gauss-Newton variant. Function names
combine both, e.g. `estimate_df_generalized_terms_marginal`.

`Φ = A_1 ⊙ ... ⊙ A_P` (Khatri-Rao/tensor-product of marginal B-spline
bases) throughout. `PhiT_list`/`PhiT_terms` always store the *transposed*
marginal bases (`Φ_pᵀ`, shape `J_p × n`), never `Φ` itself.

---

## `src/base/matrix_free_operations.cpp` — low-level Khatri-Rao kernels

| Function | Computes |
|---|---|
| `mvp_normalfactor(A, left, right, x)` | `(I_left ⊗ A ⊗ I_right) x` |
| `mvp_khatrirao(A_list, x)` | `(A_1 ⊙ ... ⊙ A_P) x` — used as `mvp_PhiT` (`A_list = PhiT_list`) |
| `mvp_transposed_khatrirao(A_list, x)` | `(A_1 ⊙ ... ⊙ A_P)ᵀ x` — used as `mvp_Phi` |
| `mvp_gram_khatrirao(A_list, x)` | `(A Aᵀ) x`, i.e. `ΦᵀΦ x` |
| `diag_gram_khatrirao(A_list)` | `diag(ΦᵀΦ)` |
| `diag_gram_khatrirao_weighted(A_list, W)` | `diag(Φᵀ diag(W) Φ)`, `W` a per-observation weight vector |

## `src/pspline/` — Gaussian, single-term

| Function | File | Computes |
|---|---|---|
| `build_univarate_bspline_basis(x, m, q, Omega)` | `pspline_matrices.R` | marginal B-spline basis `Φ_p` (`n × J_p`), truncated power form |
| `build_univarate_bspline_basis_T(...)` | `pspline_matrices.R` | `Φ_pᵀ` (what's stored in `PhiT_list`) |
| `build_penalty_difference(J, l)` | `pspline_matrices.R` | order-`l` difference penalty `L_p = DᵀD` (`J × J`) |
| `mvp_Phi(PhiT_list, x)` | `pspline_operations.R` | `Φx` |
| `mvp_PhiT(PhiT_list, x)` | `pspline_operations.R` | `Φᵀx` |
| `mvp_PhiTPhi(PhiT_list, x)` | `pspline_operations.R` | `ΦᵀΦx` |
| `mvp_Lambda(L_list, x)` | `pspline_operations.R` | `Λx`, `Λ = Σ_p I ⊗ L_p ⊗ I` |
| `mvp_A_lambda(PhiT_list, L_list, lambda, x)` | `pspline_operations.R` | `(ΦᵀΦ + λΛ)x` |
| `get_diag_Lambda(L_list)` | `pcg_solver.R` | `diag(Λ)` |
| `get_diag_PhiTPhi(PhiT_list)` | `pcg_solver.R` | `diag(ΦᵀΦ)` |
| `solve_pcg(...)` | `pcg_solver.R` | solves `(ΦᵀΦ + λΛ)α = b` via diagonally-preconditioned CG |
| `estimate_df(...)` | `parameter_estimation.R` | `df(λ) = trace(S_λ) = K − trace(A_λ⁻¹λΛ)` via Hutchinson |
| `estimate_lambda(...)` | `parameter_estimation.R` | moment-based iteration estimating `α` and `λ` jointly |

## `src/pspline_additive/` — Gaussian, additive (`Φ = Σₛ Φₛ`)

| Function | File | Computes |
|---|---|---|
| `mvp_Phi_terms(PhiT_terms, alpha_terms)` | `pspline_operations_additive.R` | `Σₛ Φₛαₛ` |
| `mvp_PhiT_terms(PhiT_terms, x)` | `pspline_operations_additive.R` | `[Φₛᵀx]ₛ` per term |
| `mvp_PhiTPhi_terms(...)` | `pspline_operations_additive.R` | `[Φₛᵀ(Σₜ Φₜαₜ)]ₛ` per term |
| `mvp_Lambda_terms(L_terms, alpha_terms)` | `pspline_operations_additive.R` | `[Λₛαₛ]ₛ` per term |
| `mvp_lambda_Lambda_terms(L_terms, lambda_vec, alpha_terms)` | `pspline_operations_additive.R` | `[λₛΛₛαₛ]ₛ` per term |
| `mvp_A_lambda_terms(...)` | `pspline_operations_additive.R` | `[Φₛᵀ(ΣₜΦₜαₜ) + λₛΛₛαₛ]ₛ`, i.e. `(ΦᵀΦ + Λ(λ))α` per term |
| `solve_pcg_terms(...)` | `pcg_solver_additive.R` | solves `(ΦᵀΦ + Λ(λ))α = b` via per-term diagonally-preconditioned CG |
| `estimate_df_terms_marginal(...)` | `parameter_estimation_additive.R` | per-term `df` treating each term in isolation (ignores cross-term coupling) — used inside the `λ_vec` update loop |
| `estimate_df_terms(...)` | `parameter_estimation_additive.R` | joint `df` of the full coupled system — used for reporting |
| `estimate_lambda_terms(...)` | `parameter_estimation_additive.R` | moment-based iteration estimating `α_terms` and `λ_vec` jointly |

## `src/pspline_generalized/` — Gauss-Newton, single-term (`y ≈ exp(Φα) + ε`)

| Function | File | Computes |
|---|---|---|
| `mvp_PhiT_W_Phi(PhiT_list, W, x)` | `pspline_operations_generalized.R` | `Φᵀ diag(W) Φ x` |
| `mvp_A_W_lambda(...)` | `pspline_operations_generalized.R` | `(Φᵀ diag(W) Φ + λΛ)x` |
| `get_diag_PhiTWPhi(PhiT_list, W)` | `pcg_solver_generalized.R` | `diag(Φᵀ diag(W) Φ)` |
| `solve_pcg_generalized(...)` | `pcg_solver_generalized.R` | solves `(Φᵀ diag(W) Φ + λΛ)α = b` via diagonally-preconditioned CG, weighted by `W` |
| `estimate_alpha_generalized(...)` | `parameter_estimation_generalized.R` | Gauss-Newton iteration for `α` at fixed `λ` (`W1=exp(Φα)`, `W2=exp(2Φα)` per step) |
| `estimate_trace_generalized(...)` | `parameter_estimation_generalized.R` | raw Hutchinson estimate of `trace(A⁻¹λΛ)` (building block, **not** `df`) |
| `estimate_df_generalized(...)` | `parameter_estimation_generalized.R` | `df(λ) = K − trace(A⁻¹λΛ)`, wraps `estimate_trace_generalized` |
| `estimate_lambda_generalized(...)` | `parameter_estimation_generalized.R` | one moment-based `λ` update at fixed `α` |
| `fit_pspline_generalized(...)` | `parameter_estimation_generalized.R` | outer loop alternating `estimate_alpha_generalized` / `estimate_lambda_generalized` |

## `src/pspline_generalized_additive/` — Gauss-Newton, additive (`y ≈ exp(Σₛ Φₛαₛ) + ε`)

| Function | File | Computes |
|---|---|---|
| `mvp_PhiT_W_Phi_terms(PhiT_terms, W, alpha_terms)` | `pspline_operations_generalized_additive.R` | `[Φₛᵀ diag(W) (Σₜ Φₜαₜ)]ₛ` per term |
| `mvp_A_W_lambda_terms(...)` | `pspline_operations_generalized_additive.R` | `(Φᵀ diag(W) Φ + Λ(λ))α` per term |
| `get_diag_PhiTWPhi_terms(PhiT_terms, W)` | `pcg_solver_generalized_additive.R` | `[diag(Φₛᵀ diag(W) Φₛ)]ₛ` per term |
| `solve_pcg_generalized_terms(...)` | `pcg_solver_generalized_additive.R` | solves `(Φᵀ diag(W) Φ + Λ(λ))α = b` via per-term diagonally-preconditioned CG |
| `estimate_alpha_generalized_terms(...)` | `parameter_estimation_generalized_additive.R` | Gauss-Newton iteration for `α_terms` at fixed `λ_vec` |
| `estimate_df_generalized_terms_marginal(...)` | `parameter_estimation_generalized_additive.R` | per-term `df`, each term in isolation (reuses `estimate_df_generalized`) — used inside the `λ_vec` update loop |
| `estimate_df_generalized_terms(...)` | `parameter_estimation_generalized_additive.R` | joint `df` of the full coupled system — used for reporting |
| `estimate_lambda_vec_generalized(...)` | `parameter_estimation_generalized_additive.R` | one moment-based `λ_vec` update at fixed `α_terms` |
| `fit_pspline_generalized_terms(...)` | `parameter_estimation_generalized_additive.R` | outer loop alternating `estimate_alpha_generalized_terms` / `estimate_lambda_vec_generalized` |

## `src/utils/`

| Function | File | Purpose |
|---|---|---|
| `rademacher_matrix(K, M, seed)` | `rademacher.R` | `K × M` matrix of ±1 draws, for Hutchinson trace estimation |
| `rademacher_matrix_terms(K_terms, M, seed)` | `rademacher.R` | per-term version of the above (list of matrices) |
| `load_lucas_data.R` | — | loads `data/LUCAS_agg.Rdata`, builds `X` (spectral bands `B2/B3/B4`), `U` (lat/long), `y` (organic carbon, range-normalized to `[0,1]`) |
| `prepare_lucas_data.R` | — | one-time script building `data/LUCAS_agg.Rdata` from the raw LUCAS source files |

## Top-level `main_*.R` scripts

Example/demo scripts applying each model to the LUCAS dataset:
`main_lm.R` (plain linear model baseline), `main_pspline.R`,
`main_pspline_additive.R`, `main_pspline_generalized.R`,
`main_pspline_generalized_additive.R`. Each ends with a validation block
printing `RSS`, `DF`, `AIC`, and the minimum fitted value.

## Known limitations (discovered empirically, not yet fixed in code)

- **Undamped Gauss-Newton**: `estimate_alpha_generalized(_terms)` takes full
  Newton steps with no step-halving/line-search. On heavily right-skewed
  targets with values near zero (e.g. `OC` in the LUCAS data), this can
  diverge over successive outer iterations rather than converge.
- **Additive term identifiability**: without a centering/sum-to-zero
  constraint, individual `alpha_terms[[s]]` are not unique in the additive
  models (only `Σₛ Φₛαₛ` is identified) — compare fitted values, not raw
  coefficients, when checking correctness.
- **Per-term diagonal preconditioner**: `solve_pcg_terms` /
  `solve_pcg_generalized_terms` precondition each term independently,
  ignoring cross-term coupling (`Φₛᵀ·Φₜ` for `s ≠ t`). Fine for weakly
  correlated terms; may need more PCG iterations when terms are strongly
  correlated.
