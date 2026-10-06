# pembed — design

Projection-based WF-in-DFT embedding on PySCF, spin-general from the first line.
Written for a study of anion redox in Li₂MnO₃.

## Why this exists

nbed 0.0.10 cannot represent n_α ≠ n_β. Its per-spin data are dense `(2, n)` arrays,
and its driver branches on `.ndim`. It also had a spin-summed-density bug in the
nonadditive xc energy (described in the paper's SI, section S2). Mn(IV) d³ needs n_α ≠ n_β, so we
need our own code.

## Core rules

1. **Per-spin data are tuples, never stacked arrays.** A `Spin[T]` is `(alpha, beta)`.
   Occupied counts, active counts and kept-orbital counts may all differ per spin.
2. **Unrestricted-native.** The global reference is always handled as UKS. A restricted
   input (RKS) is converted, so restricted is simply the α = β case. There is one code
   path.
3. **Inherit every environment effect from the global SCF object.** `h` and `E_nuc`
   come from `ks.get_hcore()` / `ks.energy_nuc()`, so QM/MM point charges (if the
   global object is `qmmm`-wrapped) and any charge or spin propagate automatically.
4. **Never trust a converged flag alone.** Every stage reports ⟨S²⟩, the embedded-SCF
   convergence, the environment-orbital projection weights, the FCI convergence, and
   the DFT-in-DFT exactness residual.

## Algorithm (Manby 2012; Monino et al. JCTC 22, 1780 (2026), eqs. 1, 7, 18–21)

Per spin σ, with D^σ the global occupied density (occupation 1):

1. **SPADE.** Take the SVD of the active-atom AO rows of S^½ C_occ^σ. The partition is
   the largest singular-value gap, or n_act^σ forced. C_loc^σ = C_occ^σ V.
2. D_A^σ = C_A C_Aᵀ, D_B^σ = C_B C_Bᵀ (D_A + D_B = D).
3. v_emb^σ = G^σ[D] − G^σ[D_A], with G from `ks.get_veff(dm=(Dα, Dβ))`. That gives J
   from the total density, plus a·K^σ (range-separated where applicable) and V_xc^σ.
   Monino eq. 7.
4. Projector:
   - μ-shift: P^σ = μ S D_B^σ S
   - Huzinaga: P^σ = −(F^σ D_B^σ S + S D_B^σ F^σ), rebuilt every SCF cycle
5. Embedded SCF (UHF for the WF path, UKS for DFT-in-DFT). It uses n_A^σ electrons and
   h_emb^σ = h + v_emb^σ (+ P^σ for μ), and is **seeded with D_A** (the nbed lesson).
6. Delete the environment orbitals per spin: the n_env^σ embedded MOs with the largest
   weight w_i = c_iᵀ S D_B S c_i. The weights are reported so the gap can be checked.
7. WF solver on the kept orbitals. FCI uses `direct_spin1` when α ≡ β and `direct_uhf`
   otherwise; DMRG goes via FCIDUMP. Outputs are E_WF^emb and γ_emb^σ in the AO basis.
8. Energies:
   - E_WF (plain h) = E_WF^emb − Σ_σ tr[γ_emb^σ (v_emb^σ + P^σ)]
   - **Eq. 1:** E = E_WF + E_DFT[D] − E_DFT[D_A] + Σ_σ tr[(γ_emb^σ − D_A^σ) v_emb^σ] + E_nuc
   - **Eq. 18:** E = E_WF + E_DFT[D_B] + E_H^nadd[D_A, D_B] + E_xc^nadd[D_A, D_B] + E_nuc,
     which equals eq. 1 minus its trace term, identically.
   - E_H^nadd = tr[ρ_A J(ρ_B)]
   - E_xc^nadd = E_xc[D] − E_xc[D_A] − E_xc[D_B]
   - DFT-in-DFT is the same expression with E_WF replaced by E_DFT[D̃_A]. It must equal
     the global energy. **This is a theorem**, so it's the primary exactness test.

## Knobs

- environment functional (from the global SCF)
- active atoms (any indices, not only the first N)
- n_act per spin (auto or forced)
- projector: `mu` / `huzinaga`
- μ
- charge (negative allowed) and spin (from the `Mole`)
- point charges (via a `qmmm`-wrapped global SCF)
- embedded-SCF `dm0` (default D_A)
- SCF macro-iterations, Δx and concentric localization: planned (A4)

## Outputs (`EmbeddingResult`)

- Energy components:
  - E_eq1, E_eq18, E_WF^emb, E_WF (plain), E_DFT[D], E_DFT[D_A], E_DFT[D_B]
  - E_H^nadd, E_xc^nadd, the trace term, E_nuc
- Partition: n_act, n_env and n_kept per spin; singular values; environment weights
- Diagnostics: ⟨S²⟩ (global, embedded SCF, FCI); convergence flags
- Correlated active 1-RDM in the AO basis, per spin
- FCIDUMP export: restricted (PySCF format); unrestricted is planned via block2

## Tests (tests/, frozen data in tests/data/)

| id | criterion |
|---|---|
| R1 | DFT-in-DFT = global DFT (water, all geometries, B3LYP5/PBE0/HF, μ and Huzinaga) ≤ 1 µHa |
| R2 | Table V FCI-in-DFT vs frozen patched-nbed ≤ 5 µHa (≤ 1 mHa at 4–5 Å) |
| R3 | Table VI likewise |
| R4 | all atoms active: FCI-in-DFT = full FCI ≤ 1 µHa |
| R5 | FCIDUMP round trip ≤ 1 µHa |
| R6 | eq. 1 (without trace) = eq. 18 ≤ 1 µHa |

## Regression notes (verified numerically, 2026-09-25)

- **MP2 natural-orbital truncation (`no_select`, tests Z9–Z10, 2026-09-29).** At cluster scale
  the concentric-localization shells of a Mn-containing region are far larger than a DMRG can
  hold: R2 has 101 / 191 orbitals for n = 0 / 1, and R3 has 137 at n = 0 (dry runs, cfg04 3.6 Å).
  `no_select=dict(lo, hi)` therefore runs UMP2 on the embedded UHF, with the environment and the
  frozen core frozen, and keeps the per-spin natural orbitals of fractional occupation.
  - **Frozen core.** NOs with occupation ≥ hi join a per-spin frozen core (H["C_core"],
    H["ncore"] = (nα, nβ)).
  - **Exactness (Z9).** Keeping every NO reproduces the untruncated FCI.
  - **Truncation (Z10).** Truncation loses dynamic correlation, 15 mEh at lo = 2e-3 on OH/6-31G,
    monotonically and variationally.

- **Per-spin site equalization is dummy padding, not trimming (test Z8, 2026-09-28).**
  - The larger spin's extra kept orbital is a genuine correlating virtual: the environment
    Mn-d that is empty for that spin. Trimming it changes the energy.
  - The smaller spin is instead padded with decoupled dummy sites (zero orbital, zero
    two-electron integrals, h1 = 10³ Eh), which is exactly invariant. It agrees with padding by
    the μ-shifted environment orbital to ≤ 1 µEh.
  - `run_dmrg(verify_starts=True)` carries every start through the second bond dimension and
    flags a spread > 0.1 mEh (`start_disagree`). Both start energies are logged for the SI.

- **Range-separated Δx (plan item 8, tests E0d–E0g, 2026-09-28).** For any functional with a
  separable correlation part, E_x^DFT = [semilocal xc − E_c] + the functional's own exact
  exchange, K_eff = hyb·K + (α − hyb)·K_lr(ω) from `numint.rsh_and_hybrid_coeff`. For HSE06 that
  is 0.25 K_sr. One routine covers HSE06, r2SCAN, SCAN and the global hybrids; it agrees with the
  explicit split to 1e-9. It is not available for ωB97X, whose B97 correlation is not separable
  in libxc.
  **DFT+U convention (methods):** U stays at the DFT level, and only exchange is corrected.
  E_U (`vhf.E_U`) is excluded from E_x^DFT, so its A/B nonadditive part is left uncorrected.

- **Open-shell DMRG-in-DFT (plan items 7 + 9, tests Z1–Z7, 2026-09-28).** `run_dmrg` runs block2
  SZ mode for any non-spin-symmetric problem, with h1e (aa, bb) and g2e (aa, ab, bb). β orbitals
  are paired with the α orbitals they overlap most. ⟨S²⟩ comes from the αβ 2-RDM and the α/β
  overlap, frozen core included (`spin_square_from_rdms`); block2's own S² MPO assumes shared
  spatial orbitals and is wrong here. Validated against PySCF `spin_square(mo_coeff, ovlp)` and
  unrestricted FCI.
  - `active_hamiltonian` supports `ncore` (mean field folded into h1/ecore, core density
    returned) and equalizes the per-spin kept counts. An open-shell environment gives
    n_env α ≠ β; the highest virtuals of the larger spin are dropped and recorded.
  - Unrestricted FCIDUMP is written through block2 (SZ), with a round-trip test.
- **DMRG start (Z7).** Single-root starts converge to EXCITED eigenstates with a discarded weight
  of ~1e-21 that looks perfect:
  - a random MPS landed 0.379 Eh high in 4/10 seeds (water, 1.0 Å);
  - HF and random starts landed 4.9 / 5.0 mEh high at 3.0 / 5.0 Å, and stronger noise does not help.

  The default is now HF-occupation + state-averaged (3-root) warm-up, keeping the lower:
  0/60 failures. A variational guard also raises if E_DMRG > E(reference determinant). SZ fixes
  M_S only, so it may return a lower-spin state (the triplet at 3.0 Å); check `extras["s2"]`.

- **MP2/CCSD/CCSD(T)-in-DFT (`run_cc`, tests C1–C3).** The embedded UHF is used with the
  environment orbitals frozen, so it works with n_α ≠ n_β and can use DF. Two findings:
  (i) the dynamic Huzinaga embedded HF can converge, from any start, to a solution with an
  electron in an environment orbital (water with only H active, weight 1.0). FCI is immune
  because it only uses the kept space; HF-in-DFT and CC are not, so both now refuse via
  `_check_no_collapse`. (ii) Huzinaga built from the global Fock (`huzinaga_fixed`, Waldrop
  et al. 2025) leaves a projector energy tr[γP] of ~10 mEh. Every result therefore also
  carries `extras["E_eq1_proj_incl"]`, which keeps that energy (nbed / Waldrop eq. 10). It
  does not depend on γ, which makes it the consistent energy for solvers without a 1-RDM.

- **Point charges / DF / DFT+U (2026-09-26, tests Q1–Q6).** QM/MM point charges reach every
  step only through `ks.get_hcore` and `ks.energy_nuc` (nucleus–MM). Every object embed builds
  itself now takes both from the global object: the embedded SCF (`energy_nuc`), CASSCF-in-DFT
  (previously the bare `mol.energy_nuc`, missing nucleus–MM) and FCIDUMP `ECORE`. The embedded
  KS evaluates the global functional through the global `get_veff`, so density fitting and
  DFT+U (UKSpU, `vhf.E_U`, counted with E_xc) are identical to the reference. A fresh `UKS`
  silently dropped both.
- **The Huzinaga projector needs bound active orbitals.** It maps an environment orbital at ε to
  −ε, and the embedded SCF occupies the lowest n_act levels. If a shifted environment level
  falls below the highest occupied active level (OH⁻ in a repulsive field, active HOMO
  +0.32 Ha), the SCF converges to garbage (−0.78 Ha) and reports success. `ProjectionEmbedding`
  now refuses Huzinaga when min(−ε_env) − max(ε_act) < 0.02 Ha (`huzinaga_gap`); μ is unaffected.
- **Eq. 18 is not variational.** With a global SCF gradient of ~1e-5 the embedded density
  differs from D_A to first order, and eq. 18 is off by the trace term (93 µHa, MnO+U, O
  active; 3 µHa at |g| = 3e-7). Eq. 1 stays exact. Converge the reference to a tight
  gradient, or test the eq. 1 − trace = eq. 18 identity.

- **The μ-shift projector is exact only as μ → ∞.** The DFT-in-DFT residual is O(1/μ):
  −17.5, −1.75, −0.175, −0.0175 µHa at μ = 10⁵…10⁸ (B3LYP5, 1.0 Å), i.e. exactly 10×
  per decade. The embedded SCF stops converging at μ = 10⁹. Huzinaga is exact to
  <0.001 µHa. So R1 uses μ = 10⁸, and R1b asserts the 1/μ law.
- **Projector-energy convention.** nbed includes tr[γ μP] in the total energy; Monino
  eq. 1 does not. Both are O(1/μ). `run_fci` reports `extras["E_proj"]`, and
  nbed ≡ E_eq1 + E_proj.
- **At r = 0.4 Å (Table VI, μ = 10⁶)** a residual −6.0 µHa difference from nbed remains
  after the convention fix. It is a finite-μ implementation difference: −0.36 µHa at
  μ = 10⁸ (R2b).
- **Single-root Davidson is unreliable on a stretched bond.** At 3.0 Å it returned the
  triplet (−74.771697) or an excited singlet (−74.771634) instead of the singlet ground
  state (−74.771826). `run_fci` solves for 3 roots and keeps the lowest (or the lowest
  with the requested ⟨S²⟩). This is the likely source of nbed's T5 residual at 3.0 Å.
- **The embedded FCI state can be a triplet** where singlet and triplet are
  near-degenerate (water, r ≥ 3 Å; degenerate at 5 Å, so D_e is unaffected). Pass
  `ss=0.0` for a spin-pure singlet.
- **nbed's CASSCF-wrapped FCI can stop above the eigenvalue.** For PBE0, Table VI, 3.0 Å,
  it is +8.8 µHa and matches no root. So at r ≥ 3 Å the comparison is one-sided:
  ours ≤ nbed.
- **Embedded SCF on a stretched bond.** DIIS can cycle (Huzinaga embedded KS at 4–5 Å;
  embedded HF at 5 Å). The fallbacks are a 0.3 Eh level shift, then second-order SCF.
  Convergence is always required and never waived.
- **Density vs energy tolerance.** The SCF converges the density only to
  ~√conv_tol ≈ 10⁻⁵, while the energy is second-order in it (exact to 10⁻⁸ Ha). So the
  density check is 10⁻⁴.
- **PySCF 2.14 tags HF veff with `.ecoul` too.** Distinguish KS from HF by `.exc`.

- **SCF macro-iteration (Monino eq. 19) is ill-posed on a dissociating bond.** For water,
  r ≤ 2 Å converges in 3 iterations. At r ≥ 3 Å the energy settles to µHa while the
  fixed-point residual ‖γ_out − γ_in‖ stays at 0.4–0.9 under damping (0.5–0.95) and under
  DIIS (6/12). The embedded FCI density switches between very different configurations
  at µHa cost (a flat, statically correlated landscape), so no fixed point exists to
  converge to. This is not solver noise: the code is deterministic at fixed v_emb, and
  exact diagonalization and Davidson agree. So report one-shot energies there. The
  convergence flag requires both the energy and the residual.
- **Exact diagonalization for small FCI spaces** (dim ≤ 4000) removes any Davidson
  guess dependence.

## Known limitations

- The WF solver needs the same number of kept orbitals in both spins (n_env^α = n_env^β).
  That holds when the environment is closed shell; otherwise a clear error is raised.

Pinned versions: PySCF 2.14.0, NumPy 2.5.3, SciPy 1.18.1, block2 0.5.3, pytest 9.1.1.

## Capped-M protocol support (5c292ac, 2026-09-30)
- `run_dmrg(on_rung=...)`: AO density after every bond dimension; production reports spins/NOs at
  the largest M with Δ(two largest M) as error bar, and checkpoints each M (a crash at a larger M
  keeps the smaller ones). `E_extrap_err`: standard error of the E(dw → 0) fit.
- `dE_mp2_fno` = E_MP2(full kept space) − E_MP2(truncated NO space). Checked (Z12) against UMP2
  computed independently on the truncated-space solver Hamiltonian (agreement 1e-10). On embedded
  OH/6-31G the discarded NOs hold correlation MP2 misses (FCI loses 15 mEh, MP2 3.5 mEh at
  lo = 2e-3), so the correction is tested per system, never assumed.
- `active_hamiltonian` memo keyed on (cl_shells, ncore, equalize, no_select, projector, mu).
- Full suite on 94: 494 passed, 5 skipped; U2[3.0-mu-scan] fails only under parallel load (SCF
  convergence), passes alone with 1 or 2 threads.

## Plain (U)HF global references (2026-09-30)
- The production 'hf' environment is a global `scf.UHF`, whose veff has no `.ecoul`/`.exc`.
  `E_dft` and `_exc` read them unconditionally, so every HF-environment correlated run crashed in
  `_assemble` after its last bond dimension. They now fall back to 1/2 tr(v D) (and E_xc = that
  minus E_J); `_embedded_scf("dft")` with a plain HF reference is HF-in-HF. Test Z13.

## S^2 penalty for low-spin targets in SZ mode (2026-10-01)
- `run_dmrg(s2_penalty=(lam, S))` adds lam [S^2 - S(S+1)] in the active space, with alpha/beta overlaps
  O = C_a^T S C_b: S-S+ adds lam (O^T O) to h1_beta and -lam O_rs O_pq to gab[p,r,s,q]; Sz^2 + Sz is a
  constant. Why: in SZ mode M_S = 1/2 also contains the quartet's component, so a Mn(IV) "doublet"
  run otherwise lands on the quartet (<S^2> 3.75) and the spin gap comes out as zero.
- A LINEAR penalty lowers every state with spin below the target, so only S = |M_S| can be targeted;
  any other S raises ValueError. The determinant guard is recomputed with the penalised Hamiltonian.
- Tests: Z14 (O2, alpha = beta: plain -> triplet = FCI, penalty -> singlet = spin-projected FCI) and
  Z15 (N atom, UHF orbitals: plain -> quartet component, penalty -> doublet = orbital-invariant FCI).
  Full suite 498 passed, 5 skipped.
