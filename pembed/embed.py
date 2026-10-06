"""Spin-general projection-based WF-in-DFT embedding (see DESIGN.md).

Usage
-----
    emb = ProjectionEmbedding(ks, active_atoms=[0, 1], projector="mu")
    res = emb.run_fci()          # FCI-in-DFT
    chk = emb.run_dft_in_dft()   # exactness check: chk.E_eq1 == ks.e_tot

`ks` is a converged global RKS or UKS object. It may be QM/MM-wrapped (point charges:
they enter through ks.get_hcore and ks.energy_nuc, both reused everywhere), density-fitted,
or DFT+U (UKSpU: the U energy vhf.E_U is counted with E_xc). All per-spin quantities are
(alpha, beta) tuples, so n_alpha != n_beta is fine.
"""
import types
from dataclasses import dataclass, field

import numpy as np
from pyscf import ao2mo, dft, fci, scf
from pyscf.tools import fcidump as _fcidump

from .spade import active_ao_indices, spade_spin, sqrt_overlap

SPINS = (0, 1)


def as_uks(ks):
    """Return a UKS view of a converged RKS/UKS object (same grids, xc, hcore)."""
    if isinstance(ks, dft.uks.UKS) or (hasattr(ks, "mo_coeff") and np.ndim(ks.mo_coeff) == 3):
        return ks
    return scf.addons.convert_to_uhf(ks)


def _e2(vhf):
    """Two-electron KS energy of a veff: Coulomb + xc (+ the DFT+U energy E_U if present)."""
    return float(vhf.ecoul + vhf.exc + getattr(vhf, "E_U", 0.0))


def _energy_elec_per_spin_hcore(mf, dm=None, h1e=None, vhf=None):
    """energy_elec that accepts a per-spin core Hamiltonian (2, n, n)."""
    if dm is None:
        dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    h1e = np.asarray(h1e)
    if h1e.ndim == 2:
        h1e = (h1e, h1e)
    e1 = sum(float(np.einsum("ij,ji->", h1e[s], dm[s]).real) for s in SPINS)
    # KS results carry .exc; PySCF >= 2.14 also tags HF veff with .ecoul, so test .exc
    if getattr(vhf, "exc", None) is not None:
        e2 = _e2(vhf)
    else:
        e2 = 0.5 * sum(float(np.einsum("ij,ji->", vhf[s], dm[s]).real) for s in SPINS)
    return e1 + e2, e2


@dataclass
class EmbeddingResult:
    method: str
    E_eq1: float
    E_eq18: float
    E_wf_emb: float            # energy of the solver in the embedded Hamiltonian (incl. E_nuc)
    E_wf_plain: float          # electronic, plain h (Monino eq. 4)
    E_dft_total: float
    E_dft_A: float
    E_dft_B: float
    E_H_nadd: float
    E_xc_nadd: float
    trace_term: float
    E_nuc: float
    n_act: tuple
    n_env: tuple
    n_kept: tuple
    sigma: tuple
    env_weight_min: tuple      # smallest projection weight among deleted orbitals (should be ~1)
    kept_weight_max: tuple     # largest weight among kept orbitals (should be ~0)
    s2_global: float
    emb_scf_converged: bool
    solver_converged: bool
    rdm1_ao: tuple = field(repr=False, default=None)
    extras: dict = field(repr=False, default_factory=dict)


class ProjectionEmbedding:
    EXACT_DIAG_MAX = 4000   # FCI dimension up to which the full matrix is diagonalised exactly

    def __init__(self, ks, active_atoms, n_act=(None, None), projector="mu", mu=1.0e6,
                 conv_tol=1e-10, max_cycle=300):
        if projector not in ("mu", "huzinaga", "huzinaga_fixed"):
            raise ValueError("projector must be 'mu', 'huzinaga' or 'huzinaga_fixed'")
        if "ROHF" in type(ks).__mro__.__repr__() or isinstance(ks, (scf.rohf.ROHF,)):
            # the embedded SCF here is spin-UNrestricted: on a restricted-open-shell reference it
            # relaxes the RO constraint inside the active region and lands 0.3-0.4 mEh BELOW the
            # global ROKS energy (OH / MnO, 2026-09-30), so DFT-in-DFT is not exact. Refuse.
            raise NotImplementedError("restricted open-shell (ROHF/ROKS) references are not supported: "
                                      "the embedded SCF is unrestricted; use a UKS/UHF reference")
        self.ks0 = ks
        self.ks = as_uks(ks)
        self.mol = self.ks.mol
        self.active_atoms = list(active_atoms)
        self.n_force = tuple(n_act)
        self.projector = projector
        self.mu = float(mu)
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self._partition()

    # ------------------------------------------------------------------ partition
    def _partition(self):
        ks, mol = self.ks, self.mol
        self.S = mol.intor_symmetric("int1e_ovlp")
        self.h = np.asarray(self.ks.get_hcore())
        self.E_nuc = float(self.ks.energy_nuc())
        s_half = sqrt_overlap(self.S)
        ao_idx = active_ao_indices(mol, self.active_atoms)
        mo, occ = ks.mo_coeff, ks.mo_occ
        cA, cB, nA, nE, sig = [], [], [], [], []
        for s in SPINS:
            c_occ = mo[s][:, occ[s] > 0.5]
            c_loc, n, sigma = spade_spin(c_occ, s_half, ao_idx, self.n_force[s])
            cA.append(c_loc[:, :n]); cB.append(c_loc[:, n:])
            nA.append(n); nE.append(c_occ.shape[1] - n); sig.append(sigma)
        self.cA, self.cB = tuple(cA), tuple(cB)
        self.n_act, self.n_env, self.sigma = tuple(nA), tuple(nE), tuple(sig)
        self.DA = tuple(c @ c.T for c in self.cA)
        self.DB = tuple(c @ c.T for c in self.cB)
        self.D = tuple(self.DA[s] + self.DB[s] for s in SPINS)
        # embedding potential (Monino eq. 7) from the global functional, per spin
        gD = self._veff(self.D)
        gA = self._veff(self.DA)
        self.v_emb = tuple(np.asarray(gD[s]) - np.asarray(gA[s]) for s in SPINS)
        self.P_mu = tuple(self.mu * self.S @ self.DB[s] @ self.S for s in SPINS)
        F_glob = tuple(self.h + np.asarray(gD[s]) for s in SPINS)
        # 'huzinaga_fixed': Huzinaga built once from the GLOBAL Fock (Waldrop et al., JCC 46
        # (2025) e70043, eq. 6) and added to the core Hamiltonian; 'huzinaga' rebuilds it from
        # the embedded Fock every SCF cycle (Hegely et al. 2016). Both vanish on the active space.
        self.P_hf = tuple(-(F_glob[s] @ self.DB[s] @ self.S + self.S @ self.DB[s] @ F_glob[s]) for s in SPINS)
        # Huzinaga validity: the projector maps an environment orbital at eps_B to -eps_B. The
        # embedded SCF occupies the lowest n_act orbitals, so every shifted environment level
        # must lie ABOVE the highest occupied active level. With an unbound (eps > 0) active
        # orbital -- an anion in a repulsive field, OH- here -- a shifted environment orbital
        # drops below it, gets occupied, and the SCF converges to garbage WITHOUT any warning
        # (-0.78 Ha; the mu projector is unaffected).
        F = tuple(self.h + np.asarray(gD[s]) for s in SPINS)
        eig = lambda c, s: np.linalg.eigvalsh(c.T @ F[s] @ c) if c.shape[1] else np.zeros(0)
        self.eps_env = tuple(eig(self.cB[s], s) for s in SPINS)
        self.eps_act = tuple(eig(self.cA[s], s) for s in SPINS)
        gaps = [float((-self.eps_env[s]).min() - self.eps_act[s].max()) for s in SPINS
                if self.eps_env[s].size and self.eps_act[s].size]
        self.huzinaga_gap = min(gaps) if gaps else np.inf
        if self.projector.startswith("huzinaga") and self.huzinaga_gap < 0.02:
            raise ValueError(f"Huzinaga projector invalid here: a shifted environment level (-eps_B) "
                             f"lies {self.huzinaga_gap:+.4f} Ha from the highest occupied active "
                             "level (needs > +0.02; unbound active orbitals?). Use projector='mu'.")
        try:
            self.s2_global = float(ks.spin_square()[0])
        except Exception:
            self.s2_global = float("nan")

    # ------------------------------------------------------------------ DFT pieces
    def _veff(self, dm):
        # bug-class guard (nbed T1b): a 2-D density here would be treated by PySCF's UKS
        # get_veff as UNPOLARISED (1/2, 1/2). Always pass per-spin (2, n, n) densities.
        dm = np.asarray(dm)
        if dm.ndim != 3 or dm.shape[0] != 2:
            raise ValueError(f"_veff needs a per-spin (2, n, n) density, got shape {dm.shape}")
        return self.ks.get_veff(self.mol, dm)

    @staticmethod
    def _e2_of(v, dm):
        """Two-electron energy of veff v at density dm. KS veffs carry .ecoul/.exc; a plain
        (U)HF veff does not (a global scf.UHF reference, the 'hf' environment): 1/2 tr(v dm)."""
        if getattr(v, "exc", None) is not None:
            return _e2(v)
        return 0.5 * sum(float(np.einsum("ij,ji->", v[s], dm[s])) for s in SPINS)

    def E_dft(self, dm):
        """Electronic KS energy of a per-spin density with the global functional."""
        v = self._veff(dm)
        return sum(float(np.einsum("ij,ji->", self.h, dm[s])) for s in SPINS) + self._e2_of(v, dm)

    def _exc(self, dm):
        """E_xc (+ E_U for DFT+U): every nonlinear, non-Hartree part of the functional."""
        v = self._veff(dm)
        if getattr(v, "exc", None) is not None:
            return float(v.exc + getattr(v, "E_U", 0.0))
        rho = np.asarray(dm[0]) + np.asarray(dm[1])          # plain HF: E_xc = exact exchange
        return self._e2_of(v, dm) - 0.5 * float(np.einsum("ij,ji->", self.ks.get_j(self.mol, rho), rho))

    def E_H_nadd(self):
        rhoA = self.DA[0] + self.DA[1]
        rhoB = self.DB[0] + self.DB[1]
        return float(np.einsum("ij,ji->", rhoA, self.ks.get_j(self.mol, rhoB)))

    # ------------------------------------------------------------------ embedded SCF
    def _embedded_mol(self):
        m = self.mol.copy()
        m.nelectron = int(sum(self.n_act))
        m.spin = int(self.n_act[0] - self.n_act[1])
        return m

    def _P_static(self):
        """Projector carried in the core Hamiltonian (None: rebuilt inside get_fock)."""
        return {"mu": self.P_mu, "huzinaga_fixed": self.P_hf}.get(self.projector)

    def _h_emb(self):
        P = self._P_static()
        if P is not None:
            return np.array([self.h + self.v_emb[s] + P[s] for s in SPINS])
        return np.array([self.h + self.v_emb[s] for s in SPINS])

    def _embedded_scf(self, kind, xc=None, dm0=None, df=False):
        if kind != "hf" and xc is None and not hasattr(self.ks, "xc"):
            # global plain (U)HF reference: its "functional" is HF, embedded exactly as HF-in-HF
            return self._embedded_scf("hf", dm0=dm0, df=df or getattr(self.ks, "with_df", None) is not None)
        m = self._embedded_mol()
        if kind == "hf":
            mf = scf.UHF(m)
            if df:
                # same auxiliary basis as the global object (Waldrop et al.: DF throughout)
                aux = getattr(getattr(self.ks, "with_df", None), "auxbasis", None)
                mf = mf.density_fit(auxbasis=aux)
        else:
            mf = dft.UKS(m)
            mf.xc = xc if xc is not None else self.ks.xc
            mf.grids = self.ks.grids
            if getattr(self.ks, "nlc", ""):
                mf.nlc = self.ks.nlc
            if xc is None or xc == self.ks.xc:
                # the global functional EXACTLY as the global object evaluates it: same density
                # fitting, same DFT+U projectors/U, same grids. A fresh UKS would silently drop
                # DF and +U and break DFT-in-DFT exactness for those references.
                ks_glob, mol_glob = self.ks, self.mol

                def get_veff(mol=None, dm=None, *args, **kwargs):
                    if dm is None:
                        dm = mf.make_rdm1()
                    return ks_glob.get_veff(mol_glob, np.asarray(dm))
                mf.get_veff = get_veff
        h_emb = self._h_emb()
        mf.get_hcore = lambda *a, **k: h_emb
        # E_nuc of the global object: includes the nucleus-point-charge term under QM/MM
        mf.energy_nuc = lambda *a, **k: self.E_nuc
        mf.energy_elec = types.MethodType(_energy_elec_per_spin_hcore, mf)
        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.verbose = 0
        if self.projector == "huzinaga":
            DB, S = self.DB, self.S
            orig = mf.get_fock

            def get_fock(h1e=None, s1e=None, vhf=None, dm=None, *args, **kwargs):
                if h1e is None:
                    h1e = mf.get_hcore()
                if dm is None:
                    dm = mf.make_rdm1()
                if vhf is None:
                    vhf = mf.get_veff(mf.mol, dm)
                f = np.asarray(h1e) + np.asarray(vhf)
                p = np.array([-(f[s] @ DB[s] @ S + S @ DB[s] @ f[s]) for s in SPINS])
                return orig(np.asarray(h1e) + p, s1e, vhf, dm, *args, **kwargs)

            mf.get_fock = get_fock
        mf.kernel(dm0=np.array(self.DA) if dm0 is None else dm0)
        if not mf.converged:
            # plain DIIS can oscillate on a stretched (near-degenerate) bond, e.g. the
            # embedded HF / Huzinaga SCF at r = 5 A; converge with a level shift, then
            # remove the shift and polish from that density
            mf.level_shift = 0.3
            mf.diis_space = 12
            mf.kernel(dm0=mf.make_rdm1())
            mf.level_shift = 0.0
            mf.kernel(dm0=mf.make_rdm1())
        if not mf.converged:
            # second-order SCF (Huzinaga embedded KS at 4-5 A still cycles under DIIS). Its
            # orbital gradient is built from mf.get_fock, i.e. includes the Huzinaga term, so
            # convergence means a genuine stationary point of the projected problem.
            so = mf.newton(); so.max_cycle = 200; so.conv_tol = self.conv_tol; so.verbose = 0
            so.kernel(dm0=mf.make_rdm1())
            for attr in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
                setattr(mf, attr, getattr(so, attr))
        return mf

    def _kept_orbitals(self, mf):
        """Delete n_env[s] orbitals with the largest environment weight, per spin."""
        kept, occs, wmin, wmax, envs = [], [], [], [], []
        for s in SPINS:
            C = mf.mo_coeff[s]
            M = self.S @ self.DB[s] @ self.S
            w = np.einsum("pi,pq,qi->i", C, M, C)
            order = np.argsort(-w)
            env = set(order[: self.n_env[s]].tolist())
            envs.append(sorted(env))
            keep = [i for i in range(C.shape[1]) if i not in env]
            kept.append(C[:, keep])
            occs.append(np.asarray(mf.mo_occ[s])[keep])
            wmin.append(float(w[order[: self.n_env[s]]].min()) if self.n_env[s] else 1.0)
            wmax.append(float(w[keep].max()) if keep else 0.0)
        self._kept_occ = tuple(occs)
        self._env_idx = tuple(envs)
        return tuple(kept), tuple(wmin), tuple(wmax)

    # ------------------------------------------------------------------ energies
    def _assemble(self, method, E_wf_emb, gamma_ao, extras, conv_scf, conv_solver,
                  n_kept, wmin, wmax, P=None):
        """Eq. 1 and eq. 18 from a (correlated or KS) active density."""
        Ps = self._P_static() if P is None else None
        h_emb_minus_h = [self.v_emb[s] + (Ps[s] if Ps is not None else 0.0) for s in SPINS]
        E_wf_plain = E_wf_emb - self.E_nuc - sum(
            float(np.einsum("ij,ji->", h_emb_minus_h[s], gamma_ao[s])) for s in SPINS)
        E_D, E_A, E_B = self.E_dft(self.D), self.E_dft(self.DA), self.E_dft(self.DB)
        trace = sum(float(np.einsum("ij,ji->", gamma_ao[s] - self.DA[s], self.v_emb[s])) for s in SPINS)
        EH = self.E_H_nadd()
        Exc = self._exc(self.D) - self._exc(self.DA) - self._exc(self.DB)
        E1 = E_wf_plain + E_D - E_A + trace + self.E_nuc
        E18 = E_wf_plain + E_B + EH + Exc + self.E_nuc
        # eq. 1 with the projector energy tr[gamma P] KEPT (nbed; Waldrop et al. eq. 10). It does
        # not depend on gamma, so it is exact for solvers without a 1-RDM (CCSD(T)); it differs
        # from E_eq1 by tr[gamma P]: O(1/mu) for mu, ~10 mEh for 'huzinaga_fixed', 0 for 'huzinaga'
        extras = dict(extras)
        extras["E_eq1_proj_incl"] = (E_wf_emb - sum(float(np.einsum("ij,ji->", self.v_emb[s], self.DA[s]))
                                                    for s in SPINS) + E_D - E_A)
        return EmbeddingResult(method, E1, E18, E_wf_emb, E_wf_plain, E_D, E_A, E_B, EH, Exc,
                               trace, self.E_nuc, self.n_act, self.n_env, n_kept, self.sigma,
                               wmin, wmax, self.s2_global, conv_scf, conv_solver,
                               tuple(gamma_ao), extras)

    # ------------------------------------------------------------------ drivers
    def run_dft_in_dft(self, xc=None):
        mf = self._embedded_scf("ks", xc=xc)
        g = tuple(np.asarray(d) for d in mf.make_rdm1())
        # E_wf_emb for KS: its own electronic energy in h_emb, plus E_nuc
        E_emb = float(mf.energy_elec(dm=np.array(g))[0]) + self.E_nuc
        # replace E_wf_plain by E_DFT[gamma] with the *global* functional
        res = self._assemble("dft-in-dft", E_emb, g, {}, bool(mf.converged), True,
                             tuple(int(mf.mo_coeff[s].shape[1] - self.n_env[s]) for s in SPINS),
                             (1.0, 1.0), (0.0, 0.0))
        E_ks_plain = self.E_dft(g)
        res.E_wf_plain = E_ks_plain
        res.E_eq1 = E_ks_plain + res.E_dft_total - res.E_dft_A + res.trace_term + self.E_nuc
        res.E_eq18 = E_ks_plain + res.E_dft_B + res.E_H_nadd + res.E_xc_nadd + self.E_nuc
        try:
            res.extras["s2_embedded"] = float(mf.spin_square()[0])
        except Exception:
            pass
        res.extras["dm_diff"] = tuple(float(np.linalg.norm(g[s] - self.DA[s])) for s in SPINS)
        return res

    def run_fci_scf(self, max_iter=60, tol=1e-8, rtol=1e-6, damping=0.0, diis=0, **fci_kw):
        """SCF-FCI-in-DFT (Monino eq. 19): macro-iterate v_emb = G[g + D_B] - G[g] with the
        correlated active 1-RDM g. Hartree (J[rho_B]) and exact-exchange (-aK[D_B]) parts are
        linear, so only the xc part responds; the projector stays built from frozen D_B.

        damping: g_in(k+1) = (1 - damping) g_out(k) + damping g_in(k). On a stretched bond the
        undamped loop can oscillate with period 2 (water r >= 3 A: +-30-90 uH, never settling).
        """
        v0 = self.v_emb
        g = self.DA
        E_old, hist, rhist = None, [], []
        hist_x, hist_r = [], []
        spin_sym = (self.n_act[0] == self.n_act[1] and np.allclose(self.DA[0], self.DA[1], atol=1e-10)
                    and np.allclose(self.DB[0], self.DB[1], atol=1e-10))
        try:
            for it in range(max_iter):
                gD, gA = self._veff(tuple(g[s] + self.DB[s] for s in SPINS)), self._veff(g)
                self.v_emb = tuple(np.asarray(gD[s]) - np.asarray(gA[s]) for s in SPINS)
                res = self.run_fci(**fci_kw)
                g_in = g
                g_out = res.rdm1_ao
                if diis:
                    # Pulay extrapolation on the fixed-point residual R = g_out - g_in: converges
                    # maps with derivative < -1, where damping cannot (the stretched-bond case)
                    xin = np.concatenate([np.ravel(g_in[s]) for s in SPINS])
                    xout = np.concatenate([np.ravel(g_out[s]) for s in SPINS])
                    hist_x.append(xin); hist_r.append(xout - xin)
                    hist_x, hist_r = hist_x[-diis:], hist_r[-diis:]
                    m = len(hist_r)
                    B = np.empty((m + 1, m + 1)); B[-1, :] = B[:, -1] = -1.0; B[-1, -1] = 0.0
                    B[:m, :m] = np.array([[a @ b for b in hist_r] for a in hist_r])
                    rhs = np.zeros(m + 1); rhs[-1] = -1.0
                    c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
                    xnew = sum(ci * (xi + 0.5 * ri) for ci, xi, ri in zip(c, hist_x, hist_r))
                    n2 = g_in[0].size
                    g = (xnew[:n2].reshape(g_in[0].shape), xnew[n2:].reshape(g_in[1].shape))
                    if spin_sym:
                        # extrapolation amplifies round-off alpha/beta differences; for a
                        # spin-symmetric reference with alpha == beta outputs, restore them equal
                        ga = 0.5 * (g[0] + g[1]); g = (ga, ga)
                else:
                    g = tuple((1.0 - damping) * g_out[s] + damping * g_in[s] for s in SPINS)
                gB = tuple(g_out[s] + self.DB[s] for s in SPINS)
                rhoA, rhoB = g_out[0] + g_out[1], self.DB[0] + self.DB[1]
                EH = float(np.einsum("ij,ji->", rhoA, self.ks.get_j(self.mol, rhoB)))
                Exc = self._exc(gB) - self._exc(g_out) - self._exc(self.DB)
                E = res.E_wf_plain + self.E_dft(self.DB) + EH + Exc + self.E_nuc     # eq. 19
                hist.append(E)
                # fixed-point residual: an energy that repeats can be a cycle landing on the same
                # phase, so convergence requires ||g_out - g_in|| small as well
                resid = max(float(np.linalg.norm(g_out[s] - g_in[s])) for s in SPINS)
                rhist.append(resid)
                if E_old is not None and abs(E - E_old) < tol and resid < rtol:
                    break
                E_old = E
            conv = bool(len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol and rhist[-1] < rtol)
            res.extras.update(E_scf=E, scf_history=hist, scf_residuals=rhist, scf_converged=conv)
            return res
        finally:
            self.v_emb = v0

    def delta_x(self, res, xc=None):
        """Monino eq. 20 nonadditive-exchange correction for a finished result (uses the
        result's correlated active 1-RDM gamma_emb and the frozen environment D_B)."""
        from .nadd import delta_x as _dx
        return _dx(self.ks, xc or self.ks.xc, res.rdm1_ao, self.DB)

    def _check_no_collapse(self, mf, tol=1e-2):
        """Reference-dependent solvers need occupied embedded orbitals OUTSIDE the environment.
        The dynamic Huzinaga embedded HF can converge (from any start) to a solution with an
        electron in an environment orbital -- water with only H active, B3LYP environment:
        weight 1.0. FCI is unaffected (it only needs the kept space), HF/MP2/CC are not."""
        w = max(float(np.max(np.einsum("pi,pq,qi->i", c, self.S @ self.DB[s] @ self.S, c), initial=0.0))
                for s in SPINS for c in [mf.mo_coeff[s][:, np.asarray(mf.mo_occ[s]) > 0.5]])
        if w > tol:
            raise RuntimeError(f"embedded SCF collapsed into the environment (occupied orbital with "
                               f"environment weight {w:.3f}); use projector='huzinaga_fixed' or 'mu'")
        return w

    def run_hf_in_dft(self):
        """HF-in-DFT: the embedded UHF density as the 'WF' (usable where FCI is not)."""
        mf = self._embedded_scf("hf")
        self._check_no_collapse(mf)
        g = tuple(np.asarray(d) for d in mf.make_rdm1())
        E_emb = float(mf.energy_elec(dm=np.array(g))[0]) + self.E_nuc
        extras = {}
        if self.projector == "mu":
            extras["E_proj"] = sum(float(np.einsum("ij,ji->", g[s], self.P_mu[s])) for s in SPINS)
        try:
            extras["s2_embedded"] = float(mf.spin_square()[0])
        except Exception:
            pass
        return self._assemble("hf-in-dft", E_emb, g, extras, bool(mf.converged), True,
                              tuple(int(mf.mo_coeff[s].shape[1] - self.n_env[s]) for s in SPINS),
                              (1.0, 1.0), (0.0, 0.0))

    def run_cc(self, method="ccsd(t)", conv_tol=1e-9, max_cycle=200, df=False, freeze_env=True):
        """UMP2 / UCCSD / UCCSD(T)-in-DFT on the embedded UHF, environment orbitals frozen.

        Spin-general (n_alpha != n_beta). The energy is eq. 1 with the embedded UHF density as
        gamma: for the Huzinaga projectors eq. 1 does not depend on gamma (its only gamma
        dependence, tr[gamma (h_emb - h - v_emb)], is zero), for mu it enters as O(1/mu).
        df=True: density-fitted embedded UHF (and hence DF-MP2/CC), auxbasis of the global.
        freeze_env=False keeps the projector-shifted environment orbitals as correlating virtuals
        (not a well-defined embedding -- those orbitals hold the environment electrons -- but
        needed to compare with codes that do not delete them).
        """
        from pyscf import cc as _cc, mp as _mp
        method = method.lower().replace("_pt", "(t)")
        if method not in ("mp2", "ccsd", "ccsd(t)"):
            raise ValueError("method must be 'mp2', 'ccsd' or 'ccsd(t)'")
        mf = self._embedded_scf("hf", df=df)
        self._check_no_collapse(mf)
        C, wmin, wmax = self._kept_orbitals(mf)
        frozen = [list(map(int, self._env_idx[s])) if freeze_env else [] for s in SPINS]
        E_hf = float(mf.energy_elec(dm=mf.make_rdm1())[0]) + self.E_nuc
        extras = {"frozen": [len(f) for f in frozen]}
        if method == "mp2":
            pt = _mp.UMP2(mf, frozen=frozen)
            e_corr = float(pt.kernel()[0]); conv = True
        else:
            mycc = _cc.UCCSD(mf, frozen=frozen)
            mycc.conv_tol, mycc.max_cycle, mycc.verbose = conv_tol, max_cycle, 0
            mycc.kernel()
            e_corr, conv = float(mycc.e_corr), bool(mycc.converged)
            extras["e_ccsd_corr"] = e_corr
            if method == "ccsd(t)":
                e_t = float(mycc.ccsd_t()); extras["e_t"] = e_t; e_corr += e_t
        extras.update(E_hf_emb=E_hf, e_corr=e_corr)
        g = tuple(np.asarray(d) for d in mf.make_rdm1())
        return self._assemble(f"{method}-in-dft", E_hf + e_corr, g, extras, bool(mf.converged), conv,
                              tuple(int(C[s].shape[1]) for s in SPINS), wmin, wmax)

    PAD_ENERGY = 1.0e3      # diagonal h1 of a decoupled dummy site (Eh): never occupied

    def active_hamiltonian(self, mf=None, cl_shells=None, ncore=0, equalize="pad", no_select=None):
        """Embedded integrals in the kept-orbital space (per spin), for FCI/DMRG/FCIDUMP.

        cl_shells=None keeps every non-environment virtual; cl_shells=n keeps the occupied
        active orbitals plus concentric-localization shells 0..n (see cl.py).
        ncore: freeze the lowest `ncore` occupied kept orbitals of EACH spin (their mean field
        is folded into h1 and ecore; their density is returned as H["D_core"]).
        equalize: an open-shell environment (e.g. other Mn in a cathode cluster) leaves the two
        spins with different numbers of kept orbitals (R2: n_env 56a/55b); SZ DMRG needs one site
        count. The extra orbital of the larger spin is a genuine correlating virtual (e.g. the
        environment Mn-d that is empty for that spin), so it must be kept:
          "pad"     (default) add DECOUPLED dummy sites to the smaller spin: zero orbital, zero
                    two-electron integrals, h1 = PAD_ENERGY -> exactly invariant (test Z8);
          "pad_env" add the smaller spin's projector-shifted environment orbital(s) of largest
                    overlap with the unmatched orbitals (inert only as mu -> inf; test only);
          "trim"    drop the highest virtuals of the larger spin (changes the energy; test only).
        no_select: dict(lo=, hi=) -> natural-orbital truncation (replaces CL). UMP2 on the embedded
          UHF (environment and the ncore core frozen), per-spin natural orbitals of the unrelaxed
          UMP2 1-RDM in the kept, non-core space: occupation >= hi -> frozen (added to the core),
          <= lo -> dropped, in between -> active. Needed at cluster scale: CL shells of a Mn-
          containing region are 100-200 orbitals (R2: 101 / 191 for n = 0 / 1).
        """
        key = None
        if mf is None:
            # memo: a repeated call with the same arguments (e.g. prod_emb sizing the space, then
            # run_dmrg) reuses the result instead of re-running the embedded UHF and the UMP2s
            key = (cl_shells, ncore, equalize, None if no_select is None else tuple(sorted(no_select.items())),
                   self.projector, getattr(self, "mu", None))
            memo = getattr(self, "_H_memo", None)
            if memo is not None and memo[0] == key:
                return memo[1]
            mf = self._embedded_scf("hf")
        C, wmin, wmax = self._kept_orbitals(mf)
        h_emb = self._h_emb()
        # Restricted if the PROBLEM is spin-symmetric (same h_emb, D_B, electron counts).
        # Without CL the kept space (complement of the environment) is then the same space for
        # both spins, so one orthonormal basis serves both even if the embedded UHF broke spin
        # symmetry (seen inside SCF macro-iterations at 3 A).
        sym_problem = (self.n_act[0] == self.n_act[1] and self.n_env[0] == self.n_env[1]
                       and np.allclose(h_emb[0], h_emb[1], atol=1e-10)
                       and np.allclose(self.DB[0], self.DB[1], atol=1e-10))
        shells = None
        no_info = None
        if no_select is not None:
            C, Cc_no, no_info = self._mp2_natural_orbitals(mf, int(ncore), no_select)
            sym_problem = False                  # per-spin NO spaces: always the unrestricted path
        if cl_shells is not None and no_select is None:
            from .cl import concentric_spin
            F = mf.get_fock()
            ao_idx = active_ao_indices(self.mol, self.active_atoms)
            dm_e = mf.make_rdm1()
            if sym_problem and np.allclose(dm_e[0], dm_e[1], atol=1e-8):
                # one CL for both spins: separate per-spin SVDs with degenerate singular values
                # return different (equally valid) bases of the same shells, which would make a
                # closed-shell problem look unrestricted (propionitrile, cc-pVDZ)
                occ = self._kept_occ[0] > 0.5
                c, sz = concentric_spin(C[0][:, occ], C[0][:, ~occ], 0.5 * (np.asarray(F[0]) + np.asarray(F[1])),
                                        self.S, ao_idx, int(cl_shells))
                C, shells = (c, c), (sz, sz)
            else:
                newC, shells = [], []
                for s in SPINS:
                    occ = self._kept_occ[s] > 0.5
                    c, sz = concentric_spin(C[s][:, occ], C[s][:, ~occ], np.asarray(F[s]), self.S,
                                            ao_idx, int(cl_shells))
                    newC.append(c); shells.append(sz)
                C = tuple(newC); shells = tuple(shells)
        trimmed, padded = (0, 0), (0, 0)
        if C[0].shape[1] != C[1].shape[1]:
            if not equalize:
                raise NotImplementedError(
                    f"kept orbitals differ per spin {C[0].shape[1]} vs {C[1].shape[1]} (open-shell environment)")
            big = int(np.argmax([C[0].shape[1], C[1].shape[1]])); sm = 1 - big
            k = C[big].shape[1] - C[sm].shape[1]
            if equalize == "trim":
                m = C[sm].shape[1]
                if m < max(self.n_act):
                    raise RuntimeError("trimming the kept spaces would drop occupied orbitals")
                trimmed = tuple(int(C[s].shape[1] - m) for s in SPINS)
                C = tuple(C[s][:, :m] for s in SPINS)      # occupied first, highest virtuals last
            elif equalize == "pad_env":
                env = self._env_idx[sm]
                Ce = mf.mo_coeff[sm][:, env]
                ov = np.abs(Ce.T @ self.S @ C[big]).max(axis=1)
                pick = np.argsort(-ov)[:k]
                Cn = list(C); Cn[sm] = np.hstack([C[sm], Ce[:, pick]]); C = tuple(Cn)
                padded = tuple(k if s == sm else 0 for s in SPINS)
            else:                                            # "pad": decoupled dummy sites
                Cn = list(C); Cn[sm] = np.hstack([C[sm], np.zeros((C[sm].shape[0], k))]); C = tuple(Cn)
                padded = tuple(k if s == sm else 0 for s in SPINS)
                self._pad = (sm, k)
        restricted = sym_problem and (shells is None or np.allclose(C[0], C[1], atol=1e-8))
        if restricted:
            C = (C[0], C[0])
        ncore = int(ncore)
        nelec = self.n_act
        D_core, ecore = (np.zeros_like(self.S), np.zeros_like(self.S)), self.E_nuc
        v_core = (0.0, 0.0)
        Cc = (np.zeros((self.S.shape[0], 0)),) * 2
        if no_select is not None:
            Cc = Cc_no                           # core + frozen high-occupation NOs, per spin
        elif ncore:
            if ncore > min(self.n_act):
                raise ValueError(f"ncore={ncore} exceeds the active occupied count {self.n_act}")
            Cc = tuple(C[s][:, :ncore] for s in SPINS)
            C = tuple(C[s][:, ncore:] for s in SPINS)
        nc_s = tuple(int(c.shape[1]) for c in Cc)
        if any(nc_s):
            D_core = tuple(c @ c.T for c in Cc)
            vj, vk = scf.hf.get_jk(self.mol, np.array(D_core), hermi=1)
            v_core = tuple(vj[0] + vj[1] - vk[s] for s in SPINS)
            ecore = self.E_nuc + sum(float(np.einsum("ij,ji->", D_core[s], h_emb[s] + 0.5 * v_core[s]))
                                     for s in SPINS)
            nelec = tuple(int(self.n_act[s] - nc_s[s]) for s in SPINS)
        n = C[0].shape[1]
        h1 = [C[s].T @ (h_emb[s] + v_core[s]) @ C[s] for s in SPINS]
        if equalize == "pad" and any(padded):
            sm = int(np.argmax(padded))
            for j in range(n - padded[sm], n):
                h1[sm][j, j] = self.PAD_ENERGY               # zero column -> zero coupling
        h1 = tuple(h1)
        if restricted:
            eri = ao2mo.restore(1, ao2mo.full(self.mol, C[0]), n)
            g = (eri, eri, eri)
        else:
            gaa = ao2mo.restore(1, ao2mo.general(self.mol, (C[0],) * 4), n)
            gbb = ao2mo.restore(1, ao2mo.general(self.mol, (C[1],) * 4), n)
            gab = ao2mo.general(self.mol, (C[0], C[0], C[1], C[1]), compact=False).reshape(n, n, n, n)
            g = (gaa, gab, gbb)
        H = dict(C=C, h1=h1, eri=g, norb=n, nelec=nelec, ecore=ecore, ncore=nc_s, C_core=Cc, D_core=D_core,
                 no_info=no_info,
                 trimmed=trimmed, padded=padded, restricted=restricted, wmin=wmin, wmax=wmax, mf=mf, cl_shells=shells)
        if key is not None:
            self._H_memo = (key, H)
        return H

    def _mp2_natural_orbitals(self, mf, ncore, sel):
        """Per-spin UMP2 natural orbitals of the embedded problem (see active_hamiltonian)."""
        from pyscf import mp as _mp
        lo, hi = float(sel.get("lo", 0.005)), float(sel.get("hi", 0.995))
        self._check_no_collapse(mf)
        self._kept_orbitals(mf)
        env = [list(map(int, self._env_idx[s])) for s in SPINS]
        occ_idx = [np.where(np.asarray(mf.mo_occ[s]) > 0.5)[0] for s in SPINS]
        core = [list(map(int, occ_idx[s][:ncore])) for s in SPINS]
        pt = _mp.UMP2(mf, frozen=[sorted(env[s] + core[s]) for s in SPINS])
        pt.verbose = 0
        e_mp2 = float(pt.kernel()[0])
        dm = pt.make_rdm1()                                   # MO basis, per spin
        Cact, Ccore, info, nos = [], [], {"e_mp2": e_mp2, "lo": lo, "hi": hi}, []
        for s in SPINS:
            keep = [i for i in range(mf.mo_coeff[s].shape[1]) if i not in env[s] and i not in core[s]]
            g = np.asarray(dm[s])[np.ix_(keep, keep)]
            n, U = np.linalg.eigh(0.5 * (g + g.T))
            order = np.argsort(-n); n, U = n[order], U[:, order]
            Cno = mf.mo_coeff[s][:, keep] @ U
            a = (n > lo) & (n < hi); f = n >= hi
            Cact.append(Cno[:, a])
            Ccore.append(np.hstack([mf.mo_coeff[s][:, core[s]], Cno[:, f]]))
            info[f"spin{s}"] = dict(n_active=int(a.sum()), n_frozen_occ=int(f.sum()), n_dropped=int((n <= lo).sum()),
                                    occ_active=[round(float(x), 5) for x in n[a]])
            nos.append((keep, U, n, a, f))
        # frozen-natural-orbital correction: UMP2 again, in the TRUNCATED space. The unrelaxed MP2
        # density has no occupied-virtual block, so every NO is occupied- or virtual-type; the kept
        # ones of each type are semicanonicalised (canonical MP2 is then exact in that space) and
        # the frozen-occupied / dropped NOs are frozen with the core and environment.
        # dE_mp2_fno = E_MP2(full kept space) - E_MP2(truncated) is added to the truncated-space
        # correlated energy (the discarded virtuals hold mostly dynamic correlation).
        fock = mf.get_fock()
        C2, frozen2 = [], []
        for s, (keep, U, n, a, f) in zip(SPINS, nos):
            C_s = np.array(mf.mo_coeff[s], copy=True)
            occ_pos = [i for i in keep if mf.mo_occ[s][i] > 0.5]
            vir_pos = [i for i in keep if mf.mo_occ[s][i] <= 0.5]
            Cno = mf.mo_coeff[s][:, keep] @ U
            is_occ = n > 0.5
            if int(is_occ.sum()) != len(occ_pos):
                raise RuntimeError("MP2 natural orbitals mix occupied and virtual MOs")
            def semican(Cb):
                if Cb.shape[1] == 0:
                    return Cb
                w_, v_ = np.linalg.eigh(Cb.T @ fock[s] @ Cb)
                return Cb @ v_
            Co_f, Co_a = Cno[:, is_occ & f], semican(Cno[:, is_occ & a])
            Cv_a, Cv_d = semican(Cno[:, ~is_occ & a]), Cno[:, ~is_occ & ~a]
            C_s[:, occ_pos] = np.hstack([Co_f, Co_a])
            C_s[:, vir_pos] = np.hstack([Cv_a, Cv_d])
            fr = env[s] + core[s] + occ_pos[:Co_f.shape[1]] + vir_pos[Cv_a.shape[1]:]
            C2.append(C_s); frozen2.append(sorted(fr))
        pt2 = _mp.UMP2(mf, frozen=frozen2, mo_coeff=C2)
        pt2.verbose = 0
        e_tr = float(pt2.kernel()[0])
        info.update(e_mp2_trunc=e_tr, dE_mp2_fno=e_mp2 - e_tr)
        return tuple(Cact), tuple(Ccore), info

    def run_fci(self, conv_tol=1e-12, max_cycle=500, ss=None, nroots=3, cl_shells=None, ncore=0, no_select=None):
        """FCI-in-DFT.

        Davidson is run for `nroots` roots and the lowest is kept: on a stretched bond a
        single root can converge to the triplet or to an excited singlet from a poor guess
        (seen at water r = 3.0 A, where it misses the singlet ground state by ~190 uH).
        ss=None: lowest state with the given M_s (may be a triplet when singlet/triplet
        are near-degenerate). ss=0.0 etc.: spin-pure target via fix_spin_.
        """
        H = self.active_hamiltonian(cl_shells=cl_shells, ncore=ncore, no_select=no_select)
        n, ne = H["norb"], H["nelec"]
        from pyscf.fci import cistring
        dim = cistring.num_strings(n, ne[0]) * cistring.num_strings(n, ne[1])
        if H["restricted"] and dim <= self.EXACT_DIAG_MAX:
            # exact diagonalisation of the full FCI matrix: no Davidson guess/convergence risk
            # (Davidson with a spin penalty returned inconsistent lowest singlets on alternate
            # calls inside SCF macro-iterations at water r = 3 A)
            addr, h0 = fci.direct_spin1.pspace(H["h1"][0], H["eri"][0], n, ne, np=dim)
            w, v = np.linalg.eigh(h0)
            na_, nb_ = cistring.num_strings(n, ne[0]), cistring.num_strings(n, ne[1])
            cs = []
            for k in range(min(max(nroots, 6), dim)):
                c = np.zeros(dim); c[addr] = v[:, k]; cs.append(c.reshape(na_, nb_))
            es = w[:len(cs)] + H["ecore"]
            s2s = [float(fci.spin_op.spin_square0(c, n, ne)[0]) for c in cs]
            solver = None
        elif H["restricted"]:
            solver = fci.direct_spin1.FCI()
            solver.conv_tol, solver.max_cycle, solver.nroots = conv_tol, max_cycle, nroots
            if ss is not None:
                solver = fci.addons.fix_spin_(solver, ss=float(ss), shift=0.5)
            es, cs = solver.kernel(H["h1"][0], H["eri"][0], n, ne, ecore=H["ecore"])
            es, cs = np.atleast_1d(es), (cs if isinstance(cs, (list, tuple)) else [cs])
            s2s = [float(fci.spin_op.spin_square0(c, n, ne)[0]) for c in cs]
        if H["restricted"]:
            if ss is not None:   # the penalty keeps others high, but pick the true target
                ok = [i for i, v in enumerate(s2s) if abs(v - ss) < 1e-3] or list(range(len(es)))
            else:
                ok = list(range(len(es)))
            i = min(ok, key=lambda j: es[j])
            e, c, s2 = float(es[i]), cs[i], s2s[i]
            r1 = fci.direct_spin1.make_rdm1s(c, n, ne)
        else:
            if ss is not None:
                raise NotImplementedError("spin-pure target for unrestricted FCI: use DMRG (SU2)")
            solver = fci.direct_uhf.FCISolver()
            solver.conv_tol, solver.max_cycle, solver.nroots = conv_tol, max_cycle, nroots
            es, cs = solver.kernel(H["h1"], H["eri"], n, ne, ecore=H["ecore"])
            es, cs = np.atleast_1d(es), (cs if isinstance(cs, (list, tuple)) else [cs])
            i = int(np.argmin(es))
            e, c, s2 = float(es[i]), cs[i], float("nan")
            r1 = solver.make_rdm1s(c, n, ne)
        gamma = tuple(H["C"][s] @ r1[s] @ H["C"][s].T + H["D_core"][s] for s in SPINS)
        if not H["restricted"]:
            r1s, r2s = solver.make_rdm12s(c, n, ne)
            s2 = self.spin_square_from_rdms(H, r1s, r2s[1])
        conv = bool(np.all(getattr(solver, "converged", True)))
        extras = {"s2_fci": s2, "fci_roots": [float(x) for x in es], "cl_shells": H["cl_shells"]}
        if self.projector == "mu":
            # projector energy tr[gamma mu P]: excluded by Monino eq. 1, INCLUDED by nbed;
            # O(1/mu), so the two conventions agree only as mu -> inf
            extras["E_proj"] = sum(float(np.einsum("ij,ji->", gamma[s], self.P_mu[s])) for s in SPINS)
        res = self._assemble("fci-in-dft", e, gamma, extras, bool(H["mf"].converged),
                             conv, (n, n), H["wmin"], H["wmax"])
        self._last_H = H
        return res

    def spin_square_from_rdms(self, H, rdm1s, rdm2ab):
        """<S^2> of a correlated wavefunction in UNRESTRICTED active orbitals (alpha != beta
        spatial orbitals), frozen core included.

        <S^2> = Sz(Sz+1) + N_b - sum O[p,q] O[p',q'] <a+_p b+_q' b_q a_p'>, with O the alpha/beta
        orbital overlap over the occupied (core + active) spaces. The frozen core enters as
        spectator occupations. rdm1s: PySCF convention; rdm2ab: PySCF make_rdm12s convention,
        dm2ab[p,q,r,s] = <a+_p b+_r b_s a_q>. (block2's SZ built-in S^2 operator assumes shared
        spatial orbitals and is wrong here.)
        """
        C, S = H["C"], self.S
        Cc = list(H.get("C_core") or [self._core_cols(H, s) for s in SPINS])
        nca, ncb = (int(x) for x in (H["ncore"] if np.ndim(H["ncore"]) else (H["ncore"],) * 2))
        # overlaps: active-active, core-active, active-core, core-core
        Oaa = C[0].T @ S @ C[1]
        g1a, g1b = np.asarray(rdm1s[0]), np.asarray(rdm1s[1])     # g[p,q] = <q+ p>, real symmetric
        na, nb = float(np.trace(g1a)) + nca, float(np.trace(g1b)) + ncb
        sz = 0.5 * (na - nb)
        T = float(np.einsum("ps,qr,pqrs->", Oaa, Oaa, rdm2ab))
        if nca or ncb:
            Ocb = Cc[0].T @ S @ np.hstack([Cc[1], C[1]])            # core alpha x (core+act) beta
            Db = np.zeros((ncb + C[1].shape[1],) * 2); Db[:ncb, :ncb] = np.eye(ncb); Db[ncb:, ncb:] = g1b
            T += float(np.einsum("pq,qr,pr->", Ocb, Db, Ocb))
            Oac = C[0].T @ S @ Cc[1]                                # active alpha x core beta
            T += float(np.einsum("pq,pr,qr->", g1a, Oac, Oac))
        return sz * (sz + 1) + nb - T

    def _core_cols(self, H, s):
        """Frozen-core orbital coefficients (recovered from the core density)."""
        nc = H["ncore"][s] if np.ndim(H["ncore"]) else H["ncore"]
        if not nc:
            return np.zeros((self.S.shape[0], 0))
        w, v = np.linalg.eigh(self.S)
        X = v @ np.diag(w ** -0.5) @ v.T
        e, u = np.linalg.eigh(X.T @ self.S @ H["D_core"][s] @ self.S @ X)
        return X @ u[:, -nc:]

    def run_dmrg(self, bond_dims=(250, 500, 1000), cl_shells=None, scratch="/tmp/pembed_dmrg",
                 n_threads=None, stack_mem=24 << 30, sweeps_per_M=6, final_sweeps=4, seed=11,
                 ncore=0, spin_adapted=None, s2=True, driver_kwargs=None, dmrg_kwargs=None,
                 starts=("hf", "sa"), noise0=1e-4, sa_roots=3, verify_starts=True, start_tol=1e-4,
                 no_select=None, stop_tol=None, on_rung=None, s2_penalty=None):
        """DMRG-in-DFT with block2, returning the correlated 1-RDM in the AO basis.

        spin-symmetric problems -> SU2 (spin-pure, 2S = n_alpha - n_beta);
        otherwise (open-shell environment / broken symmetry / n_alpha != n_beta with alpha !=
        beta orbitals) -> SZ with separate integrals h1e (aa, bb), g2e (aa, ab, bb). SZ fixes
        M_S only, so <S^2> is computed from the 1-/2-RDMs (spin_square_from_rdms).
        ncore: frozen core per spin (see active_hamiltonian). spin_adapted=False forces SZ.
        starts: the first bond dimension is run from each start and the lowest is continued:
        'hf' = embedded-HF occupations; 'sa' = state-averaged warm-up over `sa_roots` roots from a
        random MPS, then its lowest root; 'random' = single-root random MPS. Single-root starts
        alone are unsafe: random hit an excited state 0.379 Eh high in 4/10 seeds (water 1.0 A),
        HF / random stuck 4.9-5.0 mEh high in an excited singlet at 3.0-5.0 A (symmetry trapping,
        not cured by noise 1e-3); the state-averaged warm-up spans the competing states.
        Each bond dimension is its own dmrg() call; extras["M_series"] holds (M, E, discarded
        weight) and extras["E_extrap"] the linear E(dw -> 0) fit through the last three, with
        extras["E_extrap_err"] its standard error (>= 3 points; the capped-M protocol).
        on_rung(M, E, dw, gamma): called after every bond dimension of the continued state with
        its AO density (per spin, frozen core included) -- per-M properties, and a checkpoint that
        survives a crash at a larger M.
        s2_penalty=(lam, S): SZ only, S = |M_S|. Adds lam [S^2 - S(S+1)] to the active Hamiltonian (alpha/beta
        overlaps O = C_a^T S C_b), so a low-spin target (e.g. the Mn(IV) doublet) is not lost to
        the M_S = S component of a higher-spin ground state. The penalty is zero for the target
        spin, so its energy is unchanged; <S^2> from the RDMs checks the result.
        driver_kwargs / dmrg_kwargs go to DMRGDriver / dmrg (large runs: min_mpo_mem=True,
        restart_dir=...; lowmem_noise=True -- these avoided the OOM kills of C3).
        """
        import os, shutil
        from scipy.optimize import linear_sum_assignment
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
        H = self.active_hamiltonian(cl_shells=cl_shells, ncore=ncore, no_select=no_select)
        use_su2 = H["restricted"] if spin_adapted is None else bool(spin_adapted)
        if use_su2 and not H["restricted"]:
            raise ValueError("SU2 DMRG needs a spin-symmetric (restricted) Hamiltonian")
        n, ne = H["norb"], H["nelec"]
        tag = f"{scratch}_{os.getpid()}"
        drv = DMRGDriver(scratch=tag, symm_type=SymmetryTypes.SU2 if use_su2 else SymmetryTypes.SZ,
                         n_threads=int(n_threads or os.environ.get("OMP_NUM_THREADS", 8)), stack_mem=stack_mem,
                         **(driver_kwargs or {}))
        drv.initialize_system(n_sites=n, n_elec=int(sum(ne)), spin=int(ne[0] - ne[1]))
        C = [H["C"][0], H["C"][1]]
        h1 = [H["h1"][0], H["h1"][1]]
        gaa, gab, gbb = H["eri"]
        occ = [np.zeros(n), np.zeros(n)]
        for s_ in SPINS:
            occ[s_][: ne[s_]] = 1.0                       # kept orbitals: occupied first
        # reference determinant energy in this space: DMRG is variational, so a DMRG energy
        # above it means Davidson converged to an EXCITED eigenstate (seen with random MPS
        # starts: 4 of 10 seeds landed 0.379 Eh high on water/STO-3G, with dw ~ 1e-21)
        def det_energy(ecore_, h1_, gaa_, gab_, gbb_, occ_):
            oa, ob = np.where(occ_[0] > 0)[0], np.where(occ_[1] > 0)[0]
            return float(ecore_ + np.trace(h1_[0][np.ix_(oa, oa)]) + np.trace(h1_[1][np.ix_(ob, ob)])
                         + 0.5 * (np.einsum("iijj->", gaa_[np.ix_(oa, oa, oa, oa)]) - np.einsum("ijji->", gaa_[np.ix_(oa, oa, oa, oa)]))
                         + 0.5 * (np.einsum("iijj->", gbb_[np.ix_(ob, ob, ob, ob)]) - np.einsum("ijji->", gbb_[np.ix_(ob, ob, ob, ob)]))
                         + np.einsum("iijj->", gab_[np.ix_(oa, oa, ob, ob)]))
        e_det = det_energy(H["ecore"], h1, gaa, gab, gbb, occ)
        if not use_su2:
            # pair each beta orbital with the alpha orbital it overlaps most (same site)
            O = np.abs(C[0].T @ self.S @ C[1])
            _, pb = linear_sum_assignment(-O)
            occ[1] = occ[1][pb]
            C[1] = C[1][:, pb]; h1[1] = h1[1][np.ix_(pb, pb)]
            gab = gab[:, :, pb][:, :, :, pb]; gbb = gbb[np.ix_(pb, pb, pb, pb)]
        if s2_penalty is not None:
            if use_su2:
                raise ValueError("s2_penalty is for SZ mode (SU2 is spin-pure already)")
            lam, S_t = float(s2_penalty[0]), float(s2_penalty[1])
            if abs(S_t - 0.5 * abs(ne[0] - ne[1])) > 1e-9:
                # a LINEAR penalty lam[S^2 - S(S+1)] lowers every state with spin below S, so it can
                # only select the lowest spin compatible with M_S, S = |M_S| (e.g. the M_S = 1/2 doublet)
                raise ValueError(f"s2_penalty targets S = |M_S| = {0.5 * abs(ne[0] - ne[1])} only, got S = {S_t}")
            O = C[0].T @ self.S @ C[1]                      # <phi_alpha_p | phi_beta_q>, active space
            Ms = 0.5 * (ne[0] - ne[1])
            # S^2 = S-S+ + Sz^2 + Sz;  S-S+ = sum_sq (O^T O)_sq b+_s b_q - sum O_rs O_pq a+_p b+_s b_q a_r
            h1 = [h1[0], h1[1] + lam * (O.T @ O)]
            gab = gab - lam * np.einsum("rs,pq->prsq", O, O)
            H = dict(H, ecore=float(H["ecore"]) + lam * (Ms * Ms + Ms - S_t * (S_t + 1)))
            e_det = det_energy(H["ecore"], h1, gaa, gab, gbb, occ)   # the guard compares like with like
        idx = np.asarray(drv.orbital_reordering(h1[0] if use_su2 else np.array(h1), gaa))
        ix2, ix4 = np.ix_(idx, idx), np.ix_(idx, idx, idx, idx)
        if use_su2:
            mpo = drv.get_qc_mpo(h1e=h1[0][ix2], g2e=gaa[ix4], ecore=float(H["ecore"]), iprint=0)
        else:
            mpo = drv.get_qc_mpo(h1e=(h1[0][ix2], h1[1][ix2]), g2e=(gaa[ix4], gab[ix4], gbb[ix4]),
                                 ecore=float(H["ecore"]), iprint=0)
        def block(ket, k, M):
            ns = sweeps_per_M + (final_sweeps if k == len(bond_dims) - 1 else 0)
            nz = [noise0 if k == 0 else 1e-5] * (sweeps_per_M // 2) + [0.0] * (ns - sweeps_per_M // 2)
            e = float(drv.dmrg(mpo, ket, n_sweeps=ns, bond_dims=[int(M)] * ns, noises=nz, thrds=[1e-10] * ns,
                               iprint=0, **(dmrg_kwargs or {})))
            try:
                dws = drv.get_dmrg_results()[1]
                dw = float(np.max(dws[-1])) if np.ndim(dws) else float(dws)
            except Exception:
                dw = float("nan")
            return e, dw
        inv = np.argsort(idx)
        def rdm1(ket):
            """1-RDM per spin in the kept-orbital basis (PySCF convention <q+ p>)."""
            if use_su2:
                pdm = np.asarray(drv.get_1pdm(ket))[np.ix_(inv, inv)]      # spin-summed
                return pdm / 2, pdm / 2
            pdm = np.asarray(drv.get_1pdm(ket))                            # (2, n, n): <a+_i a_j>
            return tuple(pdm[s][np.ix_(inv, inv)].T for s in SPINS)
        def ao_density(r1):
            return tuple(C[s] @ r1[s] @ C[s].T + H["D_core"][s] for s in SPINS)
        def rung_done(ket):
            if on_rung is not None:
                M_, E_, dw_ = series[-1]
                on_rung(M_, E_, dw_, ao_density(rdm1(ket)))
        np.random.seed(seed)
        trials = []
        for j, st in enumerate(starts):
            if st == "sa":
                ksa = drv.get_random_mps(tag=f"KSA{j}", bond_dim=int(bond_dims[0]), nroots=int(sa_roots))
                ns = sweeps_per_M
                es = drv.dmrg(mpo, ksa, n_sweeps=ns, bond_dims=[int(bond_dims[0])] * ns,
                              noises=[noise0] * (ns // 2) + [0.0] * (ns - ns // 2), thrds=[1e-10] * ns, iprint=0)
                ir = int(np.argmin(np.atleast_1d(es)))
                k_ = drv.split_mps(ksa, ir, tag=f"K{j}")
            else:
                occs = list((occ[0] + occ[1])[idx]) if st == "hf" else None
                k_ = drv.get_random_mps(tag=f"K{j}", bond_dim=int(bond_dims[0]), nroots=1, occs=occs)
            e0, dw0 = block(k_, 0, bond_dims[0])
            trials.append((e0, dw0, f"{st}{j}", k_))
        e, dw, st_best, ket = min(trials, key=lambda t: t[0])
        series = [(int(bond_dims[0]), e, dw)]
        rung_done(ket)
        k0 = 1
        start_check = None
        if verify_starts and len(trials) > 1 and len(bond_dims) > 1:
            # carry EVERY start through the second bond dimension: agreement there is the real
            # protection against a trapped state (the determinant guard only catches gross cases)
            M1 = bond_dims[1]
            carried = [(*block(t[3], 1, M1), t[2], t[3]) for t in trials]
            start_check = {c_[2]: c_[0] for c_ in carried}
            e, dw, st_best, ket = min(carried, key=lambda t: t[0])
            series.append((int(M1), e, dw)); k0 = 2
            rung_done(ket)
        for k, M in enumerate(bond_dims[k0:], start=k0):
            if stop_tol is not None and len(series) >= 2 and abs(series[-1][1] - series[-2][1]) < stop_tol:
                break                    # adaptive: converged in M (small spaces), skip the larger M
            e, dw = block(ket, k, M)
            series.append((int(M), e, dw))
            rung_done(ket)
        if e > e_det + 1e-6:
            raise RuntimeError(f"DMRG energy {e:.8f} lies above the reference determinant {e_det:.8f}: "
                               "converged to an excited state")
        extras = {"cl_shells": H["cl_shells"], "norb": n, "bond_dims": list(bond_dims), "ncore": H["ncore"],
                  "trimmed": H["trimmed"], "mode": "SU2" if use_su2 else "SZ", "M_series": series,
                  "no_info": H.get("no_info"),
                  "E_ref_det": e_det, "start": st_best, "padded": H["padded"], "s2_penalty": s2_penalty,
                  "start_energies": {t[2]: t[0] for t in trials}, "start_check_M1": start_check}
        if start_check:
            spread = max(start_check.values()) - min(start_check.values())
            extras["start_spread_mEh"] = 1e3 * spread
            extras["start_disagree"] = bool(spread > start_tol)
        pts = [(d, E) for _, E, d in series[-3:] if np.isfinite(d)]
        if len(pts) >= 2 and np.ptp([p[0] for p in pts]) > 0:
            x, y = np.array(pts).T
            if len(pts) >= 3:
                coef, cov = np.polyfit(x, y, 1, cov="unscaled")
                dof = len(pts) - 2
                s2res = float(np.sum((y - np.polyval(coef, x)) ** 2)) / dof
                extras["E_extrap_err"] = float(np.sqrt(cov[1, 1] * s2res))    # std. error of E(dw -> 0)
            else:
                coef = np.polyfit(x, y, 1)
            extras["E_extrap"] = float(coef[1]); extras["extrap_minus_last_mEh"] = 1e3 * (extras["E_extrap"] - y[-1])
        r1 = rdm1(ket)
        if use_su2:
            S_ = 0.5 * (ne[0] - ne[1]); extras["s2"] = S_ * (S_ + 1)   # exact by construction
        else:
            if s2:
                d2 = drv.get_2pdm(ket)                                 # (aa, ab, bb), block2 order
                d2ab = np.asarray(d2[1])[ix4 if False else np.ix_(inv, inv, inv, inv)]
                # block2: <a+_i b+_j b_k a_l> = d2ab[i,j,k,l]; PySCF dm2ab[p,q,r,s] = <a+_p b+_r b_s a_q>
                dm2ab = d2ab.transpose(0, 3, 1, 2)
                Hp = dict(H, C=(C[0], C[1]))
                extras["s2"] = self.spin_square_from_rdms(Hp, r1, dm2ab)
        gamma = ao_density(r1)
        shutil.rmtree(tag, ignore_errors=True)
        if self.projector == "mu":
            extras["E_proj"] = sum(float(np.einsum("ij,ji->", gamma[s], self.P_mu[s])) for s in SPINS)
        return self._assemble("dmrg-in-dft", e, gamma, extras, bool(H["mf"].converged), True,
                              (n, n), H["wmin"], H["wmax"])

    def write_fcidump(self, path, H=None):
        """FCIDUMP of the embedded active Hamiltonian (ecore = E_nuc + frozen core).
        Restricted: PySCF format. Unrestricted: block2 SZ format (IUHF), aa/ab/bb blocks."""
        H = H or getattr(self, "_last_H", None) or self.active_hamiltonian()
        n, ne = H["norb"], H["nelec"]
        if H["restricted"]:
            _fcidump.from_integrals(path, H["h1"][0], ao2mo.restore(8, H["eri"][0], n), n,
                                    int(sum(ne)), nuc=H["ecore"], ms=int(ne[0] - ne[1]))
            return path
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
        drv = DMRGDriver(scratch=f"{path}.tmp", symm_type=SymmetryTypes.SZ, n_threads=1, stack_mem=1 << 28)
        drv.initialize_system(n_sites=n, n_elec=int(sum(ne)), spin=int(ne[0] - ne[1]))
        drv.write_fcidump(tuple(H["h1"]), tuple(H["eri"]), ecore=float(H["ecore"]), filename=path, h1e_symm=True)
        import shutil; shutil.rmtree(f"{path}.tmp", ignore_errors=True)
        return path

    def run_casscf(self, ncas, nelecas, avas_labels=None, conv_tol=1e-10, max_cycle=200):
        """CASSCF-in-DFT in the embedded (environment-deleted) space. Restricted (alpha==beta)
        closed-shell references only. Active orbitals by AVAS on `avas_labels` (e.g.
        ["0 C 2p", "1 N 2p"]) if given, else the ncas orbitals around the Fermi level."""
        from pyscf import mcscf
        mf = self._embedded_scf("hf")
        C, wmin, wmax = self._kept_orbitals(mf)
        hh = self._h_emb()
        # spin-symmetric PROBLEM (same rule as active_hamiltonian): the kept space is the same
        # for both spins, so the alpha kept orbitals serve both even if the embedded UHF broke
        # spin symmetry (propionitrile, r >= 1.45 A); CASSCF re-optimises the orbitals anyway
        if not (self.n_act[0] == self.n_act[1] and self.n_env[0] == self.n_env[1]
                and np.allclose(hh[0], hh[1], atol=1e-10) and np.allclose(self.DB[0], self.DB[1], atol=1e-10)):
            raise NotImplementedError("run_casscf: spin-symmetric (closed-shell) problems only")
        h_emb = hh[0]
        m = self._embedded_mol()
        mr = scf.RHF(m)
        mr.get_hcore = lambda *a, **k: h_emb
        mr.energy_nuc = lambda *a, **k: self.E_nuc     # incl. nucleus-point-charge (QM/MM)
        mr.mo_coeff = C[0]
        mr.mo_occ = 2.0 * np.asarray(self._kept_occ[0])
        mr.mo_energy = np.zeros(C[0].shape[1])
        mr.converged = True
        mr.e_tot = float(mr.energy_tot(dm=mr.make_rdm1()))
        mc = mcscf.CASSCF(mr, ncas, nelecas)
        mc.conv_tol, mc.max_cycle_macro, mc.verbose = conv_tol, max_cycle, 0
        mo0 = C[0]
        if avas_labels:
            from pyscf.mcscf import avas
            nc_, ne_, mo0 = avas.avas(mr, avas_labels, canonicalize=False, verbose=0)
            self._avas_found = (int(nc_), int(np.sum(ne_)) if np.ndim(ne_) else int(ne_))
        mc.kernel(mo0)
        dm = mc.make_rdm1()                   # spin-summed AO 1-RDM
        gamma = (dm / 2.0, dm / 2.0)
        res = self._assemble("casscf-in-dft", float(mc.e_tot), gamma,
                             {"avas_found": getattr(self, "_avas_found", None),
                              "E_proj": sum(float(np.einsum("ij,ji->", gamma[s], self.P_mu[s]))
                                            for s in SPINS) if self.projector == "mu" else 0.0},
                             bool(mf.converged), bool(mc.converged), (C[0].shape[1],) * 2, wmin, wmax)
        return res
