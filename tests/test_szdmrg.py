"""Z-tests (plan items 7 + 9): unrestricted active Hamiltonians, frozen core, SZ-mode DMRG-in-DFT,
<S^2> from RDMs in unrestricted orbitals, unrestricted FCIDUMP."""
import numpy as np
import pytest
from pyscf import ao2mo, dft, fci, gto, scf

from pembed import ProjectionEmbedding

UH = 1e-6
BD = (100, 200, 300)


def oh(basis="6-31G"):
    return gto.M(atom="O 0 0 0; H 0 0 0.97", basis=basis, spin=1, verbose=0)


def water(basis="6-31G"):
    return gto.M(atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0", basis=basis, verbose=0)


def conv(mf):
    mf.conv_tol = 1e-11; mf.max_cycle = 300; mf.verbose = 0; mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope="module")
def oh_ks():
    return conv(dft.UKS(oh(), xc="b3lyp5"))


def test_Z1_all_active_unrestricted_fci_is_spin_pure(oh_ks):
    """Complete space: FCI in UHF orbitals is an exact eigenstate -> <S^2> = 3/4."""
    res = ProjectionEmbedding(oh_ks, [0, 1], projector="huzinaga").run_fci()
    assert res.n_env == (0, 0)
    assert abs(res.extras["s2_fci"] - 0.75) < 1e-6, res.extras["s2_fci"]
    hf = conv(scf.UHF(oh_ks.mol))
    e_ref = fci.FCI(hf).kernel()[0]
    assert abs(res.E_eq1 - e_ref) <= 1 * UH


@pytest.mark.parametrize("ncore", [0, 1])
def test_Z2_sz_dmrg_equals_unrestricted_fci(oh_ks, ncore, tmp_path):
    emb = ProjectionEmbedding(oh_ks, [0], projector="huzinaga")
    r_fci = emb.run_fci(ncore=ncore)
    r_dmrg = emb.run_dmrg(bond_dims=BD, ncore=ncore, scratch=str(tmp_path / "d"), n_threads=4, stack_mem=2 << 30)
    assert r_dmrg.extras["mode"] == "SZ"
    assert abs(r_dmrg.E_eq1 - r_fci.E_eq1) <= 1 * UH, (r_dmrg.E_eq1 - r_fci.E_eq1) / UH
    assert abs(r_dmrg.extras["s2"] - r_fci.extras["s2_fci"]) < 1e-5
    for s in (0, 1):
        assert np.abs(r_dmrg.rdm1_ao[s] - r_fci.rdm1_ao[s]).max() < 1e-5


def test_Z2b_s2_formula_matches_pyscf(oh_ks):
    """No core: our <S^2> from RDMs == PySCF fci.spin_op.spin_square with (Ca, Cb) and S."""
    emb = ProjectionEmbedding(oh_ks, [0], projector="huzinaga")
    H = emb.active_hamiltonian()
    s = fci.direct_uhf.FCISolver(); s.conv_tol = 1e-12
    e, c = s.kernel(H["h1"], H["eri"], H["norb"], H["nelec"], ecore=H["ecore"])
    ref = fci.spin_op.spin_square(c, H["norb"], H["nelec"], mo_coeff=H["C"], ovlp=emb.S)[0]
    r1, r2 = s.make_rdm12s(c, H["norb"], H["nelec"])
    assert abs(emb.spin_square_from_rdms(H, r1, r2[1]) - ref) < 1e-8


@pytest.mark.parametrize("ncore", [0, 1])
def test_Z3_s2_of_a_determinant_with_core(oh_ks, ncore):
    """Core formula: the embedded UHF determinant written with a frozen core gives mf.spin_square."""
    emb = ProjectionEmbedding(oh_ks, [0], projector="huzinaga")
    mf = emb._embedded_scf("hf")
    H = emb.active_hamiltonian(mf=mf, ncore=ncore)
    n = H["norb"]
    occ = [np.zeros(n), np.zeros(n)]
    for s in (0, 1):
        occ[s][: H["nelec"][s]] = 1.0
    g1 = (np.diag(occ[0]), np.diag(occ[1]))
    dm2ab = np.einsum("pq,rs->pqrs", g1[0], g1[1])
    assert abs(emb.spin_square_from_rdms(H, g1, dm2ab) - mf.spin_square()[0]) < 1e-8


def test_Z4_forced_sz_equals_su2_closed_shell(tmp_path):
    ks = conv(dft.RKS(water(), xc="b3lyp5"))
    emb = ProjectionEmbedding(ks, [0], projector="huzinaga")
    kw = dict(bond_dims=BD, scratch=str(tmp_path / "d"), n_threads=4, stack_mem=2 << 30, cl_shells=0)
    r_su2 = emb.run_dmrg(**kw)
    r_sz = emb.run_dmrg(spin_adapted=False, **kw)
    assert r_su2.extras["mode"] == "SU2" and r_sz.extras["mode"] == "SZ"
    assert abs(r_su2.E_eq1 - r_sz.E_eq1) <= 2 * UH
    assert abs(r_sz.extras["s2"]) < 1e-5


def test_Z5_unrestricted_fcidump_roundtrip(oh_ks, tmp_path):
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    emb = ProjectionEmbedding(oh_ks, [0], projector="huzinaga")
    H = emb.active_hamiltonian(ncore=1)
    path = emb.write_fcidump(str(tmp_path / "FCIDUMP"), H=H)
    drv = DMRGDriver(scratch=str(tmp_path / "r"), symm_type=SymmetryTypes.SZ, n_threads=1, stack_mem=1 << 28)
    drv.read_fcidump(filename=path, iprint=0)
    assert drv.n_sites == H["norb"] and abs(drv.ecore - H["ecore"]) < 1e-10
    for s in (0, 1):
        assert np.abs(np.asarray(drv.h1e[s]) - H["h1"][s]).max() < 1e-10
    for k in range(3):
        assert np.abs(np.asarray(drv.g2e[k]).reshape(H["eri"][k].shape) - H["eri"][k]).max() < 1e-10


def test_Z6_open_shell_environment_equalized(oh_ks, tmp_path):
    """OH with only H active: the environment is an open-shell O fragment (n_env 4a/3b), so the
    kept spaces differ by one orbital; equalization pads the smaller spin with a dummy site."""
    emb = ProjectionEmbedding(oh_ks, [1], projector="huzinaga_fixed")
    assert emb.n_env[0] != emb.n_env[1]
    H = emb.active_hamiltonian()
    assert sorted(H["padded"]) == [0, 1] and H["trimmed"] == (0, 0)
    r_fci = emb.run_fci()
    r_dmrg = emb.run_dmrg(bond_dims=BD, scratch=str(tmp_path / "d"), n_threads=4, stack_mem=2 << 30)
    assert abs(r_dmrg.E_eq1 - r_fci.E_eq1) <= 1 * UH


@pytest.mark.parametrize("r", [1.0, 3.0, 5.0])
@pytest.mark.parametrize("spin_adapted", [None, False])
def test_Z7_dmrg_start_is_robust_over_seeds(r, spin_adapted, tmp_path):
    """Single-root starts were unsafe (water/STO-3G, R7 setup): a uniform random MPS converged
    (dw ~ 1e-21) to an excited state 0.379 Eh high in 4/10 seeds at 1.0 A; HF and random starts
    stuck in an excited singlet 4.9 / 5.0 mEh high at 3.0 / 5.0 A. HF + state-averaged warm-up
    fixes it (0/60). SZ fixes only M_S, so it is compared with the lowest FCI state of any spin
    (the triplet at 3.0 A, 0.065 mEh below the singlet)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import plain_rks
    emb = ProjectionEmbedding(plain_rks(r, "b3lyp5"), [0, 1], projector="huzinaga")
    ref = (emb.run_fci(ss=0.0) if spin_adapted is None else emb.run_fci()).E_eq1
    for seed in (12, 15, 16, 18, 20):
        d = emb.run_dmrg(bond_dims=(100, 200), n_threads=2, seed=seed, spin_adapted=spin_adapted,
                         scratch=str(tmp_path / f"d{seed}"))
        assert abs(d.E_eq1 - ref) <= 1 * UH, (seed, d.E_eq1 - ref)
        assert d.extras["M_series"][-1][1] <= d.extras["E_ref_det"] + 1e-9



def test_Z8_padding_invariance(oh_ks, tmp_path):
    """Plan decision 2026-09-28 (point 3): the energy must not depend on how the per-spin site
    counts are equalized, as long as the added orbital is inert. Dummy-site padding (default) vs
    padding with the mu-shifted environment orbital: <= 1 uEh. Trimming removes a genuine
    correlating virtual of the larger spin and DOES change the energy (reported, not required)."""
    emb = ProjectionEmbedding(oh_ks, [1], projector="mu", mu=1e6)
    assert emb.n_env[0] != emb.n_env[1]
    from pyscf import fci as _fci
    def e_fci(eq):
        H = emb.active_hamiltonian(equalize=eq)
        s = _fci.direct_uhf.FCISolver(); s.conv_tol = 1e-12
        return float(s.kernel(H["h1"], H["eri"], H["norb"], H["nelec"], ecore=H["ecore"])[0]), H
    e_pad, H = e_fci("pad"); e_env, He = e_fci("pad_env"); e_trim, Ht = e_fci("trim")
    assert any(H["padded"]) and any(He["padded"]) and any(Ht["trimmed"])
    assert abs(e_pad - e_env) <= 1 * UH, (e_pad - e_env) / UH
    assert e_trim > e_pad - 1e-9                     # a smaller correlation space cannot be lower
    r = emb.run_dmrg(bond_dims=BD, scratch=str(tmp_path / "d"), n_threads=4, stack_mem=2 << 30)
    assert abs(r.E_wf_emb - e_pad) <= 1 * UH
    assert not r.extras["start_disagree"]


def test_Z9_natural_orbital_space_complete_is_exact(oh_ks):
    """no_select keeping ALL natural orbitals (lo < 0, hi > 1) is only a rotation of the kept
    space -> FCI unchanged; exercises the per-spin core (frozen core + frozen NOs) bookkeeping."""
    emb = ProjectionEmbedding(oh_ks, [0], projector="mu", mu=1e8)
    e_ref = emb.run_fci(ncore=1).E_eq1
    r = emb.run_fci(ncore=1, no_select=dict(lo=-1.0, hi=2.0))
    assert abs(r.E_eq1 - e_ref) <= 1 * UH, (r.E_eq1 - e_ref) / UH


def test_Z10_natural_orbital_truncation(oh_ks, tmp_path):
    """Truncated MP2-NO space: DMRG == FCI in that space; the truncation error is small and
    variationally bounded (the NO-space energy lies above the full-space one)."""
    emb = ProjectionEmbedding(oh_ks, [0], projector="mu", mu=1e8)
    full = emb.run_fci(ncore=1)
    sel = dict(lo=2e-3, hi=0.998)
    H = emb.active_hamiltonian(ncore=1, no_select=sel)
    n_full = emb.active_hamiltonian(ncore=1)["norb"]
    assert H["norb"] < n_full and any(H["ncore"])
    r_fci = emb.run_fci(ncore=1, no_select=sel)
    r_dm = emb.run_dmrg(bond_dims=BD, ncore=1, no_select=sel, scratch=str(tmp_path / "d"), n_threads=4, stack_mem=2 << 30)
    assert abs(r_dm.E_wf_emb - r_fci.E_wf_emb) <= 1 * UH
    assert abs(r_dm.extras["s2"] - 0.75) < 0.05
    # truncation loses dynamic correlation (15 mEh here at lo = 2e-3): it must shrink monotonically
    # toward the full kept-space energy as the threshold is loosened, from above (variational)
    e = [emb.run_fci(ncore=1, no_select=dict(lo=lo, hi=1 - lo)).E_wf_emb for lo in (2e-3, 5e-4, 1e-4)]
    assert e[0] >= e[1] - 1e-9 >= e[2] - 2e-9 >= full.E_wf_emb - 3e-9, (e, full.E_wf_emb)


def test_Z11_per_rung_densities_and_extrapolation_error(oh_ks, tmp_path):
    """Capped-M protocol: on_rung sees every bond dimension of the continued state with its AO
    density; the last one is the returned density; >= 3 rungs give E(dw -> 0) and its error."""
    emb = ProjectionEmbedding(oh_ks, [0], projector="mu", mu=1e8)
    sel = dict(lo=2e-3, hi=0.998)
    seen = []
    r = emb.run_dmrg(bond_dims=(4, 8, 16), ncore=1, no_select=sel, scratch=str(tmp_path / "d"), n_threads=4,
                     stack_mem=2 << 30, on_rung=lambda M, E, dw, g: seen.append((M, E, dw, g)))
    assert [s[0] for s in seen] == [m for m, _, _ in r.extras["M_series"]] == [4, 8, 16]
    assert [s[1] for s in seen] == [e for _, e, _ in r.extras["M_series"]]
    S = oh_ks.mol.intor("int1e_ovlp")
    for _, _, _, g in seen:                        # every per-rung density holds the active + core electrons
        assert abs(sum(np.trace(g[s] @ S) for s in (0, 1)) - sum(np.trace(r.rdm1_ao[s] @ S) for s in (0, 1))) < 1e-8
    assert all(np.allclose(seen[-1][3][s], r.rdm1_ao[s], atol=1e-10) for s in (0, 1))
    assert np.isfinite(r.extras["E_extrap_err"]) and r.extras["E_extrap_err"] >= 0
    assert r.extras["E_extrap"] <= r.extras["M_series"][-1][1] + 1e-6   # extrapolates downward


def test_Z12_mp2_fno_correction(oh_ks):
    """MP2 correction for the discarded natural orbitals (dE = E_MP2(full kept) - E_MP2(truncated)):
    zero without truncation; negative with truncation; and E_MP2(truncated) equals UMP2 computed
    independently on the truncated-space Hamiltonian the DMRG/FCI solvers see. (On this toy the
    discarded orbitals hold correlation MP2 misses: FCI loses 15 mEh, MP2 3.5 mEh, at lo = 2e-3.)"""
    emb = ProjectionEmbedding(oh_ks, [0], projector="mu", mu=1e8)
    none = emb.active_hamiltonian(ncore=1, no_select=dict(lo=1e-14, hi=1 - 1e-14))["no_info"]
    assert abs(none["dE_mp2_fno"]) < 1e-9, none["dE_mp2_fno"]
    for lo in (2e-3, 1e-4):
        H = emb.active_hamiltonian(ncore=1, no_select=dict(lo=lo, hi=1 - lo))
        info, (na, nb), n = H["no_info"], H["nelec"], H["norb"]
        assert info["e_mp2_trunc"] > info["e_mp2"] and info["dE_mp2_fno"] < 0
        (ha, hb), (gaa, gab, gbb) = H["h1"], H["eri"]             # kept NOs, occupied first
        oa, ob = np.arange(na), np.arange(nb)
        fa = (ha + np.einsum("pqjj->pq", gaa[:, :, oa][:, :, :, oa]) - np.einsum("pjjq->pq", gaa[:, oa][:, :, oa])
              + np.einsum("pqjj->pq", gab[:, :, ob][:, :, :, ob]))
        fb = (hb + np.einsum("pqjj->pq", gbb[:, :, ob][:, :, :, ob]) - np.einsum("pjjq->pq", gbb[:, ob][:, :, ob])
              + np.einsum("jjpq->pq", gab[oa][:, oa]))
        def canon(f, no):
            eo, uo = np.linalg.eigh(f[:no, :no]); ev, uv = np.linalg.eigh(f[no:, no:])
            U = np.zeros_like(f); U[:no, :no] = uo; U[no:, no:] = uv
            return np.concatenate([eo, ev]), U
        (ea, Ua), (eb, Ub) = canon(fa, na), canon(fb, nb)
        tr = lambda g, A, B: np.einsum("pqrs,pi,qj,rk,sl->ijkl", g, A, A, B, B)
        def den(e1, n1, e2, n2):
            return e1[:n1, None, None, None] - e1[None, n1:, None, None] + e2[None, None, :n2, None] - e2[None, None, None, n2:]
        def same(G, e, no):
            ov = G[:no, no:, :no, no:]
            return 0.5 * np.sum(ov * (ov - ov.transpose(0, 3, 2, 1)) / den(e, no, e, no))
        ov = tr(gab, Ua, Ub)[:na, na:, :nb, nb:]
        e2 = same(tr(gaa, Ua, Ua), ea, na) + same(tr(gbb, Ub, Ub), eb, nb) + np.sum(ov * ov / den(ea, na, eb, nb))
        assert abs(e2 - info["e_mp2_trunc"]) < 1e-8, (e2, info["e_mp2_trunc"])


def test_Z13_plain_uhf_reference(tmp_path):
    """A global scf.UHF reference (the production 'hf' environment): its veff carries no
    .ecoul/.exc. E_dft must still give the UHF energy, and a correlated run must assemble
    (the HF-environment DMRG runs crashed in _assemble before this fix)."""
    mf = conv(scf.UHF(oh()))
    emb = ProjectionEmbedding(mf, [0], projector="mu", mu=1e8)
    assert abs(emb.E_dft(emb.D) + mf.energy_nuc() - mf.e_tot) < 1e-8
    sel = dict(lo=2e-3, hi=0.998)
    r_fci = emb.run_fci(ncore=1, no_select=sel)
    r_dm = emb.run_dmrg(bond_dims=BD, ncore=1, no_select=sel, scratch=str(tmp_path / "d"), n_threads=4, stack_mem=2 << 30)
    assert np.isfinite(r_dm.E_eq1) and abs(r_dm.E_wf_emb - r_fci.E_wf_emb) <= 1 * UH
    dft_in_dft = emb.run_dft_in_dft()                      # HF-in-HF: exact
    assert abs(dft_in_dft.E_eq1 - mf.e_tot) < 1e-6


def test_Z14_s2_penalty_targets_low_spin(tmp_path):
    """The doublet trap: in SZ mode M_S = S_low also contains a higher-spin
    ground state's component. O2 (triplet ground state), closed-shell RHF orbitals, all atoms
    active, M_S = 0: plain SZ DMRG finds the TRIPLET (<S^2> = 2, = triplet FCI); s2_penalty=(lam, 0)
    gives the singlet, equal to spin-projected FCI in the same space. S != |M_S| is refused."""
    mol = gto.M(atom="O 0 0 0; O 0 0 1.21", basis="sto-3g", spin=0, verbose=0)
    mf = scf.RHF(mol); mf.conv_tol = 1e-11; mf.kernel()
    emb = ProjectionEmbedding(mf, [0, 1], projector="mu", mu=1e8)
    H = emb.active_hamiltonian(ncore=2)
    n, ne = H["norb"], H["nelec"]
    ref = {}
    for S in (0.0, 1.0):
        sol = fci.addons.fix_spin_(fci.direct_spin1.FCI(), ss=S * (S + 1), shift=1.0); sol.conv_tol = 1e-12; sol.nroots = 4
        es, cs = sol.kernel(H["h1"][0], H["eri"][0], n, ne, ecore=H["ecore"])
        ok = [e for e, c in zip(np.atleast_1d(es), cs) if abs(fci.spin_op.spin_square0(c, n, ne)[0] - S * (S + 1)) < 1e-4]
        ref[S] = min(ok)
    assert ref[1.0] < ref[0.0]                                # triplet ground state
    kw = dict(bond_dims=(100, 200, 300), ncore=2, spin_adapted=False, n_threads=4, stack_mem=2 << 30)
    plain = emb.run_dmrg(scratch=str(tmp_path / "p"), **kw)
    assert abs(plain.extras["s2"] - 2.0) < 1e-3 and abs(plain.E_wf_emb - ref[1.0]) < 1e-6
    r = emb.run_dmrg(scratch=str(tmp_path / "s0"), s2_penalty=(1.0, 0.0), **kw)
    assert abs(r.extras["s2"]) < 1e-3, r.extras["s2"]
    assert abs(r.E_wf_emb - ref[0.0]) < 1e-6, r.E_wf_emb - ref[0.0]
    # a linear penalty can only select S = |M_S|: higher targets are refused, not silently wrong
    with pytest.raises(ValueError):
        emb.run_dmrg(scratch=str(tmp_path / "s1"), s2_penalty=(1.0, 1.0), **kw)


def test_Z15_s2_penalty_with_unrestricted_orbitals(tmp_path):
    """Doublet trap with alpha != beta orbitals (the Mn(IV) case). N atom (quartet ground state),
    UHF orbitals, M_S = 1/2, every orbital active: plain SZ DMRG lands on the quartet's M_S = 1/2
    component (<S^2> = 3.75); s2_penalty=(lam, 1/2) gives the lowest doublet. Reference: full CI in
    the complete space (orbital-invariant) from ROHF orbitals with a spin-pure doublet target."""
    mol = gto.M(atom="N 0 0 0", basis="sto-3g", spin=1, verbose=0)
    mf = conv(scf.UHF(mol))
    emb = ProjectionEmbedding(mf, [0], projector="mu", mu=1e8)
    H = emb.active_hamiltonian(ncore=0)
    assert not H["restricted"]
    ro = scf.ROHF(mol); ro.conv_tol = 1e-11; ro.kernel()
    n = ro.mo_coeff.shape[1]; ne = (4, 3)
    h1 = ro.mo_coeff.T @ ro.get_hcore() @ ro.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, ro.mo_coeff), n)
    sol = fci.addons.fix_spin_(fci.direct_spin1.FCI(), ss=0.75, shift=1.0); sol.conv_tol = 1e-12; sol.nroots = 6
    es, cs = sol.kernel(h1, eri, n, ne, ecore=mol.energy_nuc())
    e_doublet = min(e for e, c in zip(es, cs) if abs(fci.spin_op.spin_square0(c, n, ne)[0] - 0.75) < 1e-4)
    kw = dict(bond_dims=(50, 100), ncore=0, n_threads=4, stack_mem=2 << 30)
    plain = emb.run_dmrg(scratch=str(tmp_path / "p"), **kw)
    assert abs(plain.extras["s2"] - 3.75) < 1e-3, plain.extras["s2"]
    r = emb.run_dmrg(scratch=str(tmp_path / "d"), s2_penalty=(1.0, 0.5), **kw)
    assert abs(r.extras["s2"] - 0.75) < 1e-3, r.extras["s2"]
    assert r.E_wf_emb > plain.E_wf_emb
    assert abs(r.E_wf_emb - e_doublet) < 1e-6, r.E_wf_emb - e_doublet
