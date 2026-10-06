"""C-tests: MP2/CCSD/CCSD(T)-in-DFT (run_cc) and the fixed-Fock Huzinaga projector (O4 prep).

C1  all atoms active: run_cc = canonical UMP2 / UCCSD / UCCSD(T), open shell
C2  two active electrons: CCSD-in-DFT = FCI-in-DFT on the same embedded space (all projectors)
C3  'huzinaga_fixed' (Waldrop et al. JCC 2025 eq. 6) DFT-in-DFT = global DFT, closed + open shell
"""
import numpy as np
import pytest
from pyscf import cc, dft, gto, mp, scf

from pembed import ProjectionEmbedding

UH = 1e-6


def oh(basis="6-31G"):
    return gto.M(atom="O 0 0 0; H 0 0 0.97", basis=basis, spin=1, verbose=0)


def water(basis="6-31G"):
    return gto.M(atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0", basis=basis, verbose=0)


def conv(mf):
    mf.conv_tol = 1e-11; mf.max_cycle = 300; mf.verbose = 0; mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope="module")
def oh_ref():
    mol = oh()
    hf = conv(scf.UHF(mol))
    e_mp2 = hf.e_tot + mp.UMP2(hf).kernel()[0]
    c = cc.UCCSD(hf); c.conv_tol = 1e-10; c.verbose = 0; c.kernel()
    return mol, {"mp2": e_mp2, "ccsd": hf.e_tot + c.e_corr, "ccsd(t)": hf.e_tot + c.e_corr + c.ccsd_t()}


@pytest.mark.parametrize("method", ["mp2", "ccsd", "ccsd(t)"])
def test_C1_all_active_equals_canonical(oh_ref, method):
    mol, ref = oh_ref
    ks = conv(dft.UKS(mol, xc="b3lyp5"))
    res = ProjectionEmbedding(ks, [0, 1], projector="huzinaga").run_cc(method, conv_tol=1e-10)
    assert res.n_env == (0, 0)
    assert abs(res.E_eq1 - ref[method]) <= 1 * UH, (res.E_eq1 - ref[method]) / UH


@pytest.mark.parametrize("projector,kw", [("huzinaga_fixed", {}), ("mu", {"mu": 1e6})])
def test_C2_two_electron_ccsd_equals_fci(projector, kw):
    ks = conv(dft.RKS(water(), xc="b3lyp5"))
    emb = ProjectionEmbedding(ks, [1], projector=projector, **kw)
    assert emb.n_act == (1, 1)
    r_cc = emb.run_cc("ccsd", conv_tol=1e-11)
    r_fci = emb.run_fci()
    assert abs(r_cc.E_wf_emb - r_fci.E_wf_emb) <= 1 * UH, (r_cc.E_wf_emb - r_fci.E_wf_emb) / UH
    # the projector-inclusive eq. 1 does not depend on gamma, so it must agree exactly
    d = r_cc.extras["E_eq1_proj_incl"] - r_fci.extras["E_eq1_proj_incl"]
    assert abs(d) <= 1 * UH, d / UH


def test_C2b_dynamic_huzinaga_collapse_is_caught():
    """Water with only H active: the dynamic-Huzinaga embedded HF puts the electron in an
    environment orbital; FCI survives (kept space only), HF/CC must refuse."""
    ks = conv(dft.RKS(water(), xc="b3lyp5"))
    emb = ProjectionEmbedding(ks, [1], projector="huzinaga")
    with pytest.raises(RuntimeError, match="collapsed into the environment"):
        emb.run_cc("ccsd")
    with pytest.raises(RuntimeError, match="collapsed into the environment"):
        emb.run_hf_in_dft()
    ref = ProjectionEmbedding(ks, [1], projector="mu", mu=1e8).run_fci()
    assert abs(emb.run_fci().E_eq1 - ref.E_eq1) <= 2 * UH


@pytest.mark.parametrize("mol_fn,cls,act", [(water, dft.RKS, [0]), (water, dft.RKS, [1]),
                                            (oh, dft.UKS, [0]), (oh, dft.UKS, [1])])
def test_C3_huzinaga_fixed_dft_in_dft_exact(mol_fn, cls, act):
    ks = conv(cls(mol_fn(), xc="b3lyp5"))
    res = ProjectionEmbedding(ks, act, projector="huzinaga_fixed").run_dft_in_dft()
    assert res.emb_scf_converged
    assert abs(res.E_eq1 - ks.e_tot) <= 1 * UH, (res.E_eq1 - ks.e_tot) / UH
