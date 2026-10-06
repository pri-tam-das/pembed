"""A4 unit tests for the nonadditive exchange machinery (E0, E0b)."""
import numpy as np
import pytest

from pembed import ProjectionEmbedding
from pembed import nadd
from conftest import plain_rks, plain_uks

UH = 1e-6
SPLIT_FUNCS = ["hf", "b3lyp5", "b3lyp", "pbe0", "pbe", "bhandhlyp", "scan"]


# E0 -- exact exchange is bilinear: E_xx^nadd == E_x^HF[A+B] - E_x^HF[A] - E_x^HF[B]
@pytest.mark.parametrize("r,bs", [(1.0, False), (3.0, False), (3.0, True), (5.0, True)])
def test_E0_exx_nadd_bilinear(r, bs):
    k = plain_uks(r, "b3lyp5") if bs else plain_rks(r, "b3lyp5")
    emb = ProjectionEmbedding(k, [0, 1])
    A, B = emb.DA, emb.DB
    AB = tuple(A[s] + B[s] for s in (0, 1))
    lhs = nadd.exx_nadd(emb.ks, A, B)
    rhs = nadd.exact_exchange(emb.ks, AB) - nadd.exact_exchange(emb.ks, A) - nadd.exact_exchange(emb.ks, B)
    assert abs(lhs - rhs) <= 1 * UH, (lhs - rhs) / UH


# E0b -- the explicit x/c split reproduces PySCF's E_xc for each supported functional
@pytest.mark.parametrize("xc", SPLIT_FUNCS)
@pytest.mark.parametrize("r", [1.0, 3.0])
def test_E0b_xc_split_matches_pyscf(xc, r):
    k = plain_rks(r, xc) if xc != "scan" else plain_uks(r, "scan")
    emb = ProjectionEmbedding(k, [0, 1])
    for dm in (emb.D, emb.DA, emb.DB):
        assert abs(nadd.check_split(emb.ks, xc, dm)) <= 1e-8


def test_E0c_delta_x_vanishes_for_hf():
    """For an HF environment the DFT nonadditive exchange IS exact exchange: Delta_x = 0."""
    k = plain_rks(3.0, "hf")
    emb = ProjectionEmbedding(k, [0, 1])
    res = emb.run_fci()
    assert abs(emb.delta_x(res)) <= 1e-9


# ---------------------------------------------------------------- range separation (item 8)
from pyscf import dft as _dft, gto as _gto
from pembed import nadd as _nadd


def _oh_uks(xc):
    mol = _gto.M(atom="O 0 0 0; H 0 0 0.97", basis="6-31G*", spin=1, verbose=0)
    ks = _dft.UKS(mol, xc=xc); ks.conv_tol = 1e-10; ks.verbose = 0; ks.kernel()
    return ks


@pytest.mark.parametrize("xc", ["b3lyp5", "pbe0", "pbe", "bhandhlyp", "hf"])
def test_E0d_general_route_equals_split(xc):
    """E_x = E_xc - E_c (+ own exact exchange) reproduces the validated explicit split."""
    ks = _oh_uks(xc)
    dm = np.array(ks.make_rdm1())
    gA, gB = 0.6 * dm, 0.4 * dm
    assert abs(_nadd.ex_dft_general(ks, xc, dm) - _nadd.ex_dft(ks, xc, dm)) < 1e-9
    assert abs(_nadd.ex_dft_nadd_general(ks, xc, gA, gB) - _nadd.ex_dft_nadd(ks, xc, gA, gB)) < 1e-9


@pytest.mark.parametrize("xc", ["hse06", "b3lyp5", "r2scan"])
def test_E0e_exact_part_reproduces_pyscf_exc(xc):
    """The (omega, alpha, hyb) combination used for the functional's own exact exchange is the
    one PySCF's get_veff uses: exc(get_veff) == semilocal xc (numint) + exact_part."""
    ks = _oh_uks(xc)
    dm = np.array(ks.make_rdm1())
    ref = float(ks.get_veff(ks.mol, dm).exc)
    mine = _nadd.semilocal(ks, xc, dm) + _nadd.exact_part(ks, xc, dm)
    assert abs(mine - ref) < 1e-9, mine - ref


def test_E0f_hse06_short_range_exchange_is_bilinear():
    """E_x^SR,exact (HSE06's 0.25 K_sr) nonadditive part == -0.25 sum_s tr[gA K_sr(gB)]."""
    ks = _oh_uks("hse06")
    dm = np.array(ks.make_rdm1()); gA, gB = 0.7 * dm, 0.3 * dm
    f = lambda d: _nadd.exact_part(ks, "hse06", d)
    nadd = f(gA + gB) - f(gA) - f(gB)
    omega = ks._numint.rsh_and_hybrid_coeff("hse06", spin=1)[0]
    ksr = np.asarray(ks.get_k(ks.mol, gB, hermi=1)) - np.asarray(ks.get_k(ks.mol, gB, hermi=1, omega=omega))
    ref = -0.25 * sum(float(np.einsum("ij,ji->", gA[s], ksr[s])) for s in (0, 1))
    assert abs(nadd - ref) < 1e-10


def test_E0g_hse06_delta_x_in_embedding():
    """Delta_x with an HSE06 environment runs through ProjectionEmbedding and is finite;
    for an HF environment Delta_x vanishes identically."""
    from pembed import ProjectionEmbedding
    ks = _oh_uks("hse06")
    emb = ProjectionEmbedding(ks, [0], projector="huzinaga")
    dx = emb.delta_x(emb.run_fci())
    assert np.isfinite(dx) and abs(dx) > 1e-6
    ks_hf = _oh_uks("hf")
    emb_hf = ProjectionEmbedding(ks_hf, [0], projector="huzinaga")
    assert abs(emb_hf.delta_x(emb_hf.run_fci())) < 1e-10
