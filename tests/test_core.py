"""Exactness and regression tests R1-R6. Water / STO-3G, restricted global."""
import numpy as np
import pytest
from pyscf.tools import fcidump
from pyscf import fci

from pembed import ProjectionEmbedding
from conftest import TARGETS, plain_rks, full_fci

UH = 1e-6
PROJ = ("mu", "huzinaga")


# R1 -- DFT-in-DFT reproduces the global DFT energy (a theorem).
# Huzinaga is formally exact. The mu-shift projector is exact only as mu -> inf: its
# residual is O(1/mu) (measured: -1.75 uH at mu=1e6, -0.0175 uH at 1e8; SCF stops
# converging at 1e9). So the 1 uH bound is tested at mu=1e8, and the 1/mu law itself
# is tested separately in R1b. (The nbed comparison R2 keeps mu=1e6, as nbed and the paper.)
MU_EXACT = 1e8


@pytest.mark.parametrize("xc", ["b3lyp5", "pbe0", "hf"])
@pytest.mark.parametrize("proj", PROJ)
@pytest.mark.parametrize("r", TARGETS)
def test_R1_dft_in_dft_exact(xc, proj, r):
    ks = plain_rks(r, xc)
    res = ProjectionEmbedding(ks, [0, 1], projector=proj, mu=MU_EXACT).run_dft_in_dft()
    assert res.emb_scf_converged
    # density converges only to ~sqrt(conv_tol) ~ 1e-5 (SCF gradient threshold); the energy is
    # second-order in it, hence exact to 1e-8 Ha. Bound the density at 1e-4.
    assert max(res.extras["dm_diff"]) < 1e-4, res.extras["dm_diff"]
    assert abs(res.E_eq1 - ks.e_tot) <= 1 * UH, (res.E_eq1 - ks.e_tot) / UH


@pytest.mark.parametrize("xc,r", [("b3lyp5", 1.0), ("hf", 1.5), ("b3lyp5", 3.0)])
def test_R1b_mu_shift_residual_scales_as_inverse_mu(xc, r):
    ks = plain_rks(r, xc)
    d6, d7 = (ProjectionEmbedding(ks, [0, 1], mu=m).run_dft_in_dft().E_eq1 - ks.e_tot
              for m in (1e6, 1e7))
    assert d6 < 0 and d7 < 0                   # variational leakage lowers the energy
    assert 9.5 < d6 / d7 < 10.5, d6 / d7


# R2 / R3 -- FCI-in-DFT vs frozen patched-nbed (Table V: changing OH active; VI: fixed OH)
#
# Comparison rules, each verified numerically (see ../DESIGN.md "Regression notes"):
#  * mu projector: nbed INCLUDES the projector energy tr[gamma mu P]; Monino eq. 1 (ours) does
#    not. Both are O(1/mu) and agree as mu -> inf, so compare nbed with E_eq1 + E_proj.
#  * r >= 3 A (singlet/triplet near-degenerate): our FCI is the verified lowest of 3 roots;
#    nbed's CASSCF-wrapped FCI (conv 1e-8) can stop above the eigenvalue (PBE0, Table VI,
#    3.0 A: +8.8 uH, matching no root). Require ours <= nbed + 5 uH and nbed - ours <= 1 mH.
#  * r = 0.4 A, Table VI, mu: residual finite-mu implementation difference (-6.0 uH at mu=1e6,
#    -0.36 uH at mu=1e8, i.e. it vanishes as mu -> inf); covered live by R2b instead.
KNOWN_FINITE_MU = {("VI", "mu", 0.4)}


@pytest.mark.parametrize("xc", ["b3lyp5", "pbe0", "hf"])
@pytest.mark.parametrize("table,order", [("V", "act"), ("VI", "fix")])
@pytest.mark.parametrize("proj", PROJ)
@pytest.mark.parametrize("r", TARGETS)
def test_R2_R3_vs_nbed(nbed_frozen, xc, table, order, proj, r):
    if (table, proj, r) in KNOWN_FINITE_MU:
        pytest.skip("finite-mu implementation difference; tested at mu=1e8 in R2b")
    ref = nbed_frozen[f"{xc}|Table{table}|{proj}"][str(r)]
    ks = plain_rks(r, xc, order)
    # standing rule 3: same global SCF solution before comparing
    assert abs(ks.e_tot - ref["E_global"]) <= 1 * UH, "different global SCF solution"
    res = ProjectionEmbedding(ks, [0, 1], projector=proj).run_fci()
    assert res.solver_converged and res.emb_scf_converged
    ours = res.E_eq1 + res.extras.get("E_proj", 0.0)       # nbed's energy convention
    d = ours - ref["e_fci"]
    if r >= 3.0:
        assert d <= 5 * UH, f"ours above nbed by {d/UH:.2f} uH"
        assert -d <= 1e-3, f"nbed above ours by {-d/UH:.2f} uH"
    else:
        assert abs(d) <= 5 * UH, d / UH


@pytest.mark.parametrize("xc,table,order", [("b3lyp5", "VI", "fix"), ("b3lyp5", "V", "act")])
def test_R2b_compressed_live_vs_nbed_large_mu(xc, table, order):
    """At mu = 1e8 the two codes must agree at the most compressed geometry."""
    import sys, os
    # optional cross-check against the patched reference implementation (nbed + nbed_patches, lmo_common):
    # point PEMBED_NBED_TESTS at the folder holding those helpers; skipped otherwise
    if os.environ.get("PEMBED_NBED_TESTS"):
        sys.path.insert(0, os.environ["PEMBED_NBED_TESTS"])
    nbed_patches = pytest.importorskip("nbed_patches")  # noqa: F841
    from nbed.driver import NbedDriver
    from lmo_common import make_cfg
    ks = plain_rks(0.4, xc, order)
    d = NbedDriver(make_cfg(0.4, xc, order=order, projector="mu", mu_level_shift=1e8))
    d._global_ks = ks; d.embed()
    res = ProjectionEmbedding(ks, [0, 1], mu=1e8).run_fci()
    assert abs(res.E_eq1 + res.extras["E_proj"] - d.mu["e_fci"]) <= 1 * UH


# R4 -- all atoms active: FCI-in-DFT must equal full-system FCI.
# Compared state-by-state: the singlet (ss=0) and the lowest root, each from 3 Davidson roots
# (single-root Davidson misses the singlet ground state at 3.0 A by ~190 uH -- see run_fci).
@pytest.mark.parametrize("xc", ["b3lyp5", "pbe0"])
@pytest.mark.parametrize("r", TARGETS)
def test_R4_all_active(xc, r):
    ks = plain_rks(r, xc)
    emb = ProjectionEmbedding(ks, [0, 1, 2])
    res_s, res_l = emb.run_fci(ss=0.0), emb.run_fci()
    assert res_s.n_env == (0, 0)
    ref_s, ref_l = full_fci(r, ss=0.0), full_fci(r)
    assert abs(res_s.E_eq1 - ref_s) <= 1 * UH, (res_s.E_eq1 - ref_s) / UH
    assert abs(res_l.E_eq1 - ref_l) <= 1 * UH, (res_l.E_eq1 - ref_l) / UH


# R5 -- FCIDUMP round trip
@pytest.mark.parametrize("r", [1.0, 3.0])
def test_R5_fcidump_roundtrip(tmp_path, r):
    ks = plain_rks(r, "b3lyp5")
    emb = ProjectionEmbedding(ks, [0, 1])
    res = emb.run_fci()
    path = emb.write_fcidump(str(tmp_path / "FCIDUMP"))
    d = fcidump.read(path)
    na = (d["NELEC"] + d["MS2"]) // 2
    s = fci.direct_spin1.FCI(); s.conv_tol = 1e-12; s.nroots = 3   # same root rule as run_fci
    e = min(s.kernel(d["H1"], d["H2"], d["NORB"], (na, d["NELEC"] - na), ecore=d["ECORE"])[0])
    assert abs(e - res.E_wf_emb) <= 1 * UH, (e - res.E_wf_emb) / UH


# R6 -- eq. 1 (without its trace term) equals the decomposed eq. 18 form
@pytest.mark.parametrize("xc", ["b3lyp5", "hf"])
@pytest.mark.parametrize("r", [1.0, 3.0, 5.0])
def test_R6_eq1_equals_eq18(xc, r):
    res = ProjectionEmbedding(plain_rks(r, xc), [0, 1]).run_fci()
    assert abs((res.E_eq1 - res.trace_term) - res.E_eq18) <= 1 * UH


# R7 -- DMRG solver: energy and 1-RDM (through Delta_x) equal singlet FCI
@pytest.mark.parametrize("r", [1.0, 3.0])
def test_R7_dmrg_matches_fci(r):
    emb = ProjectionEmbedding(plain_rks(r, "b3lyp5"), [0, 1], projector="huzinaga")
    f, d = emb.run_fci(ss=0.0), emb.run_dmrg(bond_dims=(100, 200), n_threads=2)
    assert abs(d.E_eq1 - f.E_eq1) <= 1 * UH
    assert abs(emb.delta_x(d) - emb.delta_x(f)) <= 1 * UH
