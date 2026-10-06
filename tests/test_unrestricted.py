"""A2 -- broken-symmetry (spin-polarised, n_alpha == n_beta) water: tests U1-U4.

This is exactly the case where nbed 0.0.10 failed (T1b): spin-summed xc_cross, plus an
unseeded embedded SCF. Here the reference is the fully broken-symmetry UKS branch
(fragment guess, <S^2> ~ 1 at 5 A), identical to the T1b/T1c reference.
"""
import json
import os

import numpy as np
import pytest
from pyscf import dft

from pembed import ProjectionEmbedding
from conftest import DATA, TARGETS, plain_rks, plain_uks, water

UH = 1e-6
BS_FUNCS = ["b3lyp5", "pbe0", "hf", "pbe", "scan"]


def _t1c(xc):
    with open(os.path.join(DATA, f"t1c_{xc}.json")) as f:
        return json.load(f)["n=(4,4)"]["rows"]


# U1 -- closed-shell water through the unrestricted path == restricted-input results
@pytest.mark.parametrize("r", [0.798954, 1.0, 2.0])
def test_U1_closed_shell_uks_equals_rks(r):
    k_r = plain_rks(r, "b3lyp5")
    k_u = dft.UKS(water(r)); k_u.xc = "b3lyp5"; k_u.conv_tol = 1e-11; k_u.verbose = 0
    k_u.kernel(dm0=np.array([k_r.make_rdm1() / 2] * 2))
    assert abs(k_u.e_tot - k_r.e_tot) <= 1 * UH
    a = ProjectionEmbedding(k_r, [0, 1]).run_fci()
    b = ProjectionEmbedding(k_u, [0, 1]).run_fci()
    assert abs(a.E_eq1 - b.E_eq1) <= 1 * UH, (a.E_eq1 - b.E_eq1) / UH


# U3 -- same SCF solution as the T1 broken-symmetry reference (energy AND <S^2>)
@pytest.mark.parametrize("xc", BS_FUNCS)
@pytest.mark.parametrize("r", TARGETS)
def test_U3_same_solution_as_T1_reference(xc, r):
    ref = _t1c(xc)[str(r)]
    k = plain_uks(r, xc)
    assert abs(k.e_tot - ref["E_uks"]) <= 1 * UH, (k.e_tot - ref["E_uks"]) / UH
    assert abs(k.spin_square()[0] - ref["s2"]) <= 1e-3


# U2 -- DFT-in-DFT reproduces the global broken-symmetry UKS energy at every geometry
@pytest.mark.parametrize("xc", BS_FUNCS)
@pytest.mark.parametrize("proj", ["mu", "huzinaga"])
@pytest.mark.parametrize("r", TARGETS)
def test_U2_dft_in_dft_broken_symmetry(xc, proj, r):
    k = plain_uks(r, xc)
    res = ProjectionEmbedding(k, [0, 1], projector=proj, mu=1e8).run_dft_in_dft()
    assert res.emb_scf_converged
    # At r >= 4 A the H alpha electron and the OH fragment barely couple: the alpha density
    # has a nearly flat direction (1-3e-4, unchanged by a 100x tighter conv_tol) that the energy
    # does not see (exact to 0.02 uH). The energy is the criterion; bound the density at 5e-4 there.
    assert max(res.extras["dm_diff"]) < (5e-4 if r >= 4.0 else 1e-4), res.extras["dm_diff"]
    assert abs(res.E_eq1 - k.e_tot) <= 1 * UH, (res.E_eq1 - k.e_tot) / UH


# U4 -- payoff cross-validation: FCI-in-DFT on the broken-symmetry reference vs the
# independently computed patched-nbed T1c values (mu = 1e6, n_act = (4, 4)).
# Same comparison rules as R2: nbed's projector-energy convention, and at r >= 3 A ours
# must be <= nbed (verified lowest root).
@pytest.mark.parametrize("xc", BS_FUNCS)
@pytest.mark.parametrize("r", TARGETS)
def test_U4_bs_fci_in_dft_vs_patched_nbed(xc, r):
    ref = _t1c(xc)[str(r)]
    k = plain_uks(r, xc)
    res = ProjectionEmbedding(k, [0, 1], n_act=(4, 4)).run_fci()
    assert res.n_act == (4, 4)
    assert res.emb_scf_converged and res.solver_converged
    d = res.E_eq1 + res.extras["E_proj"] - ref["E_emb"]
    if r >= 3.0:
        assert d <= 5 * UH, f"ours above nbed by {d/UH:.2f} uH"
        assert -d <= 1e-3, f"nbed above ours by {-d/UH:.2f} uH"
    else:
        assert abs(d) <= 5 * UH, d / UH
