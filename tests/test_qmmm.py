"""Q-tests: point charges, negative charge, density fitting and DFT+U through every
step (global SCF -> partition -> embedded SCF -> energies -> FCI/CASSCF -> FCIDUMP).

The point charges enter only through ks.get_hcore (MM potential) and ks.energy_nuc
(nucleus-MM); these tests would catch any step that rebuilds either from the bare molecule.
"""
import numpy as np
import pytest
from pyscf import ao2mo, dft, fci, gto, mcscf, qmmm, scf
from pyscf.tools import fcidump

from pembed import ProjectionEmbedding
from pembed.embed import as_uks

UH = 1e-6
# a lopsided ring of charges (net +0.8, cation-like surroundings as in the cathode clusters):
# polarises the density and shifts every energy term, with all occupied orbitals bound
PC_XYZ = np.array([[3.0, 0.0, 0.5], [-2.5, 1.5, 0.0], [0.5, -3.0, 1.0], [0.0, 2.0, -2.8],
                   [-1.5, -1.5, 2.5], [2.2, 2.2, 2.2]])
PC_Q = np.array([0.8, -0.5, 0.4, -0.3, 0.6, -0.2])


def water(basis="6-31G"):
    return gto.M(atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0", basis=basis, verbose=0)


def with_pc(mf):
    return qmmm.mm_charge(mf, PC_XYZ, PC_Q, unit="Angstrom")


def conv(mf, dm0=None):
    mf.conv_tol = 1e-11; mf.max_cycle = 300; mf.verbose = 0
    mf.kernel(dm0=dm0)
    assert mf.converged
    return mf


def test_pc_actually_enter():
    """Sanity: the charges change h and E_nuc, and embed sees the wrapped quantities."""
    mol = water()
    ks = conv(with_pc(dft.RKS(mol, xc="b3lyp5")))
    emb = ProjectionEmbedding(ks, [0])
    assert abs(emb.E_nuc - mol.energy_nuc()) > 1e-3
    assert np.abs(emb.h - scf.hf.get_hcore(mol)).max() > 1e-3
    # the RKS -> UKS view keeps the QM/MM wrapper
    u = as_uks(ks)
    assert abs(u.energy_nuc() - ks.energy_nuc()) < 1e-12
    assert np.allclose(u.get_hcore(), ks.get_hcore())


# Q1 -- DFT-in-DFT = global DFT with point charges: closed shell, anion, open shell
CASES = {
    "water_rks": dict(atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0", charge=0, spin=0, act=[0]),
    "OH-_rks": dict(atom="O 0 0 0; H 0 0 0.97", charge=-1, spin=0, act=[1]),
    "OH_uks": dict(atom="O 0 0 0; H 0 0 0.97", charge=0, spin=1, act=[0]),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("projector,mu", [("huzinaga", None), ("mu", 1e8)])
@pytest.mark.parametrize("df", [False, True])
def test_Q1_dft_in_dft_exact_with_pc(case, projector, mu, df):
    c = CASES[case]
    mol = gto.M(atom=c["atom"], basis="6-31G*", charge=c["charge"], spin=c["spin"], verbose=0)
    ks = (dft.UKS if c["spin"] else dft.RKS)(mol, xc="b3lyp5")
    ks = with_pc(ks)
    if df:
        ks = ks.density_fit()
    conv(ks)
    kw = dict(projector=projector) if mu is None else dict(projector=projector, mu=mu)
    res = ProjectionEmbedding(ks, c["act"], **kw).run_dft_in_dft()
    assert res.emb_scf_converged
    assert abs(res.E_eq1 - ks.e_tot) <= 1 * UH, (res.E_eq1 - ks.e_tot) / UH
    assert abs(res.E_eq18 - ks.e_tot) <= 1 * UH, (res.E_eq18 - ks.e_tot) / UH


# Q2 -- all atoms active: FCI-in-DFT = full FCI of the same QM/MM Hamiltonian
def test_Q2_all_active_fci_with_pc():
    mol = water("STO-3G")
    ks = conv(with_pc(dft.RKS(mol, xc="b3lyp5")))
    res = ProjectionEmbedding(ks, [0, 1, 2]).run_fci()
    assert res.n_env == (0, 0)
    hf = conv(with_pc(scf.RHF(mol)))
    C = hf.mo_coeff; n = C.shape[1]
    h1 = C.T @ hf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.full(mol, C), n)
    s = fci.direct_spin1.FCI(); s.conv_tol = 1e-12; s.nroots = 3
    e_ref = min(s.kernel(h1, eri, n, mol.nelec, ecore=hf.energy_nuc())[0])
    assert abs(res.E_eq1 - e_ref) <= 1 * UH, (res.E_eq1 - e_ref) / UH


# Q3 -- CASSCF-in-DFT uses the QM/MM E_nuc (all active: equals plain QM/MM CASSCF)
def test_Q3_casscf_enuc_with_pc():
    mol = water()
    ks = conv(with_pc(dft.RKS(mol, xc="b3lyp5")))
    res = ProjectionEmbedding(ks, [0, 1, 2]).run_casscf(4, 4)
    hf = conv(with_pc(scf.RHF(mol)))
    mc = mcscf.CASSCF(hf, 4, 4); mc.conv_tol = 1e-10; mc.verbose = 0; mc.kernel()
    assert abs(res.E_wf_emb - mc.e_tot) <= 1 * UH, (res.E_wf_emb - mc.e_tot) / UH


# Q4 -- FCIDUMP round trip with point charges (ECORE carries nucleus-MM)
def test_Q4_fcidump_roundtrip_with_pc(tmp_path):
    ks = conv(with_pc(dft.RKS(water("STO-3G"), xc="b3lyp5")))
    emb = ProjectionEmbedding(ks, [0, 1])
    res = emb.run_fci()
    d = fcidump.read(emb.write_fcidump(str(tmp_path / "FCIDUMP")))
    na = (d["NELEC"] + d["MS2"]) // 2
    s = fci.direct_spin1.FCI(); s.conv_tol = 1e-12; s.nroots = 3
    e = min(s.kernel(d["H1"], d["H2"], d["NORB"], (na, d["NELEC"] - na), ecore=d["ECORE"])[0])
    assert abs(e - res.E_wf_emb) <= 1 * UH, (e - res.E_wf_emb) / UH


# Q5 -- DFT+U (UKSpU) + DF + point charges, high-spin MnO (n_alpha != n_beta), O or Mn active
@pytest.fixture(scope="module")
def mno_pu():
    mol = gto.M(atom="Mn 0 0 0; O 0 0 1.648", basis="def2-svp", spin=5, verbose=0)
    ks = dft.UKSpU(mol, xc="pbe", U_idx=["Mn 3d"], U_val=[3.9])
    ks = with_pc(ks).density_fit()
    ks.conv_tol = 1e-10; ks.max_cycle = 300; ks.verbose = 0
    ks.kernel()
    if not ks.converged:
        so = ks.newton(); so.conv_tol = 1e-10; so.kernel(dm0=ks.make_rdm1())
        for a in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
            setattr(ks, a, getattr(so, a))
    assert ks.converged
    return ks


@pytest.mark.parametrize("act", [[0], [1]])
def test_Q5_dft_plus_u_df_pc_exact(mno_pu, act):
    ks = mno_pu
    assert abs(getattr(ks.get_veff(), "E_U", 0.0)) > 1e-4      # the U term is really there
    res = ProjectionEmbedding(ks, act, projector="huzinaga").run_dft_in_dft()
    if act == [0]:                         # Mn active holds the five unpaired electrons
        assert res.n_act[0] - res.n_act[1] == 5
    assert abs(res.E_eq1 - ks.e_tot) <= 1 * UH, (res.E_eq1 - ks.e_tot) / UH
    # eq. 18 is not variational: at this global SCF gradient (~1e-5) the embedded density
    # differs from D_A to first order, which eq. 18 sees as the trace term (93 uH for O
    # active, 3 uH at |g| = 3e-7). Test the identity eq1 - trace == eq18 instead (as R6).
    assert abs((res.E_eq1 - res.trace_term) - res.E_eq18) <= 1 * UH


# Q6 -- guard: Huzinaga with an UNBOUND environment orbital must refuse, mu must still be exact
def test_Q6_huzinaga_refuses_unbound_environment():
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="6-31G*", charge=-1, verbose=0)
    ks = conv(qmmm.mm_charge(dft.RKS(mol, xc="b3lyp5"), PC_XYZ, -PC_Q, unit="Angstrom"))
    assert ks.mo_energy[ks.mo_occ > 0][-1] > 0            # repulsive field: unbound HOMO
    with pytest.raises(ValueError, match="Huzinaga projector invalid"):
        ProjectionEmbedding(ks, [0], projector="huzinaga")
    res = ProjectionEmbedding(ks, [0], projector="mu", mu=1e8).run_dft_in_dft()
    assert abs(res.E_eq1 - ks.e_tot) <= 1 * UH
