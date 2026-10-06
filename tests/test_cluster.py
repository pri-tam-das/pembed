"""K-tests: point-charge-embedded cluster carving (pembed.cluster)."""
import numpy as np
from pymatgen.core import Lattice, Structure

from pembed.cluster import carve


def rocksalt(shift=None):
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.44), ["Mn", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    if shift is not None:
        for i, d in shift.items():
            s.translate_sites([i], d, frac_coords=False, to_unit_cell=False)
    return s


def charges(s):
    return np.array([2.0 if x.specie.symbol == "Mn" else -2.0 for x in s])


def test_K1_qm_keys_reproduce_the_cluster():
    s = rocksalt(); o = next(i for i, x in enumerate(s) if x.specie.symbol == "O")
    c1 = carve(s, o, r_cut=2.3, r_pc=9.0, shell=2.0, site_charges=charges(s))
    c2 = carve(s, o, r_cut=2.3, r_pc=9.0, shell=2.0, site_charges=charges(s), qm_keys=c1.info["qm_keys"])
    assert c1.symbols == c2.symbols and np.allclose(c1.coords, c2.coords)
    assert np.allclose(c1.pc_charges, c2.pc_charges) and c1.formal_charge == c2.formal_charge


def test_K2_same_atoms_at_another_geometry():
    """A geometry comparison must compare the same atoms: with qm_keys the distorted cell gives the
    same atoms in the same order, moved by exactly the distortion; the plain r_cut rule need not."""
    s = rocksalt(); o = next(i for i, x in enumerate(s) if x.specie.symbol == "O")
    c1 = carve(s, o, r_cut=2.3, r_pc=9.0, shell=2.0, site_charges=charges(s))
    mn = [i for i, x in enumerate(s) if x.specie.symbol == "Mn"]
    shift = {mn[0]: [0.12, 0.0, 0.0], mn[1]: [0.0, -0.10, 0.0]}
    s2 = rocksalt(shift)
    c2 = carve(s2, o, r_cut=2.3, r_pc=9.0, shell=2.0, site_charges=charges(s2), qm_keys=c1.info["qm_keys"])
    assert c2.symbols == c1.symbols and c2.info["qm_keys"] == c1.info["qm_keys"]
    d = np.linalg.norm(c2.coords - c1.coords, axis=1)
    assert np.all(d < 0.16) and np.any(d > 0.09)          # only the shifted Mn images move
    assert abs(c2.info["total_charge"]) < 1e-6           # neutrality is kept by the refit
