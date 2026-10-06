"""SPADE occupied-orbital partition, one spin channel at a time.

Claudino & Mayhall, JCTC 15, 1053 (2019). Matches nbed 0.0.10's choice rule so
the two codes can be compared directly, but works for any set of active atoms
(nbed assumes the active atoms come first in the geometry).
"""
import numpy as np
from scipy import linalg


def active_ao_indices(mol, active_atoms):
    """AO indices belonging to the given atoms (any order, any subset)."""
    slices = mol.aoslice_by_atom()
    idx = [p for ia in active_atoms for p in range(slices[ia][2], slices[ia][3])]
    return np.asarray(idx, dtype=int)


def sqrt_overlap(s):
    w, v = linalg.eigh(s)
    return (v * np.sqrt(w)) @ v.T


def spade_spin(c_occ, s_half, ao_idx, n_force=None):
    """Partition one spin channel's occupied orbitals.

    Returns (c_loc, n_act, sigma). The columns of c_loc are the rotated occupied
    orbitals, the first n_act of which are active.
    """
    nocc = c_occ.shape[1]
    if nocc == 0:
        return c_occ, 0, np.zeros(0)
    _, sigma, vh = linalg.svd((s_half @ c_occ)[ao_idx, :])
    if n_force is not None:
        n_act = int(n_force)
        if not 0 <= n_act <= nocc:
            raise ValueError(f"forced n_act={n_act} outside [0, {nocc}]")
    elif len(sigma) == 1:
        n_act = 1
    else:
        diffs = sigma[:-1] - sigma[1:]
        n_act = len(sigma) if np.allclose(diffs, 0.0) else int(np.argmax(diffs)) + 1
    return c_occ @ vh.T, n_act, sigma
