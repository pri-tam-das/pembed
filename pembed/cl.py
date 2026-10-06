"""Concentric localization (CL) of embedded virtual orbitals, one spin at a time.

Claudino & Mayhall, JCTC 15, 6085 (2019). Conventions mirror nbed 0.0.10
(localizers/virtual/concentric.py) so the two can be cross-checked:
  shell 0 : virtual directions spanned by the projection of the active-atom AOs
            (SVD of C_v^T S_{:a} S_{aa}^-1 S_{a:} C_v; size = # nonzero singular values)
  shell k : SVD of C_kept^T F C_ker, C_kept = occupied + shells 0..k-1; the leading
            (<= n_active_AO) right singular vectors of the kernel form the next shell.
Returns the kept orbitals (occupied first, then shells 0..n_shells) and the shell sizes.
"""
import numpy as np


def concentric_spin(c_occ, c_vir, fock, S, ao_idx, n_shells, tol=1e-15):
    n_ao = len(ao_idx)
    S_aa = S[np.ix_(ao_idx, ao_idx)]
    S_a = S[ao_idx, :]
    left = np.linalg.solve(S_aa, S_a @ c_vir)
    _, sig, vh = np.linalg.svd(left.T @ S_a @ c_vir)
    k = int(np.sum(sig[:n_ao] >= tol))
    V = vh.T
    kept = [c_occ, c_vir @ V[:, :k]]
    ker = c_vir @ V[:, k:]
    sizes = [k]
    for _ in range(n_shells):
        if ker.shape[1] == 0:
            break
        if ker.shape[1] == 1:          # cannot SVD a single vector: take it and stop (as nbed)
            kept.append(ker); sizes.append(1); ker = ker[:, :0]
            break
        ck = np.concatenate(kept, axis=1)
        _, sig, vh = np.linalg.svd(ck.T @ fock @ ker)
        k = int(np.sum(sig[:n_ao] >= tol))
        if k == 0:
            break
        V = vh.T
        kept.append(ker @ V[:, :k]); sizes.append(k)
        ker = ker @ V[:, k:]
    return np.concatenate(kept, axis=1), sizes
