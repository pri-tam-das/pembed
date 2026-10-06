"""Shared fixtures: the PRA 109, 022418 Table IV water geometry and SCF branch control.

The continuation walk reproduces the one used to freeze the nbed regression data
(embedding/tests/lmo_common.walk_rks), so the global SCF solutions are identical;
test_regression asserts this before comparing any embedded energy.
"""
import functools
import json
import os
import warnings

import numpy as np
import pytest
from pyscf import dft, fci, gto, scf

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(__file__), "data")

U = np.array([0.3751747, 0.0, 0.1387225]); U /= np.linalg.norm(U)
H_FIX = np.array([-0.7493682, 0.0, 0.2770822])
TARGETS = [0.4, 0.6, 0.798954, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]
GRID = sorted(set([round(x, 4) for x in np.arange(0.4, 5.001, 0.05)] + TARGETS))


def water(r, order="act", basis="STO-3G"):
    """order='act': moving H first (Table V); 'fix': fixed H first (Table VI)."""
    hm = U * r
    atoms = ([("H", hm), ("O", np.zeros(3)), ("H", H_FIX)] if order == "act"
             else [("H", H_FIX), ("O", np.zeros(3)), ("H", hm)])
    return gto.M(atom=[[s, tuple(c)] for s, c in atoms], basis=basis, unit="angstrom", verbose=0)


def _solve_rks(r, xc, dm0, order):
    try:
        k = dft.RKS(water(r, order)); k.xc = xc; k.conv_tol = 1e-10; k.verbose = 0
        k = k.newton(); k.max_cycle = 200
        k.kernel(dm0=dm0)
        if k.converged and np.isfinite(k.e_tot):
            return k.e_tot, k.make_rdm1()
    except Exception:
        pass
    return None


CACHE = os.path.join(os.path.dirname(__file__), ".walk_cache")


@functools.lru_cache(maxsize=None)
def walk_dms(xc, order="act"):
    """Continuation walk; returns {r: dm} of the lowest converged solution.

    Cached on disk (atomic write) so parallel pytest workers share one walk.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{xc}_{order}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {float(k): z[k] for k in z.files}  # keys are str(float)
    out, prev = {}, None
    for r in GRID:
        cands = [g for g in (_solve_rks(r, xc, d0, order) for d0 in (prev, None)) if g]
        if cands:
            e, dm = min(cands, key=lambda t: t[0])
            out[r] = dm; prev = dm
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **{str(float(k)): v for k, v in out.items()})
    os.replace(tmp, path)
    return out


def plain_rks(r, xc, order="act"):
    """A PLAIN (not newton-wrapped) converged RKS object on the walked branch."""
    dm = walk_dms(xc, order)[r]
    k = dft.RKS(water(r, order)); k.xc = xc; k.conv_tol = 1e-10; k.max_cycle = 300; k.verbose = 0
    k.kernel(dm0=dm)
    if not k.converged:
        # plain DIIS can stall on the near-degenerate stretched bond (e.g. r = 3.0 A);
        # converge with second-order SCF and carry that solution in a plain object
        kn = dft.RKS(water(r, order)); kn.xc = xc; kn.verbose = 0
        kn = kn.newton(); kn.conv_tol = 1e-10; kn.max_cycle = 200
        kn.kernel(dm0=dm)
        assert kn.converged, f"RKS failed to converge at r={r} ({xc}, {order})"
        for attr in ("mo_coeff", "mo_occ", "mo_energy", "e_tot"):
            setattr(k, attr, getattr(kn, attr))
        k.converged = True
    return k


@functools.lru_cache(maxsize=None)
def full_fci(r, basis="STO-3G", ss=None):
    """Exact full-system FCI: lowest state (or lowest with <S^2> = ss).
    Small spaces are diagonalised EXACTLY: the 3-root Davidson used before sometimes missed the
    lowest root at r = 3.0 A (57 uEh high), which made R4 fail nondeterministically."""
    from pyscf import ao2mo
    from pyscf.fci import cistring
    mf = scf.RHF(water(r, basis=basis)); mf.conv_tol = 1e-11; mf.kernel()
    n, ne = mf.mo_coeff.shape[1], mf.mol.nelec
    dim = cistring.num_strings(n, ne[0]) * cistring.num_strings(n, ne[1])
    if dim <= 4000:
        C = mf.mo_coeff
        h1 = C.T @ mf.get_hcore() @ C
        eri = ao2mo.restore(1, ao2mo.full(mf.mol, C), n)
        addr, h0 = fci.direct_spin1.pspace(h1, eri, n, ne, np=dim)
        w, v = np.linalg.eigh(h0)
        na_, nb_ = cistring.num_strings(n, ne[0]), cistring.num_strings(n, ne[1])
        es, cs = [], []
        for k in range(min(8, dim)):
            c = np.zeros(dim); c[addr] = v[:, k]; cs.append(c.reshape(na_, nb_)); es.append(w[k] + mf.energy_nuc())
    else:
        s = fci.FCI(mf); s.conv_tol = 1e-12; s.nroots = 3
        if ss is not None:
            s = fci.addons.fix_spin_(s, ss=float(ss), shift=0.5)
        es, cs = s.kernel()
    s2 = [fci.spin_op.spin_square0(c, n, ne)[0] for c in cs]
    ok = [i for i in range(len(es)) if ss is None or abs(s2[i] - ss) < 1e-3]
    return float(min(es[i] for i in ok))


@pytest.fixture(scope="session")
def nbed_frozen():
    with open(os.path.join(DATA, "nbed_regression_tableV_VI.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- broken symmetry (A2)
def _frag_guess(r, xc, order="act"):
    """H (alpha) + OH radical with spins swapped: the fully broken-symmetry start (T1b)."""
    mol = water(r, order)
    iH = 0 if order == "act" else 2
    atoms = mol._atom
    hm = [atoms[iH]]
    oh = [a for i, a in enumerate(atoms) if i != iH]
    mH = gto.M(atom=hm, basis="STO-3G", spin=1, unit="bohr", verbose=0)
    mOH = gto.M(atom=oh, basis="STO-3G", spin=1, unit="bohr", verbose=0)
    kH = dft.UKS(mH); kH.xc = xc; kH.verbose = 0; kH.kernel()
    kO = dft.UKS(mOH); kO.xc = xc; kO.verbose = 0; kO.kernel()
    aH, bH = kH.make_rdm1(); aO, bO = kO.make_rdm1()
    N = mol.nao
    da = np.zeros((N, N)); db = np.zeros((N, N))
    sl = mol.aoslice_by_atom()
    idxH = list(range(sl[iH][2], sl[iH][3]))
    idxO = [p for p in range(N) if p not in idxH]
    da[np.ix_(idxH, idxH)], db[np.ix_(idxH, idxH)] = aH, bH
    da[np.ix_(idxO, idxO)], db[np.ix_(idxO, idxO)] = bO, aO
    return np.array([da, db])


def _solve_uks(r, xc, dm0, order="act"):
    try:
        k = dft.UKS(water(r, order)); k.xc = xc; k.conv_tol = 1e-10; k.verbose = 0
        k = k.newton(); k.max_cycle = 300
        k.kernel(dm0=dm0)
        if k.converged and np.isfinite(k.e_tot):
            return k.e_tot, k.make_rdm1()
    except Exception:
        pass
    return None


def _bs_from_rks(r, xc, order="act"):
    k = dft.RKS(water(r, order)); k.xc = xc; k.conv_tol = 1e-8; k.verbose = 0
    k = k.newton(); k.max_cycle = 100; k.kernel()
    mo, occ = k.mo_coeff, k.mo_occ
    h = int(np.where(occ > 0)[0][-1]); n = h + 1
    ca, cb = mo.copy(), mo.copy()
    ca[:, h] = (mo[:, h] + mo[:, h + 1]) / np.sqrt(2)
    cb[:, h] = (mo[:, h] - mo[:, h + 1]) / np.sqrt(2)
    return [np.array([ca[:, :n] @ ca[:, :n].T, cb[:, :n] @ cb[:, :n].T]),
            np.array([k.make_rdm1() / 2] * 2)]


@functools.lru_cache(maxsize=None)
def walk_bs(xc, order="act"):
    """Broken-symmetry UKS walk (T1b): forward from RKS-based BS guesses, backward from
    the fragment guess at 5 A; the lowest converged solution at every geometry."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"bs_{xc}_{order}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {float(k): z[k] for k in z.files}
    fwd, prev = {}, None
    for r in GRID:
        c = [g for g in (_solve_uks(r, xc, d, order) for d in [prev] + _bs_from_rks(r, xc, order)) if g]
        if c:
            fwd[r] = min(c, key=lambda t: t[0]); prev = fwd[r][1]
    bwd, prev = {}, None
    for r in reversed(GRID):
        c = [g for g in (_solve_uks(r, xc, d, order) for d in
                         (prev, _frag_guess(r, xc, order), fwd[r][1] if r in fwd else None)) if g]
        if c:
            bwd[r] = min(c, key=lambda t: t[0]); prev = bwd[r][1]
    out = {r: min([x for x in (fwd.get(r), bwd.get(r)) if x], key=lambda t: t[0])[1]
           for r in GRID if r in fwd or r in bwd}
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **{str(float(k)): v for k, v in out.items()})
    os.replace(tmp, path)
    return out


def plain_uks(r, xc, order="act"):
    """PLAIN converged UKS on the broken-symmetry branch, internal instabilities followed."""
    k = dft.UKS(water(r, order)); k.xc = xc; k.conv_tol = 1e-10; k.verbose = 0; k.max_cycle = 300
    k.kernel(dm0=walk_bs(xc, order)[r])
    if not k.converged:
        kn = dft.UKS(water(r, order)); kn.xc = xc; kn.verbose = 0
        kn = kn.newton(); kn.conv_tol = 1e-10; kn.max_cycle = 300
        kn.kernel(dm0=walk_bs(xc, order)[r])
        for attr in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
            setattr(k, attr, getattr(kn, attr))
    for _ in range(6):
        mo_i, _, st, _ = k.stability(internal=True, external=False, return_status=True)
        if st:
            break
        k.kernel(dm0=k.make_rdm1(mo_i, k.mo_occ))
    if not k.converged:
        # plain DIIS can lose convergence after a stability step (SCAN, 2.0 A); polish with
        # second-order SCF and carry the solution back into the plain object
        kn = dft.UKS(water(r, order)); kn.xc = xc; kn.verbose = 0
        kn = kn.newton(); kn.conv_tol = 1e-10; kn.max_cycle = 300
        kn.kernel(dm0=k.make_rdm1())
        for attr in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
            setattr(k, attr, getattr(kn, attr))
    assert k.converged, f"broken-symmetry UKS failed at r={r} ({xc})"
    return k


# ---------------------------------------------------------------- bug-class guard
class IncompatibleDM(AssertionError):
    pass


@pytest.fixture(autouse=True)
def _fail_on_unpolarised_dm(monkeypatch):
    """PySCF's UKS/UHF get_veff silently treat a 2-D density as unpolarised and only WARN
    (and only when verbose >= WARN). That is the nbed T1b bug class, so make it a hard
    failure in every test, whatever the verbosity."""
    import pyscf.lib.logger as lg
    orig = lg.warn

    def warn(obj, msg, *args):
        if "Incompatible dm dimension" in str(msg):
            raise IncompatibleDM(f"2-D density passed to a UKS/UHF routine: {msg}")
        return orig(obj, msg, *args)
    monkeypatch.setattr(lg, "warn", warn)
