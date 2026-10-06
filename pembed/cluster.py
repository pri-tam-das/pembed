"""Point-charge-embedded clusters carved from a periodic structure.

Recipe
------
1. Centre on a target atom (the redox-active O). Take every atom within `r_cut` of it.
2. Complete the octahedron of every transition-metal cation in the cluster (no dangling
   TM-O bonds), then add every Li bonded (< `r_bond`) to a cluster O as a real atom.
3. All other ions within `r_pc` become formal point charges (Li +1, Mn +4, O -2 ...).
4. The outermost point-charge shell (r > r_pc - `shell`) is refitted by least squares so
   that the external potential at every QM site reproduces the periodic Ewald Madelung
   potential (minus the formal-charge potential of the other QM ions), with the total
   (QM formal + point charges) constrained exactly neutral.

Boundary capping is NOT applied here. Use `spill_out_report` on the converged cluster
SCF to judge whether it is needed (Gate 2 decides).
"""
from dataclasses import dataclass, field

import numpy as np

FORMAL = {"Li": 1.0, "Mn": 4.0, "O": -2.0, "Ni": 2.0, "Co": 3.0, "Na": 1.0}
TM = {"Mn", "Ni", "Co", "Fe", "Cr", "V", "Ti"}


@dataclass
class EmbeddedCluster:
    symbols: list
    coords: np.ndarray          # Angstrom, QM atoms
    formal_charge: float        # sum of formal charges of the QM atoms = cluster charge
    pc_coords: np.ndarray       # Angstrom
    pc_charges: np.ndarray
    center_index: int           # index of the target atom inside `symbols`
    site_indices: list          # indices in the periodic structure
    madelung_rms: float         # rms residual (V) of the fitted external potential at QM sites
    info: dict = field(default_factory=dict)

    def pyscf_atom(self):
        return [[s, tuple(c)] for s, c in zip(self.symbols, self.coords)]


def _ewald_site_potentials(struct, q):
    """Periodic Madelung potential (V) at every site for per-site charges q (neutral cell)."""
    from pymatgen.analysis.ewald import EwaldSummation
    s = struct.copy()
    s.add_site_property("charge", list(map(float, q)))
    s.add_oxidation_state_by_site(list(map(float, q)))
    ew = EwaldSummation(s)
    # site energy E_i = 1/2 q_i phi_i  (eV, charges in e)  ->  phi_i in V
    return np.array([2.0 * ew.get_site_energy(i) / q[i] for i in range(len(s))])


def carve(struct, center, r_cut=2.3, r_bond=2.4, r_pc=15.0, shell=3.0, formal=FORMAL,
          site_charges=None, qm_charge=None, qm_keys=None):
    """Build an EmbeddedCluster around periodic site `center` of a pymatgen Structure.

    site_charges: optional per-periodic-site charges (e.g. O at -2 + its hole occupation for a
    delithiated cell); must sum to zero. qm_charge: integer total charge of the QM cluster
    (default: sum of its site charges, which must then be an integer).
    qm_keys: optional explicit QM atoms as (site index, periodic image) keys, IN ORDER, e.g. the
    info["qm_keys"] of a cluster carved from another geometry of the same supercell. Comparing two
    geometries then compares the same atoms: at a fixed r_cut the HSE06-relaxed cfg07 cell drops one
    O image (44 -> 43 atoms, charge +1 -> +3), which mixed a cluster change into the geometry row.
    """
    COUL = 14.399645  # eV * Angstrom / e^2  (potential in V from charge in e at 1 Angstrom)
    qsite = (np.asarray(site_charges, float) if site_charges is not None
             else np.array([formal[x.specie.symbol] for x in struct]))
    if abs(qsite.sum()) > 1e-6:
        raise ValueError(f"periodic cell not neutral: sum of site charges = {qsite.sum():+.4f}")
    phi_ewald = _ewald_site_potentials(struct, qsite)
    c0 = struct[center].coords
    # all periodic images within r_pc, with (site index, cartesian)
    neigh = struct.get_sites_in_sphere(c0, r_pc, include_index=True, include_image=True)
    pts = [(n.index, n.coords, tuple(np.round(n.image).astype(int)), n.specie.symbol) for n in neigh]
    key = lambda p: (p[0], p[2])
    dist = {key(p): np.linalg.norm(p[1] - c0) for p in pts}
    if qm_keys is not None:
        want = [(int(k[0]), tuple(int(x) for x in k[1])) for k in qm_keys]
        bykey = {key(p): p for p in pts}
        missing = [k for k in want if k not in bykey]
        if missing:
            raise ValueError(f"qm_keys not within r_pc of the centre: {missing[:5]}")
        qm = set(want)
        qm_pts = [bykey[k] for k in want]            # same atoms, same order as the reference
    else:
        qm = {key(p) for p in pts if dist[key(p)] <= r_cut}
        # complete TM octahedra
        for p in list(pts):
            if key(p) in qm and p[3] in TM:
                for o in pts:
                    if o[3] == "O" and np.linalg.norm(o[1] - p[1]) < r_bond:
                        qm.add(key(o))
        # Li bonded to any cluster O
        for p in pts:
            if p[3] == "Li" and any(o[3] == "O" and key(o) in qm and np.linalg.norm(o[1] - p[1]) < r_bond
                                    for o in pts):
                qm.add(key(p))
        qm_pts = [p for p in pts if key(p) in qm]
    pc_pts = [p for p in pts if key(p) not in qm]
    sym = [p[3] for p in qm_pts]
    X = np.array([p[1] for p in qm_pts])
    qf = np.array([qsite[p[0]] for p in qm_pts])
    Q = float(np.round(qf.sum())) if qm_charge is None else float(qm_charge)
    if qm_charge is None and abs(qf.sum() - Q) > 1e-6:
        raise ValueError(f"QM site charges sum to {qf.sum():.4f}; pass an integer qm_charge")
    P = np.array([p[1] for p in pc_pts])
    q = np.array([qsite[p[0]] for p in pc_pts])
    # target external potential at QM sites: Ewald minus the other QM formal charges
    target = np.empty(len(qm_pts))
    for i, p in enumerate(qm_pts):
        others = [j for j in range(len(qm_pts)) if j != i]
        target[i] = phi_ewald[p[0]] - COUL * sum(qf[j] / np.linalg.norm(X[j] - X[i]) for j in others)
    rP = np.linalg.norm(P - c0, axis=1)
    outer = rP > (r_pc - shell)
    fixed = ~outer
    A = COUL / np.linalg.norm(X[:, None, :] - P[None, outer, :], axis=2)       # (nQM, nOuter)
    b = target - COUL * (q[fixed][None, :] / np.linalg.norm(X[:, None, :] - P[None, fixed, :], axis=2)).sum(1)
    # neutrality: sum(q_outer) = -(sum qf + sum q_fixed); solve constrained least squares
    ntot = -(Q + q[fixed].sum())
    n = outer.sum()
    K = np.block([[A.T @ A, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    rhs = np.concatenate([A.T @ b, [ntot]])
    # small Tikhonov pull toward the formal charges keeps the fit physical
    lam = 1e-3 * np.trace(A.T @ A) / n
    K[:n, :n] += lam * np.eye(n)
    rhs[:n] += lam * q[outer]
    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    q_fit = q.copy(); q_fit[outer] = sol[:n]
    resid = A @ q_fit[outer] - b
    ic = next(i for i, p in enumerate(qm_pts) if p[0] == center and p[2] == (0, 0, 0))
    return EmbeddedCluster(sym, X, Q, P, q_fit, ic, [p[0] for p in qm_pts],
                           float(np.sqrt(np.mean(resid ** 2))),
                           dict(n_qm=len(qm_pts), n_pc=len(pc_pts), n_fitted=int(n),
                                total_charge=float(Q + q_fit.sum()),
                                qm_site_charge_sum=float(qf.sum()),
                                max_shift=float(np.max(np.abs(q_fit[outer] - q[outer]))) if n else 0.0,
                                r_cut=r_cut, r_pc=r_pc, qm_keys=[(int(p[0]), tuple(int(x) for x in p[2])) for p in qm_pts]))


def spill_out_report(mol, mf, cluster, r_near=2.5):
    """Electron spill-out diagnostics: Mulliken population on each QM atom, and the
    fraction of the HOMO density on atoms adjacent to (< r_near of) a positive point charge
    that replaced a real cation."""
    from pyscf import lo
    pop = mf.mulliken_pop(verbose=0)[1]
    pos = cluster.pc_coords[cluster.pc_charges > 0.5]
    near = [i for i, c in enumerate(cluster.coords)
            if len(pos) and np.min(np.linalg.norm(pos - c, axis=1)) < r_near]
    return dict(mulliken=[float(x) for x in pop], boundary_atoms=near)
