"""Nonadditive exchange: exact (E_xx^nadd) and DFT (E_x^DFT,nadd), and the Delta_x correction.

Monino et al., JCTC 22, 1780 (2026):
    eq. 21   E_xx^nadd[gA, gB] = - sum_sigma tr[ gA^sigma K(gB^sigma) ]
    eq. 20   Delta_x = - E_x^DFT,nadd[gA_emb, gB] + E_xx^nadd[gA_emb, gB]
For a hybrid, E_x^DFT = a E_x^HF + E_x^semilocal, so Delta_x swaps the semilocal part for
the missing (1 - a) of exact exchange.

The exchange/correlation split is written out explicitly per functional (libxc exposes
several hybrids only as combined XC codes); `check_split` verifies that the split plus
a * exact exchange reproduces PySCF's own E_xc.
"""
import numpy as np

# name -> (semilocal exchange code, correlation code, exact-exchange fraction a)
XC_SPLIT = {
    "hf":        (None,                     None,                        1.00),
    "b3lyp5":    (".08*LDA_X + .72*GGA_X_B88", ".81*GGA_C_LYP + .19*LDA_C_VWN",     0.20),
    "b3lyp":     (".08*LDA_X + .72*GGA_X_B88", ".81*GGA_C_LYP + .19*LDA_C_VWN_RPA", 0.20),
    "pbe0":      (".75*GGA_X_PBE",           "GGA_C_PBE",                  0.25),
    "pbe":       ("GGA_X_PBE",               "GGA_C_PBE",                  0.00),
    "bhandhlyp": (".5*GGA_X_B88",            "GGA_C_LYP",                  0.50),
    "scan":      ("MGGA_X_SCAN",             "MGGA_C_SCAN",                0.00),
}


def _split(xc):
    key = xc.lower().replace("-", "")
    if key not in XC_SPLIT:
        raise NotImplementedError(f"exchange/correlation split not defined for '{xc}' "
                                  "(combined or range-separated functional)")
    return XC_SPLIT[key]


def _dm2(dm):
    dm = np.asarray(dm)
    return dm if dm.ndim == 3 else np.array([dm / 2, dm / 2])


def exact_exchange(ks, dm):
    """E_x^HF[gamma] = -1/2 sum_sigma tr[D^s K(D^s)] (per-spin, occupation <= 1)."""
    dm = _dm2(dm)
    k = ks.get_k(ks.mol, dm, hermi=1)
    return -0.5 * sum(float(np.einsum("ij,ji->", dm[s], k[s])) for s in (0, 1))


def exx_nadd(ks, gA, gB):
    """Monino eq. 21 (bilinear, no self term)."""
    gA, gB = _dm2(gA), _dm2(gB)
    kB = ks.get_k(ks.mol, gB, hermi=1)
    return -sum(float(np.einsum("ij,ji->", gA[s], kB[s])) for s in (0, 1))


def semilocal(ks, code, dm):
    if not code:
        return 0.0
    return float(ks._numint.nr_uks(ks.mol, ks.grids, code, _dm2(dm))[1])


def ex_dft(ks, xc, dm):
    xsl, _, a = _split(xc)
    return a * exact_exchange(ks, dm) + semilocal(ks, xsl, dm)


def ex_dft_nadd(ks, xc, gA, gB):
    xsl, _, a = _split(xc)
    AB = _dm2(gA) + _dm2(gB)
    return (a * exx_nadd(ks, gA, gB)
            + semilocal(ks, xsl, AB) - semilocal(ks, xsl, gA) - semilocal(ks, xsl, gB))


# correlation part of each functional (for the general route E_x = E_xc - E_c)
CORR = {
    "hf": None, "b3lyp5": ".81*GGA_C_LYP + .19*LDA_C_VWN", "b3lyp": ".81*GGA_C_LYP + .19*LDA_C_VWN_RPA",
    "pbe0": "GGA_C_PBE", "pbe": "GGA_C_PBE", "hse06": "GGA_C_PBE", "bhandhlyp": "GGA_C_LYP",
    "scan": "MGGA_C_SCAN", "r2scan": "MGGA_C_R2SCAN", "tpssh": "MGGA_C_TPSS",
}


def exact_part(ks, xc, dm):
    """Exact-exchange energy the functional itself contains, range separation included:
    K_eff = hyb*K + (alpha - hyb)*K_lr(omega), as PySCF builds it (numint.rsh_and_hybrid_coeff).
    HSE06: hyb = 0.25, alpha = 0 -> 0.25*K_sr. Global hybrids: hyb*K. Pure/meta-GGA: 0."""
    dm = _dm2(dm)
    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(xc, spin=ks.mol.spin)
    if abs(hyb) < 1e-14 and abs(alpha) < 1e-14:
        return 0.0
    k = np.asarray(ks.get_k(ks.mol, dm, hermi=1)) * hyb
    if abs(omega) > 1e-14:
        k = k + (alpha - hyb) * np.asarray(ks.get_k(ks.mol, dm, hermi=1, omega=omega))
    return -0.5 * sum(float(np.einsum("ij,ji->", dm[s], k[s])) for s in (0, 1))


def ex_dft_general(ks, xc, dm):
    """E_x^DFT[gamma] for ANY functional with a separable correlation part: semilocal xc (numint)
    minus correlation, plus the functional's own exact exchange (range-separated if RSH).
    A DFT+U energy E_U is not part of it (convention: U stays at the DFT level)."""
    key = xc.lower().replace("-", "")
    if key not in CORR:
        raise NotImplementedError(f"correlation part not tabulated for '{xc}' (e.g. wB97X's B97 correlation "
                                  "is not separable in libxc)")
    ex = semilocal(ks, xc, dm) - semilocal(ks, CORR[key], dm)
    return ex + exact_part(ks, xc, dm)


def ex_dft_nadd_general(ks, xc, gA, gB):
    AB = _dm2(gA) + _dm2(gB)
    return ex_dft_general(ks, xc, AB) - ex_dft_general(ks, xc, gA) - ex_dft_general(ks, xc, gB)


def delta_x(ks, xc, gA_emb, gB):
    """Monino eq. 20, Delta_x = -E_x^DFT,nadd + E_xx^nadd (full-range exact exchange).
    Functionals in XC_SPLIT use the explicit split (validated); others -- range-separated HSE06,
    r2SCAN, ... -- the general E_xc - E_c route (identical for the split ones, test E0d)."""
    key = xc.lower().replace("-", "")
    nadd = ex_dft_nadd(ks, xc, gA_emb, gB) if key in XC_SPLIT else ex_dft_nadd_general(ks, xc, gA_emb, gB)
    return -nadd + exx_nadd(ks, gA_emb, gB)


def check_split(ks, xc, dm):
    """E_xc from PySCF vs (semilocal x + c + a * exact exchange) for a density."""
    xsl, c, a = _split(xc)
    ref = float(ks.get_veff(ks.mol, _dm2(dm)).exc)
    mine = semilocal(ks, xsl, dm) + semilocal(ks, c, dm) + a * exact_exchange(ks, dm)
    return mine - ref
