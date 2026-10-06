"""Bug-class guards from the nbed spin-polarisation bug (SI section S2)."""
import numpy as np
import pytest
from pyscf import dft

from pembed import ProjectionEmbedding
from conftest import IncompatibleDM, plain_uks, water


def test_guard_fires_on_2d_density_to_uks():
    k = dft.UKS(water(1.0)); k.xc = "pbe"; k.verbose = 0; k.kernel()
    with pytest.raises(IncompatibleDM):
        k.get_veff(k.mol, k.make_rdm1()[0] + k.make_rdm1()[1])


def test_veff_rejects_spin_summed_density():
    emb = ProjectionEmbedding(plain_uks(3.0, "b3lyp5"), [0, 1])
    with pytest.raises(ValueError):
        emb._veff(emb.D[0] + emb.D[1])


def test_embedded_scf_is_seeded_by_default():
    emb = ProjectionEmbedding(plain_uks(3.0, "b3lyp5"), [0, 1])
    res = emb.run_dft_in_dft()
    assert max(res.extras["dm_diff"]) < 1e-4       # converged to the global active density


def test_rohf_reference_refused():
    """A ROKS reference would be embedded with an unrestricted SCF, which relaxes the RO
    constraint (0.3-0.4 mEh below the global energy for OH / MnO): refuse it."""
    import pytest as _pt
    from pyscf import dft as _dft, gto as _gto
    from pembed import ProjectionEmbedding as _PE
    m = _gto.M(atom="O 0 0 0; H 0 0 0.97", basis="6-31G", spin=1, verbose=0)
    k = _dft.ROKS(m, xc="b3lyp5"); k.verbose = 0; k.kernel()
    with _pt.raises(NotImplementedError, match="restricted open-shell"):
        _PE(k, [0])
