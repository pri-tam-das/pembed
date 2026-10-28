# pembed

Projection-based wavefunction-in-DFT embedding on PySCF, spin-general throughout: closed-shell, broken-symmetry and
true open-shell (n_α ≠ n_β) references, with point charges (QM/MM) and DFT+U. It accompanies the paper
"⟦title⟧" (⟦journal, year⟧).

## What it does
- **Partition.** SPADE, per spin, with optional forced per-spin active counts. The forced counts fix the active-space
  M_S of a spin-unrestricted solver.
- **Projectors.** The level shift μ and the Huzinaga operator.
- **Energies.** The total energy in the eq. 1 and the decomposed eq. 18 forms of Monino et al. (JCTC 22, 1780 (2026)),
  checked against each other. The nonadditive-exchange correction Δx (their eq. 20) is included.
- **Correlated solvers.**
  - FCI and CCSD/CCSD(T) in the embedded space.
  - DMRG through block2: SU(2) for spin-symmetric problems, SZ with separate α/β integrals otherwise.
  - Multi-start DMRG, a bond-dimension ladder with a checkpoint per bond dimension, extrapolation in the discarded
    weight, and an S² penalty for low-spin targets.
- **Virtual space.** Concentric localisation, or MP2 natural-orbital truncation with a frozen-natural-orbital MP2
  correction.
- **Outputs.** The correlated 1-RDM in the AO basis (frozen core and environment included) for property analysis,
  and FCIDUMP export (restricted and unrestricted).
- **Clusters.** `pembed.cluster.carve` builds point-charge-embedded clusters from a periodic structure. It can carve
  matched clusters across geometries from an explicit list of (site, image) keys.

## Design rules
Per-spin data are tuples, never stacked arrays. One unrestricted code path serves every case. Every environment
effect (point charges, charge, spin, DFT+U) is inherited from the global SCF object. Every stage reports ⟨S²⟩ and its
exactness residual. See `DESIGN.md`.

## Install and test
Requirements: Python ≥ 3.10, PySCF 2.14, NumPy, SciPy; block2 0.5.3 for DMRG; pytest (+ pytest-xdist).
```
pip install pyscf==2.14.0 block2==0.5.3 pytest pytest-xdist
PYTHONPATH=$PWD:$PWD/tests python -m pytest -q -n 8 tests
```
The suite (≈ 500 tests) checks:
- DFT-in-DFT exactness (≤ 1 µE_h) for closed-shell, broken-symmetry and open-shell references, with and without
  point charges and DFT+U;
- agreement with a reference implementation (Nbed) on its published tables;
- the energy-form identities and the Δx bilinearity;
- FCI-in-DFT with all atoms active against full FCI;
- the SZ DMRG solver against FCI.

`run_all_tests.sh` is the cluster wrapper we used (adjust the Python path).

## Minimal example
```python
from pyscf import gto, dft
from pembed import ProjectionEmbedding
mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="sto-3g")
ks = dft.UKS(mol, xc="b3lyp5").run()
emb = ProjectionEmbedding(ks, active_atoms=[0, 1], projector="huzinaga")
res = emb.run_fci()                       # FCI-in-DFT; res.E_eq1, res.rdm1_ao, ...
print(res.E_eq1, emb.delta_x(res))
```

## License
MIT (see `LICENSE`).

## Citation
**Where Does the Anion-Redox Hole Go? Reliability of Open-Shell Projection-Based Embedding for Li-Rich Cathodes, from DFT Environments to Quantum Solvers**

Pritam Das, Satadeep Bhattacharjee, and Pil-Ryung Cha (2026).

- **arXiv:** [2610.36659](https://arxiv.org/abs/2610.36659)
- **BibTeX:**

```bibtex
@misc{das2026doesanionredoxholego,
  title        = {Where Does the Anion-Redox Hole Go? Reliability of Open-Shell Projection-Based Embedding for Li-Rich Cathodes, from DFT Environments to Quantum Solvers},
  author       = {Pritam Das and Satadeep Bhattacharjee and Pil-Ryung Cha},
  year         = {2026},
  eprint       = {2610.36659},
  archivePrefix = {arXiv},
  primaryClass = {cond-mat.mtrl-sci},
  url          = {https://arxiv.org/abs/2610.36659}
}
```
