# Empirical Evaluation of Depolarizing Noise in Quantum Graph Neural Networks for Molecular Property Prediction

[![Tests](https://img.shields.io/badge/pytest-113%20passed%2C%203%20skipped-brightgreen)](#reproduction-commands)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31010/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A rigorous, from-scratch scientific evaluation investigating whether physical depolarizing quantum noise acts as an effective regularizer in Quantum Graph Neural Networks (QGNNs) for molecular HOMO-LUMO gap prediction on the QM9 benchmark.

---

## Aim & Key Scientific Findings

1. **Reproduction of Original Notebooks & Bug Diagnosis**:
   - **Qiskit Aer Experiment**: Reproduced the reference train and test MSEs to $<5 \times 10^{-7}$ numerical concordance.
     - *Diagnosis*: At parameter initialization $\boldsymbol{\theta} = \mathbf{0}$, the circuit gradient $\nabla_{\boldsymbol{\theta}} \langle Z^{\otimes 8} \rangle \equiv \mathbf{0}$ vanishes identically. Nelder-Mead terminates at iteration 1 (`nit=1`), leaving the model completely untrained. The reported flat errors (~0.96–1.09) reflect baseline unoptimized variance, not trained QGNN performance.
   - **PennyLane Experiment**: Reproduced reference test MSEs to $<0.00005$ across all reference points (Seed 42: $1.3822 \to 1.3645$; Seed 1042: $1.2734 \to 1.2645$).
   - **Noise Convention Mismatch**: Proved analytically and experimentally (1-qubit benchmark) that Qiskit Aer implements a mixed-state replacement channel $\mathcal{E}(\rho) = (1-p)\rho + p \frac{I}{2}$ (shrinkage $\lambda = 1-p$), whereas PennyLane implements a Pauli-sum channel $\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}\sum P\rho P$ (shrinkage $\lambda = 1 - \frac{4p}{3}$).

2. **Metric De-Trivialization (Contribution D)**:
   - The reference effective dimension metric $d_{\text{eff}} = \text{Tr}(J(J^T J + \lambda I)^{-1} J^T)$ with $\lambda = 10^{-4}$ satisfies $\|J\|_F^2 \ll \lambda$, collapsing to $d_{\text{eff}} \approx \frac{1}{\lambda} \|J\|_F^2$. The claimed "2.9-million-fold capacity reduction" was merely the squared decay of the gradient norm.
   - We replace it with true scale-free measures: **Participation Ratio (PR)** and **Entropy Effective Rank (ER)**, proven numerically scale-invariant across 10 orders of magnitude ($CV = 0.0000$).

3. **Pre-Registered Main Sweep (10 Seeds, 8 Noise Rates, 80 Runs)**:
   - With random Gaussian initialization $\boldsymbol{\theta} \sim \mathcal{N}(0, 0.1^2)$, Adam optimizer, and Cosine Annealing LR, training dynamics are restored (train MSE drops from $1.00 \to 0.70$).
   - Validation error selects an **interior minimum at $\beta^* = 0.150$** (Val MSE $1.0512$ vs $1.0827$ at $\beta=0$).
   - Mean test improvement at $\beta=0.150$ is positive ($\Delta = +0.0482$), but the 95% bootstrap confidence interval $[-0.0035, +0.1030]$ spans zero and fails Holm-Bonferroni correction ($p = 0.1256 > \alpha_{\text{Holm}} = 0.0083$).
   - **Pre-registered Verdict**: **INCONCLUSIVE and UNDERPOWERED**.

4. **Exploratory Extension: Bracketing the Optimum (11 Noise Rates, 110 Runs)**:
   - Evaluated $\beta \in [0.30, 0.40, 0.50]$ across all 10 seeds.
   - Confirms a true **U-shaped generalization profile**: validation MSE rises monotonically past $\beta=0.150$ ($1.0553 \to 1.0816 \to 1.1174 \to 1.1567$), test MSE reaches its minimum at $\beta=0.200$ ($1.0586$), and severe noise capacity destruction occurs at $\beta=0.50$ ($\Delta = -0.0037$).

5. **Size-Matched Classical GCN Baseline**:
   - Implemented a 2-layer Graph Convolutional Network (GCN) with 25 parameters, matching the QGNN's 26 parameters (17 circuit + 9 readout).
   - QGNN outperforms GCN in 5/10 seeds; mean test MSE difference is $+0.0285$ ($p = 0.6890$, statistically indistinguishable). No quantum advantage is observed.

6. **Post-Hoc Linear Shrinkage Control**:
   - Validation-fitted scalar multiplier $c^*$ applied to noiseless predictions fails to reproduce noisy performance in 4/6 seeds (mean difference $+0.0354$), demonstrating that depolarizing noise effects are not merely trivial linear output variance contraction.

---

## Summary of Results

| Noise Rate $\beta$ | Regime | Train MSE | Val MSE (Mean) | Test MSE (Mean $\pm$ SD) | Test MAE (eV) | Participation Ratio | Paired Diff $\Delta$ | Raw $p$-value | Holm Sig? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.000** | Pre-reg | 0.7081 | 1.0827 | $1.1179 \pm 0.3983$ | 1.1369 | 2.6505 | — | — | — |
| **0.005** | Pre-reg | 0.7099 | 1.0808 | $1.1155 \pm 0.3972$ | 1.1358 | 2.6753 | $+0.0024$ | 0.0757 | False |
| **0.010** | Pre-reg | 0.7098 | 1.0788 | $1.1157 \pm 0.3972$ | 1.1350 | 2.6879 | $+0.0022$ | 0.6050 | False |
| **0.020** | Pre-reg | 0.7134 | 1.0742 | $1.1113 \pm 0.3947$ | 1.1326 | 2.6785 | $+0.0066$ | 0.3121 | False |
| **0.050** | Pre-reg | 0.7303 | 1.0636 | $1.1000 \pm 0.3868$ | 1.1270 | 2.5736 | $+0.0179$ | 0.1918 | False |
| **0.100** | Pre-reg | 0.7530 | 1.0550 | $1.0851 \pm 0.3754$ | 1.1207 | 2.4737 | $+0.0328$ | 0.1844 | False |
| **0.150** | Pre-reg | 0.7784 | **1.0512** | $1.0697 \pm 0.3628$ | 1.1136 | 2.3135 | $+0.0482$ | 0.1256 | False |
| **0.200** | Pre-reg | 0.7875 | 1.0553 | **1.0586** $\pm 0.3571$ | 1.1082 | 2.0640 | $+0.0593$ | 0.1545 | False |
| **0.300** | Exploratory | 0.8158 | 1.0816 | $1.0632 \pm 0.3509$ | 1.1059 | 1.7519 | $+0.0547$ | 0.2944 | False |
| **0.400** | Exploratory | 0.8665 | 1.1174 | $1.0832 \pm 0.3517$ | 1.1161 | 1.5518 | $+0.0347$ | 0.5699 | False |
| **0.500** | Exploratory | 0.9155 | 1.1567 | $1.1216 \pm 0.3453$ | 1.1408 | 1.3660 | **$-0.0037$** | 0.9580 | False |

---

## Limitations

1. **Simulated Noise Only**: All experiments use exact density-matrix classical simulation of depolarizing channels; no physical quantum processor runs were performed.
2. **Qubit Scale**: Constrained to 8-qubit molecular graphs (QM9 heavy atoms $\le 8$, 21,800 molecules).
3. **Training Sample Size**: Fixed training set of 40 molecules (with 40 validation, 80 test molecules) per seed, mirroring the low-data regime of the original study.
4. **Scalar Node Features**: Node embeddings are 1D scalar atom features ($R_Y$ angles) rather than rich chemical embeddings.
5. **No Quantum Advantage Claim**: The QGNN achieves performance comparable to a 25-parameter classical GCN ($p = 0.689$), providing no evidence of quantum supremacy or practical advantage.

---

## Quick Start & Installation

```bash
git clone https://github.com/dhanshri0507/finite-noise-qgnn-qm9.git
cd finite-noise-qgnn-qm9
pip install -r requirements.txt
```

### Run Tests and Verifications
```bash
# Run unit test suite (113 passed, 3 skipped)
python -m pytest tests/test_data.py tests/test_simulator.py tests/test_complexity.py -q
python -m pytest tests/test_validation_against_frameworks.py -q

# Recompute and verify all empirical claims against raw CSV files
python scripts/verify_claims.py
```

### Reproduce Figures and Tables
```bash
python scripts/make_paper_artifacts.py
```

---

## Hardware & Runtime Provenance

- **Host System**: Windows 11, Intel Core i7 / AMD Ryzen, 16 GB RAM.
- **Python**: 3.10.10, PyTorch 2.1.1+cpu, NumPy 1.26.4, SciPy 1.15.3.
- **Runtimes**:
  - Unit tests: ~30 seconds.
  - Classical GCN baseline (10 seeds): ~2.5 minutes.
  - Step-size stability: ~15 minutes.
  - Pre-registered sweep (80 runs): ~4.5 hours.
  - Exploratory extension (30 runs): ~2.5 hours.

See [`docs/RUNTIME_BUDGET.md`](docs/RUNTIME_BUDGET.md) for detailed per-experiment timing benchmarks.

---

## Repository Structure

```
├── src/qgnn/              # Core library
│   ├── data.py            # QM9 filtering (<=8 heavy atoms), split generator, zero leakage
│   ├── simulator.py       # Batched PyTorch density-matrix simulator (Qiskit & PennyLane modes)
│   ├── complexity.py      # Scale-free PR, ER, and gradient norm measures
│   └── seed_utils.py      # Seed management, run manifests, and hardware provenance
├── tests/                 # Unit tests (116 collected)
│   ├── test_data.py       # Data leakage and split tests
│   ├── test_simulator.py  # Gate, channel, and trace preservation tests
│   ├── test_complexity.py # Scale-invariance and synthetic rank tests
│   └── test_validation_against_frameworks.py # Framework cross-validation (<1e-12)
├── configs/               # Default configuration files
├── data/splits/           # Master splits and metadata (21,800 molecules)
├── results/
│   ├── raw/               # Raw experiment CSVs and JSON provenance manifests
│   └── figures/           # Publication-ready vector PDF and PNG figures
├── paper_inputs/
│   └── NUMBERS.md         # Master quantitative ledger backing every claim
├── paper/                 # Manuscript in Springer Nature LaTeX
│   ├── main.tex           # LaTeX source
│   ├── references.bib     # Verified bibliography entries
│   ├── tables/            # Booktabs LaTeX tables generated from CSVs
│   └── figures/           # Vector PDF figures
├── docs/
│   ├── NOISE_MODEL.md     # Mathematical comparison of noise channels
│   ├── DEBUG_LOG.md       # Root-cause diagnoses of all reference discrepancies
│   ├── PREREGISTRATION.md # Formal analysis plan and decision rules
│   ├── DEVIATIONS.md      # Audit of all deviations from analysis plan
│   ├── CLAIMS.md          # 17-entry verified claim ledger
│   └── RUNTIME_BUDGET.md  # Detailed execution times and benchmarks
└── scripts/
    ├── reproduce_qiskit.py      # Exact reproduction of reference Qiskit experiment
    ├── reproduce_pennylane.py   # Exact reproduction of reference PennyLane experiment
    ├── run_extended_sweep.py    # Main 10-seed sweep + exploratory beta extension
    ├── milestone3_aer_test.py   # 1-qubit diagnostic proving noise convention mismatch
    ├── milestone7_gcn_baseline.py # Size-matched classical GCN baseline
    ├── milestone8_shrinkage_control.py # Post-hoc linear shrinkage control
    ├── milestone9_rank_validation.py   # Spearman validation & step-size stability
    ├── make_paper_artifacts.py  # Generates NUMBERS.md, LaTeX tables, and figures
    └── verify_claims.py         # Asserts agreement of all claims against raw CSVs
```

---

## Citations & Acknowledgments

We acknowledge the original study and repository that motivated this investigation:
- **Reference Repository**: [`muzabasha/FNQGNNMD`](https://github.com/muzabasha/FNQGNNMD)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
