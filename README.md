# QGNN Depolarizing Noise Study: Empirical Reproduction and Evaluation

A reproducible, from-scratch scientific study investigating whether physical depolarizing noise acts as a regularizer in small Quantum Graph Neural Networks (QGNNs) for QM9 molecular HOMO-LUMO gap prediction.

---

## Key Scientific Findings

1. **Exact Reproduction & Optimizer Stagnation Diagnosed**:
   - The reference Qiskit experiment was reproduced to 6 decimal places across all 7 noise rates ($\beta \in [0.0, 0.20]$).
   - **Root Cause Discovered**: At parameter initialization $\boldsymbol{\theta} = \mathbf{0}$, the circuit's analytic gradient is identically zero ($\nabla_{\boldsymbol{\theta}} \langle Z^{\otimes 8} \rangle \equiv \mathbf{0}$). Nelder-Mead terminates at iteration 1 (`nit=1, nfev=9`), leaving the model untrained. Reported flat errors (~0.96–1.09) were artifacts of baseline unoptimized output.
2. **Disparate Noise Conventions Resolved**:
   - Qiskit's `depolarizing_error(p)` contracts Bloch vectors by $1-p$, whereas PennyLane's `DepolarizingChannel(p)` contracts by $1 - \frac{4}{3}p$. PennyLane applies noise only to the CNOT target wire, while Qiskit applies a 2-qubit depolarizing channel across both qubits.
3. **Effective Complexity Metric De-trivialized**:
   - The original metric $d_{\text{eff}} = \text{Tr}(J(J^TJ + \lambda I)^{-1}J^T)$ collapses to $\frac{1}{\lambda}\|J\|_F^2$ when $J^TJ \ll \lambda I$. The reported $2.9 \times 10^6$-fold capacity reduction was simply the squared decay of the gradient norm. We implemented and validated true scale-free measures: **Participation Ratio (PR)** and **Entropy Effective Rank (ER)**.
4. **Proper Training Resolves Flat-Loss Artifact**:
   - With Gaussian initialization $\boldsymbol{\theta} \sim \mathcal{N}(0, 0.1^2)$, Adam optimization, and CosineAnnealingLR, test MSE improves from $1.091$ down to $0.7037$ (Test MAE $0.9976\text{ eV}$).
5. **Pre-Registered Hypothesis Evaluation Verdict**:
   - Across 5 independent seeds and 8 noise rates ($\beta \in [0.0, 0.200]$), **no statistically significant finite-noise optimum emerges** after Holm-Bonferroni correction ($p > 0.035$, exceeding threshold $\alpha_{\text{Holm}} = 0.0071$).
   - **Final Verdict**: **NEGATIVE RESULT: NOISE-INDUCED CAPACITY SUPPRESSION**. Physical depolarizing noise shrinks quantum expectation values towards zero, suppressing model expressivity rather than providing a bona fide regularizing effect.

---

## Repository Structure

```
├── src/qgnn/              # Core library
│   ├── data.py            # QM9 filtering (<=8 heavy atoms), split generator, zero leakage
│   ├── simulator.py       # Batched PyTorch density-matrix simulator (Qiskit & PennyLane modes)
│   ├── complexity.py      # Scale-free PR, ER, and gradient norm measures
│   └── seed_utils.py      # Seed management, run manifests, and hardware provenance
├── tests/                 # Unit tests (97 passing)
│   ├── test_data.py       # Leakage and split tests
│   ├── test_simulator.py  # Gate, channel, and trace tests
│   ├── test_complexity.py # Scale-invariance and known-rank rank tests
│   └── test_validation_against_frameworks.py # PennyLane & Qiskit numerical equivalence (<1e-12)
├── configs/               # Reproducible experiment configurations
├── data/
│   ├── qm9/               # PyG QM9 raw dataset
│   ├── splits/            # Master split (train=15,259, val=3,270, test=3,271)
│   └── cache_*.pt         # Pre-processed molecular graphs and baseline features
├── results/
│   ├── raw/               # Raw result CSVs and JSON provenance manifests
│   └── figures/           # High-resolution publication figures
├── paper_inputs/
│   └── NUMBERS.md         # Master quantitative ledger backing every claim
├── docs/
│   ├── NOISE_MODEL.md     # Mathematical comparison of noise channels
│   ├── DEBUG_LOG.md       # Root-cause diagnoses of all reference discrepancies
│   ├── PREREGISTRATION.md # Pre-registered decision rules and hypothesis criteria
│   └── CLAIMS.md          # 12-entry claim ledger with empirical evidence links
└── scripts/
    ├── reproduce_qiskit.py      # Exact reproduction of reference Qiskit experiment
    ├── train_baselines.py       # Classical baselines (Mean, RidgeCV, Small MLP)
    ├── train_proper_qgnn.py     # Proper training pipeline
    ├── run_main_sweep.py        # 40-run sweep with paired bootstrap CIs and Holm test
    └── make_paper_artifacts.py  # Regenerates NUMBERS.md and figures strictly from CSVs
```

---

## Reproduction Commands

### 1. Run Complete Test Suite
```bash
python -m pytest tests/test_data.py tests/test_simulator.py tests/test_complexity.py -v
python -m pytest tests/test_validation_against_frameworks.py -v
```

### 2. Reproduce Reference Qiskit Experiment
```bash
python -u scripts/reproduce_qiskit.py
```

### 3. Run Classical Baselines
```bash
python -u scripts/train_baselines.py
```

### 4. Run Main Sweep with Statistical Testing
```bash
python -u scripts/run_main_sweep.py --epochs 15 --train_size 40
```

### 5. Regenerate Paper Artifacts and NUMBERS Ledger
```bash
python scripts/make_paper_artifacts.py
```

---

## Hardware and Provenance

- **Platform**: Windows 10 (10.0.26300 AMD64)
- **CPU**: Intel Core (Family 6 Model 126 Stepping 5)
- **Python Version**: 3.10.10
- **PyTorch Version**: 2.1.1+cpu (Zero CUDA assumed/used)
- **NumPy Version**: 1.26.4
- **Manifests**: Every run in `results/raw/` is linked to an immutable JSON run manifest containing hardware specifications, library versions, config hashes, and execution timestamps.

---

## Limitations

- **Small Scale**: 8 qubits simulating molecules with $\le 8$ heavy atoms.
- **Simulated Depolarizing Noise**: Kraus-operator based simulation; real quantum processor noise includes non-Markovian drift, cross-talk, and asymmetric $T_1/T_2$ relaxation.
- **Sample Budgets**: Focuses on small training sample regimes ($N_{\text{train}} \in [40, 320]$) representative of near-term NISQ molecular evaluations.
- **No Quantum Advantage Claim**: The QGNN does not achieve computational or statistical advantage over classical Ridge or MLP baselines on this benchmark.
