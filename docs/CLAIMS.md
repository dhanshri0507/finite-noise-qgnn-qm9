# Claims Ledger

Every quantitative claim in the paper, its status, and the evidence file that supports it.

## Format

| # | Claim | Status | Evidence File | Notes |
|---|-------|--------|---------------|-------|
| 1 | Example: Test MSE at β=0 is X.XXXX ± Y.YYYY | NOT YET RUN | — | — |

## Status values
- **SUPPORTED**: Backed by a result file with matching run ID
- **NOT SUPPORTED**: Evidence contradicts the claim
- **NOT YET RUN**: Experiment not yet conducted
- **PENDING VERIFICATION**: Result exists but not yet cross-checked

## Entries

| # | Claim | Status | Evidence File | Notes |
|---|-------|--------|---------------|-------|
| 1 | QM9 subset with $\le 8$ heavy atoms contains exactly 21,800 molecules out of 130,831 total. | SUPPORTED | `data/splits/split_metadata.json` | Exact atom filtering on atomic number $Z > 1$. |
| 2 | Train/Val/Test random split (seed 42) partitions the 21,800 molecules into 15,259 train (70%), 3,270 val (15%), and 3,271 test (15%). Target standardization parameters ($\mu=7.0670$, $\sigma=1.3164$) fit strictly on training set. | SUPPORTED | `data/splits/split_metadata.json` | Zero data leakage verified by unit test `tests/test_data.py`. |
| 3 | Qiskit Aer `depolarizing_error(p)` and PennyLane `DepolarizingChannel(p)` implement mathematically distinct physical operations: $p_{\text{qiskit}} = \frac{4}{3} p_{\text{pennylane}}$, with PennyLane inducing 33.3% stronger Bloch contraction at identical numerical $\beta$. | SUPPORTED | `docs/NOISE_MODEL.md` | Verified analytically and numerically to $<10^{-12}$ in `tests/test_simulator.py`. |
| 4 | In the reference Qiskit circuit, $\boldsymbol{\theta} = \mathbf{0}$ is a zero-gradient stationary point ($\nabla_{\boldsymbol{\theta}} \langle Z^{\otimes 8} \rangle \equiv \mathbf{0}$), causing Nelder-Mead to terminate after iteration 1 across all noise rates without training the model. | SUPPORTED | `docs/DEBUG_LOG.md` (Entry 003), `results/raw/reproduction_qiskit_results.csv` | Confirmed by exact reproduction of optimizer state `nit=1`. |
| 5 | The effective complexity metric $d_{\text{eff}} = \text{Tr}(J (J^T J + \lambda I)^{-1} J^T)$ with $\lambda = 10^{-4}$ satisfies $J^T J \ll \lambda I \implies d_{\text{eff}} \approx \frac{1}{\lambda} \|J\|_F^2$, proving the claimed $2.9 \times 10^6$-fold capacity reduction is an artifact of gradient norm squared decay. | SUPPORTED | `docs/DEBUG_LOG.md` (Entry 004), `results/raw/reproduction_qiskit_results.csv` | Mathematically and empirically verified. |
| 6 | Classical Ridge regression baseline on the 40-train / 80-test molecular split achieves Test MSE of 1.3072 and Test MAE of 1.1750 eV. | SUPPORTED | `results/raw/reproduction_ridge_results.csv` | Exact match to reference notebook cell 20 (Test MSE 1.3072, diff $3.2 \times 10^{-5}$; MAE 1.1750 eV, diff $1.3 \times 10^{-5}$). |
| 7 | Qiskit Colab notebook reproduction reproduces original reference values to 6 decimal places across all 7 noise rates: $\beta=0.000$ (Train 0.960914, Test 1.090837), $\beta=0.005$ (Train 0.963197, Test 1.091519), $\beta=0.010$ (Train 0.965101, Test 1.092106), $\beta=0.020$ (Train 0.968004, Test 1.093034), $\beta=0.050$ (Train 0.972304, Test 1.094493), $\beta=0.100$ (Train 0.973995, Test 1.095119), and $\beta=0.200$ (Train 0.974266, Test 1.095236). | SUPPORTED | `results/raw/reproduction_qiskit_comparison.csv` | Exact numerical reproduction (difference $< 5 \times 10^{-7}$) across the entire original noise grid. |
| 8 | Classical baselines on the master 21,800 split across $N_{\text{train}} \in [40, 80, 160, 320]$ achieve mean test MSEs: Mean Predictor ($0.9841 \to 0.9673$), RidgeCV ($0.9916 \to 0.8220$), and Small MLP ($0.9404 \to 0.8116$). | SUPPORTED | `results/raw/baselines_results.csv` | Evaluated across 5 random seeds with full provenance manifest. |
| 9 | Proper QGNN training with Gaussian initialization $\boldsymbol{\theta} \sim \mathcal{N}(0, 0.1^2)$, Adam, and CosineAnnealingLR resolves the zero-gradient stationary point stagnation, lowering test MSE from $>1.09$ down to $0.7037$ (Test MAE 0.9976 eV). | SUPPORTED | `results/raw/proper_training_results.csv`, `results/figures/convergence_qgnn_pennylane_beta0.0_seed42.png` | Verified that earlier flat-loss results were an artifact of undertraining and zero initialization. |
| 10 | Scale-free complexity measures (Participation Ratio and Entropy Effective Rank) are mathematically and numerically invariant under arbitrary output scaling $c \cdot J$ ($\Delta = 0.0$ to $10^{-11}$), unlike the paper's $d_{\text{eff}}$ which scales as $c^2$. | SUPPORTED | `src/qgnn/complexity.py`, `tests/test_complexity.py` | Proven on synthetic matrices of known rank $k \in \{1, 2, 4, 8, 15\}$ and scale factors $c \in [10^{-6}, 10^4]$. |
| 11 | Main sweep across 5 independent seeds and 8 noise rates ($\beta \in [0.0, 0.200]$) yields no statistically significant finite-noise optimum: all paired differences $\text{MSE}(\beta=0) - \text{MSE}(\beta)$ fail Holm-Bonferroni corrected significance ($p_{\text{raw}} \in [0.0359, 0.1434]$, exceeding Holm thresholds $0.0071 \dots 0.0500$). | SUPPORTED | `results/raw/main_sweep_statistics.csv`, `results/raw/manifest_main_sweep_20261103_103530_7dbe2d.json` | 95% paired bootstrap CIs computed over 10,000 resamples. |
| 12 | Under pre-registered decision rules (`docs/PREREGISTRATION.md`), the experimental hypothesis is rejected: physical depolarizing noise exhibits **Noise-Induced Capacity Suppression**, crushing quantum expectation values toward 0 without providing statistically validated regularization over noiseless models. | SUPPORTED | `results/raw/main_sweep_statistics.csv`, `results/figures/main_sweep_test_mse_vs_beta.png` | Adheres strictly to Rule 4 (no forced positive results). |
