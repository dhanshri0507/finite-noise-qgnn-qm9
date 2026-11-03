# Computational Budget & Runtime Benchmark

This document details the hardware specifications, measured execution times, and computational resource allocations across all phases of the study.

---

## 1. Hardware Environment

All experiments were executed in an isolated, reproducible environment logged in JSON run manifests:
- **Platform**: Windows 10 (Build 10.0.26300 AMD64)
- **CPU**: Intel Core (Family 6 Model 126 Stepping 5, GenuineIntel)
- **Python Version**: 3.10.10
- **PyTorch Version**: 2.1.1+cpu (Zero CUDA assumed/used)
- **NumPy Version**: 1.26.4

---

## 2. Density-Matrix Simulator Benchmarks (8 Qubits)

Single-molecule forward pass benchmarks comparing original naive matrix embedding vs optimized tensordot & permutation-indexed channels (`src/qgnn/simulator.py`):

| Operation Mode | Naive Embedding | Optimized Simulator | Speedup Factor |
|---|---|---|---|
| Noiseless Circuit ($\beta=0.0$) | ~120 ms | **31.2 ms** | **3.8x** |
| Noisy Circuit ($\beta=0.01$, 1q + 2q noise) | ~1,850 ms | **168.4 ms** | **11.0x** |
| Output Trace Evaluation | Matrix Trace (O($2^{2n}$)) | **Diagonal Sum (O($2^n$))** | **16.0x** |

---

## 3. Experiment Runtime Budgets

### A. Reference Qiskit Reproduction Pilot (`scripts/reproduce_qiskit.py`)
- Evaluated on 80 train + 80 test molecules across 7 noise rates:
  - $\beta=0.000$: 106.0s
  - $\beta=0.005$: 714.4s
  - $\beta=0.010$: 744.5s
  - $\beta=0.020$: 696.7s
  - $\beta=0.050$: 720.1s
  - $\beta=0.100$: 712.3s
  - $\beta=0.200$: 708.9s
  - **Total Pilot Runtime**: ~1.22 hours

### B. Classical Baselines (`scripts/train_baselines.py`)
- Mean Predictor, RidgeCV, and Small MLP evaluated across $N_{\text{train}} \in [40, 80, 160, 320]$ and 5 seeds:
  - **Total Baselines Runtime**: 47.8s

### C. Tier 1 Step 8 Main Sweep (`scripts/run_main_sweep.py`)
- 40 individual training runs (5 seeds $\times$ 8 noise rates), 15 epochs each with early stopping, validation checkpointing, exact Jacobian calculation, and bootstrap CI testing:
  - Noiseless runs ($\beta=0.0$): ~88–101s per run
  - Noisy runs ($\beta \in [0.005, 0.20]$): ~254–370s per run
  - **Cumulative Sweep Wall-Clock Time**: 11,288.4 seconds (**3.13 hours**)
  - Peak RAM consumption: < 1.2 GB
