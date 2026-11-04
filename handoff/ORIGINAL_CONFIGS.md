# Original Reference Configurations

This document details the exact configurations reverse-engineered from the two reference experiments in `reference/FNQGNNMD/` (Qiskit Aer and PennyLane `default.mixed`).

---

## 1. Qiskit Aer Reference Experiment

- **Source Notebook**: `reference/FNQGNNMD/QGNN_Final_Colab.ipynb`
- **Quantum Architecture**:
  - Qubits: 8 (one per heavy atom)
  - State Encoding: $R_y(x_i)$ rotation per qubit based on normalized atomic number
  - Entangling Topology: $CNOT(i, j) \to R_z(\theta_0) \to CNOT(i, j)$ for each chemical bond $(i, j)$
  - Parameter Count: 1 global circuit parameter $\theta_0$
  - Observable / Readout: Global $Z^{\otimes 8} = Z_0 \otimes Z_1 \otimes \dots \otimes Z_7$
- **Noise Specification**:
  - Model: Qiskit `depolarizing_error(p, 2)` applied to all 2-qubit entangling gates
  - Rate Grid $\beta$: `[0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]`
  - Physical Contraction: Bloch shrinkage $\lambda = 1 - p$
- **Optimization & Training**:
  - Optimizer: SciPy `Nelder-Mead`
  - Initialization: $\theta_0 = 0.0$ (Zero initialization)
  - **Identified Failure**: At $\theta_0 = 0.0$, the gradient $\nabla_{\theta} \langle Z^{\otimes 8} \rangle \equiv 0$ identically. Nelder-Mead terminates after 1 iteration (`nit=1, nfev=9`) across all $\beta$, leaving the circuit unoptimized.
- **Dataset Partitioning**:
  - Subset: 80 train molecules, 80 test molecules
  - Target: QM9 HOMO-LUMO gap (standardized)

---

## 2. PennyLane Reference Experiment

- **Source Notebook**: `reference/FNQGNNMD/FNQGNN_PennyLane.ipynb`
- **Quantum Architecture**:
  - Qubits: 8
  - Simulator: `default.mixed` density-matrix simulator
  - Parameter Count: 17 circuit parameters (1 entangling $R_z$ parameter + 8 $R_x$ parameters + 8 $R_z$ parameters)
  - Readout: Per-qubit local expectation values $\langle Z_i \rangle$ ($i=0, \dots, 7$) fed into a classical linear layer ($8 \to 1$)
- **Noise Specification**:
  - Model: PennyLane `qml.DepolarizingChannel(p, wires=target)` applied strictly to the target wire of CNOT gates
  - Rate Grid $\beta$: `[0.0, 0.005, 0.01, 0.02, 0.05]`
  - Physical Contraction: Bloch shrinkage $\lambda = 1 - \frac{4}{3}p$ (33.3% stronger shrinkage than Qiskit at identical numeric $p$)
- **Optimization & Training**:
  - Optimizer: PyTorch `Adam` ($\text{lr} = 0.01$)
  - Batching: Full-batch / sequential
  - Epochs: 20
- **Dataset Partitioning**:
  - Data loading: Reads the first 2,500 molecules in raw file order
  - Partition: 40 train molecules, 40 validation molecules, 80 test molecules

---

## 3. Classical Ridge Baseline

- **Model**: `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])`
- **Features**: Graph-derived node and adjacency summary features
- **Partition**: Evaluated on identical 40 train / 80 test molecules
- **Reported & Reproduced Performance**:
  - Test MSE: $1.3072$
  - Test MAE: $1.1750\text{ eV}$
