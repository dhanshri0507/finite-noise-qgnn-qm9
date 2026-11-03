# Reverse Engineering Debug Log

This log tracks every numerical discrepancy identified during reproduction of the reference experiments, along with its analytic backward-trace, root cause, and resolution.

---

## Entry 001: Framework Depolarizing Channel Convention Discrepancy (Issue a)

- **Date**: 2026-11-02
- **Components**: `qiskit_aer.noise.depolarizing_error` vs `pennylane.DepolarizingChannel`
- **Symptom**: Initial cross-validation between PyTorch simulator and Qiskit Aer produced a $\sim 0.04$ discrepancy in $\langle Z^{\otimes 4} \rangle$ at $\beta = 0.02$.
- **Backward Trace**:
  1. Prediction $\to$ Circuit expectation mismatch.
  2. Isolated single-qubit gate $R_Y(0.8) \to \text{depolarizing}(\beta=0.05)$: PyTorch gave $0.6069$, Qiskit gave $0.6288$.
  3. Inspected SuperOp matrix of `depolarizing_error(0.05, 1)`: diagonal elements of traceless subspace were $0.9500 = 1 - p$.
  4. Inspected PennyLane `DepolarizingChannel(0.05)`: output on $|0\rangle$ was $0.9333 = 1 - 4p/3$.
- **Root Cause**:
  - Qiskit Aer parameterizes depolarizing error as replacement with $I / 2^n$ with probability $p$:
    $$\mathcal{E}_{\text{qiskit}}(p)(\rho) = (1 - p)\rho + p \frac{I_{2^n}}{2^n} \implies \text{shrinkage } \lambda = 1 - p$$
  - PennyLane parameterizes depolarizing channel as applying a non-identity Pauli error with probability $p$:
    $$\mathcal{E}_{\text{PL}}(p)(\rho) = (1 - p)\rho + \frac{p}{3} \sum_{P \in \{X, Y, Z\}} P \rho P \implies \text{shrinkage } \lambda = 1 - \frac{4p}{3}$$
- **Resolution**:
  - Added explicit `convention` handling in `src/qgnn/noise.py` and `src/qgnn/simulator.py`.
  - When `experiment="qiskit"`, the exact Qiskit channel $(1 - 0.75\beta)\rho + 0.25\beta \sum P \rho P$ is used for 1q gates and $(1 - \frac{15}{16}\beta)\rho + \frac{\beta}{16} \sum P_k \rho P_k$ for 2q gates.
  - When `experiment="pennylane"`, the PennyLane channel $(1 - \beta)\rho + \frac{\beta}{3} \sum P \rho P$ is used.
  - Verified against Qiskit Aer and PennyLane default.mixed to $< 10^{-12}$ error.

---

## Entry 002: Asymmetric CNOT Gate Noise in PennyLane Reference Implementation

- **Date**: 2026-11-02
- **Components**: `Finite_Noise_QGNN_QM9_Regularization_Demonstration.ipynb`
- **Symptom**: PennyLane reference code claimed hardware noise at all gates, but CNOT noise was applied only to `wires=v` (target).
- **Backward Trace**:
  - Examined notebook cell defining `build_qnode`:
    ```python
    qml.CNOT(wires=[i, j])
    if beta > 0: qml.DepolarizingChannel(beta, wires=j)
    qml.RZ(theta[0], wires=j)
    if beta > 0: qml.DepolarizingChannel(beta, wires=j)
    qml.CNOT(wires=[i, j])
    if beta > 0: qml.DepolarizingChannel(beta, wires=j)
    ```
- **Root Cause**: The authors applied single-qubit depolarizing noise to the target qubit $j$ only, rather than a genuine 2-qubit depolarizing channel across both qubits of the CNOT.
- **Resolution**: Implemented exact gate noise placement for `experiment="pennylane"` targeting wire $v$, allowing exact bit-for-bit reproduction of that notebook.

---

## Entry 003: Stationary Point Optimization Stagnation at Zero Initialization (Issue b)

- **Date**: 2026-11-02
- **Components**: `Finite_Noise_QGNN_QM9_Colab.ipynb`
- **Symptom**: Nelder-Mead optimization terminates at iteration 1 across all noise levels; parameters remain $\approx \mathbf{0}$.
- **Backward Trace**:
  - Gradient of $\langle Z^{\otimes 8} \rangle$ at $\boldsymbol{\theta} = \mathbf{0}$ with zero features is mathematically zero:
    $$\left.\frac{\partial \langle Z^{\otimes 8} \rangle}{\partial \theta_q}\right|_{\boldsymbol{\theta}=\mathbf{0}} = 0$$
  - At $\boldsymbol{\theta} = \mathbf{0}$, $R_Z(0) = I$ and $R_X(0) = I$. The entangling block $\text{CNOT} \cdot I \cdot \text{CNOT} = I$.
  - The circuit is effectively an unentangled product of $R_Y$ rotations.
  - Nelder-Mead evaluated points within simplex tolerance (`fatol=1e-4`), found no variance in objective on flattened landscape, and exited immediately.
- **Root Cause**: The model in the Qiskit study was never effectively trained; the reported MSE ($0.961 - 0.974$) merely reflects the variance of the unoptimized baseline on standardized data ($\mu = 0, \sigma = 1$).
- **Resolution**: Documented for reproduction pilot in Step 5; random initialization with Adam will be used for Tier 1 Step 7.

---

## Entry 004: Triviality of the Effective Complexity Metric (Issue c)

- **Date**: 2026-11-02
- **Components**: Reference paper formula $d_{\text{eff}} = \text{Tr}\left(J (J^T J + \lambda I)^{-1} J^T\right)$
- **Symptom**: Paper claims a "2.9-million-fold reduction in model capacity" from $\beta = 0$ to $\beta = 0.20$.
- **Backward Trace**:
  - For small parameter Jacobian $J$ (where $\|J\|_F^2 \ll \lambda = 10^{-4}$):
    $$J^T J + \lambda I \approx \lambda I \implies d_{\text{eff}} \approx \frac{1}{\lambda} \text{Tr}(J J^T) = \frac{1}{\lambda} \|J\|_F^2$$
  - The RMS gradient norm fell from $1.17 \times 10^{-5}$ to $6.91 \times 10^{-9}$ (a $1.7 \times 10^3$ fold reduction).
  - Squaring this reduction: $(1.7 \times 10^3)^2 \approx 2.9 \times 10^6$!
- **Root Cause**: The reported "effective complexity" is simply the squared Frobenius norm of the output Jacobian divided by the arbitrary ridge constant $\lambda = 10^{-4}$. It measures gradient norm shrinkage, not an independent scale-free dimension.
- **Resolution**: Documented in Contribution D preparation.
