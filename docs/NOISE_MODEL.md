# Depolarizing Noise Model Specification

## 1. Overview & Mathematical Definition

Depolarizing noise is the canonical symmetric error channel on quantum states. In this repository, we support two documented conventions to enable faithful reproduction of both the Qiskit and PennyLane reference experiments:

### 1.1 Qiskit Aer Convention (`convention="qiskit"`)
In Qiskit Aer, `qiskit_aer.noise.depolarizing_error(p, n)` implements an $n$-qubit quantum channel that replaces the state with the maximally mixed state $I / 2^n$ with probability $p$:

$$\mathcal{E}_{\text{qiskit}}(p, n)(\rho) = (1 - p)\rho + p \frac{I_{2^n}}{2^n}$$

#### Single-Qubit Channel ($n = 1$):
$$\mathcal{E}_1(\rho) = (1 - p)\rho + p \frac{I_2}{2} = \left(1 - \frac{3p}{4}\right)\rho + \frac{p}{4} \left(X\rho X + Y\rho Y + Z\rho Z\right)$$
- **Bloch Vector Shrinkage Factor**: $\lambda = 1 - p$
- **Kraus Operators**:
  - $K_0 = \sqrt{1 - \frac{3p}{4}} \, I_2$
  - $K_1 = \sqrt{\frac{p}{4}} \, X$
  - $K_2 = \sqrt{\frac{p}{4}} \, Y$
  - $K_3 = \sqrt{\frac{p}{4}} \, Z$

#### Two-Qubit Channel ($n = 2$):
$$\mathcal{E}_2(\rho) = (1 - p)\rho + p \frac{I_4}{4} = \left(1 - \frac{15p}{16}\right)\rho + \frac{p}{16} \sum_{k=1}^{15} P_k \rho P_k$$
where $P_k \in \{I, X, Y, Z\}^{\otimes 2} \setminus \{I \otimes I\}$.
- **Two-Qubit Shrinkage Factor**: $\lambda = 1 - p$

---

### 1.2 PennyLane Convention (`convention="pennylane"`)
In PennyLane, `pennylane.DepolarizingChannel(p, wires=w)` applies an unwanted single-qubit Pauli error chosen uniformly at random from $\{X, Y, Z\}$ with total probability $p$:

$$\mathcal{E}_{\text{pennylane}}(p)(\rho) = (1 - p)\rho + \frac{p}{3}\left(X\rho X + Y\rho Y + Z\rho Z\right) = \left(1 - \frac{4p}{3}\right)\rho + \frac{4p}{3} \frac{I_2}{2}$$
- **Bloch Vector Shrinkage Factor**: $\lambda = 1 - \frac{4p}{3}$
- **Kraus Operators**:
  - $K_0 = \sqrt{1 - p} \, I_2$
  - $K_1 = \sqrt{\frac{p}{3}} \, X$
  - $K_2 = \sqrt{\frac{p}{3}} \, Y$
  - $K_3 = \sqrt{\frac{p}{3}} \, Z$

---

## 2. Resolution of Known Issue (a): Framework Parameter Discrepancy

**Known Issue (a)**: *Qiskit `depolarizing_error(p)` vs PennyLane `DepolarizingChannel(p)` may not represent the same physical noise rate.*

### Proof & Exact Relationship:
Equating the Bloch vector shrinkage factors for a single qubit:
$$1 - p_{\text{qiskit}} = 1 - \frac{4}{3} p_{\text{pennylane}}$$

$$\implies p_{\text{qiskit}} = \frac{4}{3} p_{\text{pennylane}} \iff p_{\text{pennylane}} = \frac{3}{4} p_{\text{qiskit}}$$

### Quantitative Impact:
When both frameworks are run with the nominal parameter $\beta = 0.05$:
- **Qiskit Aer ($\beta = 0.05$)**: $\lambda = 1 - 0.05 = 0.9500$ (5% shrinkage per 1q gate)
- **PennyLane ($\beta = 0.05$)**: $\lambda = 1 - \frac{4}{3}(0.05) \approx 0.9333$ (6.67% shrinkage per 1q gate)

PennyLane's nominal noise is **33.3% stronger** than Qiskit's nominal noise for the same numerical value of $\beta$.

### CNOT Gate Noise Asymmetry:
- **Qiskit**: Applies a true 2-qubit channel $\mathcal{E}_2(\beta)$ to both control and target qubits of each CNOT.
- **PennyLane**: In `Finite_Noise_QGNN_QM9_Regularization_Demonstration.ipynb`, the author placed `qml.DepolarizingChannel(beta, wires=j)` on the target wire $j$ only. The control wire received no noise.

---

## 3. Noise Placement Specification

### Qiskit Experiment (`experiment="qiskit"`)
1. **Node Feature Encoding**: $R_Y(\text{node\_signal}[q])$ on wire $q \in \{0..7\}$ followed by 1-qubit depolarizing error $\mathcal{E}_1(\beta)$.
2. **Molecular Bond Block (per edge $(u, v)$)**:
   - $\text{CNOT}(u, v) \to \mathcal{E}_2(\beta)$ on $(u, v)$
   - $R_Z(\theta_v) \to \mathcal{E}_1(\beta)$ on $v$
   - $\text{CNOT}(u, v) \to \mathcal{E}_2(\beta)$ on $(u, v)$
3. **Trainable Single-Qubit Layer**: $R_X(\theta_q) \to \mathcal{E}_1(\beta)$ on $q \in \{0..7\}$.
4. **Readout**: $Z^{\otimes 8}$ expectation value on final noisy density matrix $\rho$.

### PennyLane Experiment (`experiment="pennylane"`)
1. **Node Feature Encoding**: $R_Y(\pi \cdot x[w])$ on wire $w \in \{0..7\}$ followed by $\mathcal{E}_{\text{PL}}(\beta)$ on wire $w$.
2. **Molecular Bond Block (per edge $(u, v)$)**:
   - $\text{CNOT}(u, v) \to \mathcal{E}_{\text{PL}}(\beta)$ on target $v$
   - $R_Z(\theta_0) \to \mathcal{E}_{\text{PL}}(\beta)$ on target $v$
   - $\text{CNOT}(u, v) \to \mathcal{E}_{\text{PL}}(\beta)$ on target $v$
3. **Trainable Single-Qubit Layer**: $R_X(\theta_{1+w})$ followed by $R_Z(\theta_{1+n+w})$ on wire $w$, followed by a single $\mathcal{E}_{\text{PL}}(\beta)$ on wire $w$.
4. **Readout**: $\langle Z_w \rangle$ per wire $w \in \{0..7\}$, followed by linear readout $W \mathbf{z} + b$.

---

## 4. Verification and Validation

All channels are validated to machine precision ($< 10^{-12}$) against:
1. `qiskit_aer.noise.NoiseModel` + `AerSimulator(method="density_matrix")`
2. `pennylane.device("default.mixed", wires=8)`

Unit tests covering Kraus completeness, trace preservation, Bloch vector shrinkage, and cross-framework equivalence reside in `tests/test_simulator.py` and `tests/test_validation_against_frameworks.py`.
