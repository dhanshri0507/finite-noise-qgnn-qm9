# Pre-Registration / Formal Analysis Plan: Finite-Noise Regularization Study in QGNNs

> **Note on Protocol Terminology**: Throughout this repository and manuscript, this pre-registration protocol is designated as the formal **Analysis Plan**, committed prior to the full multi-seed evaluation sweeps (initial commit `c62d7df`).

**Study Title:** Empirical Evaluation of Depolarizing Noise Regularization in Quantum Graph Neural Networks for Molecular HOMO-LUMO Gap Prediction  
**Status:** COMMITTED PRIOR TO FINAL SWEEPS  
**Date:** 2026-11-02  
**Repository:** From-scratch clean implementation (`src/qgnn/`)  

---

## 1. Scientific Hypothesis and Core Question

### Primary Hypothesis
Does physical depolarizing quantum noise $\beta > 0$ act as an effective regularizer in Quantum Graph Neural Networks (QGNNs), improving generalization (lowering test MSE on molecular HOMO-LUMO gap prediction) relative to noiseless simulation ($\beta = 0$)?

### Competing Null Hypothesis (Noise-Induced Capacity Suppression)
Physical depolarizing noise acts monotonically to suppress output variance and gradient magnitude, contracting quantum state purity toward the maximally mixed state $I / 2^n$ without producing any statistically significant finite-noise optimum ($\beta^* > 0$). Any apparent local minimum in small unvalidated pilots is an artifact of sample noise, severe undertraining (e.g., zero initialization at stationary points), or random validation variance.

---

## 2. Strict Pre-Registered Decision Rules

A finite-noise regularizing optimum will be formally claimed **if and only if all** of the following conditions are simultaneously met:

1. **Validation Selection**:
   An optimal noise rate $\beta^* > 0$ is selected purely based on the minimum validation loss across seeds. (Test set is strictly blinded during selection).
2. **Paired Statistical Superiority on Test Set**:
   On the unseen test set, the model trained under $\beta^*$ achieves a lower Test MSE than $\beta = 0$, and the 95% paired bootstrap confidence interval:
   $$\text{CI}_{95\%}(\text{Test MSE}(\beta = 0) - \text{Test MSE}(\beta^*))$$
   **strictly excludes zero** across the independent random seeds ($N \ge 10$), with family-wise error rate controlled via Holm-Bonferroni correction across all candidate $\beta$ comparisons.
3. **U-Shaped Test Error Profile**:
   The test error curve as a function of noise rate $\beta$ must exhibit a non-trivial convex / U-shaped profile:
   $$\text{Test MSE}(\beta = 0) > \text{Test MSE}(\beta^*) < \text{Test MSE}(\beta_{\text{high}})$$
   monotonically decreasing to $\beta^*$ and monotonically increasing thereafter.
4. **Non-Triviality vs. Baseline Predictors**:
   The QGNN at $\beta^*$ must perform statistically better than a naive predict-the-training-mean baseline and match or challenge standard classical linear baselines (Ridge regression) under comparable parameter budgets.

**Negative Result Protocol**:
If any of conditions (1)–(3) fail, the paper and claim ledger will plainly declare the **negative result**: that depolarizing noise acts as pure capacity suppression and trainability degradation under the tested physical conditions, refuting preliminary unvalidated claims of beneficial quantum regularization.

---

## 3. Experimental Grid & Protocol

### 3.1 Primary Metric
- **Standardized Test Mean Squared Error (MSE)** on QM9 HOMO-LUMO gap ($\Delta \varepsilon$ in standardized units, standardized strictly using training set statistics).
- **Secondary Metrics**:
  - Test Mean Absolute Error (MAE in eV)
  - Generalization Gap ($\text{Test MSE} - \text{Train MSE}$)
  - Gradient Norm ($\|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|_2$)
  - Scale-Free Complexity (Participation Ratio $\text{PR} = \frac{(\sum_i s_i^2)^2}{\sum_i s_i^4}$ of singular values of the output Jacobian)

### 3.2 Noise Grid ($\beta$)
$$\beta \in \{0.0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150, 0.200\}$$
- Physical convention: Single documented depolarizing model defined in `docs/NOISE_MODEL.md`.

### 3.3 Seeds and Replicates
- **Number of seeds:** $N = 10$ independent random seeds:
  $$\text{Seeds} \in \{42, 1042, 2024, 3141, 5321, 6789, 7777, 8888, 9999, 12345\}$$
- Seed utility initializes `random`, `numpy`, and `torch` generators simultaneously.

### 3.4 Training Sample Sizes ($N_{\text{train}}$)
$$N_{\text{train}} \in \{40, 80, 160, 320\}$$
- Fixed test set size: $N_{\text{test}} = 3,271$ molecules (from the master random split).
- Fixed validation set size: $N_{\text{val}} = 3,270$ molecules.

### 3.5 Optimization & Training Protocol
To rule out the undertraining and stationary-point artifacts identified in the original repo (Issue b):
- **Initialization**: Gaussian random initialization $\boldsymbol{\theta} \sim \mathcal{N}(0, 0.1^2)$ (never all zeros).
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $10^{-4}$).
- **Learning Rate**: Initial $\eta = 0.03$ with cosine annealing schedule.
- **Budget**: 50 epochs.
- **Early Stopping**: Monitored on validation MSE with patience = 10 epochs. Model weights checkpointed at minimum validation error.

---

## 4. Benchmark Baselines on Identical Splits

All baselines are evaluated on the exact same train/val/test splits:
1. **Predict-the-Training-Mean**: Trivial baseline ($y_{\text{pred}} = \bar{y}_{\text{train}}$).
2. **Classical Ridge Regression**: L2-regularized linear model with identical graph-level features.
3. **Classical Multi-Layer Perceptron (MLP)**: Matching parameter budget ($\sim 25$ trainable parameters).
4. **Message-Passing GNN (GIN or GCN)**: Small 2-layer graph isomorphism or graph convolutional network with comparable parameter budget.

---

## 5. Scale-Free Complexity Measure Protocol (Contribution D)

The effective complexity measure $d_{\text{eff}} = \text{Tr}(J(J^T J + \lambda I)^{-1} J^T)$ is recognized as mathematically trivial ($\approx \frac{1}{\lambda} \|J\|_F^2$).  
We pre-register two scale-free, scale-invariant complexity candidates:
1. **Participation Ratio (PR)**:
   $$\text{PR}(J) = \frac{\left(\sum_{k=1}^r s_k^2\right)^2}{\sum_{k=1}^r s_k^4}$$
   where $s_k$ are the singular values of the output Jacobian $J$.
2. **Entropy-Based Effective Rank**:
   $$\text{ER}(J) = \exp\left( - \sum_{k=1}^r p_k \log p_k \right), \quad p_k = \frac{s_k}{\sum_{j} s_j}$$

**Validation criteria for complexity measures**:
- Scale Invariance: Must be strictly invariant under scalar multiplication $f(\mathbf{x}) \to c \cdot f(\mathbf{x})$.
- Regularization Independence: Must not depend on an arbitrary regularizer $\lambda$.
- Theoretical Validity: Evaluated alongside RMS gradient norm, plainly acknowledging that PR measures effective dimensionality while gradient norm measures overall output sensitivity.
