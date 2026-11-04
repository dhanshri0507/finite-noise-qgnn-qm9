# Protocol Deviations Ledger

This document transparently records all deviations, scope adjustments, and exploratory extensions relative to the committed pre-registration protocol in [`docs/PREREGISTRATION.md`](PREREGISTRATION.md).

---

## Summary of Protocol Execution vs. Pre-Registration

| Protocol Dimension | Pre-Registered Specification | Executed in Initial Sweep (Step 8) | Resolution / Current Status |
|---|---|---|---|
| **Random Replicates (Seeds)** | $N = 10$ seeds: `[42, 1042, 2024, 3141, 5321, 6789, 7777, 8888, 9999, 12345]` | Executed initial 5 seeds (`[42, 1042, 2024, 3141, 5321]`) due to CPU runtime budget | **Executing remaining 5 seeds** to complete the full $N=10$ pre-registered sample. |
| **Noise Rate Grid ($\beta$)** | $\beta \in [0.0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150, 0.200]$ | Tested exact 8-point grid | **Added exploratory extension**: $\beta \in [0.30, 0.40, 0.50]$ labeled explicitly as *Exploratory* to test if the optimum is bracketed. |
| **Training Set Sizes ($N_{\text{train}}$)** | $N_{\text{train}} \in \{40, 80, 160, 320\}$ | Executed $N_{\text{train}} = 40$ (matching reference pilot) | Classical baselines evaluated on all 4 sizes; QGNN sweep focused on $N_{\text{train}} = 40$. |
| **Evaluation Set Partitioning** | Pre-registration text referenced master split ($N_{\text{val}}=3,270, N_{\text{test}}=3,271$) | Evaluated on $N_{\text{val}}=40, N_{\text{test}}=80$ (matching PennyLane partition) | Main sweep used $N_{\text{val}}=40, N_{\text{test}}=80$ per seed for simulation speed. Master split was used for classical baselines. |
| **Epoch Budget & Early Stopping** | 50 epochs, patience = 10 | 15 epochs, patience = 10 | Models converged early (best epoch typically between 1 and 10). |
| **Graph Neural Network Baseline** | GIN or GCN with comparable parameter budget | Not included in initial baselines script (only Mean, Ridge, MLP) | **Implemented**: GCN baseline added with paired statistical table. |
| **Complexity Measure Validation** | PR and ER scale invariance and correlation with test metrics | Scale invariance and synthetic known-rank proven | **Added**: Spearman correlation with bootstrap 95% CIs across metrics and finite-difference delta stability. |

---

## Detailed Notes on Deviations

### 1. Statistical Power and Replicate Count
The initial main sweep was terminated after 5 seeds ($N=5$) due to computational duration (~3.13 hours on CPU). However, $N=5$ is severely underpowered for paired tests after family-wise error correction (Holm threshold $\alpha / 7 = 0.0071$). Running the remaining 5 pre-registered seeds (`[6789, 7777, 8888, 9999, 12345]`) satisfies the pre-registered sample size of $N=10$.

### 2. Exploratory Noise Extension ($\beta \in [0.30, 0.40, 0.50]$)
In the 5-seed sweep, the validation loss was lowest at $\beta=0.200$ (the upper boundary of the pre-registered grid). Because the minimum landed on the boundary, the optimum was **not bracketed**. To determine whether this represents true regularization or monotonic output collapse (shrinkage towards 0), an exploratory sweep across $\beta \in [0.30, 0.40, 0.50]$ is evaluated. In accordance with open-science standards, this extension is strictly labeled as **exploratory / post-hoc** and not conflated with the pre-registered hypothesis test.

### 3. Post-Hoc Output Shrinkage Control
A critical methodological addition not originally listed in the pre-registration is a **post-hoc shrinkage control**:
If physical depolarizing noise merely acts as an isotropic contraction $\langle Z_i \rangle_\beta \approx (1-\gamma) \langle Z_i \rangle_0$, then scaling the predictions of a noiseless model by a scalar shrinkage factor $c \in [0, 1]$ fitted on validation data should reproduce the entire effect without quantum noise. We include this control directly in the evaluation.
