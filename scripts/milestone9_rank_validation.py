"""
Milestone 9: Rank Metric Validation — Spearman Correlations with Bootstrap CIs.

Validates whether Participation Ratio (PR) and Entropy Effective Rank (ER) are
meaningful predictors of generalization quality.

Protocol:
  For each (seed, beta) pair in main sweep:
    - Collect PR, ER, rms_gradient_norm, test_mse_std, gap, beta
  Compute Spearman rank correlation (and bootstrap 95% CI) between:
    - PR vs test_mse_std
    - PR vs gap
    - PR vs rms_gradient_norm
    - ER vs test_mse_std
    - ER vs gap
    - ER vs rms_gradient_norm
  
  Also: Step-size stability analysis.
  Run FD-Jacobian computation with delta in [1e-5, 1e-4, 1e-3, 1e-2]
  on a restricted subset of (seed=42, all betas) using the saved models.
  Since we don't save model weights, we retrain 1 epoch from best state
  and use the EXACT same test samples for all delta values.
  
  Outputs:
    results/raw/rank_metric_spearman.csv    — correlation table
    results/raw/step_size_stability.csv     — PR/ER at different delta steps
    results/figures/spearman_heatmap.png
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qgnn.complexity import participation_ratio, entropy_effective_rank, rms_gradient_norm
from src.qgnn.simulator import DensityMatrixSimulator
from src.qgnn.seed_utils import set_global_seed
from scripts.reproduce_pennylane import build_compact_graphs

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "results" / "raw"
FIG_DIR = ROOT_DIR / "results" / "figures"
RAW_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_spearman_ci(x, y, n_boot=5000, ci=0.95, seed=42):
    """Bootstrap 95% CI for Spearman rank correlation."""
    rng = np.random.default_rng(seed)
    n = len(x)
    x, y = np.array(x), np.array(y)
    rho_obs, p_obs = stats.spearmanr(x, y)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            r, _ = stats.spearmanr(x[idx], y[idx])
            boot_rhos.append(r)
        except Exception:
            pass
    boot_rhos = np.array(boot_rhos)
    alpha = (1.0 - ci) / 2.0
    ci_lo = float(np.percentile(boot_rhos, 100 * alpha))
    ci_hi = float(np.percentile(boot_rhos, 100 * (1 - alpha)))
    return float(rho_obs), float(p_obs), ci_lo, ci_hi


def run_rank_metric_validation():
    print("=" * 70)
    print("Milestone 9: Rank Metric Spearman Validation")
    print("=" * 70)

    main_file = RAW_DIR / "main_sweep_results.csv"
    if not main_file.exists():
        print("ERROR: main_sweep_results.csv not found.")
        return

    df = pd.read_csv(main_file)
    print(f"Loaded {len(df)} runs from main sweep.")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Betas: {sorted(df['beta'].unique())}")

    # Variables for correlation analysis
    metrics_x = ["participation_ratio", "entropy_effective_rank", "rms_gradient_norm", "beta"]
    metrics_y = ["test_mse_std", "gap"]

    corr_records = []
    for x_col in metrics_x:
        for y_col in metrics_y:
            x = df[x_col].values
            y = df[y_col].values
            # Remove any nan
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 5:
                continue
            rho, p_val, ci_lo, ci_hi = bootstrap_spearman_ci(x, y)
            print(f"  Spearman({x_col[:20]} vs {y_col[:15]}): rho={rho:+.3f} p={p_val:.4f} CI=[{ci_lo:+.3f},{ci_hi:+.3f}]")
            corr_records.append({
                "predictor": x_col,
                "outcome": y_col,
                "n": int(len(x)),
                "spearman_rho": rho,
                "p_value": p_val,
                "ci_95_lower": ci_lo,
                "ci_95_upper": ci_hi,
                "significant_p05": p_val < 0.05,
                "ci_excludes_zero": (ci_lo > 0 or ci_hi < 0),
            })

    df_corr = pd.DataFrame(corr_records)
    corr_file = RAW_DIR / "rank_metric_spearman.csv"
    df_corr.to_csv(corr_file, index=False)
    print(f"\nSpearman results saved to {corr_file}")

    # Heatmap figure
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        pivot_rho  = df_corr.pivot(index="predictor", columns="outcome", values="spearman_rho")
        pivot_sig  = df_corr.pivot(index="predictor", columns="outcome", values="significant_p05")

        fig, ax = plt.subplots(figsize=(6, 4))
        cmap = plt.cm.RdBu_r
        im = ax.imshow(pivot_rho.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot_rho.columns)))
        ax.set_xticklabels(pivot_rho.columns, rotation=20, ha="right")
        ax.set_yticks(range(len(pivot_rho.index)))
        ax.set_yticklabels(pivot_rho.index)
        plt.colorbar(im, ax=ax, label="Spearman ρ")

        for i, row in enumerate(pivot_rho.index):
            for j, col in enumerate(pivot_rho.columns):
                val = pivot_rho.loc[row, col]
                sig = pivot_sig.loc[row, col]
                marker = "*" if sig else ""
                ax.text(j, i, f"{val:+.2f}{marker}", ha="center", va="center",
                        fontsize=9, color="white" if abs(val) > 0.5 else "black")

        ax.set_title("Spearman ρ: Rank Metrics vs Generalization Outcomes\n(* p<0.05)", fontsize=11)
        fig.tight_layout()
        fig_path = FIG_DIR / "spearman_heatmap.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"Heatmap saved to {fig_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    return df_corr


def run_step_size_stability(train_size=40, val_size=40, test_size=80, epochs=15,
                            test_seed=42, betas=None, max_molecules=10):
    """
    Test PR/ER stability across finite-difference steps.
    Uses a restricted subset (1 seed, all betas, only max_molecules test samples).
    Retrains the model from scratch for this test (no saved weights).
    """
    if betas is None:
        betas = [0.0, 0.05, 0.10, 0.20]  # restricted subset for speed
    deltas = [1e-5, 1e-4, 1e-3, 1e-2]

    print("\n" + "=" * 70)
    print(f"Milestone 9: Step-size Stability (seed={test_seed}, {max_molecules} molecules)")
    print("=" * 70)

    import torch, torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    class ProperQGNNRegressor(nn.Module):
        def __init__(self, sim, seed=42):
            super().__init__()
            self.sim = sim
            gen = torch.Generator().manual_seed(seed)
            n_theta = 1 + 2 * sim.n_qubits
            self.theta = nn.Parameter(0.1 * torch.randn(n_theta, generator=gen, dtype=torch.float64))
            self.readout = nn.Linear(sim.n_qubits, 1, dtype=torch.float64)
            nn.init.xavier_uniform_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)
        def forward(self, x, edges):
            z = self.sim.run_circuit(x, edges, self.theta)
            return self.readout(z).reshape(())

    def fd_jacobian(model, samples, delta):
        """Finite-difference Jacobian (not autograd) for stability test."""
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        n_theta = model.theta.numel()
        grads = []
        for s in samples[:max_molecules]:
            grad_row = np.zeros(n_theta)
            theta_orig = model.theta.detach().clone()
            for k in range(n_theta):
                theta_p = theta_orig.clone(); theta_p[k] += delta
                theta_m = theta_orig.clone(); theta_m[k] -= delta
                model.theta.data = theta_p
                fp = model(s["x"], s["edges"]).item()
                model.theta.data = theta_m
                fm = model(s["x"], s["edges"]).item()
                model.theta.data = theta_orig
                grad_row[k] = (fp - fm) / (2 * delta)
            grads.append(grad_row)
        return np.array(grads)

    graphs = build_compact_graphs(dataset_root="data/qm9", max_load=2500, max_nodes=8)
    rng = np.random.default_rng(test_seed)
    shuffled = rng.permutation(len(graphs))
    tr_idx = shuffled[:train_size]
    va_idx = shuffled[train_size:train_size + val_size]
    te_idx = shuffled[train_size + val_size:train_size + val_size + test_size]

    tr_targets = np.array([graphs[int(i)]["y"] for i in tr_idx]).reshape(-1, 1)
    scaler = StandardScaler().fit(tr_targets)

    def make_sample(i):
        g = graphs[int(i)]
        y_std = float(scaler.transform([[g["y"]]])[0, 0])
        return {"x": torch.tensor(g["x"], dtype=torch.float64),
                "edges": g["edges"], "y_std": torch.tensor(y_std, dtype=torch.float64),
                "y_raw": g["y"]}

    train_data = [make_sample(i) for i in tr_idx]
    val_data   = [make_sample(i) for i in va_idx]
    test_data  = [make_sample(i) for i in te_idx]

    records = []
    for beta in betas:
        print(f"\n  Beta={beta:.3f} — training...")
        t0 = time.time()
        set_global_seed(test_seed)
        sim = DensityMatrixSimulator(n_qubits=8, experiment='pennylane', beta=beta)
        model = ProperQGNNRegressor(sim=sim, seed=test_seed)
        opt = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        best_val = float("inf")
        best_state = None
        for ep in range(1, epochs + 1):
            model.train()
            order = np.random.default_rng(test_seed + ep).permutation(len(train_data))
            for idx in order:
                opt.zero_grad()
                s = train_data[idx]
                pred = model(s["x"], s["edges"])
                loss = (pred - s["y_std"]) ** 2
                loss.backward()
                opt.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                va_mse = np.mean([
                    ((model(s["x"], s["edges"]) - s["y_std"]) ** 2).item()
                    for s in val_data
                ])
            if va_mse < best_val:
                best_val = va_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        print(f"    Trained in {time.time()-t0:.1f}s")

        for delta in deltas:
            t1 = time.time()
            J_fd = fd_jacobian(model, test_data, delta)
            pr = participation_ratio(J_fd)
            er = entropy_effective_rank(J_fd)
            gn = rms_gradient_norm(J_fd)
            # Also autograd for reference at delta=N/A
            elapsed = time.time() - t1
            print(f"    delta={delta:.0e}: PR={pr:.4f}, ER={er:.4f}, GradNorm={gn:.4e} ({elapsed:.1f}s)")
            records.append({
                "seed": test_seed, "beta": beta, "delta": delta,
                "pr_fd": pr, "er_fd": er, "grad_norm_fd": gn,
            })

    df_stab = pd.DataFrame(records)
    stab_file = RAW_DIR / "step_size_stability.csv"
    df_stab.to_csv(stab_file, index=False)
    print(f"\nStep-size stability saved to {stab_file}")

    # Check stability: CV of PR across deltas per beta
    print("\n--- PR stability (CV across delta steps) ---")
    for beta in betas:
        sub = df_stab[df_stab["beta"] == beta]["pr_fd"].values
        cv = np.std(sub) / max(np.mean(sub), 1e-12)
        print(f"  Beta={beta:.3f}: PR CV={cv:.4f} ({'STABLE' if cv < 0.05 else 'UNSTABLE'})")

    return df_stab


if __name__ == "__main__":
    df_corr = run_rank_metric_validation()
    df_stab = run_step_size_stability()
    print("\nMilestone 9 complete.")
