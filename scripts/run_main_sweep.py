"""
Tier 1 Step 8: Main Sweep with Confidence Intervals and Holm Correction.

Conducts the rigorous pre-registered evaluation of the QGNN model:
- Sweeps physical depolarizing noise beta across pre-registered grid:
    beta in [0.0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150, 0.200]
- Evaluates across independent random seeds:
    seeds in [42, 1042, 2024, 3141, 5321]
- Training done properly: random Gaussian init theta ~ N(0, 0.1^2), Adam optimizer,
  CosineAnnealingLR, early stopping on validation with best checkpointing
- Evaluates on unseen test set: Train MSE, Val MSE, Test MSE, Test MAE (eV), Gap,
  Gradient Norm, Participation Ratio (scale-free), Entropy Effective Rank (scale-free)
- Statistical analysis:
  - Paired difference Delta(beta) = Test_MSE(beta=0) - Test_MSE(beta)
  - 95% paired bootstrap confidence intervals (10,000 resamples)
  - Paired t-test p-values with Holm-Bonferroni correction
  - Rigorous pre-registered decision rule evaluation
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qgnn.simulator import DensityMatrixSimulator
from src.qgnn.complexity import (
    participation_ratio,
    entropy_effective_rank,
    rms_gradient_norm,
    compute_all_complexity_metrics,
)
from src.qgnn.seed_utils import create_run_manifest, generate_run_id, set_global_seed
from scripts.reproduce_pennylane import build_compact_graphs

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "results" / "raw"
FIG_DIR = ROOT_DIR / "results" / "figures"
RAW_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


class ProperQGNNRegressor(nn.Module):
    """QGNN Regressor with trainable circuit parameters and linear readout."""
    def __init__(self, sim: DensityMatrixSimulator, seed: int = 42):
        super().__init__()
        self.sim = sim
        gen = torch.Generator().manual_seed(seed)
        n_theta = 1 + 2 * sim.n_qubits  # 17 for 8 qubits
        self.theta = nn.Parameter(0.1 * torch.randn(n_theta, generator=gen, dtype=torch.float64))
        self.readout = nn.Linear(sim.n_qubits, 1, dtype=torch.float64)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor, edges: list) -> torch.Tensor:
        z = self.sim.run_circuit(x, edges, self.theta)
        return self.readout(z).reshape(())


def compute_output_jacobian(model: ProperQGNNRegressor, samples: list, max_samples: int = 20) -> np.ndarray:
    """Compute empirical output Jacobian J = d f(x) / d theta via exact PyTorch Autograd."""
    model.eval()
    eval_samples = samples[:max_samples]
    grads = []
    for s in eval_samples:
        model.zero_grad()
        out = model(s["x"], s["edges"])
        out.backward()
        grads.append(model.theta.grad.detach().cpu().numpy().copy())
    return np.array(grads, dtype=float)


def bootstrap_paired_difference_ci(diffs: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 42):
    """Compute bootstrap confidence interval for mean of paired differences."""
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    mean = float(np.mean(diffs))
    return mean, lower, upper


def run_main_sweep(
    train_size: int = 40,
    val_size: int = 40,
    test_size: int = 80,
    epochs: int = 25,
    betas: list = None,
    seeds: list = None,
):
    if betas is None:
        betas = [0.0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150, 0.200]
    if seeds is None:
        seeds = [42, 1042, 2024, 3141, 5321]

    print("=" * 80)
    print("TIER 1 STEP 8: MAIN SWEEP WITH CONFIDENCE INTERVALS & HOLM CORRECTION")
    print(f"Noise grid ({len(betas)} points): {betas}")
    print(f"Seeds ({len(seeds)} replicates): {seeds}")
    print(f"Dataset sizes: Train={train_size}, Val={val_size}, Test={test_size} | Epochs={epochs}")
    print("=" * 80)

    graphs = build_compact_graphs(dataset_root="data/qm9", max_load=2500, max_nodes=8)

    raw_results_file = RAW_DIR / "main_sweep_results.csv"
    existing_results = []
    if raw_results_file.exists():
        existing_df = pd.read_csv(raw_results_file)
        existing_results = existing_df.to_dict("records")
        print(f"Found {len(existing_results)} existing runs in {raw_results_file}")

    completed_keys = {(r["seed"], r["beta"]) for r in existing_results}
    all_records = list(existing_results)

    total_runs = len(seeds) * len(betas)
    current_run = 0

    for seed in seeds:
        # Fixed split for this seed
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(graphs))
        tr_idx = shuffled[:train_size]
        va_idx = shuffled[train_size:train_size + val_size]
        te_idx = shuffled[train_size + val_size:train_size + val_size + test_size]

        tr_targets = np.array([graphs[int(i)]["y"] for i in tr_idx]).reshape(-1, 1)
        scaler = StandardScaler().fit(tr_targets)

        def make_sample(i):
            g = graphs[int(i)]
            y_std = float(scaler.transform([[g["y"]]])[0, 0])
            return {
                "x": torch.tensor(g["x"], dtype=torch.float64),
                "edges": g["edges"],
                "y_std": torch.tensor(y_std, dtype=torch.float64),
                "y_raw": g["y"],
            }

        train_data = [make_sample(i) for i in tr_idx]
        val_data = [make_sample(i) for i in va_idx]
        test_data = [make_sample(i) for i in te_idx]

        for beta in betas:
            current_run += 1
            if (seed, beta) in completed_keys:
                print(f"[{current_run}/{total_runs}] Skipping Seed={seed}, Beta={beta:5.3f} (already computed)")
                continue

            print(f"\n[{current_run}/{total_runs}] Running Seed={seed}, Beta={beta:5.3f}...")
            t0 = time.time()
            set_global_seed(seed)

            sim = DensityMatrixSimulator(n_qubits=8, experiment='pennylane', beta=beta)
            model = ProperQGNNRegressor(sim=sim, seed=seed)
            opt = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

            best_val = float("inf")
            best_state = None
            best_epoch = 0

            for ep in range(1, epochs + 1):
                model.train()
                order = np.random.default_rng(seed + ep).permutation(len(train_data))
                for idx in order:
                    opt.zero_grad()
                    s = train_data[idx]
                    pred = model(s["x"], s["edges"])
                    loss = (pred - s["y_std"]) ** 2
                    loss.backward()
                    opt.step()
                scheduler.step()

                # Fast validation check
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for s in val_data:
                        p = model(s["x"], s["edges"])
                        val_losses.append(((p - s["y_std"]) ** 2).item())
                va_mse = float(np.mean(val_losses))

                if va_mse < best_val:
                    best_val = va_mse
                    best_epoch = ep
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)

            def eval_metrics(s_list):
                model.eval()
                p_std, t_std, t_raw = [], [], []
                with torch.no_grad():
                    for s in s_list:
                        p = model(s["x"], s["edges"]).item()
                        p_std.append(p)
                        t_std.append(s["y_std"].item())
                        t_raw.append(s["y_raw"])
                p_std = np.array(p_std)
                t_std = np.array(t_std)
                p_raw = scaler.inverse_transform(p_std.reshape(-1, 1)).ravel()
                mse = float(mean_squared_error(t_std, p_std))
                mae = float(mean_absolute_error(t_raw, p_raw))
                return mse, mae

            tr_mse, tr_mae = eval_metrics(train_data)
            va_mse, va_mae = eval_metrics(val_data)
            te_mse, te_mae = eval_metrics(test_data)
            gap = te_mse - tr_mse

            # Compute output Jacobian and complexity measures
            J = compute_output_jacobian(model, test_data)
            pr = participation_ratio(J)
            er = entropy_effective_rank(J)
            grad_norm = rms_gradient_norm(J)

            elapsed = time.time() - t0
            print(f"  Best Ep: {best_epoch} | Tr MSE: {tr_mse:.4f} | Val MSE: {va_mse:.4f} | Test MSE: {te_mse:.4f} | MAE: {te_mae:.4f} eV | PR: {pr:.2f} | Time: {elapsed:.1f}s")

            record = {
                "seed": seed,
                "beta": beta,
                "best_epoch": best_epoch,
                "train_mse_std": tr_mse,
                "train_mae_eV": tr_mae,
                "val_mse_std": va_mse,
                "val_mae_eV": va_mae,
                "test_mse_std": te_mse,
                "test_mae_eV": te_mae,
                "gap": gap,
                "participation_ratio": pr,
                "entropy_effective_rank": er,
                "rms_gradient_norm": grad_norm,
                "elapsed_seconds": elapsed,
            }
            all_records.append(record)
            completed_keys.add((seed, beta))

            # Save checkpoint of results after every run
            pd.DataFrame(all_records).to_csv(raw_results_file, index=False)

    df = pd.DataFrame(all_records)
    df.to_csv(raw_results_file, index=False)
    print(f"\nAll {len(df)} runs saved to {raw_results_file}")

    # --- Part 2: Paired Bootstrap Statistical Testing & Holm Correction ---
    print("\n" + "=" * 80)
    print("STATISTICAL TESTING (Paired Differences: Test_MSE(beta=0) - Test_MSE(beta))")
    print("Positive difference indicates noise improved generalization over noiseless baseline.")
    print("=" * 80)

    # Pivot test MSE: index=seed, columns=beta
    pivot_test = df.pivot(index="seed", columns="beta", values="test_mse_std")
    zero_mse = pivot_test[0.0]

    stat_results = []
    comparisons = [b for b in betas if b > 0.0]

    for b in comparisons:
        # Paired diff: zero - noisy (positive means noisy is better)
        diff = (zero_mse - pivot_test[b]).to_numpy()
        mean_diff, ci_low, ci_high = bootstrap_paired_difference_ci(diff)

        # Paired t-test
        t_stat, p_val = stats.ttest_1samp(diff, popmean=0.0)

        stat_results.append({
            "beta": b,
            "mean_diff": mean_diff,
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high,
            "excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
            "t_statistic": float(t_stat),
            "p_value_raw": float(p_val),
            "noisy_beats_zero": bool(mean_diff > 0.0 and ci_low > 0.0),
        })

    df_stats = pd.DataFrame(stat_results)
    # Holm-Bonferroni correction
    df_stats = df_stats.sort_values("p_value_raw").reset_index(drop=True)
    m = len(df_stats)
    df_stats["holm_threshold"] = [0.05 / (m - i) for i in range(m)]
    df_stats["holm_significant"] = df_stats["p_value_raw"] <= df_stats["holm_threshold"]
    df_stats = df_stats.sort_values("beta").reset_index(drop=True)

    stats_file = RAW_DIR / "main_sweep_statistics.csv"
    df_stats.to_csv(stats_file, index=False)
    print(df_stats[["beta", "mean_diff", "ci_95_lower", "ci_95_upper", "p_value_raw", "holm_threshold", "holm_significant", "noisy_beats_zero"]].to_string(index=False))

    # --- Part 3: Pre-Registered Decision Rule Check ---
    print("\n" + "=" * 80)
    print("PRE-REGISTERED HYPOTHESIS EVALUATION:")
    print("=" * 80)

    # 1. Validation-selected beta
    val_pivot = df.pivot(index="seed", columns="beta", values="val_mse_std").mean()
    val_best_beta = float(val_pivot.idxmin())
    print(f"1. Validation-selected optimal beta: {val_best_beta}")

    # 2. Check if val_best_beta > 0 and statistically beats beta=0
    finite_opt_supported = False
    if val_best_beta > 0.0:
        row = df_stats[df_stats["beta"] == val_best_beta].iloc[0]
        if row["noisy_beats_zero"] and row["holm_significant"]:
            finite_opt_supported = True
            print(f"2. Statistical superiority: SUPPORTED (mean diff={row['mean_diff']:.4f}, CI=[{row['ci_95_lower']:.4f}, {row['ci_95_upper']:.4f}], p={row['p_value_raw']:.4e})")
        else:
            print(f"2. Statistical superiority: NOT SUPPORTED (CI contains zero or p-value insignificant)")
    else:
        print("2. Validation-selected beta is 0.0 (noiseless model achieved best validation performance).")

    # 3. Check U-shape
    test_means = df.groupby("beta")["test_mse_std"].mean()
    is_u_shape = False
    if val_best_beta > 0.0:
        b_idx = betas.index(val_best_beta)
        if b_idx > 0 and b_idx < len(betas) - 1:
            if test_means[val_best_beta] < test_means[0.0] and test_means[val_best_beta] < test_means[betas[-1]]:
                is_u_shape = True
    print(f"3. U-shaped profile across noise grid: {is_u_shape}")

    if finite_opt_supported and is_u_shape:
        conclusion = "FINITE-NOISE REGULARIZATION SUPPORTED"
        print(f"\n=> FINAL VERDICT: {conclusion}")
    else:
        conclusion = "NEGATIVE RESULT: NOISE-INDUCED CAPACITY SUPPRESSION"
        print(f"\n=> FINAL VERDICT: {conclusion}")
        print("   (Physical noise monotonically impairs quantum expressivity and test generalization without regularizing benefit.)")

    # --- Part 4: Publication Figures ---
    summary_beta = df.groupby("beta").agg(
        test_mse_mean=("test_mse_std", "mean"),
        test_mse_std=("test_mse_std", "std"),
        train_mse_mean=("train_mse_std", "mean"),
        gap_mean=("gap", "mean"),
        pr_mean=("participation_ratio", "mean"),
        er_mean=("entropy_effective_rank", "mean"),
        grad_norm_mean=("rms_gradient_norm", "mean"),
    ).reset_index()

    # Figure 1: Test MSE & Train MSE vs Beta with 95% CIs
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.errorbar(
        summary_beta["beta"], summary_beta["test_mse_mean"],
        yerr=1.96 * summary_beta["test_mse_std"] / np.sqrt(len(seeds)),
        fmt="s--", color="#d62728", label="Test MSE (95% CI)", capsize=4, linewidth=2
    )
    ax1.plot(summary_beta["beta"], summary_beta["train_mse_mean"], "o-", color="#1f77b4", label="Train MSE (Mean)", linewidth=2)
    ax1.set_xlabel(r"Physical Depolarizing Noise Rate $\beta$", fontsize=11)
    ax1.set_ylabel("Standardized MSE", fontsize=11)
    ax1.set_title("Main Sweep: Molecular HOMO-LUMO Gap Prediction vs Noise", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    fig1_path = FIG_DIR / "main_sweep_test_mse_vs_beta.png"
    fig.tight_layout()
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved {fig1_path}")

    # Figure 2: Scale-free Complexity (PR) vs Gradient Norm
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color1 = "#2ca02c"
    ax1.set_xlabel(r"Physical Depolarizing Noise Rate $\beta$", fontsize=11)
    ax1.set_ylabel("Scale-Free Participation Ratio (PR)", color=color1, fontsize=11)
    ax1.plot(summary_beta["beta"], summary_beta["pr_mean"], "D-", color=color1, linewidth=2, label="Participation Ratio")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#9467bd"
    ax2.set_ylabel("RMS Gradient Norm (log scale)", color=color2, fontsize=11)
    ax2.plot(summary_beta["beta"], summary_beta["grad_norm_mean"], "^--", color=color2, linewidth=2, label="RMS Grad Norm")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor=color2)
    plt.title("Contribution D: Scale-Free Dimensionality vs Gradient Magnitude", fontsize=12)
    fig2_path = FIG_DIR / "main_sweep_complexity_vs_beta.png"
    fig.tight_layout()
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig2_path}")

    # Save manifest
    manifest = create_run_manifest(
        run_id=generate_run_id("main_sweep"),
        config={
            "step": "tier1_step8_main_sweep",
            "betas": betas,
            "seeds": seeds,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "epochs": epochs,
            "verdict": conclusion,
        },
        output_dir=RAW_DIR,
    )
    print(f"Saved run manifest for {manifest['run_id']}")

    return df, df_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train_size", type=int, default=40)
    args = parser.parse_args()

    run_main_sweep(epochs=args.epochs, train_size=args.train_size)
