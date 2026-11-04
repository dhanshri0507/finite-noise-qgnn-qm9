"""
Milestone 7: GCN Baseline — Size-matched classical Graph Convolutional Network.

Implements a 2-layer GCN with comparable parameter budget to the QGNN (~25 parameters).
Evaluates on the SAME train/val/test splits as the main QGNN sweep (all 10 seeds).
Produces:
  results/raw/gcn_baseline_results.csv    — per-seed metrics
  results/raw/gcn_vs_qgnn_comparison.csv — paired comparison table
  results/figures/gcn_vs_qgnn_test_mse.png
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qgnn.seed_utils import set_global_seed, create_run_manifest, generate_run_id
from scripts.reproduce_pennylane import build_compact_graphs

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "results" / "raw"
FIG_DIR = ROOT_DIR / "results" / "figures"
RAW_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


class SmallGCN(nn.Module):
    """
    2-layer GCN for molecular graph regression.
    
    Parameter budget matches QGNN (~25 total):
      QGNN: n_theta=17 (circuit) + 8*1+1=9 (readout linear) = 26 params
      GCN:  hidden_dim=3: input_proj(1->3,no bias)=3 + GCN1(3->3)=9 + GCN2(3->3)=9 + 
            readout(3->1+bias)=4 = 25 params
    
    Graph convolution: h_v = ReLU(W * mean_{u in N(v)+{v}} h_u)
    Uses symmetric normalized adjacency (self-loop inclusive).
    """
    def __init__(self, in_features=1, hidden=3, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.proj = nn.Linear(in_features, hidden, bias=False)
        self.gcn1 = nn.Linear(hidden, hidden, bias=False)
        self.gcn2 = nn.Linear(hidden, hidden, bias=False)
        self.readout = nn.Linear(hidden, 1)
        # Xavier init
        for m in [self.proj, self.gcn1, self.gcn2]:
            nn.init.xavier_uniform_(m.weight.float())
            m.weight = nn.Parameter(m.weight.double())
        nn.init.xavier_uniform_(self.readout.weight.float())
        self.readout.weight = nn.Parameter(self.readout.weight.double())
        self.readout.bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, x, adj):
        """
        x: (n_nodes, in_features) float64
        adj: (n_nodes, n_nodes) pre-built adjacency matrix (binary)
        Returns: scalar prediction
        """
        n_nodes = x.shape[0]
        # Add self-loops to adjacency
        adj_sl = adj + torch.eye(n_nodes, dtype=torch.float64)
        # Row-normalize
        deg = adj_sl.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj_norm = adj_sl / deg

        h = self.proj(x)  # (n, hidden)
        h = torch.relu(self.gcn1(adj_norm @ h))
        h = torch.relu(self.gcn2(adj_norm @ h))
        # Global mean pool
        h_pool = h.mean(dim=0)
        return self.readout(h_pool).squeeze()

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def run_gcn_baseline(train_size=40, val_size=40, test_size=80, epochs=200, seeds=None):
    if seeds is None:
        seeds = [42, 1042, 2024, 3141, 5321, 6789, 7777, 8888, 9999, 12345]

    print("=" * 70)
    print("Milestone 7: GCN Baseline (size-matched to QGNN)")
    print(f"Seeds: {seeds}")
    print(f"Train={train_size}, Val={val_size}, Test={test_size}, Epochs={epochs}")
    print("=" * 70)

    graphs = build_compact_graphs(dataset_root="data/qm9", max_load=2500, max_nodes=8)

    # Count QGNN params for reference
    gcn_ref = SmallGCN(in_features=1, hidden=3, seed=42)
    n_gcn_params = gcn_ref.count_params()
    print(f"GCN parameter count: {n_gcn_params}")
    print(f"QGNN parameter count: 17 (circuit) + 9 (readout) = 26")

    records = []
    for seed in seeds:
        t0 = time.time()
        set_global_seed(seed)

        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(graphs))
        tr_idx = shuffled[:train_size]
        va_idx = shuffled[train_size:train_size + val_size]
        te_idx = shuffled[train_size + val_size:train_size + val_size + test_size]

        tr_targets = np.array([graphs[int(i)]["y"] for i in tr_idx]).reshape(-1, 1)
        scaler = StandardScaler().fit(tr_targets)

        def make_sample(i):
            g = graphs[int(i)]
            # x is (8,) flat vector — reshape to (n_nodes, 1)
            x = torch.tensor(g["x"], dtype=torch.float64).unsqueeze(-1)  # (8, 1)
            # adj is (8, 8) numpy array — use it for GCN
            adj = torch.tensor(g["adj"], dtype=torch.float64)  # (8, 8)
            y_std = float(scaler.transform([[g["y"]]])[0, 0])
            return {
                "x": x,
                "adj": adj,
                "edges": g["edges"],
                "n_nodes": x.shape[0],
                "y_std": torch.tensor(y_std, dtype=torch.float64),
                "y_raw": g["y"],
            }

        train_data = [make_sample(i) for i in tr_idx]
        val_data   = [make_sample(i) for i in va_idx]
        test_data  = [make_sample(i) for i in te_idx]

        model = SmallGCN(in_features=train_data[0]["x"].shape[1], hidden=3, seed=seed)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        best_val = float("inf")
        best_state = None
        best_epoch = 0

        for ep in range(1, epochs + 1):
            model.train()
            order = np.random.default_rng(seed + ep).permutation(len(train_data))
            for idx in order:
                s = train_data[idx]
                opt.zero_grad()
                pred = model(s["x"], s["adj"])
                loss = (pred - s["y_std"]) ** 2
                loss.backward()
                opt.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                va_mse = np.mean([
                    ((model(s["x"], s["adj"]) - s["y_std"]) ** 2).item()
                    for s in val_data
                ])
            if va_mse < best_val:
                best_val = va_mse
                best_epoch = ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()

        def eval_set(dataset):
            ps, ts, tr = [], [], []
            with torch.no_grad():
                for s in dataset:
                    p = model(s["x"], s["adj"]).item()
                    ps.append(p)
                    ts.append(s["y_std"].item())
                    tr.append(s["y_raw"])
            ps = np.array(ps)
            ts = np.array(ts)
            pr = scaler.inverse_transform(ps.reshape(-1, 1)).ravel()
            return float(mean_squared_error(ts, ps)), float(mean_absolute_error(tr, pr))

        tr_mse, tr_mae = eval_set(train_data)
        va_mse, va_mae = eval_set(val_data)
        te_mse, te_mae = eval_set(test_data)
        elapsed = time.time() - t0

        print(f"  Seed={seed}: BestEp={best_epoch} | Tr={tr_mse:.4f} | Val={va_mse:.4f} | Test={te_mse:.4f} | MAE={te_mae:.4f}eV | t={elapsed:.1f}s")
        records.append({
            "model": "GCN",
            "seed": seed,
            "n_params": n_gcn_params,
            "best_epoch": best_epoch,
            "train_mse_std": tr_mse,
            "train_mae_eV": tr_mae,
            "val_mse_std": va_mse,
            "val_mae_eV": va_mae,
            "test_mse_std": te_mse,
            "test_mae_eV": te_mae,
            "gap": te_mse - tr_mse,
            "elapsed_seconds": elapsed,
        })

    df_gcn = pd.DataFrame(records)
    out_file = RAW_DIR / "gcn_baseline_results.csv"
    df_gcn.to_csv(out_file, index=False)
    print(f"\nGCN results saved to {out_file}")

    # Build paired comparison with QGNN best (beta=0.0, noiseless)
    # Load main sweep — use beta=0 as the noiseless QGNN baseline
    main_file = RAW_DIR / "main_sweep_results.csv"
    if main_file.exists():
        df_qgnn = pd.read_csv(main_file)
        df_qgnn0 = df_qgnn[df_qgnn["beta"] == 0.0].copy()

        # Merge on seed
        merged = pd.merge(
            df_gcn[["seed", "test_mse_std", "test_mae_eV", "train_mse_std"]].rename(
                columns={"test_mse_std": "gcn_test_mse", "test_mae_eV": "gcn_test_mae",
                         "train_mse_std": "gcn_train_mse"}
            ),
            df_qgnn0[["seed", "test_mse_std", "test_mae_eV", "train_mse_std"]].rename(
                columns={"test_mse_std": "qgnn_test_mse", "test_mae_eV": "qgnn_test_mae",
                         "train_mse_std": "qgnn_train_mse"}
            ),
            on="seed", how="inner"
        )
        merged["diff_test_mse"] = merged["gcn_test_mse"] - merged["qgnn_test_mse"]  # positive = GCN worse
        merged["qgnn_better"] = (merged["diff_test_mse"] > 0).astype(int)

        comp_file = RAW_DIR / "gcn_vs_qgnn_comparison.csv"
        merged.to_csv(comp_file, index=False)
        print(f"\nGCN vs QGNN comparison saved to {comp_file}")

        print("\n--- Paired QGNN (beta=0) vs GCN Comparison ---")
        print(f"{'Seed':>6} | {'QGNN Test MSE':>14} | {'GCN Test MSE':>13} | {'Diff (GCN-QGNN)':>16} | {'QGNN Better?':>13}")
        print("-" * 70)
        for _, row in merged.iterrows():
            better = "YES" if row["qgnn_better"] else "NO"
            print(f"{int(row['seed']):>6} | {row['qgnn_test_mse']:>14.4f} | {row['gcn_test_mse']:>13.4f} | {row['diff_test_mse']:>16.4f} | {better:>13}")

        n_seeds = len(merged)
        n_qgnn_better = int(merged["qgnn_better"].sum())
        mean_diff = merged["diff_test_mse"].mean()
        _, p_val = stats.ttest_rel(merged["gcn_test_mse"].values, merged["qgnn_test_mse"].values)
        print(f"\nSummary: QGNN beats GCN in {n_qgnn_better}/{n_seeds} seeds")
        print(f"Mean GCN-QGNN diff: {mean_diff:.4f} (positive=QGNN better)")
        print(f"Paired t-test p-value: {p_val:.4f}")

        # Save summary stats
        summary = {
            "n_seeds": n_seeds,
            "qgnn_mean_test_mse": float(merged["qgnn_test_mse"].mean()),
            "qgnn_std_test_mse": float(merged["qgnn_test_mse"].std()),
            "gcn_mean_test_mse": float(merged["gcn_test_mse"].mean()),
            "gcn_std_test_mse": float(merged["gcn_test_mse"].std()),
            "mean_diff_gcn_minus_qgnn": float(mean_diff),
            "qgnn_better_count": n_qgnn_better,
            "paired_ttest_pvalue": float(p_val),
            "verdict": (
                f"QGNN outperforms GCN in {n_qgnn_better}/{n_seeds} seeds. "
                f"Mean test MSE: QGNN={merged['qgnn_test_mse'].mean():.4f}, GCN={merged['gcn_test_mse'].mean():.4f}. "
                f"p={p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'})."
            )
        }
        with open(RAW_DIR / "gcn_vs_qgnn_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))

        gcn_mean = df_gcn.groupby("seed")["test_mse_std"].mean()
        if main_file.exists():
            qgnn_mean = df_qgnn0.set_index("seed")["test_mse_std"]
            seeds_common = sorted(set(gcn_mean.index) & set(qgnn_mean.index))
            x = range(len(seeds_common))
            ax.bar([i - 0.2 for i in x], [qgnn_mean[s] for s in seeds_common],
                   0.4, label="QGNN (β=0)", color="#1f77b4", alpha=0.8)
            ax.bar([i + 0.2 for i in x], [gcn_mean[s] for s in seeds_common],
                   0.4, label="GCN (classical)", color="#ff7f0e", alpha=0.8)
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(s) for s in seeds_common], rotation=30, ha="right")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Test MSE (standardized)")
        ax.set_title("QGNN (noiseless, β=0) vs GCN Baseline — Per-seed Test MSE")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig_path = FIG_DIR / "gcn_vs_qgnn_test_mse.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"\nFigure saved to {fig_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    return df_gcn


if __name__ == "__main__":
    # Use only completed seeds from main sweep; fallback to first 5 if sweep not done
    import pandas as pd
    main_file = Path("results/raw/main_sweep_results.csv")
    if main_file.exists():
        completed_seeds = pd.read_csv(main_file)["seed"].unique().tolist()
    else:
        completed_seeds = [42, 1042, 2024, 3141, 5321]
    run_gcn_baseline(seeds=completed_seeds)
