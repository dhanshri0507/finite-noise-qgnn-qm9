"""
Milestone 8: Noiseless QGNN + Post-hoc Output Shrinkage Control.

Control experiment to test whether the apparent noise improvement in QGNN is
simply due to trivial output variance shrinkage (multiplicative rescaling of predictions).

Protocol:
  For each seed:
    1. Train noiseless QGNN (beta=0) with same training protocol as main sweep.
    2. Fit optimal scalar shrinkage c* on VALIDATION set:
       c* = argmin_c MSE(c * pred_val, y_val) = sum(pred_val * y_val) / sum(pred_val^2)
       (closed-form OLS solution for scaling only)
    3. Evaluate: c* * pred_test on test set.
    4. Compare test MSE of [noiseless, shrunk-noiseless, best-noisy] models.

If shrunk noiseless ≈ best noisy: noise effect is JUST trivial output shrinkage.
If shrunk noiseless > best noisy: noise provides genuine regularization beyond shrinkage.

Saves: results/raw/shrinkage_control_results.csv
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qgnn.simulator import DensityMatrixSimulator
from src.qgnn.seed_utils import set_global_seed
from scripts.reproduce_pennylane import build_compact_graphs

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


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


def fit_shrinkage(pred_val, y_val):
    """
    Closed-form OLS: fit c* = argmin_c MSE(c * pred_val, y_val).
    c* = (pred_val . y_val) / (pred_val . pred_val)
    """
    p = np.array(pred_val)
    y = np.array(y_val)
    if np.dot(p, p) < 1e-12:
        return 1.0
    return float(np.dot(p, y) / np.dot(p, p))


def run_shrinkage_control(train_size=40, val_size=40, test_size=80, epochs=15, seeds=None):
    if seeds is None:
        seeds = [42, 1042, 2024, 3141, 5321]

    print("=" * 70)
    print("Milestone 8: Post-hoc Output Shrinkage Control")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    graphs = build_compact_graphs(dataset_root="data/qm9", max_load=2500, max_nodes=8)
    
    # Load best noisy results for comparison (validation-selected per seed)
    main_file = RAW_DIR / "main_sweep_results.csv"
    if main_file.exists():
        df_main = pd.read_csv(main_file)
        # Validation-selected best beta per seed (must not look at test)
        best_noisy = (df_main.groupby("seed")
                     .apply(lambda g: g.loc[g["val_mse_std"].idxmin()])
                     .reset_index(drop=True))
        best_noisy_dict = best_noisy.set_index("seed")["test_mse_std"].to_dict()
        best_beta_dict = best_noisy.set_index("seed")["beta"].to_dict()
    else:
        best_noisy_dict = {}
        best_beta_dict = {}

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
            y_std = float(scaler.transform([[g["y"]]])[0, 0])
            return {"x": torch.tensor(g["x"], dtype=torch.float64),
                    "edges": g["edges"],
                    "y_std": torch.tensor(y_std, dtype=torch.float64),
                    "y_raw": g["y"]}

        train_data = [make_sample(i) for i in tr_idx]
        val_data   = [make_sample(i) for i in va_idx]
        test_data  = [make_sample(i) for i in te_idx]

        # Train NOISELESS model (beta=0)
        sim = DensityMatrixSimulator(n_qubits=8, experiment='pennylane', beta=0.0)
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

            model.eval()
            with torch.no_grad():
                va_mse = np.mean([
                    ((model(s["x"], s["edges"]) - s["y_std"]) ** 2).item()
                    for s in val_data
                ])
            if va_mse < best_val:
                best_val = va_mse
                best_epoch = ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()

        # Get raw predictions on val and test
        with torch.no_grad():
            pred_val_std = np.array([model(s["x"], s["edges"]).item() for s in val_data])
            y_val_std    = np.array([s["y_std"].item() for s in val_data])
            pred_test_std = np.array([model(s["x"], s["edges"]).item() for s in test_data])
            y_test_std    = np.array([s["y_std"].item() for s in test_data])
            y_test_raw    = np.array([s["y_raw"] for s in test_data])

        # Noiseless test MSE (no shrinkage)
        noiseless_test_mse = float(mean_squared_error(y_test_std, pred_test_std))

        # Fit shrinkage on validation
        c_star = fit_shrinkage(pred_val_std, y_val_std)

        # Evaluate shrunk model on test
        pred_test_shrunk = c_star * pred_test_std
        shrunk_test_mse = float(mean_squared_error(y_test_std, pred_test_shrunk))
        pred_test_shrunk_raw = scaler.inverse_transform(pred_test_shrunk.reshape(-1, 1)).ravel()
        shrunk_test_mae = float(mean_absolute_error(y_test_raw, pred_test_shrunk_raw))

        # Noiseless val MSE
        noiseless_val_mse = float(mean_squared_error(y_val_std, pred_val_std))
        shrunk_val_mse = float(mean_squared_error(y_val_std, pred_val_std * c_star))

        # Best noisy from main sweep
        best_noisy_mse = best_noisy_dict.get(seed, float('nan'))
        best_beta = best_beta_dict.get(seed, float('nan'))

        elapsed = time.time() - t0
        print(f"\nSeed={seed}: BestEp={best_epoch} | t={elapsed:.1f}s")
        print(f"  Noiseless test MSE:      {noiseless_test_mse:.4f}")
        print(f"  Shrinkage c*={c_star:.4f} | Shrunk test MSE: {shrunk_test_mse:.4f}")
        print(f"  Best noisy (beta={best_beta:.3f}) test MSE: {best_noisy_mse:.4f}")

        shrinkage_explains = shrunk_test_mse <= best_noisy_mse + 0.001  # within 0.001
        print(f"  Shrinkage explains noise benefit? {'YES' if shrinkage_explains else 'NO'}")

        records.append({
            "seed": seed,
            "noiseless_val_mse": noiseless_val_mse,
            "noiseless_test_mse": noiseless_test_mse,
            "shrinkage_c_star": c_star,
            "shrunk_val_mse": shrunk_val_mse,
            "shrunk_test_mse": shrunk_test_mse,
            "shrunk_test_mae_eV": shrunk_test_mae,
            "best_noisy_beta": best_beta,
            "best_noisy_test_mse": best_noisy_mse,
            "diff_shrunk_minus_noisy": shrunk_test_mse - best_noisy_mse,
            "shrinkage_explains": shrinkage_explains,
            "best_epoch_noiseless": best_epoch,
        })

    df = pd.DataFrame(records)
    out_file = RAW_DIR / "shrinkage_control_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\nShrinkage control results saved to {out_file}")

    print("\n--- Summary ---")
    print(df[["seed", "noiseless_test_mse", "shrinkage_c_star", "shrunk_test_mse",
              "best_noisy_test_mse", "diff_shrunk_minus_noisy", "shrinkage_explains"]].to_string(index=False))

    n_explains = df["shrinkage_explains"].sum()
    n_total = len(df)
    mean_diff = df["diff_shrunk_minus_noisy"].mean()
    print(f"\nShrinkage explains benefit in {n_explains}/{n_total} seeds.")
    print(f"Mean diff (shrunk - noisy): {mean_diff:.4f} (negative = noisy is still better than shrunk noiseless)")

    if n_explains >= n_total * 0.5:
        verdict = "CAUTION: Post-hoc shrinkage of noiseless model matches or exceeds best noisy model in most seeds. The apparent noise benefit may be trivial output contraction, not genuine regularization."
    else:
        verdict = "INFORMATIVE: Shrunk noiseless model does NOT match best noisy model in most seeds, suggesting noise provides regularization beyond trivial output contraction."

    print(f"\nVERDICT: {verdict}")
    df.attrs["verdict"] = verdict
    return df


if __name__ == "__main__":
    import pandas as pd
    main_file = Path("results/raw/main_sweep_results.csv")
    if main_file.exists():
        completed_seeds = sorted(pd.read_csv(main_file)["seed"].unique().tolist())
    else:
        completed_seeds = [42, 1042, 2024, 3141, 5321]
    run_shrinkage_control(seeds=completed_seeds)
