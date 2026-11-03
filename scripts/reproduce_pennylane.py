"""
Tier 1 Step 5: Reproduction Pilot for the PennyLane Experiment.

Reproduces the exact configuration from reference/FNQGNNMD/Finite_Noise_QGNN_QM9_Regularization_Demonstration.ipynb:
- 8 qubits, RY encoding (pi * x), CNOT-RZ-CNOT per bond with single-qubit noise on target wire, RX/RZ trainable layer
- Adam optimizer (lr=0.03, weight_decay=1e-4), 8 epochs, Xavier init on readout, random 0.1*randn on theta (17 params)
- First 2500 QM9 molecules in dataset order, filtered for <=8 heavy atoms
- Split: random_state=2026 for test (80), random_state=2027 for val (max(10, 0.2*trainval)), 40 train molecules (seed 42)
- Target standardized with training statistics only
- Seeds 42 and 1042 across beta in [0.0, 0.005, 0.01, 0.02, 0.05] (PennyLane convention)
- Classical Ridge regression baseline on the identical 40 train / 80 test split
- Saves raw results to results/raw/ and prints side-by-side comparison with original numbers.
"""

import sys
import os
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.datasets import QM9

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qgnn.simulator import DensityMatrixSimulator
from src.qgnn.seed_utils import create_run_manifest, generate_run_id, set_global_seed

# Original saved reference numbers from notebook cell outputs
ORIGINAL_PENNYLANE_RESULTS = {
    (42, 0.000): {"test_mse": 1.3822, "val_mse": 1.1546},
    (42, 0.050): {"test_mse": 1.3645, "val_mse": 1.1390},
    (1042, 0.000): {"test_mse": 1.2734, "val_mse": 1.0920},
    (1042, 0.050): {"test_mse": 1.2645, "val_mse": 1.0780},
}
ORIGINAL_RIDGE_RESULTS = {
    "test_mse": 1.3072,
    "test_mae_eV": 1.1750,
}


def build_compact_graphs(dataset_root="data/qm9", max_load=2500, max_nodes=8):
    """
    Build graph records identically to cell 7 of the reference notebook:
    - First max_load molecules from QM9
    - Atom type indicators: first 5 columns of x
    - type_map: 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00
    - feat[k] = type_map + 0.05 * min(degree, 4)
    - adj: (8, 8) float32
    - y: d.y[0, 4]
    """
    cache_path = Path("data/cache_pennylane_graphs.pt")
    if cache_path.exists():
        print(f"Loading cached PennyLane graphs from {cache_path}...")
        return torch.load(cache_path)

    print(f"Loading QM9 dataset for PennyLane pilot (first {max_load} molecules)...")
    dataset = QM9(root=dataset_root)
    limit = min(len(dataset), max_load) if max_load is not None else len(dataset)
    graphs = []

    type_map = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}

    for i in range(limit):
        d = dataset[i]
        x = d.x.detach().cpu().numpy()
        atom_type = np.argmax(x[:, :5], axis=1)
        heavy = np.where(atom_type != 0)[0]
        if len(heavy) == 0 or len(heavy) > max_nodes:
            continue

        slot = {int(idx): k for k, idx in enumerate(heavy)}
        adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        edges = []
        for u, v in d.edge_index.detach().cpu().numpy().T:
            u_int, v_int = int(u), int(v)
            if u_int in slot and v_int in slot and u_int != v_int:
                su, sv = slot[u_int], slot[v_int]
                adj[su, sv] = 1.0
                if su < sv and (su, sv) not in edges:
                    edges.append((su, sv))

        feat = np.zeros(max_nodes, dtype=np.float32)
        for idx, k in slot.items():
            feat[k] = type_map[int(atom_type[idx])] + 0.05 * min(adj[k].sum(), 4.0)

        y = float(d.y[0, 4].item())
        if not np.isfinite(y):
            continue

        graphs.append({
            "x": feat,
            "adj": adj,
            "edges": edges,
            "y": y,
            "source_idx": i,
        })

    print(f"Eligible molecules from first {limit}: {len(graphs)}")
    torch.save(graphs, cache_path)
    return graphs


class QGNNRegressorPennyLane(nn.Module):
    """
    QGNN Regressor matching cell 13 of the PennyLane reference notebook:
    - 8 qubits
    - theta: 17 parameters (1 RZ + 8 RX + 8 RZ) initialized as 0.1 * randn(17)
    - readout: Linear(8, 1) with Xavier uniform weight and zero bias
    """
    def __init__(self, sim: DensityMatrixSimulator, seed: int):
        super().__init__()
        self.sim = sim
        gen = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(0.1 * torch.randn(17, generator=gen, dtype=torch.float64))
        self.readout = nn.Linear(8, 1, dtype=torch.float64)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor, edges: list) -> torch.Tensor:
        z = self.sim.run_circuit(x, edges, self.theta)
        return self.readout(z).reshape(())


def run_pennylane_reproduction(dataset_root="data/qm9"):
    print("=" * 75)
    print("TIER 1 STEP 5: REPRODUCING PENNYLANE EXPERIMENT")
    print("=" * 75)

    graphs = build_compact_graphs(dataset_root=dataset_root, max_load=2500, max_nodes=8)

    # Recreate the exact split from cells 9 & 20
    CONFIG = {
        "seed": 42,
        "train_size": 40,
        "test_size": 80,
        "epochs": 8,
        "learning_rate": 0.03,
        "weight_decay": 1e-4,
        "noise_rates": [0.0, 0.005, 0.01, 0.02, 0.05],
        "seeds": [42, 1042],
    }

    indices = np.arange(len(graphs))
    trainval_idx, test_idx = train_test_split(indices, test_size=CONFIG["test_size"], random_state=2026)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=max(10, int(0.2 * len(trainval_idx))), random_state=2027)

    rng = np.random.default_rng(CONFIG["seed"])
    if len(train_idx) > CONFIG["train_size"]:
        train_idx = np.sort(rng.choice(train_idx, size=CONFIG["train_size"], replace=False))

    print(f"Data split: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

    # Target scaling fitted strictly on training set targets
    train_targets = np.array([graphs[int(i)]["y"] for i in train_idx]).reshape(-1, 1)
    scaler = StandardScaler().fit(train_targets)

    def y_scaled(i):
        return float(scaler.transform([[graphs[int(i)]["y"]]])[0, 0])

    def y_array(ids, scaled=True):
        y = np.array([graphs[int(i)]["y"] for i in ids]).reshape(-1, 1)
        return scaler.transform(y).ravel() if scaled else y.ravel()

    # Pre-extract tensors
    graph_tensors = []
    for g in graphs:
        graph_tensors.append({
            "x": torch.tensor(g["x"], dtype=torch.float64),
            "edges": g["edges"],
            "y": g["y"],
        })

    # --- Part 1: Ridge Baseline ---
    print("\n--- Running Classical Ridge Baseline ---")
    def extract_features(graph_list):
        X = []
        for g in graph_list:
            mean_feat = np.mean(g["x"])
            std_feat = np.std(g["x"])
            adj_upper = g["adj"][np.triu_indices(8, k=1)]
            vec = np.hstack([[mean_feat, std_feat], adj_upper])
            X.append(vec)
        return np.array(X)

    X_all = extract_features(graphs)
    y_all_scaled = y_array(np.arange(len(graphs)), scaled=True)

    X_train, y_train = X_all[train_idx], y_all_scaled[train_idx]
    X_test, y_test = X_all[test_idx], y_all_scaled[test_idx]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    ridge_train_pred = ridge.predict(X_train)
    ridge_test_pred = ridge.predict(X_test)

    ridge_test_pred_eV = scaler.inverse_transform(ridge_test_pred.reshape(-1, 1)).ravel()
    y_test_eV = y_array(test_idx, scaled=False)

    ridge_train_mse = float(mean_squared_error(y_train, ridge_train_pred))
    ridge_test_mse = float(mean_squared_error(y_test, ridge_test_pred))
    ridge_test_mae = float(mean_absolute_error(y_test_eV, ridge_test_pred_eV))

    print(f"Ridge Train MSE: {ridge_train_mse:.4f} | Test MSE: {ridge_test_mse:.4f} | Test MAE: {ridge_test_mae:.4f} eV")
    print(f"Original Notebook Ridge -> Test MSE: {ORIGINAL_RIDGE_RESULTS['test_mse']} | Test MAE: {ORIGINAL_RIDGE_RESULTS['test_mae_eV']}")

    # --- Part 2: QGNN Adam Reproduction ---
    results_records = []
    comparison_records = []

    def predict_model(model, ids):
        model.eval()
        preds = []
        with torch.no_grad():
            for idx in ids:
                gt = graph_tensors[int(idx)]
                p = model(gt["x"], gt["edges"]).item()
                preds.append(p)
        return np.array(preds, dtype=float)

    for seed in CONFIG["seeds"]:
        for beta in CONFIG["noise_rates"]:
            print(f"\n--- Running Seed={seed}, Beta={beta:5.3f} ---")
            t0 = time.time()

            # Set global and local seeds identically to reference
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            sim = DensityMatrixSimulator(n_qubits=8, experiment='pennylane', beta=beta)
            model = QGNNRegressorPennyLane(sim=sim, seed=seed)
            opt = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

            best_val = float("inf")
            best_state = None
            best_epoch = 0
            history = []

            for epoch in range(CONFIG["epochs"]):
                model.train()
                order = np.random.default_rng(seed + epoch).permutation(train_idx)
                for idx in order:
                    opt.zero_grad()
                    gt = graph_tensors[int(idx)]
                    pred = model(gt["x"], gt["edges"])
                    target = torch.tensor(y_scaled(idx), dtype=torch.float64)
                    loss = (pred - target) ** 2
                    loss.backward()
                    opt.step()

                # Validation checkpointing
                train_pred = predict_model(model, train_idx)
                val_pred = predict_model(model, val_idx)
                tr_mse = float(mean_squared_error(y_array(train_idx), train_pred))
                va_mse = float(mean_squared_error(y_array(val_idx), val_pred))
                history.append({"epoch": epoch + 1, "train_mse": tr_mse, "val_mse": va_mse})

                if va_mse < best_val:
                    best_val = va_mse
                    best_epoch = epoch + 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Restore best checkpoint
            model.load_state_dict(best_state)
            tr_pred = predict_model(model, train_idx)
            va_pred = predict_model(model, val_idx)
            te_pred = predict_model(model, test_idx)

            te_eV = scaler.inverse_transform(te_pred.reshape(-1, 1)).ravel()
            y_te_eV = y_array(test_idx, scaled=False)

            train_mse_std = float(mean_squared_error(y_array(train_idx), tr_pred))
            val_mse_std = float(mean_squared_error(y_array(val_idx), va_pred))
            test_mse_std = float(mean_squared_error(y_array(test_idx), te_pred))
            test_mae_eV = float(mean_absolute_error(y_te_eV, te_eV))
            gap = test_mse_std - train_mse_std

            t1 = time.time()
            print(f"  Best epoch: {best_epoch} | Time: {t1-t0:.1f}s")
            print(f"  Train MSE: {train_mse_std:.4f} | Val MSE: {val_mse_std:.4f} | Test MSE: {test_mse_std:.4f} | MAE: {test_mae_eV:.4f} eV")

            rec = {
                "seed": seed,
                "beta": beta,
                "best_epoch": best_epoch,
                "train_mse_standardized": train_mse_std,
                "val_mse_standardized": val_mse_std,
                "test_mse_standardized": test_mse_std,
                "test_mae_eV": test_mae_eV,
                "generalization_gap": gap,
                "elapsed_seconds": t1 - t0,
            }
            results_records.append(rec)

            orig = ORIGINAL_PENNYLANE_RESULTS.get((seed, beta), {})
            comparison_records.append({
                "seed": seed,
                "beta": beta,
                "orig_test_mse": orig.get("test_mse", np.nan),
                "new_test_mse": test_mse_std,
                "test_mse_diff": abs(test_mse_std - orig.get("test_mse", np.nan)),
                "orig_val_mse": orig.get("val_mse", np.nan),
                "new_val_mse": val_mse_std,
            })

    # Save results
    results_dir = Path("results/raw")
    results_dir.mkdir(parents=True, exist_ok=True)

    df_res = pd.DataFrame(results_records)
    df_comp = pd.DataFrame(comparison_records)
    df_res.to_csv(results_dir / "reproduction_pennylane_results.csv", index=False)
    df_comp.to_csv(results_dir / "reproduction_pennylane_comparison.csv", index=False)

    # Save Ridge results
    df_ridge = pd.DataFrame([{
        "model": "Ridge",
        "train_mse_standardized": ridge_train_mse,
        "test_mse_standardized": ridge_test_mse,
        "test_mae_eV": ridge_test_mae,
        "orig_test_mse": ORIGINAL_RIDGE_RESULTS["test_mse"],
        "orig_test_mae_eV": ORIGINAL_RIDGE_RESULTS["test_mae_eV"],
    }])
    df_ridge.to_csv(results_dir / "reproduction_ridge_results.csv", index=False)

    print("\n" + "=" * 75)
    print("PENNYLANE REPRODUCTION SUMMARY (Test MSE standardized):")
    print("=" * 75)
    print(df_comp.to_string(index=False))

    manifest = create_run_manifest(
        run_id=generate_run_id("reproduce_pennylane"),
        config=CONFIG,
        output_dir=results_dir,
    )
    print(f"\nManifest saved for run {manifest['run_id']}")
    return df_res, df_comp


if __name__ == "__main__":
    run_pennylane_reproduction()
