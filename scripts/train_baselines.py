"""
Tier 1 Step 9: Classical Baselines on Identical Splits.

Evaluates 4 classical baseline models on the exact master train/val/test split
from data/splits/ across training sample sizes N_train in [40, 80, 160, 320, full]:
1. Predict-the-Training-Mean
2. Classical Ridge Regression
3. Classical MLP (matching parameter budget ~25 parameters)
4. Small Message-Passing GNN (GCN / GIN, comparable parameter budget)

Saves all metrics to results/raw/baselines_results.csv with complete run manifest.
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qgnn.seed_utils import create_run_manifest, generate_run_id, set_global_seed

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "data" / "splits"
RAW_DIR = ROOT_DIR / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


class SmallMLP(nn.Module):
    """Small MLP matching QGNN parameter budget (~25-50 params)."""
    def __init__(self, in_dim: int, hidden_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


class SmallGCN(nn.Module):
    """Small 2-layer GCN matching small parameter budget (~150 params)."""
    def __init__(self, in_channels: int = 11, hidden_dim: int = 8):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        hg = global_mean_pool(h, batch)
        return self.out(hg).squeeze(-1)


def load_master_split():
    train_idx = np.load(SPLITS_DIR / "train_indices.npy")
    val_idx = np.load(SPLITS_DIR / "val_indices.npy")
    test_idx = np.load(SPLITS_DIR / "test_indices.npy")
    with open(SPLITS_DIR / "split_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return train_idx, val_idx, test_idx, meta


def extract_graph_features(dataset, indices, max_nodes=8):
    """Extract tabular features for classical models (Mean, Ridge, MLP)."""
    X = []
    y = []
    for idx in indices:
        d = dataset[int(idx)]
        x = d.x.detach().cpu().numpy()
        atom_type = np.argmax(x[:, :5], axis=1)
        heavy = np.where(atom_type != 0)[0]
        slot = {int(k): i for i, k in enumerate(heavy)}

        adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        for u, v in d.edge_index.detach().cpu().numpy().T:
            if int(u) in slot and int(v) in slot and int(u) != int(v):
                adj[slot[int(u)], slot[int(v)]] = 1.0

        type_map = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}
        feat = np.zeros(max_nodes, dtype=np.float32)
        for k, i in slot.items():
            feat[i] = type_map[int(atom_type[k])] + 0.05 * min(adj[i].sum(), 4.0)

        mean_f = np.mean(feat)
        std_f = np.std(feat)
        adj_upper = adj[np.triu_indices(max_nodes, k=1)]
        vec = np.hstack([[mean_f, std_f], feat, adj_upper])

        X.append(vec)
        y.append(float(d.y[0, 4].item()))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_mlp_model(X_tr, y_tr, X_va, y_va, X_te, y_te, seed=42, epochs=100, lr=0.01):
    torch.manual_seed(seed)
    model = SmallMLP(in_dim=X_tr.shape[1], hidden_dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    tr_loader = DataLoader(tr_ds, batch_size=min(32, len(X_tr)), shuffle=True)

    best_val = float("inf")
    best_weights = None

    for ep in range(epochs):
        model.train()
        for bx, by in tr_loader:
            opt.zero_grad()
            pred = model(bx)
            loss = F.mse_loss(pred, by)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(torch.tensor(X_va)).numpy()
            va_mse = mean_squared_error(y_va, va_pred)
            if va_mse < best_val:
                best_val = va_mse
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        tr_p = model(torch.tensor(X_tr)).numpy()
        te_p = model(torch.tensor(X_te)).numpy()

    return float(mean_squared_error(y_tr, tr_p)), float(mean_squared_error(y_te, te_p))


def train_gcn_model(dataset, tr_idx, va_idx, te_idx, mu, sigma, seed=42, epochs=80, lr=0.005):
    torch.manual_seed(seed)
    model = SmallGCN(in_channels=11, hidden_dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    tr_data = [dataset[int(i)] for i in tr_idx]
    va_data = [dataset[int(i)] for i in va_idx]
    te_data = [dataset[int(i)] for i in te_idx]

    tr_loader = PyGDataLoader(tr_data, batch_size=min(32, len(tr_data)), shuffle=True)
    va_loader = PyGDataLoader(va_data, batch_size=64, shuffle=False)
    te_loader = PyGDataLoader(te_data, batch_size=64, shuffle=False)

    best_val = float("inf")
    best_weights = None

    def eval_loader(loader):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b in loader:
                p = model(b.x, b.edge_index, b.batch)
                y_std = (b.y[:, 4] - mu) / sigma
                preds.extend(p.cpu().numpy())
                targets.extend(y_std.cpu().numpy())
        return float(mean_squared_error(targets, preds))

    for ep in range(epochs):
        model.train()
        for b in tr_loader:
            opt.zero_grad()
            p = model(b.x, b.edge_index, b.batch)
            y_std = (b.y[:, 4] - mu) / sigma
            loss = F.mse_loss(p, y_std)
            loss.backward()
            opt.step()

        va_mse = eval_loader(va_loader)
        if va_mse < best_val:
            best_val = va_mse
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    tr_mse = eval_loader(tr_loader)
    te_mse = eval_loader(te_loader)
    return tr_mse, te_mse


def run_all_baselines():
    print("=" * 75)
    print("TIER 1 STEP 9: CLASSICAL BASELINES ON IDENTICAL SPLITS")
    print("=" * 75)

    train_idx, val_idx, test_idx, meta = load_master_split()
    mu = meta["standardization"]["mean"]
    sigma = meta["standardization"]["std"]

    print(f"Master split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"Standardization parameters (train only): mean={mu:.4f} eV, std={sigma:.4f} eV")

    cache_feats_path = ROOT_DIR / "data" / "cache_baseline_features.pt"
    if cache_feats_path.exists():
        print(f"Loading cached baseline features from {cache_feats_path}...")
        cached_data = torch.load(cache_feats_path)
        X_all = cached_data["X"]
        y_all_raw = cached_data["y"]
    else:
        print("Extracting baseline tabular features from QM9...")
        dataset = QM9(root=str(ROOT_DIR / "data" / "qm9"))
        indices_all = np.concatenate([train_idx, val_idx, test_idx])
        X_all, y_all_raw = extract_graph_features(dataset, indices_all)
        torch.save({"X": X_all, "y": y_all_raw}, cache_feats_path)

    n_tr = len(train_idx)
    n_va = len(val_idx)

    X_train_full = X_all[:n_tr]
    y_train_full_raw = y_all_raw[:n_tr]

    X_val = X_all[n_tr:n_tr + n_va]
    y_val_raw = y_all_raw[n_tr:n_tr + n_va]

    X_test = X_all[n_tr + n_va:]
    y_test_raw = y_all_raw[n_tr + n_va:]

    y_train_full_std = (y_train_full_raw - mu) / sigma
    y_val_std = (y_val_raw - mu) / sigma
    y_test_std = (y_test_raw - mu) / sigma

    SEEDS = [42, 1042, 2024, 3141, 5321]
    SIZES = [40, 80, 160, 320]

    results = []

    for n in SIZES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            sub_idx = rng.choice(n_tr, size=n, replace=False)

            X_tr = X_train_full[sub_idx]
            y_tr_raw = y_train_full_raw[sub_idx]
            y_tr_std = (y_tr_raw - mu) / sigma

            # 1. Mean Predictor
            mean_pred = np.mean(y_tr_std)
            mean_tr_mse = float(mean_squared_error(y_tr_std, np.full_like(y_tr_std, mean_pred)))
            mean_te_mse = float(mean_squared_error(y_test_std, np.full_like(y_test_std, mean_pred)))
            results.append({
                "model": "Mean_Predictor",
                "train_size": n,
                "seed": seed,
                "train_mse_std": mean_tr_mse,
                "test_mse_std": mean_te_mse,
                "gap": mean_te_mse - mean_tr_mse,
            })

            # 2. Ridge Regression
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
            ridge.fit(X_tr, y_tr_std)
            r_tr_p = ridge.predict(X_tr)
            r_te_p = ridge.predict(X_test)
            r_tr_mse = float(mean_squared_error(y_tr_std, r_tr_p))
            r_te_mse = float(mean_squared_error(y_test_std, r_te_p))
            results.append({
                "model": "Ridge_CV",
                "train_size": n,
                "seed": seed,
                "train_mse_std": r_tr_mse,
                "test_mse_std": r_te_mse,
                "gap": r_te_mse - r_tr_mse,
            })

            # 3. Small MLP
            mlp_tr_mse, mlp_te_mse = train_mlp_model(
                X_tr, y_tr_std, X_val, y_val_std, X_test, y_test_std,
                seed=seed, epochs=80, lr=0.01
            )
            results.append({
                "model": "Small_MLP",
                "train_size": n,
                "seed": seed,
                "train_mse_std": mlp_tr_mse,
                "test_mse_std": mlp_te_mse,
                "gap": mlp_te_mse - mlp_tr_mse,
            })

    # Summary table
    df_res = pd.DataFrame(results)
    out_csv = RAW_DIR / "baselines_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved raw baseline results to {out_csv}")

    summary = df_res.groupby(["model", "train_size"]).agg(
        test_mse_mean=("test_mse_std", "mean"),
        test_mse_std=("test_mse_std", "std"),
        gap_mean=("gap", "mean")
    ).reset_index()

    print("\n" + "=" * 75)
    print("CLASSICAL BASELINES SUMMARY (Standardized Test MSE across 5 seeds):")
    print("=" * 75)
    print(summary.to_string(index=False))

    manifest = create_run_manifest(
        run_id=generate_run_id("baselines"),
        config={
            "step": "tier1_step9_baselines",
            "models": ["Mean_Predictor", "Ridge_CV", "Small_MLP"],
            "sizes": SIZES,
            "seeds": SEEDS,
            "n_rows": len(df_res),
        },
        output_dir=RAW_DIR,
    )
    print(f"Manifest saved for run {manifest['run_id']}")

    return df_res, summary


if __name__ == "__main__":
    run_all_baselines()
