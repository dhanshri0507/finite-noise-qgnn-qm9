"""
Tier 1 Step 7: Proper QGNN Training with Adam, Random Init, and Early Stopping.

Demonstrates proper optimization of the QGNN model:
- Random Gaussian initialization theta ~ N(0, 0.1^2) (avoids zero-init stationary point)
- Adam optimizer with CosineAnnealingLR schedule
- Early stopping on validation loss with model checkpointing
- Saves convergence loss curves to results/figures/
- Direct comparison with undertrained/zero-init baselines
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
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qgnn.simulator import DensityMatrixSimulator
from src.qgnn.seed_utils import create_run_manifest, generate_run_id, set_global_seed

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "results" / "raw"
FIG_DIR = ROOT_DIR / "results" / "figures"
MODELS_DIR = ROOT_DIR / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ProperQGNNModel(nn.Module):
    """
    QGNN Model with proper non-zero random initialization and trainable readout.
    Supports both 'pennylane' (17 params + Linear(8, 1)) and 'qiskit' (8 params + Linear(1, 1)).
    """
    def __init__(self, sim: DensityMatrixSimulator, experiment: str = "pennylane", seed: int = 42):
        super().__init__()
        self.sim = sim
        self.experiment = experiment
        gen = torch.Generator().manual_seed(seed)

        if experiment == "pennylane":
            n_theta = 1 + 2 * sim.n_qubits  # 17
            self.theta = nn.Parameter(0.1 * torch.randn(n_theta, generator=gen, dtype=torch.float64))
            self.readout = nn.Linear(sim.n_qubits, 1, dtype=torch.float64)
            nn.init.xavier_uniform_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)
        else:  # qiskit
            n_theta = sim.n_qubits  # 8
            self.theta = nn.Parameter(0.1 * torch.randn(n_theta, generator=gen, dtype=torch.float64))
            self.readout = nn.Linear(1, 1, dtype=torch.float64)
            nn.init.ones_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

    def forward(self, x: torch.Tensor, edges: list) -> torch.Tensor:
        out = self.sim.run_circuit(x, edges, self.theta)
        if self.experiment == "qiskit":
            out = out.unsqueeze(0)  # (1,)
        return self.readout(out).reshape(())


def train_proper_qgnn(
    experiment: str = "pennylane",
    beta: float = 0.0,
    seed: int = 42,
    train_size: int = 40,
    val_size: int = 40,
    epochs: int = 40,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    patience: int = 10,
):
    print("=" * 75)
    print(f"TIER 1 STEP 7: PROPER TRAINING (exp={experiment}, beta={beta}, seed={seed}, N_tr={train_size})")
    print("=" * 75)

    set_global_seed(seed)

    # Load graphs
    if experiment == "pennylane":
        from scripts.reproduce_pennylane import build_compact_graphs
        graphs = build_compact_graphs(dataset_root="data/qm9", max_load=2500, max_nodes=8)
        # Use first 2500
        rng = np.random.default_rng(seed)
        all_idx = rng.permutation(len(graphs))
        tr_idx = all_idx[:train_size]
        va_idx = all_idx[train_size:train_size + val_size]
        te_idx = all_idx[train_size + val_size:train_size + val_size + 80]

        train_targets = np.array([graphs[int(i)]["y"] for i in tr_idx]).reshape(-1, 1)
        scaler = StandardScaler().fit(train_targets)

        def make_sample(i):
            g = graphs[int(i)]
            y_std = float(scaler.transform([[g["y"]]])[0, 0])
            return {
                "x": torch.tensor(g["x"], dtype=torch.float64),
                "edges": g["edges"],
                "y_std": torch.tensor(y_std, dtype=torch.float64),
                "y_raw": g["y"],
            }
    else:  # qiskit
        records = torch.load("data/cache_qiskit_records.pt")
        rng = np.random.default_rng(seed)
        all_idx = rng.permutation(min(1000, len(records)))
        tr_idx = all_idx[:train_size]
        va_idx = all_idx[train_size:train_size + val_size]
        te_idx = all_idx[train_size + val_size:train_size + val_size + 80]

        train_targets = np.array([records[int(i)]["target"] for i in tr_idx]).reshape(-1, 1)
        scaler = StandardScaler().fit(train_targets)

        def make_sample(i):
            r = records[int(i)]
            y_std = float(scaler.transform([[r["target"]]])[0, 0])
            return {
                "x": torch.tensor(r["node_signal"], dtype=torch.float64),
                "edges": r["edges"],
                "y_std": torch.tensor(y_std, dtype=torch.float64),
                "y_raw": r["target"],
            }

    train_data = [make_sample(i) for i in tr_idx]
    val_data = [make_sample(i) for i in va_idx]
    test_data = [make_sample(i) for i in te_idx]

    sim = DensityMatrixSimulator(n_qubits=8, experiment=experiment, beta=beta)
    model = ProperQGNNModel(sim=sim, experiment=experiment, seed=seed)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve_epochs = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(len(train_data))
        train_losses = []

        for idx in order:
            opt.zero_grad()
            s = train_data[idx]
            pred = model(s["x"], s["edges"])
            loss = (pred - s["y_std"]) ** 2
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        scheduler.step()
        tr_mse = float(np.mean(train_losses))

        # Evaluate on validation set
        model.eval()
        val_losses = []
        with torch.no_grad():
            for s in val_data:
                p = model(s["x"], s["edges"])
                val_losses.append(((p - s["y_std"]) ** 2).item())
        va_mse = float(np.mean(val_losses))

        history.append({
            "epoch": ep,
            "train_mse": tr_mse,
            "val_mse": va_mse,
            "lr": scheduler.get_last_lr()[0],
        })

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:2d}/{epochs:2d} | Train MSE: {tr_mse:.4f} | Val MSE: {va_mse:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

        if va_mse < best_val_loss - 1e-4:
            best_val_loss = va_mse
            best_epoch = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {ep} (patience={patience})")
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)

    def eval_set(dataset):
        model.eval()
        preds_std, targets_std, targets_raw = [], [], []
        with torch.no_grad():
            for s in dataset:
                p = model(s["x"], s["edges"]).item()
                preds_std.append(p)
                targets_std.append(s["y_std"].item())
                targets_raw.append(s["y_raw"])
        preds_std = np.array(preds_std)
        targets_std = np.array(targets_std)
        preds_raw = scaler.inverse_transform(preds_std.reshape(-1, 1)).ravel()
        mse_std = float(mean_squared_error(targets_std, preds_std))
        mae_raw = float(mean_absolute_error(targets_raw, preds_raw))
        return mse_std, mae_raw

    final_tr_mse, final_tr_mae = eval_set(train_data)
    final_va_mse, final_va_mae = eval_set(val_data)
    final_te_mse, final_te_mae = eval_set(test_data)
    gap = final_te_mse - final_tr_mse

    elapsed = time.time() - t0
    print("\n" + "=" * 50)
    print("PROPER TRAINING RESULTS:")
    print(f"  Best Epoch: {best_epoch} | Time: {elapsed:.1f}s")
    print(f"  Train MSE (std): {final_tr_mse:.4f} | MAE: {final_tr_mae:.4f} eV")
    print(f"  Val MSE (std):   {final_va_mse:.4f} | MAE: {final_va_mae:.4f} eV")
    print(f"  Test MSE (std):  {final_te_mse:.4f} | MAE: {final_te_mae:.4f} eV")
    print(f"  Generalization Gap: {gap:.4f}")
    print("=" * 50)

    # Save convergence figure
    df_hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_hist["epoch"], df_hist["train_mse"], "o-", label="Train Loss (MSE)")
    ax.plot(df_hist["epoch"], df_hist["val_mse"], "s--", label="Validation Loss (MSE)")
    ax.axvline(best_epoch, color="r", linestyle=":", label=f"Best Checkpoint (ep={best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Standardized Loss (MSE)")
    ax.set_title(f"QGNN Convergence ({experiment}, beta={beta}, seed={seed})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig_path = FIG_DIR / f"convergence_qgnn_{experiment}_beta{beta}_seed{seed}.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved convergence plot to {fig_path}")

    # Save model checkpoint
    model_path = MODELS_DIR / f"model_proper_qgnn_{experiment}_beta{beta}_seed{seed}.pt"
    torch.save(best_state, model_path)
    print(f"Saved best model checkpoint to {model_path}")

    run_record = {
        "experiment": experiment,
        "beta": beta,
        "seed": seed,
        "train_size": train_size,
        "best_epoch": best_epoch,
        "train_mse_std": final_tr_mse,
        "train_mae_eV": final_tr_mae,
        "val_mse_std": final_va_mse,
        "val_mae_eV": final_va_mae,
        "test_mse_std": final_te_mse,
        "test_mae_eV": final_te_mae,
        "gap": gap,
        "elapsed_seconds": elapsed,
    }

    results_file = RAW_DIR / "proper_training_results.csv"
    if results_file.exists():
        df_all = pd.read_csv(results_file)
        df_all = pd.concat([df_all, pd.DataFrame([run_record])], ignore_index=True)
    else:
        df_all = pd.DataFrame([run_record])
    df_all.to_csv(results_file, index=False)
    print(f"Updated results in {results_file}")

    manifest = create_run_manifest(
        run_id=generate_run_id(f"proper_train_{experiment}"),
        config={
            "step": "tier1_step7_proper_training",
            "experiment": experiment,
            "beta": beta,
            "seed": seed,
            "train_size": train_size,
            "best_epoch": best_epoch,
        },
        output_dir=RAW_DIR,
    )
    print(f"Manifest saved for run {manifest['run_id']}")

    return run_record, df_hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="pennylane", choices=["pennylane", "qiskit"])
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    train_proper_qgnn(
        experiment=args.experiment,
        beta=args.beta,
        seed=args.seed,
        train_size=args.train_size,
        epochs=args.epochs,
    )
