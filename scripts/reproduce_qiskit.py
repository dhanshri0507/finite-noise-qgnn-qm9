"""
Tier 1 Step 5: Reproduction Pilot for the Qiskit Experiment.

Runs the exact configuration from reference/FNQGNNMD/Finite_Noise_QGNN_QM9_Colab.ipynb:
- 8 qubits, RY encoding, CNOT-RZ-CNOT per bond, RX trainable layer (8 shared parameters)
- Nelder-Mead optimization (maxiter=8, xatol=1e-3, fatol=1e-4), zero-initialization
- 80 training molecules, 80 test molecules
- Depolarizing noise beta in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20] (Qiskit convention)
- Metrics: Train MSE, Test MSE, Gap, Train MAE, Test MAE, Gradient Norm, Effective Complexity

Saves raw results to results/raw/ and prints side-by-side comparison with original numbers.
"""

import sys
import os
import time
import json
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.datasets import QM9

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qgnn.simulator import DensityMatrixSimulator
from src.qgnn.seed_utils import create_run_manifest, generate_run_id, set_global_seed

# Original saved reference numbers from notebook cell outputs
ORIGINAL_QISKIT_RESULTS = {
    0.000: {"train_mse": 0.960914, "test_mse": 1.090837, "gap": 0.129922, "grad_norm": 1.168322e-05, "d_eff": 8.730077e-04},
    0.005: {"train_mse": 0.963197, "test_mse": 1.091519, "gap": 0.128323, "grad_norm": 9.709968e-06, "d_eff": 6.031381e-04},
    0.010: {"train_mse": 0.965101, "test_mse": 1.092106, "gap": 0.127005, "grad_norm": 8.068629e-06, "d_eff": 4.165258e-04},
    0.020: {"train_mse": 0.968004, "test_mse": 1.093034, "gap": 0.125030, "grad_norm": 5.568270e-06, "d_eff": 1.984060e-04},
    0.050: {"train_mse": 0.972304, "test_mse": 1.094493, "gap": 0.122188, "grad_norm": 1.820219e-06, "d_eff": 2.120411e-05},
    0.100: {"train_mse": 0.973995, "test_mse": 1.095119, "gap": 0.121124, "grad_norm": 2.758058e-07, "d_eff": 4.868402e-07},
    0.200: {"train_mse": 0.974266, "test_mse": 1.095236, "gap": 0.120970, "grad_norm": 6.907544e-09, "d_eff": 3.053707e-10},
}


def extract_graph_record(data, max_qubits=8):
    heavy_idx = torch.where(data.z > 1)[0].tolist()
    if len(heavy_idx) == 0 or len(heavy_idx) > max_qubits:
        return None

    old_to_new = {old: new for new, old in enumerate(heavy_idx)}
    z = data.z[heavy_idx].cpu().numpy().astype(float)

    edges = []
    edge_index = data.edge_index.cpu().numpy()
    for a, b in edge_index.T:
        a = int(a)
        b = int(b)
        if a in old_to_new and b in old_to_new:
            u = old_to_new[a]
            v = old_to_new[b]
            if u != v:
                pair = tuple(sorted((u, v)))
                if pair not in edges:
                    edges.append(pair)

    degree = np.zeros(len(heavy_idx), dtype=float)
    for u, v in edges:
        degree[u] += 1.0
        degree[v] += 1.0

    node_signal = np.zeros(max_qubits, dtype=float)
    for i in range(len(heavy_idx)):
        z_scaled = (z[i] - 1.0) / 5.0
        d_scaled = degree[i] / 4.0
        node_signal[i] = 0.70 * z_scaled + 0.30 * d_scaled

    y = float(data.y[0, 4].item())
    return {
        "node_signal": node_signal,
        "edges": edges,
        "n_heavy": len(heavy_idx),
        "target": y,
    }


def predict_batch(sim, records, theta):
    theta_t = torch.tensor(theta, dtype=torch.float64)
    preds = []
    for r in records:
        f = torch.tensor(r["node_signal"], dtype=torch.float64)
        p = sim.run_circuit(f, r["edges"], theta_t).item()
        preds.append(p)
    return np.array(preds, dtype=float)


def gradient_norm_estimate(sim, records, theta, delta=1e-3):
    base_theta = np.asarray(theta, dtype=float)
    grads = []
    for j in range(len(base_theta)):
        t_plus = base_theta.copy()
        t_minus = base_theta.copy()
        t_plus[j] += delta
        t_minus[j] -= delta

        pred_plus = predict_batch(sim, records, t_plus)
        pred_minus = predict_batch(sim, records, t_minus)

        grad_j = (pred_plus - pred_minus) / (2.0 * delta)
        grads.append(grad_j)

    J = np.column_stack(grads)
    norm = float(np.sqrt(np.mean(J**2)))
    return norm, J


def effective_complexity_from_jacobian(J, ridge=1e-4):
    J = np.asarray(J, dtype=float)
    q = J.shape[1]
    gram = J.T @ J
    H = J @ np.linalg.inv(gram + ridge * np.eye(q)) @ J.T
    return float(np.trace(H))


def run_qiskit_reproduction(data_root="data/qm9", seed=42):
    print("=" * 75)
    print("TIER 1 STEP 5: REPRODUCING QISKIT EXPERIMENT")
    print("=" * 75)

    set_global_seed(seed)

    cache_path = Path("data/cache_qiskit_records.pt")
    if cache_path.exists():
        print(f"Loading cached records from {cache_path}...")
        records = torch.load(cache_path)
        print(f"Loaded {len(records)} eligible molecules from cache.")
    else:
        print("Loading QM9 dataset...")
        qm9 = QM9(root=data_root)
        print(f"Total QM9 molecules: {len(qm9)}")

        print("Extracting heavy-atom records (<= 8 heavy atoms)...")
        records = []
        for i in range(len(qm9)):
            rec = extract_graph_record(qm9[i], max_qubits=8)
            if rec is not None:
                records.append(rec)
        print(f"Eligible molecules (<= 8 heavy atoms): {len(records)}")
        torch.save(records, cache_path)

    # Fixed subset of 600 molecules with seed=42
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(records), size=min(600, len(records)), replace=False)
    records = [records[int(i)] for i in chosen]
    y_raw = np.array([r["target"] for r in records], dtype=float)

    # 75/25 split with random_state=42
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=seed)

    # Standardize using training set only
    y_scaler = StandardScaler()
    y_scaler.fit(y_raw[train_idx].reshape(-1, 1))

    # Slice to 80 train and 80 test
    MAX_TRAIN = 80
    MAX_TEST = 80
    train_records = [records[int(i)] for i in train_idx[:MAX_TRAIN]]
    test_records = [records[int(i)] for i in test_idx[:MAX_TEST]]

    y_train_eval = y_scaler.transform(np.array([r["target"] for r in train_records]).reshape(-1, 1)).ravel()
    y_test_eval = y_scaler.transform(np.array([r["target"] for r in test_records]).reshape(-1, 1)).ravel()

    print(f"Working evaluation subset: {len(train_records)} train, {len(test_records)} test molecules")
    print(f"Train standardized targets mean: {y_train_eval.mean():.4f}, std: {y_train_eval.std():.4f}")

    noise_rates = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    maxiter = 8
    theta_start = np.zeros(8)

    reproduced_results = []
    comparison_rows = []

    for beta in noise_rates:
        print(f"\n--- Running beta = {beta:5.3f} ---")
        t0 = time.time()
        sim = DensityMatrixSimulator(n_qubits=8, experiment='qiskit', beta=beta)

        # Objective function for Nelder-Mead
        def objective(theta):
            pred = predict_batch(sim, train_records, theta)
            return float(mean_squared_error(y_train_eval, pred))

        opt_res = minimize(
            objective,
            x0=theta_start.copy(),
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4, "disp": False}
        )
        theta_star = opt_res.x
        train_pred = predict_batch(sim, train_records, theta_star)
        test_pred = predict_batch(sim, test_records, theta_star)

        train_mse = float(mean_squared_error(y_train_eval, train_pred))
        test_mse = float(mean_squared_error(y_test_eval, test_pred))
        gap = test_mse - train_mse

        # eV metrics
        train_pred_eV = y_scaler.inverse_transform(train_pred.reshape(-1, 1)).ravel()
        test_pred_eV = y_scaler.inverse_transform(test_pred.reshape(-1, 1)).ravel()
        y_train_raw = np.array([r["target"] for r in train_records])
        y_test_raw = np.array([r["target"] for r in test_records])
        train_mae_eV = float(mean_absolute_error(y_train_raw, train_pred_eV))
        test_mae_eV = float(mean_absolute_error(y_test_raw, test_pred_eV))

        # Gradient norm & effective complexity
        grad_norm, J = gradient_norm_estimate(sim, train_records, theta_star, delta=1e-3)
        d_eff = effective_complexity_from_jacobian(J, ridge=1e-4)

        t1 = time.time()
        print(f"  Converged in {opt_res.nit} iter(s) | Elapsed: {t1-t0:.1f}s")
        print(f"  Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f} | Gap: {gap:.6f}")
        print(f"  Grad Norm: {grad_norm:.4e} | d_eff: {d_eff:.4e}")

        reproduced_results.append({
            "beta": beta,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "generalization_gap": gap,
            "train_mae_eV": train_mae_eV,
            "test_mae_eV": test_mae_eV,
            "gradient_norm": grad_norm,
            "d_eff": d_eff,
            "optimizer_iterations": opt_res.nit,
            "optimizer_success": opt_res.success,
        })

        orig = ORIGINAL_QISKIT_RESULTS.get(beta, {})
        comparison_rows.append({
            "beta": beta,
            "orig_train_mse": orig.get("train_mse", np.nan),
            "new_train_mse": train_mse,
            "train_mse_diff": abs(train_mse - orig.get("train_mse", np.nan)),
            "orig_test_mse": orig.get("test_mse", np.nan),
            "new_test_mse": test_mse,
            "test_mse_diff": abs(test_mse - orig.get("test_mse", np.nan)),
            "orig_grad_norm": orig.get("grad_norm", np.nan),
            "new_grad_norm": grad_norm,
            "orig_d_eff": orig.get("d_eff", np.nan),
            "new_d_eff": d_eff,
        })

        # Warm start
        theta_start = theta_star.copy()

    # Save results
    results_dir = Path("results/raw")
    results_dir.mkdir(parents=True, exist_ok=True)

    df_rep = pd.DataFrame(reproduced_results)
    df_comp = pd.DataFrame(comparison_rows)

    df_rep.to_csv(results_dir / "reproduction_qiskit_results.csv", index=False)
    df_comp.to_csv(results_dir / "reproduction_qiskit_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE REPRODUCTION COMPARISON (QISKIT EXPERIMENT):")
    print("=" * 80)
    fmt_comp = df_comp[["beta", "orig_train_mse", "new_train_mse", "train_mse_diff",
                         "orig_test_mse", "new_test_mse", "test_mse_diff",
                         "orig_grad_norm", "new_grad_norm", "orig_d_eff", "new_d_eff"]]
    print(fmt_comp.to_string(index=False))

    # Create run manifest
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["reproduction_qiskit"] = {
        "dataset": "QM9",
        "n_qubits": 8,
        "n_train": MAX_TRAIN,
        "n_test": MAX_TEST,
        "optimizer": "Nelder-Mead",
        "noise_rates": noise_rates,
        "max_diff_test_mse": float(df_comp["test_mse_diff"].max()),
    }
    manifest = create_run_manifest(
        run_id=generate_run_id("reproduce_qiskit"),
        config=cfg,
        output_dir=str(results_dir)
    )
    print(f"\nRun manifest saved: {manifest['run_id']}")
    return df_comp


if __name__ == "__main__":
    run_qiskit_reproduction()
