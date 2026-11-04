"""
Extended Sweep: Remaining 5 pre-registered seeds + exploratory beta extension.

Pre-registered seeds remaining: [6789, 7777, 8888, 9999, 12345]
Exploratory betas (labeled): [0.30, 0.40, 0.50]

Logs every run to results/raw/extended_sweep.log in addition to stdout.
Results checkpoint to results/raw/main_sweep_results.csv (same file as main sweep —
checkpointing skips already-completed (seed, beta) pairs).
Exploratory runs go to results/raw/exploratory_sweep_results.csv.

Usage:
    python -u scripts/run_extended_sweep.py --epochs 15 --train_size 40 2>&1 | Tee-Object results/raw/extended_sweep.log
"""

import sys
import os
import time
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
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

LOG_FILE = RAW_DIR / "extended_sweep.log"


def setup_logging():
    log = logging.getLogger("extended_sweep")
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


class ProperQGNNRegressor(nn.Module):
    def __init__(self, sim: DensityMatrixSimulator, seed: int = 42):
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


def compute_output_jacobian(model, samples, max_samples=20):
    model.eval()
    eval_samples = samples[:max_samples]
    grads = []
    for s in eval_samples:
        model.zero_grad()
        out = model(s["x"], s["edges"])
        out.backward()
        grads.append(model.theta.grad.detach().cpu().numpy().copy())
    return np.array(grads, dtype=float)


def bootstrap_paired_difference_ci(diffs, n_boot=10000, ci=0.95, seed=42):
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


def run_one_seed_beta(seed, beta, train_data, val_data, test_data, scaler, epochs, log):
    """Train and evaluate a single (seed, beta) run. Returns a record dict."""
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

    J = compute_output_jacobian(model, test_data)
    pr = participation_ratio(J)
    er = entropy_effective_rank(J)
    grad_norm = rms_gradient_norm(J)

    elapsed = time.time() - t0
    log.info(f"  DONE Seed={seed}, Beta={beta:.3f} | BestEp={best_epoch} | Tr={tr_mse:.4f} | Val={va_mse:.4f} | Test={te_mse:.4f} | MAE={te_mae:.4f}eV | PR={pr:.2f} | t={elapsed:.1f}s")

    return {
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


def run_extended_sweep(train_size=40, val_size=40, test_size=80, epochs=15):
    log = setup_logging()

    PREREGISTERED_SEEDS = [6789, 7777, 8888, 9999, 12345]
    REGISTERED_BETAS = [0.0, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150, 0.200]
    EXPLORATORY_BETAS = [0.30, 0.40, 0.50]

    log.info("=" * 80)
    log.info("EXTENDED SWEEP: Remaining 5 Pre-registered Seeds + Exploratory Beta Extension")
    log.info(f"Pre-registered seeds: {PREREGISTERED_SEEDS}")
    log.info(f"Registered betas: {REGISTERED_BETAS}")
    log.info(f"Exploratory betas: {EXPLORATORY_BETAS}")
    log.info(f"Sizes: Train={train_size}, Val={val_size}, Test={test_size} | Epochs={epochs}")
    log.info("=" * 80)

    log.info("Loading graphs...")
    graphs = build_compact_graphs(dataset_root="data/qm9", max_load=2500, max_nodes=8)
    log.info(f"Loaded {len(graphs)} graphs.")

    # --- PHASE A: Pre-registered seeds, registered betas ---
    main_file = RAW_DIR / "main_sweep_results.csv"
    if main_file.exists():
        existing_df = pd.read_csv(main_file)
        existing_records = existing_df.to_dict("records")
        log.info(f"Loaded {len(existing_records)} existing runs from main_sweep_results.csv")
    else:
        existing_records = []

    completed_keys = {(r["seed"], r["beta"]) for r in existing_records}
    all_records = list(existing_records)

    total_phase_a = len(PREREGISTERED_SEEDS) * len(REGISTERED_BETAS)
    done_phase_a = 0

    for seed in PREREGISTERED_SEEDS:
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
                    "edges": g["edges"], "y_std": torch.tensor(y_std, dtype=torch.float64),
                    "y_raw": g["y"]}

        train_data = [make_sample(i) for i in tr_idx]
        val_data = [make_sample(i) for i in va_idx]
        test_data = [make_sample(i) for i in te_idx]

        for beta in REGISTERED_BETAS:
            done_phase_a += 1
            if (seed, beta) in completed_keys:
                log.info(f"[PhaseA {done_phase_a}/{total_phase_a}] SKIP Seed={seed} Beta={beta:.3f} (already done)")
                continue
            log.info(f"[PhaseA {done_phase_a}/{total_phase_a}] START Seed={seed} Beta={beta:.3f}")
            rec = run_one_seed_beta(seed, beta, train_data, val_data, test_data, scaler, epochs, log)
            all_records.append(rec)
            completed_keys.add((seed, beta))
            pd.DataFrame(all_records).to_csv(main_file, index=False)

    df_main = pd.DataFrame(all_records)
    df_main.to_csv(main_file, index=False)
    log.info(f"Phase A complete. {len(df_main)} total runs saved to {main_file}")

    # --- PHASE B: Exploratory betas across ALL 10 seeds ---
    exp_file = RAW_DIR / "exploratory_sweep_results.csv"
    if exp_file.exists():
        exp_existing = pd.read_csv(exp_file).to_dict("records")
        log.info(f"Loaded {len(exp_existing)} existing exploratory runs")
    else:
        exp_existing = []

    exp_done_keys = {(r["seed"], r["beta"]) for r in exp_existing}
    exp_records = list(exp_existing)

    ALL_10_SEEDS = [42, 1042, 2024, 3141, 5321] + PREREGISTERED_SEEDS
    total_phase_b = len(ALL_10_SEEDS) * len(EXPLORATORY_BETAS)
    done_phase_b = 0

    for seed in ALL_10_SEEDS:
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
                    "edges": g["edges"], "y_std": torch.tensor(y_std, dtype=torch.float64),
                    "y_raw": g["y"]}

        train_data = [make_sample(i) for i in tr_idx]
        val_data = [make_sample(i) for i in va_idx]
        test_data = [make_sample(i) for i in te_idx]

        for beta in EXPLORATORY_BETAS:
            done_phase_b += 1
            if (seed, beta) in exp_done_keys:
                log.info(f"[PhaseB {done_phase_b}/{total_phase_b}] SKIP Seed={seed} Beta={beta:.2f} (already done)")
                continue
            log.info(f"[PhaseB {done_phase_b}/{total_phase_b}] START Seed={seed} Beta={beta:.2f} [EXPLORATORY]")
            rec = run_one_seed_beta(seed, beta, train_data, val_data, test_data, scaler, epochs, log)
            rec["split"] = "exploratory"
            exp_records.append(rec)
            exp_done_keys.add((seed, beta))
            pd.DataFrame(exp_records).to_csv(exp_file, index=False)

    df_exp = pd.DataFrame(exp_records)
    df_exp.to_csv(exp_file, index=False)
    log.info(f"Phase B complete. {len(df_exp)} exploratory runs saved to {exp_file}")
    log.info("EXTENDED SWEEP COMPLETE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--train_size", type=int, default=40)
    args = parser.parse_args()
    run_extended_sweep(epochs=args.epochs, train_size=args.train_size)
