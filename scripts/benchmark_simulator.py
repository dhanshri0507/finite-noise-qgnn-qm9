"""
Benchmark script for the PyTorch DensityMatrixSimulator.
Measures simulation runtime across different circuit sizes, batch sizes, and noise levels.
"""

import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.qgnn.simulator import DensityMatrixSimulator, compute_features_qiskit

def benchmark():
    print("=" * 60)
    print("BENCHMARK: PyTorch DensityMatrixSimulator (8 Qubits)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Note: Running on CPU (no CUDA GPU available on this system)")

    n_qubits = 8
    dim = 1 << n_qubits
    print(f"Hilbert space dimension: {dim} x {dim} ({dim*dim*16 / (1024*1024):.2f} MB per density matrix)")

    # Sample molecular circuit
    z = np.array([6, 6, 6, 7, 8, 6, 6, 6], dtype=float)
    deg = np.array([1, 2, 3, 2, 1, 2, 2, 1], dtype=float)
    feats = torch.tensor(compute_features_qiskit(z, deg, n_qubits=n_qubits), dtype=torch.float64, device=device)
    # Parameters for each experiment
    theta_qiskit = torch.tensor([0.1] * n_qubits, dtype=torch.float64, device=device)
    theta_pennylane = torch.tensor([0.1] * (1 + 2 * n_qubits), dtype=torch.float64, device=device)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7)]

    # 1. Single Circuit Latency
    for exp in ['qiskit', 'pennylane']:
        th = theta_qiskit if exp == 'qiskit' else theta_pennylane
        for beta in [0.0, 0.01, 0.05]:
            sim = DensityMatrixSimulator(n_qubits=n_qubits, experiment=exp, beta=beta, device=device)
            # Warmup
            _ = sim.run_circuit(feats, edges, th)
            
            # Timing (5 runs)
            n_reps = 5
            t0 = time.perf_counter()
            for _ in range(n_reps):
                res = sim.run_circuit(feats, edges, th)
            t1 = time.perf_counter()
            avg_ms = ((t1 - t0) / n_reps) * 1000.0
            print(f"[{exp.upper():9s}] beta={beta:5.3f} | Latency: {avg_ms:6.2f} ms/circuit")

    # 2. Batch Throughput (80 molecules - pilot dataset size)
    batch_size = 40
    batch_feats = [feats for _ in range(batch_size)]
    batch_edges = [edges for _ in range(batch_size)]

    print("-" * 60)
    print(f"BATCH RUNTIME (N = {batch_size} molecules, chunk_size = 32):")
    for exp in ['qiskit', 'pennylane']:
        th = theta_qiskit if exp == 'qiskit' else theta_pennylane
        for beta in [0.0, 0.02]:
            sim = DensityMatrixSimulator(n_qubits=n_qubits, experiment=exp, beta=beta, device=device, chunk_size=32)
            t0 = time.perf_counter()
            res = sim.run_batch(batch_feats, batch_edges, th)
            t1 = time.perf_counter()
            total_s = t1 - t0
            rate = batch_size / total_s
            print(f"[{exp.upper():9s}] beta={beta:5.3f} | Total: {total_s:5.2f} s | Throughput: {rate:5.1f} mol/s")
    print("=" * 60)

if __name__ == '__main__':
    benchmark()
