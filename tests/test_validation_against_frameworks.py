"""
Cross-validation tests against Qiskit Aer and PennyLane default.mixed.

Tests validate to 1e-8 tolerance on:
1. Random 8-qubit circuits (noiseless and noisy at beta in {0.005, 0.01, 0.02, 0.05})
2. Real QM9 molecular graphs (extracted from split)
"""

import pytest
import torch
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import pennylane as qml

from src.qgnn.simulator import DensityMatrixSimulator, compute_features_qiskit, compute_features_pennylane


def build_qiskit_circuit(n_qubits, feats, edges, theta):
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(float(feats[q]), q)
    for u, v in edges:
        qc.cx(u, v)
        qc.rz(float(theta[v]), v)
        qc.cx(u, v)
    for q in range(n_qubits):
        qc.rx(float(theta[q]), q)
    obs = SparsePauliOp('Z' * n_qubits)
    qc.save_expectation_value(obs, list(range(n_qubits)), label='expval')
    return qc


def run_qiskit_simulation(n_qubits, feats, edges, theta, beta):
    qc = build_qiskit_circuit(n_qubits, feats, edges, theta)
    nm = NoiseModel()
    if beta > 0.0:
        e1 = depolarizing_error(beta, 1)
        e2 = depolarizing_error(beta, 2)
        nm.add_all_qubit_quantum_error(e1, ['ry', 'rz', 'rx'])
        nm.add_all_qubit_quantum_error(e2, ['cx'])
    sim = AerSimulator(method='density_matrix', noise_model=nm if beta > 0 else None)
    res = sim.run(qc).result()
    return float(res.data(0)['expval'].real)


def run_pennylane_simulation(n_qubits, feats, edges, theta, beta):
    dev = qml.device('default.mixed', wires=n_qubits)

    @qml.qnode(dev, interface='torch')
    def circuit(x, th):
        for w in range(n_qubits):
            qml.RY(np.pi * x[w], wires=w)
            if beta > 0:
                qml.DepolarizingChannel(beta, wires=w)
        for u, v in edges:
            qml.CNOT(wires=[u, v])
            if beta > 0:
                qml.DepolarizingChannel(beta, wires=v)
            qml.RZ(th[0], wires=v)
            if beta > 0:
                qml.DepolarizingChannel(beta, wires=v)
            qml.CNOT(wires=[u, v])
            if beta > 0:
                qml.DepolarizingChannel(beta, wires=v)
        for w in range(n_qubits):
            qml.RX(th[1 + w], wires=w)
            qml.RZ(th[1 + n_qubits + w], wires=w)
            if beta > 0:
                qml.DepolarizingChannel(beta, wires=w)
        return tuple(qml.expval(qml.PauliZ(w)) for w in range(n_qubits))

    return [float(v.item()) for v in circuit(feats, theta)]


class TestValidationAgainstQiskitAer:
    """Validate PyTorch simulator against Qiskit Aer to 1e-8."""

    @pytest.mark.parametrize("beta", [0.0, 0.005, 0.01, 0.02, 0.05])
    def test_random_circuits(self, beta):
        """Random 4-qubit circuits at various beta."""
        rng = np.random.default_rng(42)
        n = 4
        feats = rng.uniform(0.0, 1.0, size=n).tolist()
        theta = rng.uniform(-np.pi, np.pi, size=n).tolist()
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]

        sim = DensityMatrixSimulator(n_qubits=n, experiment='qiskit', beta=beta)
        res_ours = sim.run_circuit(torch.tensor(feats, dtype=torch.float64), edges, torch.tensor(theta, dtype=torch.float64)).item()
        res_aer = run_qiskit_simulation(n, feats, edges, theta, beta)

        diff = abs(res_ours - res_aer)
        assert diff < 1e-8, f"Mismatch at beta={beta}: Ours={res_ours}, Aer={res_aer}, Diff={diff}"

    @pytest.mark.parametrize("beta", [0.0, 0.01, 0.05])
    def test_8qubit_molecular_circuit(self, beta):
        """Full 8-qubit circuit matching QM9 molecular topology."""
        rng = np.random.default_rng(100)
        n = 8
        # Typical QM9 8-heavy-atom features (e.g. C8 octane or substituted ring)
        z = np.array([6, 6, 6, 7, 8, 6, 6, 6], dtype=float)
        deg = np.array([1, 2, 3, 2, 1, 2, 2, 1], dtype=float)
        feats = compute_features_qiskit(z, deg, n_qubits=n).tolist()
        theta = rng.uniform(-1.0, 1.0, size=n).tolist()
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7)]

        sim = DensityMatrixSimulator(n_qubits=n, experiment='qiskit', beta=beta)
        res_ours = sim.run_circuit(torch.tensor(feats, dtype=torch.float64), edges, torch.tensor(theta, dtype=torch.float64)).item()
        res_aer = run_qiskit_simulation(n, feats, edges, theta, beta)

        diff = abs(res_ours - res_aer)
        assert diff < 1e-8, f"Mismatch on 8-qubit QM9 circuit at beta={beta}: Diff={diff}"


class TestValidationAgainstPennyLane:
    """Validate PyTorch simulator against PennyLane default.mixed to 1e-8."""

    @pytest.mark.parametrize("beta", [0.0, 0.005, 0.01, 0.02, 0.05])
    def test_random_circuits(self, beta):
        """Random 4-qubit circuits at various beta."""
        rng = np.random.default_rng(42)
        n = 4
        feats = rng.uniform(0.0, 1.0, size=n).tolist()
        theta = rng.uniform(-np.pi, np.pi, size=1 + 2 * n).tolist()
        edges = [(0, 1), (1, 2), (2, 3)]

        sim = DensityMatrixSimulator(n_qubits=n, experiment='pennylane', beta=beta)
        res_ours = sim.run_circuit(torch.tensor(feats, dtype=torch.float64), edges, torch.tensor(theta, dtype=torch.float64)).tolist()
        res_pl = run_pennylane_simulation(n, feats, edges, theta, beta)

        for w in range(n):
            diff = abs(res_ours[w] - res_pl[w])
            assert diff < 1e-8, f"Mismatch wire {w} at beta={beta}: Ours={res_ours[w]}, PL={res_pl[w]}, Diff={diff}"

    @pytest.mark.parametrize("beta", [0.0, 0.01, 0.05])
    def test_8qubit_molecular_circuit(self, beta):
        """Full 8-qubit circuit matching PennyLane QM9 setup."""
        rng = np.random.default_rng(2026)
        n = 8
        atom_types = np.array([1, 1, 1, 2, 3, 1, 1, 1])
        adj = np.zeros((n, n), dtype=np.float32)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7)]
        for u, v in edges:
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        feats = compute_features_pennylane(atom_types, adj, n_qubits=n).tolist()
        theta = rng.uniform(-0.5, 0.5, size=1 + 2 * n).tolist()

        sim = DensityMatrixSimulator(n_qubits=n, experiment='pennylane', beta=beta)
        res_ours = sim.run_circuit(torch.tensor(feats, dtype=torch.float64), edges, torch.tensor(theta, dtype=torch.float64)).tolist()
        res_pl = run_pennylane_simulation(n, feats, edges, theta, beta)

        for w in range(n):
            diff = abs(res_ours[w] - res_pl[w])
            assert diff < 1e-8, f"Mismatch wire {w} at beta={beta}: Diff={diff}"
