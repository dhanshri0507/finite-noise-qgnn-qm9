"""
Fast batched PyTorch density-matrix simulator for the 8-qubit QGNN circuit.

Supports two experiment configurations:
- 'qiskit': Matches the Qiskit/Aer notebook exactly
  * Feature encoding: node_signal directly (no pi factor)
  * Entangling: CNOT(u,v) -> 2q depolarizing(beta, 2) -> RZ(theta[v]) -> 1q depolarizing(beta, 1) -> CNOT(u,v) -> 2q depolarizing(beta, 2)
  * Noise model: Qiskit Aer depolarizing_error convention (shrinkage = 1 - beta)
  * Final layer: RX(theta[q]) -> 1q depolarizing(beta, 1)
  * Readout: global parity observable Z^n = Z tensor Z ... tensor Z
- 'pennylane': Matches the PennyLane notebook exactly
  * Feature encoding: pi * node_signal
  * Entangling: CNOT(u,v) -> 1q depolarizing(beta, target=v) -> RZ(theta[0]) -> 1q depolarizing(beta, target=v) -> CNOT(u,v) -> 1q depolarizing(beta, target=v)
  * Noise model: PennyLane DepolarizingChannel convention (shrinkage = 1 - 4*beta/3)
  * Final layer: RX(theta[1+q]) -> RZ(theta[1+n+q]) -> 1q depolarizing(beta)
  * Readout: vector of single-qubit <Z_w> expectations for w in 0..n-1
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from src.qgnn.noise import I2, X, Y, Z


def ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RY(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]"""
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    row0 = torch.stack([c, -s])
    row1 = torch.stack([s, c])
    return torch.stack([row0, row1]).to(dtype=torch.complex128)


def rx_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RX(theta) = [[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]"""
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    row0 = torch.stack([c, -1j * s])
    row1 = torch.stack([-1j * s, c])
    return torch.stack([row0, row1]).to(dtype=torch.complex128)


def rz_matrix(theta: torch.Tensor) -> torch.Tensor:
    """RZ(theta) = [[exp(-i*t/2), 0], [0, exp(i*t/2)]]"""
    phase = torch.exp(-1j * theta / 2.0)
    phase_conj = torch.exp(1j * theta / 2.0)
    zero = torch.zeros_like(phase)
    row0 = torch.stack([phase, zero])
    row1 = torch.stack([zero, phase_conj])
    return torch.stack([row0, row1]).to(dtype=torch.complex128)


class DensityMatrixSimulator:
    """
    Fast batched PyTorch density-matrix simulator for the QGNN circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 8).
    experiment : str
        Either 'qiskit' or 'pennylane' to match the original experiment.
    beta : float
        Depolarizing noise parameter. 0.0 = noiseless.
    device : str
        PyTorch device ('cpu' or 'cuda').
    chunk_size : int
        Batch chunk size for memory bounding.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        experiment: str = 'qiskit',
        beta: float = 0.0,
        device: str = 'cpu',
        chunk_size: int = 32,
    ):
        if experiment not in ('qiskit', 'pennylane'):
            raise ValueError(f"experiment must be 'qiskit' or 'pennylane', got '{experiment}'")

        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits  # 2**n_qubits
        self.experiment = experiment
        self.beta = float(beta)
        self.device = device
        self.chunk_size = chunk_size

        # Precompute embedded Paulis and ultra-fast permutation/sign structures for noise
        self._paulis_1q: Dict[str, List[torch.Tensor]] = {'X': [], 'Y': [], 'Z': []}
        self._x_perms: List[torch.Tensor] = []
        self._z_signs: List[torch.Tensor] = []

        idx_all = torch.arange(self.dim, device=device)
        for q in range(n_qubits):
            self._paulis_1q['X'].append(self._embed_1q_operator(X.to(device), q))
            self._paulis_1q['Y'].append(self._embed_1q_operator(Y.to(device), q))
            self._paulis_1q['Z'].append(self._embed_1q_operator(Z.to(device), q))

            # Ultra-fast 1q Pauli structures (avoids matmul)
            bit_shift = n_qubits - 1 - q
            self._x_perms.append(idx_all ^ (1 << bit_shift))
            z_diag = torch.where(((idx_all >> bit_shift) & 1) == 0, 1.0, -1.0).to(torch.complex128)
            self._z_signs.append(z_diag[:, None] * z_diag[None, :])

        # Precompute CNOT permutations for all (u, v) pairs
        self._cnot_perms: Dict[Tuple[int, int], torch.Tensor] = {}
        for u in range(n_qubits):
            for v in range(n_qubits):
                if u != v:
                    idx = torch.arange(self.dim, device=device)
                    ctrl = (idx >> (n_qubits - 1 - u)) & 1
                    perm = idx ^ (ctrl << (n_qubits - 1 - v))
                    self._cnot_perms[(u, v)] = perm

        # Precompute observable(s)
        if experiment == 'qiskit':
            self._observable = self._build_z_tensor_product()
            self._diag_observable = torch.diag(self._observable).real
        else:
            self._single_z_observables = [self._paulis_1q['Z'][q] for q in range(n_qubits)]
            self._diag_single_z = [torch.diag(ob).real for ob in self._single_z_observables]

    def _apply_1q_gate(self, rho: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
        """
        Apply a 2x2 unitary gate to qubit without constructing full 2^n x 2^n matrix.
        Uses fast tensordot contraction on tensor indices.
        """
        n = self.n_qubits
        r = rho.reshape((2,) * (2 * n))
        g = gate.to(self.device)
        r = torch.movedim(torch.tensordot(g, r, dims=([1], [qubit])), 0, qubit)
        r = torch.movedim(torch.tensordot(g.conj(), r, dims=([1], [n + qubit])), 0, n + qubit)
        return r.reshape(self.dim, self.dim)

    def _embed_1q_operator(self, op: torch.Tensor, qubit: int) -> torch.Tensor:
        """Embed a 2x2 operator into the full n-qubit Hilbert space at specified qubit."""
        n = self.n_qubits
        left_dim = 1 << qubit
        right_dim = 1 << (n - 1 - qubit)

        res = op.to(self.device)
        if left_dim > 1:
            left = torch.eye(left_dim, dtype=torch.complex128, device=self.device)
            res = torch.kron(left, res)
        if right_dim > 1:
            right = torch.eye(right_dim, dtype=torch.complex128, device=self.device)
            res = torch.kron(res, right)

        return res

    def _apply_cnot(self, rho: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """Apply CNOT gate via precomputed index permutation."""
        perm = self._cnot_perms[(control, target)]
        return rho[perm, :][:, perm]

    def _apply_noise_1q(self, rho: torch.Tensor, qubit: int) -> torch.Tensor:
        """
        Apply 1-qubit depolarizing noise on specified qubit via ultra-fast sign/perm indexing.
        """
        if self.beta <= 0.0:
            return rho

        perm = self._x_perms[qubit]
        z_signs = self._z_signs[qubit]

        rho_z = z_signs * rho
        rho_x = rho[perm, :][:, perm]
        rho_y = rho_z[perm, :][:, perm]
        pauli_sum = rho_x + rho_y + rho_z

        if self.experiment == 'qiskit':
            return (1.0 - 0.75 * self.beta) * rho + (self.beta / 4.0) * pauli_sum
        else:
            return (1.0 - self.beta) * rho + (self.beta / 3.0) * pauli_sum

    def _apply_noise_2q(self, rho: torch.Tensor, qubit1: int, qubit2: int) -> torch.Tensor:
        """
        Apply 2-qubit depolarizing noise on (qubit1, qubit2) (Qiskit convention).
        Exact match to Qiskit depolarizing_error(beta, 2), using ultra-fast
        zero-matmul permutation and sign indexing for all 15 non-identity Paulis.
        """
        if self.beta <= 0.0:
            return rho

        perm_u = self._x_perms[qubit1]
        perm_v = self._x_perms[qubit2]
        z_u = self._z_signs[qubit1]
        z_v = self._z_signs[qubit2]

        # 1. Non-identity Paulis on qubit2 only (I_u \otimes P_v)
        zv_rho = z_v * rho
        xv_rho = rho[perm_v, :][:, perm_v]
        yv_rho = zv_rho[perm_v, :][:, perm_v]
        sum_v = zv_rho + xv_rho + yv_rho

        # 2. Non-identity Paulis on qubit1 only (P_u \otimes I_v)
        zu_rho = z_u * rho
        xu_rho = rho[perm_u, :][:, perm_u]
        yu_rho = zu_rho[perm_u, :][:, perm_u]
        sum_u = zu_rho + xu_rho + yu_rho

        # 3. Two-qubit cross terms (P_u \otimes P_v)
        # Apply the 3 Pauli operations of qubit1 to each of the 3 results from qubit2
        sum_uv = torch.zeros_like(rho)
        for rv in (zv_rho, xv_rho, yv_rho):
            z_rv = z_u * rv
            x_rv = rv[perm_u, :][:, perm_u]
            y_rv = z_rv[perm_u, :][:, perm_u]
            sum_uv = sum_uv + (z_rv + x_rv + y_rv)

        pauli_sum = sum_v + sum_u + sum_uv
        return (1.0 - (15.0 / 16.0) * self.beta) * rho + (self.beta / 16.0) * pauli_sum

    def _build_z_tensor_product(self) -> torch.Tensor:
        """Build Z^n = Z tensor Z tensor ... tensor Z."""
        res = Z.to(self.device).clone()
        for _ in range(self.n_qubits - 1):
            res = torch.kron(res, Z.to(self.device))
        return res

    def run_circuit(
        self,
        node_features: torch.Tensor,
        edges: List[Tuple[int, int]],
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the QGNN circuit on a single molecule.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features, shape (n_qubits,).
        edges : list of (int, int)
            Molecular bond edges as (u, v) pairs with u < v.
        theta : torch.Tensor
            Trainable parameters.
            Qiskit: shape (n_qubits,)
            PennyLane: shape (1 + 2*n_qubits,)

        Returns
        -------
        torch.Tensor
            Qiskit: scalar Z^n expectation value.
            PennyLane: vector of n individual <Z_w> values.
        """
        n = self.n_qubits

        # Initialize |0><0|^n
        rho = torch.zeros(self.dim, self.dim, dtype=torch.complex128, device=self.device)
        rho[0, 0] = 1.0

        # --- Stage 1: Feature encoding RY ---
        for q in range(n):
            if self.experiment == 'pennylane':
                angle = torch.pi * node_features[q]
            else:
                angle = node_features[q]

            gate = ry_matrix(angle)
            rho = self._apply_1q_gate(rho, gate, q)

            if self.beta > 0.0:
                rho = self._apply_noise_1q(rho, q)

        # --- Stage 2: Graph entangling CNOT-RZ-CNOT ---
        for u, v in edges:
            # First CNOT
            rho = self._apply_cnot(rho, u, v)

            if self.experiment == 'qiskit':
                if self.beta > 0.0:
                    rho = self._apply_noise_2q(rho, u, v)
            elif self.experiment == 'pennylane':
                if self.beta > 0.0:
                    rho = self._apply_noise_1q(rho, v)

            # RZ rotation
            if self.experiment == 'qiskit':
                rz_angle = theta[v]
            else:
                rz_angle = theta[0]

            gate = rz_matrix(rz_angle)
            rho = self._apply_1q_gate(rho, gate, v)

            if self.beta > 0.0:
                rho = self._apply_noise_1q(rho, v)

            # Second CNOT
            rho = self._apply_cnot(rho, u, v)

            if self.experiment == 'qiskit':
                if self.beta > 0.0:
                    rho = self._apply_noise_2q(rho, u, v)
            elif self.experiment == 'pennylane':
                if self.beta > 0.0:
                    rho = self._apply_noise_1q(rho, v)

        # --- Stage 3: Trainable single-qubit layer ---
        for q in range(n):
            if self.experiment == 'qiskit':
                gate = rx_matrix(theta[q])
                rho = self._apply_1q_gate(rho, gate, q)

                if self.beta > 0.0:
                    rho = self._apply_noise_1q(rho, q)

            else:  # pennylane
                gate_rx = rx_matrix(theta[1 + q])
                rho = self._apply_1q_gate(rho, gate_rx, q)

                gate_rz = rz_matrix(theta[1 + n + q])
                rho = self._apply_1q_gate(rho, gate_rz, q)

                if self.beta > 0.0:
                    rho = self._apply_noise_1q(rho, q)

        # --- Stage 4: Readout ---
        diag_rho = torch.diagonal(rho).real
        if self.experiment == 'qiskit':
            expval = torch.sum(self._diag_observable * diag_rho)
            return expval
        else:
            expvals = [torch.sum(diag_ob * diag_rho) for diag_ob in self._diag_single_z]
            return torch.stack(expvals)

    def run_batch(
        self,
        batch_features: List[torch.Tensor],
        batch_edges: List[List[Tuple[int, int]]],
        theta: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Run the circuit on a batch of molecules with chunking.

        Parameters
        ----------
        batch_features : list of torch.Tensor
            Node features for each molecule.
        batch_edges : list of list of (int, int)
            Edges for each molecule.
        theta : torch.Tensor
            Trainable parameters.

        Returns
        -------
        list of torch.Tensor
            Predictions for each molecule.
        """
        results = []
        n_molecules = len(batch_features)

        for start in range(0, n_molecules, self.chunk_size):
            end = min(start + self.chunk_size, n_molecules)
            for i in range(start, end):
                pred = self.run_circuit(batch_features[i], batch_edges[i], theta)
                results.append(pred)

        return results


def compute_features_qiskit(z: np.ndarray, degree: np.ndarray, n_qubits: int = 8) -> np.ndarray:
    """
    Compute node features using the Qiskit notebook convention.

    node_signal[i] = 0.70 * (Z_i - 1.0) / 5.0 + 0.30 * degree_i / 4.0
    """
    n_atoms = len(z)
    node_signal = np.zeros(n_qubits, dtype=np.float64)
    for i in range(min(n_atoms, n_qubits)):
        z_scaled = (float(z[i]) - 1.0) / 5.0
        d_scaled = float(degree[i]) / 4.0
        node_signal[i] = 0.70 * z_scaled + 0.30 * d_scaled
    return node_signal


def compute_features_pennylane(atom_types: np.ndarray, adj: np.ndarray, n_qubits: int = 8) -> np.ndarray:
    """
    Compute node features using the PennyLane notebook convention.

    type_map = {C: 0.25, N: 0.50, O: 0.75, F: 1.00}
    feat[k] = type_map[atom_type[k]] + 0.05 * min(degree[k], 4)
    """
    type_map = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}
    feat = np.zeros(n_qubits, dtype=np.float64)
    for k in range(len(atom_types)):
        if k >= n_qubits:
            break
        at = int(atom_types[k])
        if at in type_map:
            feat[k] = type_map[at] + 0.05 * min(float(np.sum(adj[k])), 4.0)
    return feat
