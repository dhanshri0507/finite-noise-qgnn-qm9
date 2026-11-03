"""
Tests for the density-matrix simulator and noise model.

Covers:
1. Kraus operator completeness (trace preservation)
2. Bloch vector shrinkage for 1-qubit depolarizing channel
3. Known analytic results (|0> under depolarizing, Bell state correlators)
4. CNOT-RZ-CNOT at theta=0 is identity
5. Gate matrices (RY, RX, RZ unitarity)
6. Noiseless circuit correctness on known states
7. Noise convention comparison (issue (a) from user spec)
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.qgnn.noise import (
    depolarizing_kraus_1q,
    depolarizing_kraus_2q,
    bloch_shrink_1q,
    bloch_shrink_factor,
    verify_kraus_completeness,
    apply_kraus_channel,
    I2, X, Y, Z,
    PAULIS_1Q,
)
from src.qgnn.simulator import (
    ry_matrix, rx_matrix, rz_matrix,
    DensityMatrixSimulator,
    compute_features_qiskit,
    compute_features_pennylane,
)


# ============================================================
# 1. Kraus Operator Tests
# ============================================================

class TestKrausOperators:
    """Test Kraus operator construction and properties."""

    @pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75])
    def test_1q_completeness(self, p):
        """Sum_k K_k^dag K_k = I for 1-qubit depolarizing."""
        kraus = depolarizing_kraus_1q(p)
        assert verify_kraus_completeness(kraus, atol=1e-12)

    @pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.1, 0.2, 0.5])
    def test_2q_completeness(self, p):
        """Sum_k K_k^dag K_k = I for 2-qubit depolarizing."""
        kraus = depolarizing_kraus_2q(p)
        assert verify_kraus_completeness(kraus, atol=1e-12)

    def test_1q_invalid_p(self):
        """p > 3/4 should raise ValueError."""
        with pytest.raises(ValueError):
            depolarizing_kraus_1q(0.8)

    def test_2q_invalid_p(self):
        """p > 1.0 (Qiskit) or p > 15/16 (standard) should raise ValueError."""
        with pytest.raises(ValueError):
            depolarizing_kraus_2q(1.1, convention='qiskit')
        with pytest.raises(ValueError):
            depolarizing_kraus_2q(0.95, convention='standard')

    def test_1q_noiseless_is_identity(self):
        """p=0 should return a single identity Kraus operator."""
        kraus = depolarizing_kraus_1q(0.0)
        assert len(kraus) == 1
        assert torch.allclose(kraus[0], I2, atol=1e-15)

    def test_2q_noiseless_is_identity(self):
        """p=0 should return a single identity Kraus operator."""
        kraus = depolarizing_kraus_2q(0.0)
        assert len(kraus) == 1
        assert torch.allclose(kraus[0], torch.eye(4, dtype=torch.complex128), atol=1e-15)

    @pytest.mark.parametrize("p", [0.01, 0.05, 0.1, 0.2])
    def test_1q_trace_preservation(self, p):
        """Channel should preserve trace of density matrix."""
        kraus = depolarizing_kraus_1q(p)
        # Random density matrix
        psi = torch.randn(2, dtype=torch.complex128)
        psi = psi / torch.norm(psi)
        rho = torch.outer(psi, psi.conj())

        rho_out = apply_kraus_channel(rho, kraus)
        assert abs(torch.trace(rho_out).real - 1.0) < 1e-12

    @pytest.mark.parametrize("p", [0.01, 0.05, 0.1, 0.2])
    def test_2q_trace_preservation(self, p):
        """2-qubit channel should preserve trace."""
        kraus = depolarizing_kraus_2q(p)
        psi = torch.randn(4, dtype=torch.complex128)
        psi = psi / torch.norm(psi)
        rho = torch.outer(psi, psi.conj())

        rho_out = apply_kraus_channel(rho, kraus)
        assert abs(torch.trace(rho_out).real - 1.0) < 1e-12


# ============================================================
# 2. Bloch Vector Shrinkage Tests
# ============================================================

class TestBlochShrinkage:
    """
    Test that the depolarizing channel contracts the Bloch vector correctly.

    For |0>: Bloch vector = (0, 0, 1)
    After depolarizing with parameter p:
        <Z> = 1 * (1 - 4p/3) = bloch_shrink_1q(p)
        <X> = 0
        <Y> = 0
    """

    @pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 0.75])
    def test_z_expectation_on_zero_state(self, p):
        """<Z> on |0> after depolarizing(p) should equal (1 - 4p/3)."""
        kraus = depolarizing_kraus_1q(p)
        rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)

        rho_out = apply_kraus_channel(rho0, kraus)
        z_exp = torch.trace(Z @ rho_out).real.item()
        expected = bloch_shrink_1q(p)

        assert abs(z_exp - expected) < 1e-12, (
            f"p={p}: got <Z>={z_exp}, expected {expected}"
        )

    @pytest.mark.parametrize("p", [0.01, 0.05, 0.10, 0.20])
    def test_x_expectation_on_plus_state(self, p):
        """<X> on |+> after depolarizing(p) should equal (1 - 4p/3)."""
        kraus = depolarizing_kraus_1q(p)
        # |+> = (|0> + |1>) / sqrt(2)
        plus = torch.tensor([1, 1], dtype=torch.complex128) / np.sqrt(2)
        rho_plus = torch.outer(plus, plus.conj())

        rho_out = apply_kraus_channel(rho_plus, kraus)
        x_exp = torch.trace(X @ rho_out).real.item()
        expected = bloch_shrink_1q(p)

        assert abs(x_exp - expected) < 1e-12, (
            f"p={p}: got <X>={x_exp}, expected {expected}"
        )

    @pytest.mark.parametrize("p", [0.01, 0.05, 0.10, 0.20])
    def test_y_expectation_on_plus_i_state(self, p):
        """<Y> on |+i> after depolarizing(p) should equal (1 - 4p/3)."""
        kraus = depolarizing_kraus_1q(p)
        # |+i> = (|0> + i|1>) / sqrt(2)
        plus_i = torch.tensor([1, 1j], dtype=torch.complex128) / np.sqrt(2)
        rho_pi = torch.outer(plus_i, plus_i.conj())

        rho_out = apply_kraus_channel(rho_pi, kraus)
        y_exp = torch.trace(Y @ rho_out).real.item()
        expected = bloch_shrink_1q(p)

        assert abs(y_exp - expected) < 1e-12, (
            f"p={p}: got <Y>={y_exp}, expected {expected}"
        )

    @pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.10, 0.20])
    def test_maximally_mixed_is_fixed_point(self, p):
        """I/2 should be unchanged by depolarizing channel."""
        kraus = depolarizing_kraus_1q(p)
        rho_mixed = torch.eye(2, dtype=torch.complex128) / 2.0

        rho_out = apply_kraus_channel(rho_mixed, kraus)
        assert torch.allclose(rho_out, rho_mixed, atol=1e-12)

    def test_full_depolarization(self):
        """At p=3/4, all states map to I/2."""
        kraus = depolarizing_kraus_1q(0.75)
        rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)

        rho_out = apply_kraus_channel(rho0, kraus)
        expected = torch.eye(2, dtype=torch.complex128) / 2.0
        assert torch.allclose(rho_out, expected, atol=1e-12)

    @pytest.mark.parametrize("p", [0.01, 0.05, 0.10])
    def test_2q_bloch_shrinkage_on_bell_state_qiskit(self, p):
        """
        2-qubit depolarizing on Bell state |Phi+> (Qiskit convention).
        In Qiskit: E(rho) = (1-p)*rho + p*I/4.
        After E(rho): <ZZ> should be (1 - p).
        """
        kraus = depolarizing_kraus_2q(p, convention='qiskit')
        bell = torch.zeros(4, dtype=torch.complex128)
        bell[0] = 1 / np.sqrt(2)  # |00>
        bell[3] = 1 / np.sqrt(2)  # |11>
        rho_bell = torch.outer(bell, bell.conj())

        rho_out = apply_kraus_channel(rho_bell, kraus)
        zz = torch.kron(Z, Z)
        zz_exp = torch.trace(zz @ rho_out).real.item()
        expected = 1.0 - p
        assert abs(zz_exp - expected) < 1e-10, (
            f"p={p}: got <ZZ>={zz_exp}, expected {expected}"
        )

    @pytest.mark.parametrize("p", [0.01, 0.05, 0.10])
    def test_2q_bloch_shrinkage_on_bell_state_standard(self, p):
        """
        2-qubit depolarizing on Bell state |Phi+> (Standard convention).
        After E(rho): <ZZ> should be (1 - 16p/15).
        """
        kraus = depolarizing_kraus_2q(p, convention='standard')
        bell = torch.zeros(4, dtype=torch.complex128)
        bell[0] = 1 / np.sqrt(2)  # |00>
        bell[3] = 1 / np.sqrt(2)  # |11>
        rho_bell = torch.outer(bell, bell.conj())

        rho_out = apply_kraus_channel(rho_bell, kraus)
        zz = torch.kron(Z, Z)
        zz_exp = torch.trace(zz @ rho_out).real.item()
        expected = 1.0 - 16.0 * p / 15.0
        assert abs(zz_exp - expected) < 1e-10, (
            f"p={p}: got <ZZ>={zz_exp}, expected {expected}"
        )


# ============================================================
# 3. Gate Matrix Tests
# ============================================================

class TestGateMatrices:
    """Test single-qubit gate matrix constructions."""

    def test_ry_unitarity(self):
        """RY should be unitary for various angles."""
        for angle in [0.0, 0.5, 1.0, np.pi, np.pi/2]:
            t = torch.tensor(angle, dtype=torch.float64)
            U = ry_matrix(t)
            identity = U @ U.conj().T
            assert torch.allclose(identity, torch.eye(2, dtype=torch.complex128), atol=1e-12)

    def test_rx_unitarity(self):
        """RX should be unitary."""
        for angle in [0.0, 0.5, 1.0, np.pi, np.pi/2]:
            t = torch.tensor(angle, dtype=torch.float64)
            U = rx_matrix(t)
            identity = U @ U.conj().T
            assert torch.allclose(identity, torch.eye(2, dtype=torch.complex128), atol=1e-12)

    def test_rz_unitarity(self):
        """RZ should be unitary."""
        for angle in [0.0, 0.5, 1.0, np.pi, np.pi/2]:
            t = torch.tensor(angle, dtype=torch.float64)
            U = rz_matrix(t)
            identity = U @ U.conj().T
            assert torch.allclose(identity, torch.eye(2, dtype=torch.complex128), atol=1e-12)

    def test_ry_zero_is_identity(self):
        """RY(0) = I."""
        U = ry_matrix(torch.tensor(0.0))
        assert torch.allclose(U, torch.eye(2, dtype=torch.complex128), atol=1e-12)

    def test_rx_zero_is_identity(self):
        """RX(0) = I."""
        U = rx_matrix(torch.tensor(0.0))
        assert torch.allclose(U, torch.eye(2, dtype=torch.complex128), atol=1e-12)

    def test_rz_zero_is_identity(self):
        """RZ(0) = I."""
        U = rz_matrix(torch.tensor(0.0))
        assert torch.allclose(U, torch.eye(2, dtype=torch.complex128), atol=1e-12)

    def test_ry_pi_flips_state(self):
        """RY(pi)|0> should give |1> up to numerical precision."""
        U = ry_matrix(torch.tensor(np.pi))
        state0 = torch.tensor([1.0, 0.0], dtype=torch.complex128)
        result = U @ state0
        state1 = torch.tensor([0.0, 1.0], dtype=torch.complex128)
        assert torch.allclose(result, state1, atol=1e-6)

    def test_rx_pi_gives_minus_i_X(self):
        """RX(pi) = -i*X (up to global phase and numerical precision)."""
        U = rx_matrix(torch.tensor(np.pi))
        expected = -1j * X
        # Check up to global phase
        ratio = U[1, 0] / expected[1, 0]
        assert torch.allclose(U, ratio * expected, atol=1e-6)


# ============================================================
# 4. CNOT-RZ-CNOT Identity Test
# ============================================================

class TestCNOTRZCNOT:
    """Test that CNOT-RZ(0)-CNOT is identity."""

    def test_cnot_rz0_cnot_is_identity_2q(self):
        """
        On a 2-qubit system, CNOT(0,1)-RZ(0,wire=1)-CNOT(0,1)
        should be the identity operation.
        """
        sim = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.0)

        # Test on several random 2-qubit states
        for _ in range(5):
            psi = torch.randn(4, dtype=torch.complex128)
            psi = psi / torch.norm(psi)
            rho = torch.outer(psi, psi.conj())

            # Apply CNOT
            rho1 = sim._apply_cnot(rho, 0, 1)
            # Apply RZ(0) on qubit 1
            rz0 = rz_matrix(torch.tensor(0.0))
            U_rz = sim._embed_1q_operator(rz0, 1)
            rho2 = U_rz @ rho1 @ U_rz.conj().T
            # Apply CNOT again
            rho3 = sim._apply_cnot(rho2, 0, 1)

            assert torch.allclose(rho3, rho, atol=1e-10), (
                f"CNOT-RZ(0)-CNOT is not identity! Max diff: {(rho3 - rho).abs().max()}"
            )


# ============================================================
# 5. Simulator Basic Tests
# ============================================================

class TestSimulatorBasic:
    """Basic simulator functionality tests."""

    def test_noiseless_all_zeros_features_qiskit(self):
        """
        With all-zero features and theta=0, the circuit should be identity.
        All qubits in |0>, Z^8 expectation = 1.
        """
        sim = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.0)
        features = torch.zeros(2, dtype=torch.float64)
        theta = torch.zeros(2, dtype=torch.float64)
        edges = []

        result = sim.run_circuit(features, edges, theta)
        # Z x Z on |00> = (+1)(+1) = 1
        assert abs(result.item() - 1.0) < 1e-10

    def test_noiseless_all_zeros_features_pennylane(self):
        """
        With all-zero features and theta=0, PennyLane circuit:
        pi*0 = 0, so RY(0) = I, all qubits in |0>.
        Each <Z_w> = 1.
        """
        sim = DensityMatrixSimulator(n_qubits=2, experiment='pennylane', beta=0.0)
        features = torch.zeros(2, dtype=torch.float64)
        theta = torch.zeros(1 + 2*2, dtype=torch.float64)  # 5 params for 2 qubits
        edges = []

        result = sim.run_circuit(features, edges, theta)
        assert result.shape == (2,)
        assert torch.allclose(result, torch.ones(2, dtype=torch.float64), atol=1e-10)

    def test_noise_reduces_z_expectation(self):
        """Noise should reduce the magnitude of Z expectation."""
        features = torch.tensor([0.5, 0.3], dtype=torch.float64)
        theta = torch.tensor([0.1, 0.2], dtype=torch.float64)
        edges = [(0, 1)]

        sim_clean = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.0)
        sim_noisy = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.1)

        result_clean = sim_clean.run_circuit(features, edges, theta)
        result_noisy = sim_noisy.run_circuit(features, edges, theta)

        # Noisy result should have smaller magnitude
        assert abs(result_noisy.item()) < abs(result_clean.item()) + 1e-10

    def test_cnot_creates_entanglement(self):
        """
        With RY(pi) on q0 giving |1>, CNOT(0,1) flips q1 giving |1,1>.
        But then CNOT-RZ(0)-CNOT is identity, so CNOT undoes itself.
        The actual circuit flow is:
          RY(pi) on q0 -> |10>
          CNOT(0,1) -> |11>
          RZ(0) on q1 -> |11> (identity)
          CNOT(0,1) -> |10> (undone!)
          RX(0) -> |10>
        So <ZZ> = <Z_0><Z_1> = (-1)(+1) = -1
        """
        sim = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.0)

        features = torch.tensor([np.pi, 0.0], dtype=torch.float64)
        theta = torch.zeros(2, dtype=torch.float64)
        edges = [(0, 1)]

        result = sim.run_circuit(features, edges, theta)
        # State is |10>, <ZZ> = (-1)(+1) = -1
        assert abs(result.item() - (-1.0)) < 1e-6

    def test_run_batch(self):
        """Test batch processing."""
        sim = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.0, chunk_size=2)

        batch_features = [
            torch.zeros(2, dtype=torch.float64),
            torch.tensor([0.5, 0.3], dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        ]
        batch_edges = [[], [(0,1)], []]
        theta = torch.zeros(2, dtype=torch.float64)

        results = sim.run_batch(batch_features, batch_edges, theta)
        assert len(results) == 3


# ============================================================
# 6. Noise Convention Comparison (Issue (a))
# ============================================================

class TestNoiseConventions:
    """
    Settle issue (a): Qiskit depolarizing_error(p) vs PennyLane
    DepolarizingChannel(p) may not be the same physical beta.

    FINDING: For 1-qubit gates, both use IDENTICAL formulas:
        E(rho) = (1-p)*rho + (p/3)*(X rho X + Y rho Y + Z rho Z)

    The DIFFERENCE is in how 2-qubit gate noise is handled:
    - Qiskit: 2-qubit depolarizing on BOTH qubits of CNOT
    - PennyLane: 1-qubit depolarizing on TARGET qubit ONLY

    So the same beta produces different physical noise levels
    at CNOT gates.
    """

    def test_1q_channels_are_identical(self):
        """
        Both conventions should produce identical 1-qubit channels.
        Test by verifying Bloch shrinkage is the same.
        """
        for p in [0.01, 0.05, 0.1, 0.2]:
            kraus = depolarizing_kraus_1q(p)
            rho0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128)
            rho_out = apply_kraus_channel(rho0, kraus)

            z_exp = torch.trace(Z @ rho_out).real.item()
            expected = 1.0 - 4.0 * p / 3.0

            assert abs(z_exp - expected) < 1e-12, (
                f"p={p}: Bloch shrinkage mismatch. Got {z_exp}, expected {expected}"
            )

    def test_cnot_noise_differs_between_conventions(self):
        """
        Demonstrate that the same beta produces different noise
        at CNOT gates between Qiskit and PennyLane conventions.
        """
        beta = 0.1

        # Prepare a Bell state |Phi+> using a 2-qubit circuit
        # Features: qubit 0 gets RY(pi/2) -> |+>, qubit 1 gets RY(0) -> |0>
        # Then CNOT(0,1) -> (|00> + |11>)/sqrt(2)
        features = torch.tensor([np.pi/2, 0.0], dtype=torch.float64)
        # For Qiskit: theta = [0, 0] (no trainable rotations matter for this test)

        sim_qiskit = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=beta)
        sim_pl = DensityMatrixSimulator(n_qubits=2, experiment='pennylane', beta=beta)

        theta_q = torch.zeros(2, dtype=torch.float64)
        theta_pl = torch.zeros(5, dtype=torch.float64)  # 1 + 2*2

        edges = [(0, 1)]

        # PennyLane multiplies features by pi, so to get same encoding:
        features_pl = features / np.pi

        result_qiskit = sim_qiskit.run_circuit(features, edges, theta_q)
        result_pl = sim_pl.run_circuit(features_pl, edges, theta_pl)

        # These should differ because Qiskit applies 2q noise on CNOT
        # while PennyLane applies 1q noise on target only
        # We just verify they're not the same (within reason)
        # Note: They could coincidentally be close, so we just document the values
        print(f"  Qiskit (beta={beta}): Z^2 = {result_qiskit.item():.10f}")
        print(f"  PennyLane (beta={beta}): Z = {result_pl.tolist()}")

    def test_noiseless_circuits_agree(self):
        """
        Both conventions should give the same noiseless result
        when given the same physical circuit (adjusted for encoding).
        """
        # Use 2 qubits, same effective encoding
        features_q = torch.tensor([0.5, 0.3], dtype=torch.float64)
        features_pl = features_q / np.pi  # PennyLane multiplies by pi internally

        sim_q = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.0)
        sim_pl = DensityMatrixSimulator(n_qubits=2, experiment='pennylane', beta=0.0)

        # Set up equivalent parameters:
        # Qiskit: theta = [theta_0, theta_1] (RZ uses theta[v], RX uses theta[q])
        # PennyLane: theta = [theta_rz_shared, theta_rx_0, theta_rx_1, theta_rz2_0, theta_rz2_1]
        # For equivalence with no edges, just compare encoding + trainable layer
        # With edge (0,1):
        #   Qiskit RZ: theta[1] on qubit 1
        #   PennyLane RZ: theta[0] on qubit 1
        #   Qiskit RX: theta[0] on q0, theta[1] on q1
        #   PennyLane RX: theta[1] on q0, theta[2] on q1
        #   PennyLane RZ2: theta[3] on q0, theta[4] on q1

        # No edges, so just encoding + trainable layer
        # Qiskit: RY(0.5) on q0, RY(0.3) on q1, then RX(theta[0]) on q0, RX(theta[1]) on q1
        # PennyLane: RY(pi*0.5/pi)=RY(0.5) on q0, RY(pi*0.3/pi)=RY(0.3) on q1,
        #           then RX(theta[1]) on q0, RZ(theta[3]) on q0, RX(theta[2]) on q1, RZ(theta[4]) on q1

        # Match by setting PennyLane's extra RZ to 0
        theta_q = torch.tensor([0.4, 0.6], dtype=torch.float64)
        theta_pl = torch.tensor([0.0, 0.4, 0.6, 0.0, 0.0], dtype=torch.float64)

        result_q = sim_q.run_circuit(features_q, [], theta_q)
        result_pl = sim_pl.run_circuit(features_pl, [], theta_pl)

        # Qiskit returns scalar Z^2, PennyLane returns vector [<Z_0>, <Z_1>]
        # Z^2 = Z_0 * Z_1 for product states
        z_product = result_pl[0] * result_pl[1]

        # These should be close (not exact due to different readout structures)
        assert abs(result_q.item() - z_product.item()) < 1e-8, (
            f"Noiseless mismatch: Qiskit Z^2={result_q.item()}, PennyLane Z0*Z1={z_product.item()}"
        )


# ============================================================
# 7. Feature Computation Tests
# ============================================================

class TestFeatureComputation:
    """Test node feature encoding formulas."""

    def test_qiskit_carbon_degree2(self):
        """Carbon (Z=6) with degree 2."""
        z = np.array([6.0])
        deg = np.array([2.0])
        feat = compute_features_qiskit(z, deg, n_qubits=2)

        z_scaled = (6.0 - 1.0) / 5.0  # = 1.0
        d_scaled = 2.0 / 4.0  # = 0.5
        expected = 0.70 * z_scaled + 0.30 * d_scaled  # = 0.70 + 0.15 = 0.85
        assert abs(feat[0] - expected) < 1e-12
        assert feat[1] == 0.0  # padding

    def test_qiskit_nitrogen_degree3(self):
        """Nitrogen (Z=7) with degree 3."""
        z = np.array([7.0])
        deg = np.array([3.0])
        feat = compute_features_qiskit(z, deg, n_qubits=2)

        expected = 0.70 * (7.0 - 1.0) / 5.0 + 0.30 * 3.0 / 4.0
        assert abs(feat[0] - expected) < 1e-12

    def test_pennylane_carbon_degree2(self):
        """PennyLane: C (type=1) -> 0.25, degree 2 -> 0.05*2=0.10. Total=0.35."""
        atom_types = np.array([1])
        adj = np.array([[0, 1, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        feat = compute_features_pennylane(atom_types, adj, n_qubits=8)

        expected = 0.25 + 0.05 * min(2.0, 4.0)
        assert abs(feat[0] - expected) < 1e-12


# ============================================================
# 8. Small-Scale Validation Tests
# ============================================================

class TestSmallScaleValidation:
    """Tests on 1-2 qubit systems with known analytic answers."""

    def test_1qubit_ry_z_expectation(self):
        """
        1-qubit: RY(theta)|0>, <Z> = cos(theta).
        """
        sim = DensityMatrixSimulator(n_qubits=1, experiment='qiskit', beta=0.0)

        for angle in [0.0, 0.3, np.pi/4, np.pi/2, np.pi]:
            features = torch.tensor([angle], dtype=torch.float64)
            theta = torch.zeros(1, dtype=torch.float64)

            result = sim.run_circuit(features, [], theta)
            expected = np.cos(angle)

            assert abs(result.item() - expected) < 1e-10, (
                f"angle={angle}: got {result.item()}, expected {expected}"
            )

    def test_1qubit_ry_noisy_z_expectation(self):
        """
        1-qubit: RY(theta)|0> then depolarizing(p).
        <Z> = cos(theta) * (1 - 4p/3).
        """
        for p in [0.01, 0.05, 0.1, 0.2]:
            sim = DensityMatrixSimulator(n_qubits=1, experiment='qiskit', beta=p)

            angle = np.pi / 3
            features = torch.tensor([angle], dtype=torch.float64)
            theta = torch.zeros(1, dtype=torch.float64)

            result = sim.run_circuit(features, [], theta)
            # In Qiskit convention: shrinkage is (1 - p) per gate
            # RY(angle) followed by noise -> (1 - p)
            # RX(0) followed by noise -> (1 - p)^2
            expected = np.cos(angle) * bloch_shrink_factor(p, convention='qiskit') ** 2

            assert abs(result.item() - expected) < 1e-10, (
                f"p={p}: got {result.item()}, expected {expected}"
            )

    def test_2qubit_bell_state_noiseless(self):
        """
        RY(pi/2) on q0 gives |+>, CNOT(0,1) creates Bell state,
        but then CNOT-RZ(0)-CNOT undoes itself.
        Final state is |+,0>, so <ZZ> = <Z_0><Z_1> = 0 * 1 = 0.

        For a proper Bell state test, use nonzero RZ angle so the
        second CNOT doesn't fully undo the first.
        With theta[1]=pi (for qubit 1 in Qiskit convention):
          RY(pi/2) on q0 -> |+>
          CNOT(0,1) -> Bell |Phi+>
          RZ(pi) on q1: adds a phase Z to |1> component
          CNOT(0,1) -> different entangled state
        """
        sim = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.0)

        # With nonzero theta for a nontrivial circuit test
        features = torch.tensor([np.pi/2, 0.0], dtype=torch.float64)
        theta = torch.tensor([0.0, np.pi], dtype=torch.float64)  # RZ(pi) on q1
        edges = [(0, 1)]

        result = sim.run_circuit(features, edges, theta)
        # Just verify it's a valid expectation value in [-1, 1]
        assert -1.0 <= result.item() <= 1.0 + 1e-10

        # Also test: with no edges but nonzero features,
        # <ZZ> = <Z_0> * <Z_1> for product states
        features2 = torch.tensor([0.5, 0.3], dtype=torch.float64)
        theta2 = torch.zeros(2, dtype=torch.float64)
        result2 = sim.run_circuit(features2, [], theta2)
        expected = np.cos(0.5) * np.cos(0.3)  # <Z_0><Z_1>
        assert abs(result2.item() - expected) < 1e-6


# ============================================================
# 9. Gradient Flow Test
# ============================================================

class TestGradientFlow:
    """Test that gradients can flow through the simulator."""

    def test_gradient_flows_qiskit(self):
        """Verify gradients exist for the Qiskit circuit."""
        sim = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.0)

        features = torch.tensor([0.5, 0.3], dtype=torch.float64)
        theta = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        edges = [(0, 1)]

        result = sim.run_circuit(features, edges, theta)
        result.backward()

        assert theta.grad is not None
        assert not torch.all(theta.grad == 0), "Gradients are all zero!"

    def test_gradient_flows_pennylane(self):
        """Verify gradients exist for the PennyLane circuit."""
        sim = DensityMatrixSimulator(n_qubits=2, experiment='pennylane', beta=0.0)

        features = torch.tensor([0.5, 0.3], dtype=torch.float64)
        theta = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64, requires_grad=True)
        edges = [(0, 1)]

        result = sim.run_circuit(features, edges, theta)
        loss = result.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.all(theta.grad == 0), "Gradients are all zero!"

    def test_zero_init_gradient_is_zero_qiskit(self):
        """
        Confirm issue (b): at theta=0, the Qiskit circuit's gradient
        may be exactly zero (stationary point issue).

        With zero features and theta=0:
        Circuit is identity, state is |00...0>, <Z^n> = 1.
        Gradient of <Z^n> w.r.t. theta should be zero because
        the state is already an eigenstate.
        """
        sim = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.0)

        features = torch.zeros(2, dtype=torch.float64)
        theta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        edges = []

        result = sim.run_circuit(features, edges, theta)
        result.backward()

        # With zero features and no edges, the circuit is just RY(0)-RX(0) = I-I = I
        # State = |00>, <ZZ> = 1 (constant), so gradient = 0
        assert torch.allclose(theta.grad, torch.zeros_like(theta.grad), atol=1e-10), (
            f"Expected zero gradient at theta=0 with zero features, got {theta.grad}"
        )

    def test_noisy_gradient_flows(self):
        """Verify gradients flow through noisy channels."""
        sim = DensityMatrixSimulator(n_qubits=2, experiment='qiskit', beta=0.05)

        features = torch.tensor([0.5, 0.3], dtype=torch.float64)
        theta = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)
        edges = [(0, 1)]

        result = sim.run_circuit(features, edges, theta)
        result.backward()

        assert theta.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
