"""
Depolarizing noise channels with explicit Kraus operators and convention mappings.

Settles known issue (a): Qiskit depolarizing_error(p) vs PennyLane DepolarizingChannel(p).

Two explicit conventions:
1. 'qiskit':
   In Qiskit Aer, depolarizing_error(p, n) replaces the state with the maximally
   mixed state I / 2^n with probability p:
       E(rho) = (1 - p)*rho + p * (I / 2^n)
   Bloch-vector / traceless operator shrinkage factor:
       lambda = 1 - p  (for ANY number of qubits n)

   Expressed in Pauli basis:
   - 1-qubit (n=1):
       E_1(rho) = (1 - 3p/4)*rho + (p/4)*(X rho X + Y rho Y + Z rho Z)
   - 2-qubit (n=2):
       E_2(rho) = (1 - 15p/16)*rho + (p/16)*sum_{k=1..15} P_k rho P_k

2. 'pennylane':
   In PennyLane, DepolarizingChannel(p, wires=w) applies a non-identity Pauli
   error with total probability p (each Pauli X, Y, Z with probability p/3):
       E(rho) = (1 - p)*rho + (p/3)*(X rho X + Y rho Y + Z rho Z)
              = (1 - 4p/3)*rho + (4p/3)*(I / 2)
   Bloch-vector shrinkage factor:
       lambda = 1 - 4p/3

Parameter mapping between conventions for identical physical 1-qubit shrinkage:
    p_qiskit = (4/3) * p_pennylane
    p_pennylane = (3/4) * p_qiskit
"""

import torch
import numpy as np
from typing import List

# Fundamental Pauli matrices (complex128)
I2 = torch.eye(2, dtype=torch.complex128)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
PAULIS_1Q = [I2, X, Y, Z]


def depolarizing_kraus_1q(p: float, convention: str = 'pennylane') -> List[torch.Tensor]:
    """
    Return 1-qubit depolarizing Kraus operators under specified convention.

    Parameters
    ----------
    p : float
        Noise parameter.
    convention : str
        'pennylane': E(rho) = (1-p)*rho + (p/3)*(X rho X + Y rho Y + Z rho Z)
                     p in [0, 3/4]
        'qiskit':    E(rho) = (1-p)*rho + p*(I/2)
                     p in [0, 1]

    Returns
    -------
    list of torch.Tensor (four 2x2 matrices)
    """
    if convention == 'pennylane':
        if p < 0 or p > 0.75:
            raise ValueError(f"PennyLane 1q depolarizing p={p} must be in [0, 3/4]")
        if p == 0.0:
            return [I2.clone()]
        return [
            np.sqrt(1.0 - p) * I2,
            np.sqrt(p / 3.0) * X,
            np.sqrt(p / 3.0) * Y,
            np.sqrt(p / 3.0) * Z,
        ]
    elif convention == 'qiskit':
        if p < 0 or p > 1.0:
            raise ValueError(f"Qiskit 1q depolarizing p={p} must be in [0, 1]")
        if p == 0.0:
            return [I2.clone()]
        return [
            np.sqrt(1.0 - 0.75 * p) * I2,
            np.sqrt(p / 4.0) * X,
            np.sqrt(p / 4.0) * Y,
            np.sqrt(p / 4.0) * Z,
        ]
    else:
        raise ValueError(f"Unknown convention: {convention}")


def depolarizing_kraus_2q(p: float, convention: str = 'qiskit') -> List[torch.Tensor]:
    """
    Return 2-qubit depolarizing Kraus operators.

    Parameters
    ----------
    p : float
        Noise parameter.
    convention : str
        'qiskit': E(rho) = (1-p)*rho + p*(I/4), p in [0, 1]
        'standard': E(rho) = (1-p)*rho + (p/15)*sum P_k rho P_k, p in [0, 15/16]

    Returns
    -------
    list of torch.Tensor (sixteen 4x4 matrices)
    """
    if convention == 'qiskit':
        if p < 0 or p > 1.0:
            raise ValueError(f"Qiskit 2q depolarizing p={p} must be in [0, 1]")
        if p == 0.0:
            return [torch.eye(4, dtype=torch.complex128)]
        k0_scale = np.sqrt(1.0 - 15.0 * p / 16.0)
        kp_scale = np.sqrt(p / 16.0)
    elif convention == 'standard':
        if p < 0 or p > 15.0 / 16.0:
            raise ValueError(f"Standard 2q depolarizing p={p} must be in [0, 15/16]")
        if p == 0.0:
            return [torch.eye(4, dtype=torch.complex128)]
        k0_scale = np.sqrt(1.0 - p)
        kp_scale = np.sqrt(p / 15.0)
    else:
        raise ValueError(f"Unknown convention: {convention}")

    kraus = []
    for i, pi in enumerate(PAULIS_1Q):
        for j, pj in enumerate(PAULIS_1Q):
            pij = torch.kron(pi, pj)
            if i == 0 and j == 0:
                kraus.append(k0_scale * pij)
            else:
                kraus.append(kp_scale * pij)
    return kraus


def bloch_shrink_factor(p: float, convention: str = 'pennylane') -> float:
    """
    Bloch vector shrinkage factor for single-qubit depolarizing channel.

    After channel E(rho), Bloch vector r -> lambda * r.
    """
    if convention == 'pennylane':
        return 1.0 - 4.0 * p / 3.0
    elif convention == 'qiskit':
        return 1.0 - p
    else:
        raise ValueError(f"Unknown convention: {convention}")


# Alias for backwards compatibility
bloch_shrink_1q = bloch_shrink_factor


def verify_kraus_completeness(kraus_ops: List[torch.Tensor], atol: float = 1e-12) -> bool:
    """Verify trace preservation sum_k K_k^dag K_k = I."""
    d = kraus_ops[0].shape[0]
    total = torch.zeros(d, d, dtype=torch.complex128)
    for k in kraus_ops:
        total += k.conj().T @ k
    identity = torch.eye(d, dtype=torch.complex128)
    return torch.allclose(total, identity, atol=atol)


def apply_kraus_channel(rho: torch.Tensor, kraus_ops: List[torch.Tensor]) -> torch.Tensor:
    """Apply quantum channel E(rho) = sum_k K_k rho K_k^dag."""
    result = torch.zeros_like(rho)
    for k in kraus_ops:
        result = result + k @ rho @ k.conj().T
    return result
