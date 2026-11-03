"""
Contribution D: Scale-Free and Magnitude Complexity Measures for QGNNs.

Implements and compares:
1. Participation Ratio (PR): Scale-free effective rank based on Jacobian singular values.
2. Entropy-Based Effective Rank (ER): Scale-free spectral entropy exponent.
3. Paper's Effective Complexity (d_eff): Trivial ridge hat-matrix trace (shown to be scale-dependent).
4. RMS Gradient Norm: Pure magnitude measure of output sensitivity.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import linalg


def compute_jacobian_singular_values(J: np.ndarray) -> np.ndarray:
    """Compute non-negative singular values of Jacobian matrix J, sorted descending."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2:
        raise ValueError(f"Jacobian must be 2D, got shape {J.shape}")
    s = linalg.svd(J, compute_uv=False)
    # Clip negative numerical noise
    return np.maximum(s, 0.0)


def participation_ratio(J: np.ndarray, eps: float = 1e-15) -> float:
    """
    Participation Ratio (PR) of Jacobian singular values:
        PR(J) = (sum_k s_k^2)^2 / sum_k s_k^4

    Properties:
    - Scale invariant: PR(c * J) == PR(J) for all c != 0.
    - Bounded: 1.0 <= PR(J) <= rank(J) <= min(N, q).
    - If k singular values are equal and rest are 0, PR == k.
    """
    s = compute_jacobian_singular_values(J)
    max_s = np.max(s) if len(s) > 0 else 0.0
    if max_s < eps:
        return 0.0
    # Normalize by max_s to eliminate underflow/overflow for extreme scales
    s_norm = s / max_s
    s2 = s_norm ** 2
    sum_s2 = np.sum(s2)
    sum_s4 = np.sum(s2 ** 2)
    return float((sum_s2 ** 2) / sum_s4)


def entropy_effective_rank(J: np.ndarray, eps: float = 1e-15) -> float:
    """
    Entropy-based Effective Rank (Roy & Vetterli, 2007):
        ER(J) = exp( - sum_k p_k * ln(p_k) ),  where p_k = s_k / sum_j s_j

    Properties:
    - Scale invariant: ER(c * J) == ER(J) for all c != 0.
    - Continuous measure of effective dimensionality.
    - If k singular values are equal and rest are 0, ER == k.
    """
    s = compute_jacobian_singular_values(J)
    total_s = np.sum(s)
    if total_s < eps:
        return 0.0
    p = s / total_s
    # Filter positive probabilities for entropy
    p_pos = p[p > eps]
    entropy = -np.sum(p_pos * np.log(p_pos))
    return float(np.exp(entropy))


def paper_effective_complexity(J: np.ndarray, ridge: float = 1e-4) -> float:
    """
    Effective complexity definition from the original paper/repo:
        d_eff = Tr( J (J^T J + ridge * I)^(-1) J^T )

    Note: When J^T J << ridge * I, this is mathematically approximately:
        d_eff ~ (1 / ridge) * ||J||_F^2
    which is a rescaled squared gradient norm, NOT an independent scale-free dimension.
    """
    J = np.asarray(J, dtype=float)
    q = J.shape[1]
    gram = J.T @ J
    H = J @ np.linalg.inv(gram + ridge * np.eye(q)) @ J.T
    return float(np.trace(H))


def rms_gradient_norm(J: np.ndarray) -> float:
    """
    RMS Gradient Norm (magnitude measure):
        RMS(J) = sqrt( mean( J^2 ) )
    Measures overall magnitude of output sensitivity across samples and parameters.
    """
    J = np.asarray(J, dtype=float)
    return float(np.sqrt(np.mean(J ** 2)))


def compute_all_complexity_metrics(J: np.ndarray, ridge: float = 1e-4) -> Dict[str, float]:
    """
    Compute all complexity metrics side-by-side:
    - Scale-free dimension measures: Participation Ratio, Entropy Effective Rank
    - Magnitude measure: RMS Gradient Norm
    - Reference unnormalized measure: Paper Effective Complexity
    """
    return {
        "participation_ratio": participation_ratio(J),
        "entropy_effective_rank": entropy_effective_rank(J),
        "rms_gradient_norm": rms_gradient_norm(J),
        "paper_effective_complexity": paper_effective_complexity(J, ridge=ridge),
    }
