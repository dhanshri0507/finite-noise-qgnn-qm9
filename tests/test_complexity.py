"""
Unit tests for Contribution D: Scale-free and magnitude complexity measures.
Validates:
(i) Synthetic models of known rank give the exact correct rank.
(ii) Rescaling the Jacobian leaves PR and ER unchanged (scale invariance).
(iii) Paper d_eff scales with c^2 (proving lack of scale invariance).
(iv) Stability across small singular values.
"""

import numpy as np
import pytest
from src.qgnn.complexity import (
    participation_ratio,
    entropy_effective_rank,
    paper_effective_complexity,
    rms_gradient_norm,
    compute_all_complexity_metrics,
)


def test_synthetic_exact_rank_participation_ratio():
    """If k singular values are equal and the rest are zero, PR must be exactly k."""
    N, q = 50, 20
    for k in [1, 2, 4, 8, 15]:
        # Construct orthogonal matrix with exactly k equal singular values
        U, _ = np.linalg.qr(np.random.randn(N, k))
        V, _ = np.linalg.qr(np.random.randn(q, k))
        s = np.ones(k) * 2.5
        J = U @ np.diag(s) @ V.T

        pr = participation_ratio(J)
        assert np.isclose(pr, k, atol=1e-10), f"Expected PR={k}, got {pr}"


def test_synthetic_exact_rank_entropy_effective_rank():
    """If k singular values are equal and the rest are zero, ER must be exactly k."""
    N, q = 50, 20
    for k in [1, 2, 4, 8, 15]:
        U, _ = np.linalg.qr(np.random.randn(N, k))
        V, _ = np.linalg.qr(np.random.randn(q, k))
        s = np.ones(k) * 3.7
        J = U @ np.diag(s) @ V.T

        er = entropy_effective_rank(J)
        assert np.isclose(er, k, atol=1e-10), f"Expected ER={k}, got {er}"


def test_scale_invariance_participation_ratio():
    """PR must be strictly invariant under scalar multiplication J -> c * J."""
    np.random.seed(42)
    J = np.random.randn(80, 8)
    base_pr = participation_ratio(J)

    for c in [1e-6, 1e-3, 0.5, 2.0, 10.0, 1e4]:
        pr_scaled = participation_ratio(c * J)
        assert np.isclose(pr_scaled, base_pr, atol=1e-11), (
            f"PR scale invariance failed for c={c}: base={base_pr}, scaled={pr_scaled}"
        )


def test_scale_invariance_entropy_effective_rank():
    """ER must be strictly invariant under scalar multiplication J -> c * J."""
    np.random.seed(42)
    J = np.random.randn(80, 8)
    base_er = entropy_effective_rank(J)

    for c in [1e-6, 1e-3, 0.5, 2.0, 10.0, 1e4]:
        er_scaled = entropy_effective_rank(c * J)
        assert np.isclose(er_scaled, base_er, atol=1e-11), (
            f"ER scale invariance failed for c={c}: base={base_er}, scaled={er_scaled}"
        )


def test_paper_deff_lacks_scale_invariance():
    """Prove that paper d_eff scales quadratic with norm when J^T J << ridge * I."""
    np.random.seed(42)
    # Small Jacobian typical of QGNN with noise (norm ~ 1e-5)
    J = 1e-5 * np.random.randn(80, 8)
    deff_1 = paper_effective_complexity(J, ridge=1e-4)

    # Scaling J by 2 should scale d_eff by approximately 4 (2^2)
    deff_2 = paper_effective_complexity(2.0 * J, ridge=1e-4)
    ratio = deff_2 / deff_1

    # Ratio should be ~4.0, proving quadratic dependence on scale
    assert np.isclose(ratio, 4.0, atol=1e-3), f"Expected d_eff ratio ~4.0, got {ratio}"


def test_zero_jacobian_handling():
    """All metrics should return clean 0.0 on an all-zero Jacobian without crashing."""
    J_zero = np.zeros((80, 8))
    metrics = compute_all_complexity_metrics(J_zero)
    assert metrics["participation_ratio"] == 0.0
    assert metrics["entropy_effective_rank"] == 0.0
    assert metrics["rms_gradient_norm"] == 0.0
    assert np.isclose(metrics["paper_effective_complexity"], 0.0, atol=1e-12)
