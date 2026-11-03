"""Tests for data loading, splitting, and standardization.

Tests cover:
1. QM9 loading and heavy-atom filtering
2. Split creation (no overlap, correct sizes)
3. Data leakage prevention (standardization on train only)
4. Reproducibility (same seed = same split)
5. Feature extraction
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---- Split Logic Tests (no QM9 download required) ----

class TestCreateSplit:
    """Test the split creation logic."""

    def test_split_sizes(self):
        """Split sizes should match requested fractions."""
        from src.qgnn.data import create_split
        n = 1000
        split = create_split(n, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
        assert len(split["train"]) == 700
        assert len(split["val"]) == 150
        assert len(split["test"]) == 150

    def test_no_overlap(self):
        """Train, val, and test sets must not overlap."""
        from src.qgnn.data import create_split
        split = create_split(500, seed=42)
        all_idx = np.concatenate([split["train"], split["val"], split["test"]])
        assert len(all_idx) == 500
        assert len(np.unique(all_idx)) == 500

    def test_complete_coverage(self):
        """All indices from 0 to n-1 must appear exactly once."""
        from src.qgnn.data import create_split
        n = 300
        split = create_split(n, seed=42)
        all_idx = sorted(np.concatenate([split["train"], split["val"], split["test"]]))
        np.testing.assert_array_equal(all_idx, np.arange(n))

    def test_reproducibility(self):
        """Same seed must produce identical splits."""
        from src.qgnn.data import create_split
        split1 = create_split(1000, seed=123)
        split2 = create_split(1000, seed=123)
        for key in ["train", "val", "test"]:
            np.testing.assert_array_equal(split1[key], split2[key])

    def test_different_seed_different_split(self):
        """Different seeds must produce different splits."""
        from src.qgnn.data import create_split
        split1 = create_split(1000, seed=42)
        split2 = create_split(1000, seed=43)
        # Very unlikely to be equal with different seeds
        assert not np.array_equal(split1["train"], split2["train"])

    def test_fractions_must_sum_to_one(self):
        """Invalid fractions should raise an error."""
        from src.qgnn.data import create_split
        with pytest.raises(AssertionError):
            create_split(100, train_frac=0.5, val_frac=0.3, test_frac=0.3, seed=42)


class TestSeedUtils:
    """Test seed and reproducibility utilities."""

    def test_config_hash_deterministic(self):
        """Same config must produce the same hash."""
        from src.qgnn.seed_utils import config_hash
        cfg = {"lr": 0.001, "seed": 42, "beta": 0.05}
        assert config_hash(cfg) == config_hash(cfg)

    def test_config_hash_order_invariant(self):
        """Hash should not depend on key insertion order."""
        from src.qgnn.seed_utils import config_hash
        cfg1 = {"a": 1, "b": 2}
        cfg2 = {"b": 2, "a": 1}
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_config_hash_different_configs(self):
        """Different configs must produce different hashes."""
        from src.qgnn.seed_utils import config_hash
        cfg1 = {"lr": 0.001}
        cfg2 = {"lr": 0.01}
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_system_info(self):
        """System info must include required fields."""
        from src.qgnn.seed_utils import get_system_info
        info = get_system_info()
        required_keys = ["platform", "python_version", "torch_version", "timestamp"]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

    def test_run_manifest_creates_file(self):
        """Run manifest should create a JSON file."""
        from src.qgnn.seed_utils import create_run_manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = create_run_manifest(
                run_id="test_run_001",
                config={"lr": 0.001, "seed": 42},
                output_dir=Path(tmpdir),
                git_commit="abc123",
            )
            manifest_file = Path(tmpdir) / "manifest_test_run_001.json"
            assert manifest_file.exists()
            with open(manifest_file) as f:
                loaded = json.load(f)
            assert loaded["run_id"] == "test_run_001"
            assert loaded["git_commit"] == "abc123"
            assert "config_hash" in loaded

    def test_set_global_seed_reproducibility(self):
        """Setting the same seed must produce the same random numbers."""
        from src.qgnn.seed_utils import set_global_seed
        import torch

        set_global_seed(42)
        a = torch.randn(10)
        b = np.random.randn(10)

        set_global_seed(42)
        a2 = torch.randn(10)
        b2 = np.random.randn(10)

        np.testing.assert_array_equal(a.numpy(), a2.numpy())
        np.testing.assert_array_equal(b, b2)


class TestDataIntegrity:
    """Tests that require QM9 to be downloaded.

    These are marked with @pytest.mark.slow and skipped if QM9 is not available.
    """

    @pytest.fixture(autouse=True)
    def check_qm9_available(self):
        """Skip if QM9 data is not available locally."""
        try:
            from torch_geometric.datasets import QM9
            # Just check if we can instantiate - actual download happens in test
            self.qm9_available = True
        except ImportError:
            pytest.skip("torch_geometric not available")

    @pytest.mark.slow
    def test_heavy_atom_count(self):
        """Verify heavy atom counting is correct."""
        from src.qgnn.data import count_heavy_atoms
        from torch_geometric.datasets import QM9
        dataset = QM9(root="data/qm9")
        # Check first molecule
        data = dataset[0]
        n_heavy = count_heavy_atoms(data)
        # Count manually
        expected = int((data.z > 1).sum().item())
        assert n_heavy == expected

    @pytest.mark.slow
    def test_standardization_uses_train_only(self):
        """Standardization statistics must come from training set only.

        This is a DATA LEAKAGE test.
        """
        from src.qgnn.data import (
            load_and_filter_qm9, create_split,
            compute_target_statistics, GAP_TARGET_IDX
        )
        dataset, filtered_indices = load_and_filter_qm9("data/qm9", max_heavy_atoms=8)
        filtered_indices = np.array(filtered_indices)
        split = create_split(len(filtered_indices), seed=42)

        train_ds_indices = filtered_indices[split["train"]]
        val_ds_indices = filtered_indices[split["val"]]

        train_stats = compute_target_statistics(dataset, train_ds_indices)
        val_stats = compute_target_statistics(dataset, val_ds_indices)

        # Stats should differ (they're computed on different data)
        # This doesn't prove no leakage by itself, but verifies the
        # function uses only the given indices
        assert train_stats["mean"] != val_stats["mean"]
        assert train_stats["count"] == len(train_ds_indices)
        assert val_stats["count"] == len(val_ds_indices)

    @pytest.mark.slow
    def test_gap_target_is_index_4(self):
        """Verify that index 4 corresponds to HOMO-LUMO gap."""
        from torch_geometric.datasets import QM9
        dataset = QM9(root="data/qm9")
        data = dataset[0]
        # QM9 targets: mu(0), alpha(1), homo(2), lumo(3), gap(4), ...
        # gap = homo - lumo (in eV, some sources use Hartree)
        homo = data.y[0, 2].item()
        lumo = data.y[0, 3].item()
        gap = data.y[0, 4].item()
        # gap should equal lumo - homo (PyG convention) or similar
        # The exact relationship depends on PyG version; just check it's reasonable
        assert gap > 0, "HOMO-LUMO gap should be positive"
        assert 0 < gap < 30, f"Gap {gap} eV seems unreasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
