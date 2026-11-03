"""QM9 data loading, filtering, splitting, and standardization.

Design decisions:
- Uses PyTorch Geometric's QM9 dataset.
- Filters to molecules with <= 8 heavy atoms (non-hydrogen).
- RANDOM subset selection (not file order) using a fixed seed.
- Train/val/test split saved to disk for exact reproducibility.
- Targets standardized using training set statistics ONLY.
- HOMO-LUMO gap is target index 4 in PyG's QM9 (column 'gap').
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.datasets import QM9

from .seed_utils import set_global_seed

logger = logging.getLogger(__name__)

# QM9 target index for HOMO-LUMO gap (eV)
# PyG QM9 targets: mu, alpha, homo, lumo, gap, r2, zpve, u0, u298, h298, g298, cv, ...
# gap is index 4
GAP_TARGET_IDX = 4


def count_heavy_atoms(data) -> int:
    """Count non-hydrogen atoms in a PyG Data object.

    In QM9 from PyG, data.z contains atomic numbers.
    Heavy atoms are those with Z > 1 (not hydrogen).
    """
    return int((data.z > 1).sum().item())


def load_and_filter_qm9(
    data_dir: str = "data/qm9",
    max_heavy_atoms: int = 8,
) -> Tuple:
    """Load QM9 and filter to molecules with <= max_heavy_atoms heavy atoms.

    Args:
        data_dir: Directory to download/cache QM9.
        max_heavy_atoms: Maximum number of heavy (non-H) atoms.

    Returns:
        Tuple of (dataset, filtered_indices, all_targets).
    """
    logger.info(f"Loading QM9 from {data_dir}...")
    dataset = QM9(root=data_dir)
    logger.info(f"Full QM9 size: {len(dataset)}")

    # Filter by heavy atom count
    filtered_indices = []
    for i in range(len(dataset)):
        data = dataset[i]
        n_heavy = count_heavy_atoms(data)
        if n_heavy <= max_heavy_atoms:
            filtered_indices.append(i)

    logger.info(
        f"Molecules with <= {max_heavy_atoms} heavy atoms: {len(filtered_indices)} "
        f"out of {len(dataset)}"
    )
    return dataset, filtered_indices


def create_split(
    n_samples: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create a random train/val/test split.

    Args:
        n_samples: Total number of samples.
        train_frac, val_frac, test_frac: Split fractions (must sum to 1).
        seed: Random seed for the permutation.

    Returns:
        Dictionary with 'train', 'val', 'test' index arrays.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, \
        f"Split fractions must sum to 1, got {train_frac + val_frac + test_frac}"

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)

    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    # Rest goes to test (handles rounding)
    n_test = n_samples - n_train - n_val

    split = {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train:n_train + n_val]),
        "test": np.sort(perm[n_train + n_val:]),
    }

    # Verify no overlap
    all_indices = np.concatenate([split["train"], split["val"], split["test"]])
    assert len(all_indices) == n_samples
    assert len(np.unique(all_indices)) == n_samples, "Split has overlapping indices!"

    return split


def compute_target_statistics(
    dataset,
    indices: np.ndarray,
    target_idx: int = GAP_TARGET_IDX,
) -> Dict[str, float]:
    """Compute mean and std of target values for given indices.

    IMPORTANT: Must be called with TRAINING indices only to prevent data leakage.

    Args:
        dataset: PyG QM9 dataset.
        indices: Array of dataset indices (should be training set).
        target_idx: Which target column to use.

    Returns:
        Dictionary with 'mean' and 'std'.
    """
    targets = []
    for idx in indices:
        data = dataset[int(idx)]
        targets.append(data.y[0, target_idx].item())
    targets = np.array(targets)
    return {
        "mean": float(np.mean(targets)),
        "std": float(np.std(targets)),
        "min": float(np.min(targets)),
        "max": float(np.max(targets)),
        "count": int(len(targets)),
    }


def prepare_qm9_data(
    data_dir: str = "data/qm9",
    output_dir: str = "data/splits",
    max_heavy_atoms: int = 8,
    split_seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Dict:
    """Full data preparation pipeline.

    1. Load and filter QM9.
    2. Create random split.
    3. Compute training-set statistics.
    4. Save everything to disk.

    Args:
        data_dir: QM9 cache directory.
        output_dir: Directory to save split and statistics.
        max_heavy_atoms: Filter threshold.
        split_seed: Seed for random split.
        train_frac, val_frac, test_frac: Split fractions.

    Returns:
        Dictionary with all metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and filter
    dataset, filtered_indices = load_and_filter_qm9(data_dir, max_heavy_atoms)
    filtered_indices = np.array(filtered_indices)

    # Step 2: Create split (indices into filtered_indices array)
    split = create_split(
        n_samples=len(filtered_indices),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=split_seed,
    )

    # Map split indices back to dataset indices
    dataset_split = {
        key: filtered_indices[idx_array]
        for key, idx_array in split.items()
    }

    # Step 3: Compute training-set statistics (TRAIN ONLY)
    train_stats = compute_target_statistics(
        dataset, dataset_split["train"], GAP_TARGET_IDX
    )

    # Also compute stats on val/test for reporting (NOT for standardization)
    val_stats = compute_target_statistics(
        dataset, dataset_split["val"], GAP_TARGET_IDX
    )
    test_stats = compute_target_statistics(
        dataset, dataset_split["test"], GAP_TARGET_IDX
    )

    # Step 4: Save to disk
    # Save split indices (dataset-level indices)
    for key, indices in dataset_split.items():
        np.save(output_path / f"{key}_indices.npy", indices)

    # Save filtered indices
    np.save(output_path / "filtered_indices.npy", filtered_indices)

    # Save metadata
    metadata = {
        "dataset": "QM9",
        "target": "HOMO-LUMO gap (eV)",
        "target_index": GAP_TARGET_IDX,
        "max_heavy_atoms": max_heavy_atoms,
        "split_seed": split_seed,
        "total_qm9_size": len(dataset),
        "filtered_size": len(filtered_indices),
        "train_size": len(dataset_split["train"]),
        "val_size": len(dataset_split["val"]),
        "test_size": len(dataset_split["test"]),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "train_target_stats": train_stats,
        "val_target_stats": val_stats,
        "test_target_stats": test_stats,
        "standardization": {
            "method": "z-score using TRAINING set only",
            "mean": train_stats["mean"],
            "std": train_stats["std"],
        },
    }

    with open(output_path / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Data prepared and saved to {output_path}")
    logger.info(f"  Train: {metadata['train_size']}")
    logger.info(f"  Val:   {metadata['val_size']}")
    logger.info(f"  Test:  {metadata['test_size']}")
    logger.info(f"  Train target mean: {train_stats['mean']:.4f}, std: {train_stats['std']:.4f}")

    return metadata


def load_split(split_dir: str = "data/splits") -> Dict:
    """Load a previously saved split from disk.

    Args:
        split_dir: Directory containing split files.

    Returns:
        Dictionary with split indices, metadata, and standardization info.
    """
    split_path = Path(split_dir)

    with open(split_path / "split_metadata.json", "r") as f:
        metadata = json.load(f)

    result = {
        "metadata": metadata,
        "train_indices": np.load(split_path / "train_indices.npy"),
        "val_indices": np.load(split_path / "val_indices.npy"),
        "test_indices": np.load(split_path / "test_indices.npy"),
        "filtered_indices": np.load(split_path / "filtered_indices.npy"),
    }

    return result


def extract_molecule_features(data, target_idx: int = GAP_TARGET_IDX):
    """Extract features from a single QM9 molecule for the QGNN circuit.

    For the 8-qubit circuit, we need:
    - Atom features (atomic number) for RY encoding
    - Bond information (edge_index) for CNOT-RZ-CNOT blocks
    - Target value (HOMO-LUMO gap)

    Args:
        data: PyG Data object from QM9.
        target_idx: Target column index.

    Returns:
        Dictionary with extracted features.
    """
    return {
        "z": data.z.numpy(),  # Atomic numbers
        "pos": data.pos.numpy() if data.pos is not None else None,  # 3D coordinates
        "edge_index": data.edge_index.numpy(),  # Bond connectivity
        "edge_attr": data.edge_attr.numpy() if data.edge_attr is not None else None,
        "target": data.y[0, target_idx].item(),
        "num_atoms": data.z.shape[0],
        "num_heavy_atoms": int((data.z > 1).sum().item()),
    }
