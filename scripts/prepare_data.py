"""Script to prepare QM9 data: download, filter, split, and save.

Usage:
    python scripts/prepare_data.py
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qgnn.data import prepare_qm9_data
from src.qgnn.seed_utils import set_global_seed, create_run_manifest, generate_run_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    set_global_seed(42)

    run_id = generate_run_id("data_prep")
    logger.info(f"Run ID: {run_id}")

    metadata = prepare_qm9_data(
        data_dir="data/qm9",
        output_dir="data/splits",
        max_heavy_atoms=8,
        split_seed=42,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
    )

    # Create run manifest
    manifest = create_run_manifest(
        run_id=run_id,
        config={
            "task": "data_preparation",
            "max_heavy_atoms": 8,
            "split_seed": 42,
            "train_frac": 0.7,
            "val_frac": 0.15,
            "test_frac": 0.15,
        },
        output_dir=Path("results/raw"),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Config hash: {manifest['config_hash']}")
    print(f"\nFull QM9 size: {metadata['total_qm9_size']}")
    print(f"Filtered (<={metadata['max_heavy_atoms']} heavy atoms): {metadata['filtered_size']}")
    print(f"\nSplit (seed={metadata['split_seed']}):")
    print(f"  Train: {metadata['train_size']} ({metadata['train_frac']*100:.0f}%)")
    print(f"  Val:   {metadata['val_size']} ({metadata['val_frac']*100:.0f}%)")
    print(f"  Test:  {metadata['test_size']} ({metadata['test_frac']*100:.0f}%)")
    print(f"\nTarget: {metadata['target']} (index {metadata['target_index']})")
    print(f"\nTraining set target statistics:")
    ts = metadata['train_target_stats']
    print(f"  Mean: {ts['mean']:.6f}")
    print(f"  Std:  {ts['std']:.6f}")
    print(f"  Min:  {ts['min']:.6f}")
    print(f"  Max:  {ts['max']:.6f}")
    print(f"\nVal set target statistics:")
    vs = metadata['val_target_stats']
    print(f"  Mean: {vs['mean']:.6f}")
    print(f"  Std:  {vs['std']:.6f}")
    print(f"\nTest set target statistics:")
    tes = metadata['test_target_stats']
    print(f"  Mean: {tes['mean']:.6f}")
    print(f"  Std:  {tes['std']:.6f}")
    print(f"\nStandardization (TRAINING set only):")
    print(f"  z = (y - {metadata['standardization']['mean']:.6f}) / {metadata['standardization']['std']:.6f}")
    print(f"\nFiles saved to: data/splits/")
    print(f"Manifest saved to: results/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()
