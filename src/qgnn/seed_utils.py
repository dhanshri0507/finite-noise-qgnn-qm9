"""Seed and reproducibility utilities.

Every random operation in the project must go through these utilities
so that full reproducibility is guaranteed across runs.
"""
import os
import random
import hashlib
import json
import platform
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def config_hash(config: Dict[str, Any]) -> str:
    """Compute a SHA-256 hash of a configuration dictionary.

    Args:
        config: Configuration dictionary (must be JSON-serializable).

    Returns:
        First 12 hex characters of the SHA-256 hash.
    """
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def get_system_info() -> Dict[str, str]:
    """Collect system information for run manifests."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def create_run_manifest(
    run_id: str,
    config: Dict[str, Any],
    output_dir: Path,
    git_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and save a run manifest with full provenance information.

    Args:
        run_id: Unique identifier for this run.
        config: Full configuration used for this run.
        output_dir: Directory to save the manifest.
        git_commit: Git commit hash (auto-detected if None).

    Returns:
        The manifest dictionary.
    """
    if git_commit is None:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=str(Path(__file__).parent.parent.parent)
            )
            git_commit = result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            git_commit = "unknown"

    manifest = {
        "run_id": run_id,
        "git_commit": git_commit,
        "config_hash": config_hash(config),
        "config": config,
        "system_info": get_system_info(),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_{run_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def generate_run_id(prefix: str = "run") -> str:
    """Generate a unique run ID with timestamp.

    Args:
        prefix: String prefix for the run ID.

    Returns:
        Run ID string like 'run_20240101_120000_abc123'.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(
        str(random.random()).encode()
    ).hexdigest()[:6]
    return f"{prefix}_{timestamp}_{random_suffix}"
