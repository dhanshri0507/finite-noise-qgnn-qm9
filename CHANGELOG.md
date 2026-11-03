# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-11-02

### Added
- Repository skeleton: src/, tests/, configs/, results/, docs/, reference/
- Seed and reproducibility utilities (`src/qgnn/seed_utils.py`)
- QM9 data loading, filtering (≤8 heavy atoms), random splitting, standardization (`src/qgnn/data.py`)
- Default configuration (`configs/default.yaml`)
- Unit tests for split logic, seed utilities, and data integrity (`tests/test_data.py`)
- Pinned requirements (`requirements.txt`)
- Documentation stubs: NOISE_MODEL.md, DEBUG_LOG.md, CLAIMS.md
