# Tier 1 — Steps 1 & 2: Repo Skeleton + Data

## Status: ✅ COMPLETE

---

## Repository Tree

```
d:\Projects\Quantum\
├── configs/
│   └── default.yaml              # Full config (all result-affecting params)
├── data/
│   ├── qm9/                      # Downloaded QM9 dataset (130,831 molecules)
│   └── splits/
│       ├── filtered_indices.npy   # 21,800 indices (≤8 heavy atoms)
│       ├── train_indices.npy      # 15,259 training indices
│       ├── val_indices.npy        # 3,270 validation indices
│       ├── test_indices.npy       # 3,271 test indices
│       └── split_metadata.json    # Full metadata + statistics
├── docs/
│   ├── CLAIMS.md                  # Claims ledger (empty, populated during experiments)
│   ├── DEBUG_LOG.md               # Reverse-engineering debug log
│   └── NOISE_MODEL.md            # Canonical noise definition (draft)
├── paper_inputs/                  # Will contain NUMBERS.md, tables, figures
├── reference/                     # Will contain read-only clone of original repo
├── results/
│   ├── figures/
│   └── raw/
│       └── manifest_data_prep_20261102_184646_6bbeff.json
├── scripts/
│   └── prepare_data.py            # Data preparation script
├── src/
│   ├── __init__.py
│   └── qgnn/
│       ├── __init__.py
│       ├── data.py                # Data loading, filtering, splitting
│       └── seed_utils.py          # Seeds, config hashing, run manifests
├── tests/
│   ├── __init__.py
│   └── test_data.py               # 15 tests (split, seed, leakage, QM9)
├── CHANGELOG.md
├── conftest.py                    # pytest config with --runslow marker
├── README.md
└── requirements.txt               # Pinned dependencies
```

---

## Split Statistics

| Metric | Value |
|--------|-------|
| **Full QM9** | 130,831 molecules |
| **Filtered (≤8 heavy atoms)** | 21,800 molecules (16.7%) |
| **Split seed** | 42 |
| **Selection method** | RANDOM permutation (not file order) |

### Split Sizes

| Split | Count | Fraction |
|-------|-------|----------|
| Train | 15,259 | 70% |
| Val | 3,270 | 15% |
| Test | 3,271 | 15% |

### HOMO-LUMO Gap Target Statistics (eV)

| Statistic | Train | Val | Test |
|-----------|-------|-----|------|
| Mean | 7.0670 | 7.0386 | 7.0742 |
| Std | 1.3164 | 1.3241 | 1.2932 |
| Min | 2.6749 | 2.0844 | 3.0232 |
| Max | 16.9282 | 11.6955 | 11.7172 |

> [!IMPORTANT]
> Standardization uses **training-set statistics only**: z = (y − 7.0670) / 1.3164.
> This is enforced by design and verified by `test_standardization_uses_train_only`.

---

## Test Results

**15/15 passed** (45.11s)

```
tests/test_data.py::TestCreateSplit::test_split_sizes PASSED
tests/test_data.py::TestCreateSplit::test_no_overlap PASSED
tests/test_data.py::TestCreateSplit::test_complete_coverage PASSED
tests/test_data.py::TestCreateSplit::test_reproducibility PASSED
tests/test_data.py::TestCreateSplit::test_different_seed_different_split PASSED
tests/test_data.py::TestCreateSplit::test_fractions_must_sum_to_one PASSED
tests/test_data.py::TestSeedUtils::test_config_hash_deterministic PASSED
tests/test_data.py::TestSeedUtils::test_config_hash_order_invariant PASSED
tests/test_data.py::TestSeedUtils::test_config_hash_different_configs PASSED
tests/test_data.py::TestSeedUtils::test_system_info PASSED
tests/test_data.py::TestSeedUtils::test_run_manifest_creates_file PASSED
tests/test_data.py::TestSeedUtils::test_set_global_seed_reproducibility PASSED
tests/test_data.py::TestDataIntegrity::test_heavy_atom_count PASSED
tests/test_data.py::TestDataIntegrity::test_standardization_uses_train_only PASSED
tests/test_data.py::TestDataIntegrity::test_gap_target_is_index_4 PASSED
```

### What the tests verify:
- **No overlap** between train/val/test
- **Complete coverage** (all filtered indices appear exactly once)
- **Reproducibility** (same seed → identical split)
- **Different seeds** produce different splits
- **Invalid fractions** are rejected
- **Config hashing** is deterministic and order-invariant
- **System info** collection works
- **Run manifests** are saved with provenance
- **Seed setting** produces reproducible random numbers
- **Heavy atom counting** is correct
- **Standardization** uses only the provided indices (data leakage guard)
- **Target index 4** corresponds to a positive, physically reasonable HOMO-LUMO gap

---

## Provenance

| Field | Value |
|-------|-------|
| Run ID | `data_prep_20261102_184646_6bbeff` |
| Config hash | `4e175ac96fd8` |
| Manifest | [manifest](file:///d:/Projects/Quantum/results/raw/manifest_data_prep_20261102_184646_6bbeff.json) |
| Split metadata | [split_metadata.json](file:///d:/Projects/Quantum/data/splits/split_metadata.json) |

---

## Ready for Next Step?

Tier 1 Steps 1–2 are complete. The next step is **Step 3**: Fast batched PyTorch density-matrix simulator for the 8-qubit circuit. This requires:
1. Cloning the reference repo into `reference/`
2. Implementing the density-matrix simulator
3. Validating against PennyLane `default.mixed` and Qiskit Aer
