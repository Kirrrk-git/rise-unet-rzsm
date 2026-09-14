#!/usr/bin/env python3
"""
scripts/07_generate_dataset_splits.py
----------------------------------
Sub-Phase 21K, Step 21K.2: Split Segregation and Protection Manifest Generator.

Partitions the verified 1,154-cycle production case calendar into strictly segregated,
leakage-free datasets:
  1. TRAIN:       2015-01-01 to 2021-12-30 (735 forecast cycles, 63.7%)
  2. VAL:         2022-01-03 to 2023-12-28 (210 forecast cycles, 18.2%)
  3. SEALED_TEST: 2024-01-01 to 2025-12-29 (209 forecast cycles, 18.1%)
                  - 202 USABLE_SEALED cycles
                  - 7 TARGET_OUT_OF_BOUNDS cycles (W4 target extends into Jan 2026)

Enforces:
  - Strict chronological quarantine (Train < Val < Test).
  - Cryptographic integrity (SHA-256 digests per split manifest).
  - Zero date overlap between partitions.
  - Emission of split CSVs and machine-readable split_summary.json.
"""

import hashlib
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CALENDAR_PATH = REPO_ROOT / "manifests" / "production_case_calendar.csv"
SPLITS_DIR = REPO_ROOT / "manifests" / "splits"


def sha256_file(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_splits():
    print("=" * 80)
    print("[PIPELINE] Generating Dataset Split Manifests (Train / Val / Sealed Test)")
    print("=" * 80)

    if not CALENDAR_PATH.exists():
        raise FileNotFoundError(f"Calendar manifest not found: {CALENDAR_PATH}")

    df_cal = pd.read_csv(CALENDAR_PATH)
    total_cases = len(df_cal)
    print(f"Loaded calendar manifest: {total_cases} total cycles from {CALENDAR_PATH.name}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Filter splits
    train_df = df_cal[df_cal["split"] == "TRAIN"].copy()
    val_df = df_cal[df_cal["split"] == "VAL"].copy()
    test_df = df_cal[df_cal["split"] == "SEALED_TEST"].copy()

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    print(f"Split distribution:")
    print(f"  - TRAIN:       {n_train} cycles ({n_train / total_cases * 100:.1f}%)")
    print(f"  - VAL:         {n_val} cycles ({n_val / total_cases * 100:.1f}%)")
    print(f"  - SEALED_TEST: {n_test} cycles ({n_test / total_cases * 100:.1f}%)")

    # Assertions
    assert n_train == 735, f"Expected 735 train cycles, got {n_train}"
    assert n_val == 210, f"Expected 210 val cycles, got {n_val}"
    assert n_test == 209, f"Expected 209 test cycles, got {n_test}"
    assert n_train + n_val + n_test == total_cases, "Sum of splits does not equal total cycles"

    # Leakage assertions
    train_dates = set(train_df["issue_date"])
    val_dates = set(val_df["issue_date"])
    test_dates = set(test_df["issue_date"])

    assert len(train_dates & val_dates) == 0, "Leakage detected between TRAIN and VAL!"
    assert len(train_dates & test_dates) == 0, "Leakage detected between TRAIN and SEALED_TEST!"
    assert len(val_dates & test_dates) == 0, "Leakage detected between VAL and SEALED_TEST!"

    assert train_df["issue_date"].max() < val_df["issue_date"].min(), "Train/Val chronological inversion!"
    assert val_df["issue_date"].max() < test_df["issue_date"].min(), "Val/Test chronological inversion!"

    # 2. Write split CSV files
    train_csv = SPLITS_DIR / "train_cases.csv"
    val_csv = SPLITS_DIR / "val_cases.csv"
    test_csv = SPLITS_DIR / "test_cases_sealed.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"\nWritten split manifests:")
    print(f"  - {train_csv.name} ({train_csv.stat().st_size:,} bytes)")
    print(f"  - {val_csv.name} ({val_csv.stat().st_size:,} bytes)")
    print(f"  - {test_csv.name} ({test_csv.stat().st_size:,} bytes)")

    # 3. Compute SHA-256 hashes
    train_sha = sha256_file(train_csv)
    val_sha = sha256_file(val_csv)
    test_sha = sha256_file(test_csv)
    cal_sha = sha256_file(CALENDAR_PATH)

    # 4. Generate Machine-Readable Summary
    summary = {
        "calendar_source": CALENDAR_PATH.name,
        "calendar_sha256": cal_sha,
        "total_cycles": total_cases,
        "splits": {
            "TRAIN": {
                "manifest_file": train_csv.name,
                "sha256": train_sha,
                "cycle_count": n_train,
                "percentage": round(n_train / total_cases * 100, 2),
                "start_issue_date": str(train_df["issue_date"].min()),
                "end_issue_date": str(train_df["issue_date"].max()),
                "operational_years": [2015, 2016, 2017, 2018, 2019, 2020, 2021],
                "purpose": "Model A0 weight optimization and gradient updates",
                "normalization_authority": "Sole source of all scaling parameters",
                "usable_ready_count": int((train_df["usable_status"] == "USABLE_READY").sum()),
                "queued_s2s_count": int((train_df["usable_status"] == "QUEUED_S2S_DOWNLOAD").sum()),
            },
            "VAL": {
                "manifest_file": val_csv.name,
                "sha256": val_sha,
                "cycle_count": n_val,
                "percentage": round(n_val / total_cases * 100, 2),
                "start_issue_date": str(val_df["issue_date"].min()),
                "end_issue_date": str(val_df["issue_date"].max()),
                "operational_years": [2022, 2023],
                "purpose": "Hyperparameter tuning, early stopping, and checkpoint minimum-CRPS selection",
                "usable_ready_count": int((val_df["usable_status"] == "USABLE_READY").sum()),
                "queued_s2s_count": int((val_df["usable_status"] == "QUEUED_S2S_DOWNLOAD").sum()),
            },
            "SEALED_TEST": {
                "manifest_file": test_csv.name,
                "sha256": test_sha,
                "cycle_count": n_test,
                "percentage": round(n_test / total_cases * 100, 2),
                "start_issue_date": str(test_df["issue_date"].min()),
                "end_issue_date": str(test_df["issue_date"].max()),
                "operational_years": [2024, 2025],
                "purpose": "Final thesis evaluation; strictly locked and sealed until Phase 26",
                "queued_s2s_count": int((test_df["usable_status"] == "QUEUED_S2S_DOWNLOAD").sum()),
                "target_out_of_bounds_count": int((test_df["usable_status"] == "TARGET_OUT_OF_BOUNDS").sum()),
                "quarantine_policy": "STRICT_SEALED_NO_PEEKING",
            },
        },
        "non_leakage_guarantees": {
            "train_val_overlap_count": len(train_dates & val_dates),
            "train_test_overlap_count": len(train_dates & test_dates),
            "val_test_overlap_count": len(val_dates & test_dates),
            "chronological_ordering_verified": True,
            "temporal_gap_days": {
                "train_end_to_val_start": (pd.Timestamp(val_df["issue_date"].min()) - pd.Timestamp(train_df["issue_date"].max())).days,
                "val_end_to_test_start": (pd.Timestamp(test_df["issue_date"].min()) - pd.Timestamp(val_df["issue_date"].max())).days,
            },
        },
    }

    summary_json = SPLITS_DIR / "split_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved split summary: {summary_json.name}")
    print(f"  - TRAIN SHA-256:       {train_sha}")
    print(f"  - VAL SHA-256:         {val_sha}")
    print(f"  - SEALED_TEST SHA-256: {test_sha}")
    print("=" * 80)
    print("STEP 21K.2 SPLIT GENERATION: SUCCESS")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    generate_splits()
