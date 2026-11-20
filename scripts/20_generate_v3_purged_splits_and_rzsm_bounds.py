#!/usr/bin/env python3
"""
scripts/20_generate_v3_purged_splits_and_rzsm_bounds.py
------------------------------------------------------
Phase 21R Non-Destructive Offline Execution Script:
  1. Generates Thesis v3.0 chronological 80/10/10 purged split manifests (`851 / 100 / 108`)
     and 9-year OOF fold plans in `manifests/splits_v3/` without altering legacy manifests.
  2. Fits training-only (`time <= 2023-10-20`) RZSM seasonal climatology (DJF/MAM/JJA/SON),
     active-cell scalar min-max bounds, and strict-less-than 20th-percentile drought thresholds
     tau(s, k) directly from local `era5_land_rzsm_production_2015_2025.nc`.
  3. Quarantines Validation (> 2023-10-20) and Sealed Final Test completely untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.cloud_lake import resolve_rzsm_target_cube, resolve_spatial_mask
from src.data.splits_v3 import fit_training_rzsm_v3_contract, write_v3_split_manifests


def main() -> None:
    audit_csv = REPO_ROOT / "manifests" / "splits" / "cases_availability_audit.csv"
    out_splits_dir = REPO_ROOT / "manifests" / "splits_v3"
    out_yaml = REPO_ROOT / "contracts" / "A0" / "rzsm_normalization_v3_purged.yaml"
    out_json = REPO_ROOT / "contracts" / "A0" / "rzsm_normalization_v3_purged.json"
    out_clim_nc = REPO_ROOT / "processed" / "rzsm" / "v3_purged_train_climatology_and_thresholds.nc"

    print("=" * 80)
    print("[PHASE 21R] Generating Thesis v3.0 Purged 80/10/10 Splits & Train-Only RZSM Bounds")
    print("=" * 80)

    train_df, val_df, test_df, summary = write_v3_split_manifests(audit_csv, out_splits_dir)
    pm = summary["partition_metadata"]
    oof = summary["oof_cross_fitting_plan"]

    print(f"Usable issues before purging: {pm['usable_total']}")
    print(f"Unpurged split counts       : {pm['unpurged_counts']}")
    print(f"Purged boundary removals    : {pm['purged_removed']}")
    print(f"Retained counts (851/100/108): {pm['retained_counts']}")
    print(f"Max W4 end dates            : {pm['max_w4_dates']}")
    print(
        f"9-year OOF prefix plan      : naive={oof['naive_oof_stage_fits']} fits -> "
        f"deduplicated={oof['deduplicated_unique_stage_fits']} unique fits per seed"
    )

    rzsm_path = resolve_rzsm_target_cube()
    mask_path = resolve_spatial_mask()
    contract = fit_training_rzsm_v3_contract(
        rzsm_cube_path=rzsm_path,
        mask_path=mask_path,
        train_df=train_df,
        out_yaml_path=out_yaml,
        out_json_path=out_json,
        out_clim_nc_path=out_clim_nc,
    )

    anom_p = contract["rzsm_parameters"]["seasonal_anomaly"]
    print(
        f"Train RZSM anomaly (<= {contract['training_partition_scope']['train_w4_window_end_inclusive']}): "
        f"min={anom_p['min']:+.6f}, max={anom_p['max']:+.6f}, std={anom_p['std']:.6f}"
    )
    print(f"Dry thresholds tau(s, k) written to: {out_clim_nc.relative_to(REPO_ROOT)}")
    print(f"Contract written to: {out_yaml.relative_to(REPO_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
