#!/usr/bin/env python3
"""
09_build_pilot_manifest_and_cases.py
---------------------------------
Step 21G: 8-Case Pilot Ladder Manifest Generation & Batch Case Assembly.

Executes:
1. End-to-end data ingestion for 8 consecutive winter/spring 2015 forecast cycles.
2. Zero-tolerance S2S harmonization under allow_step0_fallback=False.
3. Multi-lead tensor hierarchy assembly conforming strictly to parent EX29 contracts.
4. Comprehensive case-to-case quality and masking census across 126 evaluation cells.
5. Serialization to compressed NPZ files under processed/cases/pilot/.
6. Deterministic SHA-256 checksum computation.
7. Emission of:
   - Member-level 88-row manifest: manifests/cases_pilot_v001.csv (8 cases x 11 members)
   - Case-level 8-row summary manifest: manifests/cases_pilot_summary_v001.csv
"""

import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.s2s import parse_ecmwf_s2s_grib_messages, harmonize_s2s_cycle
from src.data.case_builder import assemble_single_a0_case

# Candidate 8 cycles
PILOT_CYCLES = [
    ("CASE_20150115_W01", "2015-01-15"),
    ("CASE_20150122_W02", "2015-01-22"),
    ("CASE_20150129_W03", "2015-01-29"),
    ("CASE_20150205_W04", "2015-02-05"),
    ("CASE_20150212_W05", "2015-02-12"),
    ("CASE_20150219_W06", "2015-02-19"),
    ("CASE_20150226_W07", "2015-02-26"),
    ("CASE_20150304_W08", "2015-03-04"),
]

def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 85)
    print("[PIPELINE] 8-Case Pilot Ladder Manifest & Batch Assembly Pipeline")
    print("=" * 85)

    # 1. Setup paths
    s2s_ladder_dir = Path("scratch/pilot_s2s_ladder")
    if not s2s_ladder_dir.exists():
        s2s_ladder_dir = REPO_ROOT.parent / "scratch" / "pilot_s2s_ladder"

    rzsm_path = REPO_ROOT / "processed" / "rzsm" / "production" / "era5_land_rzsm_production_2015_2025.nc"
    mask_path = REPO_ROOT / "processed" / "grid" / "mindanao_eval_mask_025.nc"
    grid_path = REPO_ROOT / "processed" / "grid" / "mindanao_025deg.nc"
    
    atmos_files = [
        REPO_ROOT / "processed" / "atmospheric" / "pilot" / "era5_atmospheric_pilot_2015_01.nc",
        REPO_ROOT / "processed" / "atmospheric" / "pilot" / "era5_atmospheric_pilot_2015_02.nc",
        REPO_ROOT / "processed" / "atmospheric" / "pilot" / "era5_atmospheric_pilot_2015_03.nc",
    ]
    for af in atmos_files:
        if not af.exists():
            raise FileNotFoundError(f"Missing required atmospheric pilot file: {af}")

    out_cases_dir = REPO_ROOT / "processed" / "cases" / "pilot"
    out_cases_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = REPO_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # 2. Open continuous base datasets
    print("Loading continuous base datasets (RZSM production cube, multi-month atmospheric, evaluation mask)...")
    ds_rzsm = xr.open_dataset(rzsm_path)
    ds_atmos = xr.open_mfdataset(atmos_files, combine="by_coords")
    ds_mask = xr.open_dataset(mask_path)
    eval_mask = ds_mask["evaluation_mask"].values.astype(bool)
    assert np.sum(eval_mask) == 126, "Evaluation mask must have exactly 126 active cells"

    case_summary_records = []
    member_manifest_rows = []

    print("\nStarting batch assembly across all 8 forecast cycles...\n")

    for case_id, issue_date_str in PILOT_CYCLES:
        t0 = pd.Timestamp(issue_date_str)
        print(f"--> Processing {case_id} (Issue Date: {issue_date_str}, {t0.strftime('%A')})...")

        cf_file = s2s_ladder_dir / issue_date_str / f"s2s_cf_{issue_date_str}.grib"
        pf_file = s2s_ladder_dir / issue_date_str / f"s2s_pf_{issue_date_str}.grib"

        if not cf_file.exists() or not pf_file.exists():
            raise FileNotFoundError(f"Missing S2S GRIB files for {case_id}: {cf_file} or {pf_file}")

        # S2S message parsing & verification
        cf_msgs = parse_ecmwf_s2s_grib_messages(cf_file)
        pf_msgs = parse_ecmwf_s2s_grib_messages(pf_file)
        assert len(cf_msgs) == 42, f"Expected 42 CF messages, got {len(cf_msgs)}"
        assert len(pf_msgs) == 420, f"Expected 420 PF messages, got {len(pf_msgs)}"

        # S2S Harmonization under strict production mode
        s2s_ds = harmonize_s2s_cycle(
            cf_path=cf_file,
            pf_path=pf_file,
            eval_mask_path=mask_path,
            allow_step0_fallback=False
        )
        assert not s2s_ds.attrs["contains_step0_fallback"], "Fallback logic triggered!"
        hdate = s2s_ds.attrs.get("hdate", issue_date_str)
        model_ver_date = s2s_ds.attrs.get("model_version_date", "2020-01-16")

        # Assemble Case
        case = assemble_single_a0_case(
            issue_date=issue_date_str,
            rzsm_cube_ds=ds_rzsm,
            atmospheric_ds=ds_atmos,
            s2s_ds=s2s_ds,
            eval_mask=eval_mask
        )

        # ---------------------------------------------------------------------
        # Rigorous Case Assertions
        # ---------------------------------------------------------------------
        # 1. Shape assertions
        assert case.x_w1.shape == (11, 32, 48, 11), f"Invalid x_w1 shape: {case.x_w1.shape}"
        assert case.x_w2_base.shape == (11, 32, 48, 11), f"Invalid x_w2_base shape: {case.x_w2_base.shape}"
        assert case.x_w3_base.shape == (11, 32, 48, 3), f"Invalid x_w3_base shape: {case.x_w3_base.shape}"
        assert case.x_w4_base.shape == (11, 32, 48, 3), f"Invalid x_w4_base shape: {case.x_w4_base.shape}"
        assert case.y_w1.shape == (1, 32, 48, 1), f"Invalid y_w1 shape: {case.y_w1.shape}"
        assert case.y_w2.shape == (1, 32, 48, 1), f"Invalid y_w2 shape: {case.y_w2.shape}"
        assert case.y_w3.shape == (1, 32, 48, 1), f"Invalid y_w3 shape: {case.y_w3.shape}"
        assert case.y_w4.shape == (1, 32, 48, 1), f"Invalid y_w4 shape: {case.y_w4.shape}"

        # 2. Target date offsets verification [6, 13, 20, 27]
        expected_w1 = (t0 + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        expected_w2 = (t0 + pd.Timedelta(days=13)).strftime("%Y-%m-%d")
        expected_w3 = (t0 + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        expected_w4 = (t0 + pd.Timedelta(days=27)).strftime("%Y-%m-%d")
        assert case.target_dates[1] == expected_w1, f"W1 target mismatch: {case.target_dates[1]} != {expected_w1}"
        assert case.target_dates[2] == expected_w2, f"W2 target mismatch: {case.target_dates[2]} != {expected_w2}"
        assert case.target_dates[3] == expected_w3, f"W3 target mismatch: {case.target_dates[3]} != {expected_w3}"
        assert case.target_dates[4] == expected_w4, f"W4 target mismatch: {case.target_dates[4]} != {expected_w4}"

        # 3. Quality & Masking Census over 126 evaluation cells
        for name, arr in [
            ("X_w1", case.x_w1), ("X_w2_base", case.x_w2_base), 
            ("X_w3_base", case.x_w3_base), ("X_w4_base", case.x_w4_base),
            ("Y_w1", case.y_w1), ("Y_w2", case.y_w2), ("Y_w3", case.y_w3), ("Y_w4", case.y_w4)
        ]:
            eval_vals = arr[:, eval_mask]
            ocean_vals = arr[:, ~eval_mask]
            assert np.isnan(eval_vals).sum() == 0, f"{name} contains NaNs over evaluation cells!"
            assert np.isinf(eval_vals).sum() == 0, f"{name} contains Infs over evaluation cells!"
            assert np.max(np.abs(ocean_vals)) == 0.0, f"{name} non-zero values in ocean buffer!"

        lag_1d = (t0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        lag_7d = (t0 - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        lag_14d = (t0 - pd.Timedelta(days=14)).strftime("%Y-%m-%d")

        # 4. Serialize to NPZ
        npz_filename = f"{case_id}.npz"
        npz_path = out_cases_dir / npz_filename
        np.savez_compressed(
            npz_path,
            x_w1=case.x_w1,
            x_w2_base=case.x_w2_base,
            x_w3_base=case.x_w3_base,
            x_w4_base=case.x_w4_base,
            y_w1=case.y_w1,
            y_w2=case.y_w2,
            y_w3=case.y_w3,
            y_w4=case.y_w4,
            case_id=case_id,
            issue_date=issue_date_str,
            hdate=hdate,
            model_version_date=model_ver_date,
            target_dates=np.array([expected_w1, expected_w2, expected_w3, expected_w4]),
            antecedent_lags=np.array([lag_1d, lag_7d, lag_14d])
        )

        npz_checksum = sha256_file(npz_path)
        npz_size_kb = npz_path.stat().st_size / 1024

        print(f"    [OK] Saved {npz_filename} ({npz_size_kb:.1f} KB, SHA-256: {npz_checksum[:16]}...)")
        print(f"         Targets: W1={expected_w1}, W2={expected_w2}, W3={expected_w3}, W4={expected_w4}")

        # Record Case-Level Summary
        case_summary_records.append({
            "case_id": case_id,
            "issue_time": issue_date_str,
            "s2s_hdate": hdate,
            "model_version_date": model_ver_date,
            "ensemble_members_count": 11,
            "split": "TRAIN",
            "cf_source_file": str(cf_file.relative_to(REPO_ROOT.parent) if REPO_ROOT.parent in cf_file.parents else cf_file.name),
            "pf_source_file": str(pf_file.relative_to(REPO_ROOT.parent) if REPO_ROOT.parent in pf_file.parents else pf_file.name),
            "npz_artifact_path": f"processed/cases/pilot/{npz_filename}",
            "artifact_size_bytes": npz_path.stat().st_size,
            "sha256_checksum": npz_checksum,
            "lag_1d_date": lag_1d,
            "lag_7d_date": lag_7d,
            "lag_14d_date": lag_14d,
            "target_w1_date": expected_w1,
            "target_w2_date": expected_w2,
            "target_w3_date": expected_w3,
            "target_w4_date": expected_w4,
            "active_eval_cells": 126,
            "nan_count": 0,
            "status": "VALID"
        })

        # Record Member-Level Rows (11 rows per case)
        gcs_cf_uri = f"gs://rise-unet-rzsm/raw/ecmwf_s2s/production/{issue_date_str}/s2s_cf_{issue_date_str}.grib"
        gcs_pf_uri = f"gs://rise-unet-rzsm/raw/ecmwf_s2s/production/{issue_date_str}/s2s_pf_{issue_date_str}.grib"
        
        for mem_idx in range(11):
            mem_type = "CF" if mem_idx == 0 else "PF"
            source_grib = gcs_cf_uri if mem_idx == 0 else gcs_pf_uri
            member_manifest_rows.append({
                "case_id": case_id,
                "ensemble_member": mem_idx,
                "member_type": mem_type,
                "issue_time": issue_date_str,
                "s2s_hdate": hdate,
                "model_version_date": model_ver_date,
                "split": "TRAIN",
                "source_grib_uri": source_grib,
                "rzsm_source_cube": "processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc",
                "lag_1d_date": lag_1d,
                "lag_7d_date": lag_7d,
                "lag_14d_date": lag_14d,
                "target_w1_date": expected_w1,
                "target_w2_date": expected_w2,
                "target_w3_date": expected_w3,
                "target_w4_date": expected_w4,
                "preprocessing_version": "v1.0.0",
                "grid_version": "Candidate_A_0.25deg_32x48",
                "A0_contract_version": "EX29_A0_11_12_5_6",
                "status": "VALID",
                "case_tensor_checksum": npz_checksum
            })

    # 3. Export Manifests
    df_summary = pd.DataFrame(case_summary_records)
    df_members = pd.DataFrame(member_manifest_rows)

    summary_csv = manifest_dir / "cases_pilot_summary_v001.csv"
    members_csv = manifest_dir / "cases_pilot_v001.csv"

    df_summary.to_csv(summary_csv, index=False)
    df_members.to_csv(members_csv, index=False)

    print("\n" + "=" * 85)
    print("[PIPELINE] Pilot Ladder Manifest Generation: SUCCESS")
    print("=" * 85)
    print(f"Case-Level Summary Manifest (8 rows)   : {summary_csv}")
    print(f"Member-Level Manifest (88 rows)         : {members_csv}")
    print(f"All 8 Case Tensors Serialized in        : {out_cases_dir}")
    print("Zero NaNs, Zero Infs, Zero Missing Members, Strict [6, 13, 20, 27] Lead Targets.")
    print("=" * 85)

if __name__ == "__main__":
    main()
