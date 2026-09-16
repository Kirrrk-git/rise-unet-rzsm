#!/usr/bin/env python3
"""
scripts/17_build_production_cases.py
------------------------------------
Authoritative Production Case Assembly Pipeline for Mindanao RISE-UNet (Step 21K).

Executes:
1. Multi-source data ingestion:
   - ERA5-Land RZSM Production Cube (2015-2025 daily continuous)
   - ERA5 Daily Surface Atmospheric Drivers (5-channel)
   - ECMWF S2S Harmonized Reforecast/Forecast Cycles (11-member, Leads 1 & 2)
   - Authoritative 126-cell Mindanao binary evaluation mask
2. Production tensor hierarchy assembly conforming strictly to Kyle Lesinger's EX29 contracts:
   - x_w1: (11, 32, 48, 11) -> 3 RZSM lags + 5 ERA5 atm + 3 S2S W1
   - x_w2_base: (11, 32, 48, 11) -> 3 RZSM lags + 5 ERA5 atm + 3 S2S W2
   - x_w3_base: (11, 32, 48, 3) -> 3 RZSM lags
   - x_w4_base: (11, 32, 48, 3) -> 3 RZSM lags
   - y_w1, y_w2, y_w3, y_w4: (1, 32, 48, 1) -> Ground truth at +6d, +13d, +20d, +27d
3. Rigorous domain-wide normalization scaling into [0, 1] via contracts/A0/normalization_parameters.yaml
   with ocean buffer zero-filling invariant (~eval_mask -> 0.0).
4. Full quality census: Zero NaNs/Infs over all 126 evaluation cells.
5. Deterministic NPZ serialization with SHA-256 checksums to processed/cases/production/.
6. Optional automatic synchronization to Google Cloud Storage (gs://rise-unet-rzsm/processed/cases/production/).
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import xarray as xr

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.case_builder import (
    assemble_single_a0_case,
    load_frozen_normalization_contract,
    CaseTensorHierarchy,
)
from src.data.s2s import harmonize_s2s_cycle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("17_build_production_cases")

GCS_BUCKET = "gs://rise-unet-rzsm"


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_file_local_or_gcs(
    local_path: Path,
    gcs_uri: str,
    desc: str = "file",
) -> bool:
    """Ensures a file exists locally, downloading from GCS if absent."""
    if local_path.is_file() and local_path.stat().st_size > 0:
        return True

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {desc} from {gcs_uri} -> {local_path}...")

    # Try gcloud storage cp first, then gsutil cp
    try:
        res = subprocess.run(
            ["gcloud", "storage", "cp", gcs_uri, str(local_path)],
            capture_output=True, text=True, check=False,
        )
        if res.returncode == 0 and local_path.is_file() and local_path.stat().st_size > 0:
            return True
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["gsutil", "cp", gcs_uri, str(local_path)],
            capture_output=True, text=True, check=False,
        )
        if res.returncode == 0 and local_path.is_file() and local_path.stat().st_size > 0:
            return True
    except Exception:
        pass

    return local_path.is_file() and local_path.stat().st_size > 0


def upload_file_to_gcs(local_path: Path, gcs_uri: str) -> bool:
    """Uploads a local file to GCS."""
    try:
        res = subprocess.run(
            ["gcloud", "storage", "cp", str(local_path), gcs_uri],
            capture_output=True, text=True, check=False,
        )
        if res.returncode == 0:
            return True
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["gsutil", "cp", str(local_path), gcs_uri],
            capture_output=True, text=True, check=False,
        )
        if res.returncode == 0:
            return True
    except Exception:
        pass

    return False


class AtmosphericDatasetProvider:
    """Manages lazy-loading, caching, and GCS retrieval of monthly ERA5 atmospheric NetCDFs."""

    def __init__(self, repo_root: Path, auto_download: bool = True):
        self.repo_root = repo_root
        self.auto_download = auto_download
        self.cache: Dict[Tuple[int, int], xr.Dataset] = {}
        self.atmos_dir = repo_root / "processed" / "atmospheric"
        self.pilot_dir = self.atmos_dir / "pilot"

    def get_dataset_for_date(self, dt: pd.Timestamp) -> Optional[xr.Dataset]:
        """Returns open xarray Dataset containing atmospheric variables for given date."""
        key = (dt.year, dt.month)
        if key in self.cache:
            return self.cache[key]

        month_str = f"{dt.year}_{dt.month:02d}"
        candidate_paths = [
            self.atmos_dir / f"era5_atmospheric_{month_str}.nc",
            self.pilot_dir / f"era5_atmospheric_pilot_{month_str}.nc",
        ]

        target_path = None
        for p in candidate_paths:
            if p.is_file() and p.stat().st_size > 0:
                target_path = p
                break

        if target_path is None and self.auto_download:
            # Try to download standard monthly file from GCS
            dest = self.atmos_dir / f"era5_atmospheric_{month_str}.nc"
            gcs_uri = f"{GCS_BUCKET}/processed/atmospheric/era5_atmospheric_{month_str}.nc"
            if ensure_file_local_or_gcs(dest, gcs_uri, desc=f"ERA5 atmospheric {month_str}"):
                target_path = dest

        if target_path is None:
            logger.warning(f"Atmospheric NetCDF not found for {month_str}")
            return None

        ds = xr.open_dataset(target_path)
        self.cache[key] = ds
        return ds


def locate_or_fetch_s2s_gribs(
    issue_date: str,
    repo_root: Path,
    auto_download: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locates CF and PF GRIB files for an issue date, downloading from GCS if needed.
    """
    cf_name = f"s2s_cf_{issue_date}.grib"
    pf_name = f"s2s_pf_{issue_date}.grib"

    candidate_dirs = [
        repo_root / "Data" / "raw_downloads" / "ECMWF_S2S" / "production" / issue_date,
        repo_root / "raw" / "ecmwf_s2s" / "production" / issue_date,
        repo_root.parent / "scratch" / "pilot_s2s_ladder" / issue_date,
        repo_root / "scratch" / "pilot_s2s_ladder" / issue_date,
        Path("/content/s2s_cache") / issue_date,
    ]

    cf_path, pf_path = None, None
    for cdir in candidate_dirs:
        c_cand = cdir / cf_name
        p_cand = cdir / pf_name
        if c_cand.is_file() and p_cand.is_file() and c_cand.stat().st_size > 0 and p_cand.stat().st_size > 0:
            cf_path, pf_path = c_cand, p_cand
            break

    if (cf_path is None or pf_path is None) and auto_download:
        dest_dir = repo_root / "raw" / "ecmwf_s2s" / "production" / issue_date
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_cf = dest_dir / cf_name
        dest_pf = dest_dir / pf_name

        gcs_cf = f"{GCS_BUCKET}/raw/ecmwf_s2s/production/{issue_date}/{cf_name}"
        gcs_pf = f"{GCS_BUCKET}/raw/ecmwf_s2s/production/{issue_date}/{pf_name}"

        ok_cf = ensure_file_local_or_gcs(dest_cf, gcs_cf, desc=f"S2S CF {issue_date}")
        ok_pf = ensure_file_local_or_gcs(dest_pf, gcs_pf, desc=f"S2S PF {issue_date}")

        if ok_cf and ok_pf:
            cf_path, pf_path = dest_cf, dest_pf

    return cf_path, pf_path


def verify_assembled_case(
    case: CaseTensorHierarchy,
    eval_mask: np.ndarray,
    expected_issue_date: str,
) -> None:
    """Strictly validates shapes, normalization bounds, target offsets, and evaluation masking."""
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
    t0 = pd.Timestamp(expected_issue_date)
    expected_w1 = (t0 + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    expected_w2 = (t0 + pd.Timedelta(days=13)).strftime("%Y-%m-%d")
    expected_w3 = (t0 + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    expected_w4 = (t0 + pd.Timedelta(days=27)).strftime("%Y-%m-%d")
    assert case.target_dates[1] == expected_w1, f"W1 target mismatch: {case.target_dates[1]} != {expected_w1}"
    assert case.target_dates[2] == expected_w2, f"W2 target mismatch: {case.target_dates[2]} != {expected_w2}"
    assert case.target_dates[3] == expected_w3, f"W3 target mismatch: {case.target_dates[3]} != {expected_w3}"
    assert case.target_dates[4] == expected_w4, f"W4 target mismatch: {case.target_dates[4]} != {expected_w4}"

    # 3. Quality, Normalization & Masking Census over 126 evaluation cells
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
        if case.is_normalized:
            assert np.min(eval_vals) >= -1e-6, f"{name} normalized values < 0.0: {np.min(eval_vals)}"
            assert np.max(eval_vals) <= 1.0 + 1e-6, f"{name} normalized values > 1.0: {np.max(eval_vals)}"


def build_single_case(
    case_id: str,
    issue_date: str,
    ds_rzsm: xr.Dataset,
    atmos_provider: AtmosphericDatasetProvider,
    eval_mask: np.ndarray,
    mask_path: Path,
    norm_params: Dict[str, Any],
    out_dir: Path,
    repo_root: Path,
    target_var_name: str = "rzsm_0_100_seasonal_anomaly",
    auto_download: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Builds, normalizes, validates, and serializes a single forecast case.
    """
    # Primary artifact path
    clean_id = case_id if case_id.startswith("CASE_") else f"CASE_{case_id}"
    out_npz = out_dir / f"{clean_id}.npz"

    if out_npz.is_file() and not force:
        try:
            with np.load(out_npz) as d:
                if "x_w1" in d and "y_w1" in d:
                    return {
                        "case_id": clean_id,
                        "issue_date": issue_date,
                        "status": "EXISTING_VALID",
                        "path": str(out_npz),
                        "sha256": compute_sha256(out_npz),
                        "bytes": out_npz.stat().st_size,
                    }
        except Exception:
            pass

    t0 = pd.Timestamp(issue_date)

    # 1. Locate S2S GRIB files
    cf_path, pf_path = locate_or_fetch_s2s_gribs(issue_date, repo_root, auto_download=auto_download)
    if cf_path is None or pf_path is None:
        return {
            "case_id": clean_id,
            "issue_date": issue_date,
            "status": "SKIPPED_NO_S2S",
            "path": None,
            "sha256": None,
            "bytes": 0,
        }

    # 2. Get Atmospheric Dataset covering t0
    ds_atmos = atmos_provider.get_dataset_for_date(t0)
    if ds_atmos is None:
        return {
            "case_id": clean_id,
            "issue_date": issue_date,
            "status": "SKIPPED_NO_ATMOS",
            "path": None,
            "sha256": None,
            "bytes": 0,
        }

    # 3. Harmonize S2S Cycle
    s2s_ds = harmonize_s2s_cycle(
        cf_path=cf_path,
        pf_path=pf_path,
        eval_mask_path=mask_path,
        allow_step0_fallback=False,
    )
    assert not s2s_ds.attrs["contains_step0_fallback"], f"Step0 fallback triggered on {issue_date}!"
    hdate = str(s2s_ds.attrs.get("hdate", issue_date))
    model_ver_date = str(s2s_ds.attrs.get("model_version_date", "unknown"))

    # 4. Assemble and Normalize Case
    case = assemble_single_a0_case(
        issue_date=issue_date,
        rzsm_cube_ds=ds_rzsm,
        atmospheric_ds=ds_atmos,
        s2s_ds=s2s_ds,
        target_var_name=target_var_name,
        eval_mask=eval_mask,
        normalize=True,
        norm_params=norm_params,
    )

    # 5. Verify Case Invariants
    verify_assembled_case(case, eval_mask, issue_date)

    # 6. Save compressed NPZ
    lag_1d = (t0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lag_7d = (t0 - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    lag_14d = (t0 - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    tgt_dates = np.array([case.target_dates[i] for i in (1, 2, 3, 4)])

    np.savez_compressed(
        out_npz,
        x_w1=case.x_w1,
        x_w2_base=case.x_w2_base,
        x_w3_base=case.x_w3_base,
        x_w4_base=case.x_w4_base,
        y_w1=case.y_w1,
        y_w2=case.y_w2,
        y_w3=case.y_w3,
        y_w4=case.y_w4,
        case_id=clean_id,
        issue_date=issue_date,
        hdate=hdate,
        model_version_date=model_ver_date,
        target_dates=tgt_dates,
        antecedent_lags=np.array([lag_1d, lag_7d, lag_14d]),
        is_normalized=case.is_normalized,
        normalization_contract_used=case.normalization_contract_used,
    )

    checksum = compute_sha256(out_npz)
    size_bytes = out_npz.stat().st_size

    return {
        "case_id": clean_id,
        "issue_date": issue_date,
        "status": "ASSEMBLED_OK",
        "path": str(out_npz),
        "sha256": checksum,
        "bytes": size_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description="Mindanao RISE-UNet: Step 21K Production Case Builder")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"],
                        help="Dataset splits to assemble (default: train val).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of cases to assemble per split.")
    parser.add_argument("--out-dir", type=str, default="processed/cases/production",
                        help="Output directory for assembled case NPZs.")
    parser.add_argument("--target-var", type=str, default="rzsm_0_100_seasonal_anomaly",
                        help="Target RZSM variable (default: rzsm_0_100_seasonal_anomaly).")
    parser.add_argument("--gcs-sync", action="store_true",
                        help="Automatically synchronize assembled cases to GCS lake.")
    parser.add_argument("--no-auto-download", action="store_true",
                        help="Disable automatic on-demand GCS downloads for raw files.")
    parser.add_argument("--force", action="store_true",
                        help="Force reassembly of existing NPZ files.")
    args = parser.parse_args()

    print("=" * 85)
    print("MINDANAO RISE-UNET: STEP 21K PRODUCTION CASE ASSEMBLY ENGINE")
    print("=" * 85)
    print(f"Splits Target     : {args.splits}")
    print(f"Target Variable   : {args.target_var}")
    print(f"Output Directory  : {args.out_dir}")
    print(f"GCS Synchronization: {args.gcs_sync}")
    print(f"Limit per Split   : {args.limit or 'All Available'}")
    print("=" * 85)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base Paths & Verification
    mask_path = REPO_ROOT / "processed" / "grid" / "mindanao_eval_mask_025.nc"
    ensure_file_local_or_gcs(mask_path, f"{GCS_BUCKET}/processed/grid/mindanao_eval_mask_025.nc", desc="evaluation mask")
    ds_mask = xr.open_dataset(mask_path)
    eval_mask = ds_mask["evaluation_mask"].values.astype(bool)
    assert np.sum(eval_mask) == 126, f"Evaluation mask must have 126 active cells, got {np.sum(eval_mask)}"

    norm_path = REPO_ROOT / "contracts" / "A0" / "normalization_parameters.yaml"
    ensure_file_local_or_gcs(norm_path, f"{GCS_BUCKET}/contracts/A0/normalization_parameters.yaml", desc="normalization contract")
    norm_params = load_frozen_normalization_contract(norm_path)

    rzsm_path = REPO_ROOT / "processed" / "rzsm" / "production" / "era5_land_rzsm_production_2015_2025.nc"
    ensure_file_local_or_gcs(rzsm_path, f"{GCS_BUCKET}/processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc", desc="RZSM production cube")
    logger.info("Opening RZSM production continuous cube...")
    ds_rzsm = xr.open_dataset(rzsm_path)

    atmos_provider = AtmosphericDatasetProvider(REPO_ROOT, auto_download=not args.no_auto_download)

    # 2. Iterate over designated splits
    manifest_map = {
        "train": REPO_ROOT / "manifests" / "splits" / "train_cases.csv",
        "val": REPO_ROOT / "manifests" / "splits" / "val_cases.csv",
        "test": REPO_ROOT / "manifests" / "splits" / "test_cases_sealed.csv",
    }

    all_results = []
    total_start_time = time.time()

    for split_name in args.splits:
        mpath = manifest_map[split_name]
        ensure_file_local_or_gcs(mpath, f"{GCS_BUCKET}/manifests/splits/{mpath.name}", desc=f"{split_name} manifest")
        df_split = pd.read_csv(mpath)
        logger.info(f"\n[{split_name.upper()}] Processing split with {len(df_split)} cataloged cycles...")

        if args.limit is not None and args.limit > 0:
            df_split = df_split.iloc[:args.limit]
            logger.info(f"[{split_name.upper()}] Limited to first {len(df_split)} cycles.")

        assembled_count = 0
        existing_count = 0
        skipped_count = 0

        for idx, row in df_split.iterrows():
            cid = str(row["case_id"])
            issue_date = str(row["issue_date"])

            t_case_start = time.time()
            res = build_single_case(
                case_id=cid,
                issue_date=issue_date,
                ds_rzsm=ds_rzsm,
                atmos_provider=atmos_provider,
                eval_mask=eval_mask,
                mask_path=mask_path,
                norm_params=norm_params,
                out_dir=out_dir,
                repo_root=REPO_ROOT,
                target_var_name=args.target_var,
                auto_download=not args.no_auto_download,
                force=args.force,
            )
            res["split"] = split_name
            all_results.append(res)

            elapsed = time.time() - t_case_start
            st = res["status"]
            if st == "ASSEMBLED_OK":
                assembled_count += 1
                logger.info(f"  [{idx+1}/{len(df_split)}] {cid} ({issue_date}) -> [ASSEMBLED] ({res['bytes']/1024:.1f} KB, {elapsed:.2f}s)")
                if args.gcs_sync and res["path"]:
                    upload_file_to_gcs(Path(res["path"]), f"{GCS_BUCKET}/processed/cases/production/{Path(res['path']).name}")
            elif st == "EXISTING_VALID":
                existing_count += 1
                if idx % 25 == 0 or idx == len(df_split) - 1:
                    logger.info(f"  [{idx+1}/{len(df_split)}] {cid} ({issue_date}) -> [EXISTING_OK]")
            else:
                skipped_count += 1
                logger.warning(f"  [{idx+1}/{len(df_split)}] {cid} ({issue_date}) -> [{st}]")

        logger.info(f"[{split_name.upper()} COMPLETE] Assembled: {assembled_count} | Existing: {existing_count} | Skipped: {skipped_count}")

    total_time = time.time() - total_start_time

    # 3. Emit Production Summary Manifest
    summary_csv = REPO_ROOT / "manifests" / "cases_production_summary.csv"
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\n[MANIFEST] Saved summary manifest to {summary_csv}")
    if args.gcs_sync:
        upload_file_to_gcs(summary_csv, f"{GCS_BUCKET}/manifests/cases_production_summary.csv")

    # 4. Final Census Table
    print("\n" + "=" * 85)
    print("STEP 21K PRODUCTION CASE ASSEMBLY SUMMARY")
    print("=" * 85)
    for sname in args.splits:
        sub = summary_df[summary_df["split"] == sname]
        ok_count = len(sub[sub["status"].isin(["ASSEMBLED_OK", "EXISTING_VALID"])])
        skip_count = len(sub[~sub["status"].isin(["ASSEMBLED_OK", "EXISTING_VALID"])])
        print(f"  Split: {sname:<8} | Total Scheduled: {len(sub):<4} | Valid Cases: {ok_count:<4} | Skipped: {skip_count:<4}")
    print(f"Total Pipeline Runtime: {total_time:.1f} seconds")
    print("=" * 85)


if __name__ == "__main__":
    main()
