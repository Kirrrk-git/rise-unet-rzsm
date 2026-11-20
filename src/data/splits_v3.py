"""
src/data/splits_v3.py
---------------------
Revised Thesis v3.0 chronological 80/10/10 dataset partitioning with target-window
boundary purging, 9-year interval-purged nested OOF cross-fitting planner, and
training-only RZSM climatology, normalization bounds, and dry-event threshold fitting.

Quarantine & Leakage Invariants:
  1. Target-window purging removes every Train/Val issue whose 28-day window
     [t0, t0 + 27d] reaches or crosses the earliest issue date of the next split.
  2. Training RZSM climatology, min-max scaling bounds, and 20th-percentile drought
     thresholds tau(s, k) are derived STRICTLY from `time <= max_train_w4_date`
     (2023-10-20), never reading Validation or Sealed Final Test observations.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SEASONS: Tuple[str, ...] = ("DJF", "MAM", "JJA", "SON")


def date_to_season(dates: Union[pd.DatetimeIndex, pd.Series, Sequence[str]]) -> np.ndarray:
    """Maps dates to 3-month meteorological seasons: DJF, MAM, JJA, SON."""
    months = pd.DatetimeIndex(pd.to_datetime(dates)).month.to_numpy()
    out = np.empty(len(months), dtype=object)
    out[np.isin(months, [12, 1, 2])] = "DJF"
    out[np.isin(months, [3, 4, 5])] = "MAM"
    out[np.isin(months, [6, 7, 8])] = "JJA"
    out[np.isin(months, [9, 10, 11])] = "SON"
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_chronological_purged_splits(
    audit_csv_path: Union[str, Path],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    target_horizon_days: int = 27,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Constructs chronological 80/10/10 partitions with target-window boundary purging.

    Returns (train_df, val_df, test_df, metadata_summary).
    """
    df_all = pd.read_csv(audit_csv_path)
    if "status" not in df_all.columns or "issue_date" not in df_all.columns:
        raise KeyError("Availability audit CSV must contain 'issue_date' and 'status' columns.")

    usable = df_all[df_all["status"] == "USABLE_AVAILABLE"].copy()
    usable["_t0"] = pd.to_datetime(usable["issue_date"])
    usable = usable.sort_values("_t0").reset_index(drop=True)
    usable["w4_end_date"] = (usable["_t0"] + pd.Timedelta(days=target_horizon_days)).dt.strftime("%Y-%m-%d")
    usable["w1_midpoint_date"] = (usable["_t0"] + pd.Timedelta(days=3)).dt.strftime("%Y-%m-%d")
    usable["w1_midpoint_season"] = date_to_season(usable["w1_midpoint_date"])

    n_total = len(usable)
    n_train_raw = int(np.floor(train_frac * n_total))
    n_val_raw = int(np.floor((train_frac + val_frac) * n_total)) - n_train_raw

    train_raw = usable.iloc[:n_train_raw].copy()
    val_raw = usable.iloc[n_train_raw : n_train_raw + n_val_raw].copy()
    test_raw = usable.iloc[n_train_raw + n_val_raw :].copy()

    val_min_t0 = val_raw["_t0"].min()
    test_min_t0 = test_raw["_t0"].min()

    train_keep = (train_raw["_t0"] + pd.Timedelta(days=target_horizon_days)) < val_min_t0
    val_keep = (val_raw["_t0"] + pd.Timedelta(days=target_horizon_days)) < test_min_t0

    train_purged_df = train_raw[train_keep].copy().drop(columns=["_t0"]).reset_index(drop=True)
    val_purged_df = val_raw[val_keep].copy().drop(columns=["_t0"]).reset_index(drop=True)
    test_df = test_raw.copy().drop(columns=["_t0"]).reset_index(drop=True)

    train_purged_df["split"] = "TRAIN"
    val_purged_df["split"] = "VAL"
    test_df["split"] = "SEALED_TEST"

    meta = {
        "usable_total": n_total,
        "unpurged_counts": {
            "train": len(train_raw),
            "val": len(val_raw),
            "test": len(test_raw),
        },
        "purged_removed": {
            "train": int((~train_keep).sum()),
            "val": int((~val_keep).sum()),
            "test": 0,
        },
        "retained_counts": {
            "train": len(train_purged_df),
            "val": len(val_purged_df),
            "test": len(test_df),
        },
        "max_w4_dates": {
            "train": str(train_purged_df["w4_end_date"].max()),
            "val": str(val_purged_df["w4_end_date"].max()),
            "test": str(test_df["w4_end_date"].max()),
        },
    }
    return train_purged_df, val_purged_df, test_df, meta


def filter_interval_purged_years(
    issue_dates: Sequence[str],
    excluded_years: Iterable[int],
    horizon_days: int = 27,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (keep_mask, target_in_excluded_mask) for 9-year interval-purged OOF fitting.

    An issue t0 is excluded from training a prefix model whenever ANY day in its
    target window [t0, t0 + horizon_days] falls inside any year in `excluded_years`.
    """
    excl = set(int(y) for y in excluded_years)
    t0 = pd.to_datetime(issue_dates)
    w4 = t0 + pd.Timedelta(days=horizon_days)

    overlap = np.zeros(len(t0), dtype=bool)
    for yr in excl:
        yr_start = pd.Timestamp(f"{yr}-01-01")
        yr_end = pd.Timestamp(f"{yr}-12-31")
        overlap |= (t0 <= yr_end) & (w4 >= yr_start)

    in_excl = np.array([int(d.year) in excl for d in t0], dtype=bool)
    keep_mask = ~overlap
    return keep_mask, in_excl


def plan_nested_yearly_oof_folds(
    train_issue_dates: Sequence[str],
    years: Sequence[int] = tuple(range(2015, 2024)),
    horizon_days: int = 27,
) -> Dict[str, Any]:
    """
    Plans interval-purged outer and nested inner OOF folds across all 9 training years
    and computes exact deduplication by `(stage_lead, frozenset(excluded_years))`:
      - W1 unique fits: C(9,1) + C(9,2) + C(9,3) = 9 + 36 + 84 = 129
      - W2 unique fits: C(9,1) + C(9,2)          = 9 + 36      = 45
      - W3 unique fits: C(9,1)                   = 9
      - Total unique stage fits = 183 (vs. 675 naive un-cached fits)
    """
    dates = list(train_issue_dates)
    unique_stage_fits: Dict[Tuple[int, frozenset], np.ndarray] = {}
    subset_cache: Dict[frozenset, np.ndarray] = {}
    naive_stage_fits = 0

    def get_keep(excl_key: frozenset) -> np.ndarray:
        if excl_key not in subset_cache:
            subset_cache[excl_key] = filter_interval_purged_years(dates, excl_key, horizon_days)[0]
        return subset_cache[excl_key]

    for y_out in years:
        # W1 outer OOF (stage 1)
        key_1 = frozenset([y_out])
        naive_stage_fits += 1
        unique_stage_fits[(1, key_1)] = get_keep(key_1)

        # W2 outer OOF (stage 2) + inner W1 needed inside fold y_out
        naive_stage_fits += 1
        unique_stage_fits[(2, key_1)] = get_keep(key_1)
        inner_years = [y for y in years if y != y_out]
        for y_in1 in inner_years:
            key_12 = frozenset([y_out, y_in1])
            naive_stage_fits += 1
            unique_stage_fits[(1, key_12)] = get_keep(key_12)

        # W3 outer OOF (stage 3) + inner W2 + doubly-nested W1 inside y_out
        naive_stage_fits += 1
        unique_stage_fits[(3, key_1)] = get_keep(key_1)
        for y_in2 in inner_years:
            key_2 = frozenset([y_out, y_in2])
            naive_stage_fits += 1
            unique_stage_fits[(2, key_2)] = get_keep(key_2)
            for y_in1 in [y for y in inner_years if y != y_in2]:
                key_123 = frozenset([y_out, y_in2, y_in1])
                naive_stage_fits += 1
                unique_stage_fits[(1, key_123)] = get_keep(key_123)

    outer_fold_counts = {int(y): int(subset_cache[frozenset([y])].sum()) for y in years}
    purged_per_outer_fold = {int(y): len(dates) - int(subset_cache[frozenset([y])].sum()) for y in years}
    return {
        "years": [int(y) for y in years],
        "naive_oof_stage_fits": naive_stage_fits,
        "deduplicated_unique_stage_fits": len(unique_stage_fits),
        "unique_stage_fits_by_lead": {
            "W1": sum(1 for lead, _ in unique_stage_fits if lead == 1),
            "W2": sum(1 for lead, _ in unique_stage_fits if lead == 2),
            "W3": sum(1 for lead, _ in unique_stage_fits if lead == 3),
        },
        "unique_year_subsets": len(subset_cache),
        "outer_fold_retained_counts": outer_fold_counts,
        "outer_fold_purged_counts": purged_per_outer_fold,
    }


def fit_training_rzsm_v3_contract(
    rzsm_cube_path: Union[str, Path],
    mask_path: Union[str, Path],
    train_df: pd.DataFrame,
    out_yaml_path: Union[str, Path],
    out_json_path: Union[str, Path],
    out_clim_nc_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Fits Thesis v3.0 training-only RZSM seasonal climatology, domain-wide active-cell
    scaling bounds, and per-cell x midpoint-season 20th-percentile drought thresholds
    using STRICTLY timesteps `<= max_train_w4_date` (no Val or Sealed Test access).
    """
    ds_mask = xr.open_dataset(mask_path)
    eval_mask = ds_mask["evaluation_mask"].values == 1
    assert int(np.sum(eval_mask)) == 126, "Evaluation mask must contain exactly 126 active cells."

    max_train_w4 = str(train_df["w4_end_date"].max())
    min_train_t0 = str(train_df["issue_date"].min())

    # Quarantine slice: strictly discard everything beyond max_train_w4 before any computation
    ds_rzsm_full = xr.open_dataset(rzsm_cube_path)
    try:
        ds_train = ds_rzsm_full.sel(time=slice(min_train_t0, max_train_w4)).load()
    finally:
        ds_rzsm_full.close()

    rolling_train = ds_train["rzsm_0_100_rolling_7d"]
    t_coord = pd.to_datetime(rolling_train["time"].values)
    season_arr = date_to_season(t_coord)
    h, w = eval_mask.shape

    # 1. Fit DJF/MAM/JJA/SON seasonal climatology on 2015-01-01 .. max_train_w4
    clim_out = np.zeros((len(SEASONS), h, w), dtype=np.float32)
    anom_vals = np.empty_like(rolling_train.values, dtype=np.float32)
    for s_idx, s_name in enumerate(SEASONS):
        s_mask = season_arr == s_name
        s_clim = np.mean(rolling_train.values[s_mask], axis=0).astype(np.float32)
        s_clim[~eval_mask] = 0.0
        clim_out[s_idx] = s_clim
        anom_vals[s_mask] = rolling_train.values[s_mask] - s_clim[np.newaxis, :, :]

    active_anom = anom_vals[:, eval_mask]
    anom_min = float(np.min(active_anom))
    anom_max = float(np.max(active_anom))
    anom_mean = float(np.mean(active_anom))
    anom_std = float(np.std(active_anom))

    active_vol = rolling_train.values[:, eval_mask]
    vol_min = float(np.min(active_vol))
    vol_max = float(np.max(active_vol))

    # 2. Compute per-cell x season 20th-percentile drought thresholds tau(s, k) [m^3/m^3]
    tau_20 = np.zeros((len(SEASONS), h, w), dtype=np.float32)
    tau_summary: Dict[str, Dict[str, float]] = {}
    for s_idx, s_name in enumerate(SEASONS):
        s_vals = rolling_train.values[season_arr == s_name]
        s_p20 = np.percentile(s_vals, 20.0, axis=0).astype(np.float32)
        s_p20[~eval_mask] = 0.0
        tau_20[s_idx] = s_p20
        act = s_p20[eval_mask]
        tau_summary[s_name] = {
            "active_cell_min": float(np.min(act)),
            "active_cell_median": float(np.median(act)),
            "active_cell_max": float(np.max(act)),
        }

    # Write seasonal climatology + drought thresholds NetCDF cleanly
    out_clim = Path(out_clim_nc_path)
    out_clim.parent.mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        data_vars={
            "climatology_rolling_7d": (("season", "lat", "lon"), clim_out),
            "dry_threshold_p20": (("season", "lat", "lon"), tau_20),
            "evaluation_mask": (("lat", "lon"), ds_mask["evaluation_mask"].values.astype(np.uint8)),
        },
        coords={
            "season": list(SEASONS),
            "lat": ds_train["lat"].values,
            "lon": ds_train["lon"].values,
        },
        attrs={
            "contract": "thesis_revised_baseline_and_data_contract v3.0.0",
            "fit_window_start": min_train_t0,
            "fit_window_end_inclusive": max_train_w4,
            "train_issue_count": int(len(train_df)),
            "active_cell_count": 126,
            "dry_event_inequality": "strict_less_than (< tau)",
        },
    )
    ds_out.to_netcdf(out_clim)

    contract = {
        "contract_version": "v3.0.0-purged-train-rzsm",
        "governance_authority": "contracts/A0/thesis_revised_baseline_and_data_contract.yaml",
        "training_partition_scope": {
            "train_issue_date_range": [str(train_df["issue_date"].min()), str(train_df["issue_date"].max())],
            "train_w4_window_end_inclusive": max_train_w4,
            "train_issue_count": int(len(train_df)),
            "daily_rzsm_timesteps_used": int(len(t_coord)),
            "active_evaluation_cells": 126,
            "sealed_test_and_val_accessed": False,
        },
        "rzsm_parameters": {
            "seasonal_anomaly": {
                "min": anom_min,
                "max": anom_max,
                "mean": anom_mean,
                "std": anom_std,
                "range": float(anom_max - anom_min),
                "formula": "(x - min) / (max - min)",
            },
            "volumetric_rolling_7d": {
                "min": vol_min,
                "max": vol_max,
                "units": "m^3/m^3",
            },
        },
        "dry_event_thresholds_p20_summary": tau_summary,
        "climatology_and_threshold_netcdf": str(out_clim.relative_to(REPO_ROOT)).replace("\\", "/")
        if REPO_ROOT in out_clim.parents
        else str(out_clim),
    }

    out_yaml = Path(out_yaml_path)
    out_json = Path(out_json_path)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    out_json.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def write_v3_split_manifests(
    audit_csv_path: Union[str, Path],
    out_dir: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Writes Thesis v3.0 purged 80/10/10 split manifests and OOF metadata to `out_dir`."""
    train_df, val_df, test_df, meta = build_chronological_purged_splits(audit_csv_path)
    oof_plan = plan_nested_yearly_oof_folds(train_df["issue_date"].tolist())

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_csv = out_path / "train_cases_v3_purged.csv"
    val_csv = out_path / "val_cases_v3_purged.csv"
    test_csv = out_path / "test_cases_sealed_v3.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    summary = {
        "schema_version": "v3.0.0-thesis-revised",
        "source_audit_csv": str(Path(audit_csv_path).name),
        "partition_metadata": meta,
        "oof_cross_fitting_plan": oof_plan,
        "files": {
            "TRAIN": {
                "filename": train_csv.name,
                "sha256": _sha256_file(train_csv),
                "count": len(train_df),
                "issue_range": [str(train_df["issue_date"].min()), str(train_df["issue_date"].max())],
            },
            "VAL": {
                "filename": val_csv.name,
                "sha256": _sha256_file(val_csv),
                "count": len(val_df),
                "issue_range": [str(val_df["issue_date"].min()), str(val_df["issue_date"].max())],
            },
            "SEALED_TEST": {
                "filename": test_csv.name,
                "sha256": _sha256_file(test_csv),
                "count": len(test_df),
                "issue_range": [str(test_df["issue_date"].min()), str(test_df["issue_date"].max())],
                "quarantine_policy": "STRICT_SEALED_NO_PEEKING",
            },
        },
    }
    (out_path / "split_summary_v3.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return train_df, val_df, test_df, summary
