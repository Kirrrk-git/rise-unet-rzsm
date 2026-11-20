"""
src/data/preprocessing_v3.py
----------------------------
Remediated Thesis v3.0 predictor preprocessing and tensor assembly engine.

Directly resolves legacy Phase 21 preprocessing defects:
  1. Removes all hardcoded atmospheric/S2S bounds (which clipped 100% of ch10 s2s_tcw
     to 1.0000 in legacy production tensors). Fits all scaling parameters strictly from
     empirical training-partition extrema over the 126 active evaluation cells.
  2. Applies backward 7-day trailing rolling averages to daily ERA5 atmospheric variables
     and samples them strictly at lag = -1d (t0 - 1 day), preventing same-day t0 leakage.
  3. Subtracts training-fitted DJF/MAM/JJA/SON seasonal climatologies from both ERA5
     atmospheric predictors and ECMWF S2S lead forecasts (t2m, d2m, tcw).
  4. Enforces `eps_floor = 1e-6` denominator stability when a training predictor is constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.data.case_builder import CaseTensorHierarchy
from src.data.splits_v3 import SEASONS, date_to_season

EPS_FLOOR: float = 1e-6
ATM_VARS: Tuple[str, ...] = ("pwat", "spfh", "tmax", "diff_temp", "hgt_pres")
S2S_VARS: Tuple[str, ...] = ("t2m", "d2m", "tcw")


def empirical_min_max_scale(
    arr: np.ndarray,
    train_min: float,
    train_max: float,
    eps_floor: float = EPS_FLOOR,
    clip: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Scales array using empirical training-only scalar extrema with eps_floor guard.
    Never uses hardcoded constants; reports out-of-[0, 1] excursions if `clip=False`.
    """
    denom = max(float(train_max) - float(train_min), eps_floor)
    scaled = ((arr.astype(np.float64) - float(train_min)) / denom).astype(np.float32)
    excursion_low = int(np.sum(scaled < 0.0))
    excursion_high = int(np.sum(scaled > 1.0))
    if clip:
        scaled = np.clip(scaled, 0.0, 1.0).astype(np.float32)
    return scaled, {
        "below_zero_count": excursion_low,
        "above_one_count": excursion_high,
        "clipped": bool(clip),
    }


@dataclass
class V3PredictorNormalizationParams:
    """Empirical training-only climatology and scalar extrema for all v3.0 predictors."""
    rzsm_min: float
    rzsm_max: float
    atm_climatology: Dict[str, Dict[str, np.ndarray]]   # var -> season -> (H, W)
    atm_bounds: Dict[str, Tuple[float, float]]          # var -> (min, max)
    s2s_climatology: Dict[int, Dict[str, Dict[str, np.ndarray]]]  # lead -> var -> season -> (H, W)
    s2s_bounds: Dict[int, Dict[str, Tuple[float, float]]]          # lead -> var -> (min, max)


def fit_and_transform_era5_atmos(
    atmos_ds: xr.Dataset,
    eval_mask: np.ndarray,
    max_train_date: str,
    window_days: int = 7,
    min_check_date: Optional[str] = None,
    allow_incomplete_warmup: bool = False,
) -> Tuple[xr.Dataset, Dict[str, Dict[str, np.ndarray]], Dict[str, Tuple[float, float]]]:
    """
    Computes backward 7-day trailing rolling means on daily ERA5 atmospheric predictors,
    fits DJF/MAM/JJA/SON seasonal climatologies strictly on `time <= max_train_date`,
    and fits empirical min-max anomaly bounds over active cells (`eval_mask == 1`).
    """
    m_bool = np.asarray(eval_mask) == 1
    times = pd.to_datetime(atmos_ds["time"].values)
    max_dt = pd.Timestamp(max_train_date)
    train_time_mask = times <= max_dt
    if not np.any(train_time_mask):
        raise ValueError(f"No atmospheric timesteps found on or before {max_train_date}.")

    check_mask = train_time_mask if min_check_date is None else (train_time_mask & (times >= pd.Timestamp(min_check_date)))
    seasons = date_to_season(times)
    clim_out: Dict[str, Dict[str, np.ndarray]] = {}
    bounds_out: Dict[str, Tuple[float, float]] = {}
    anom_vars: Dict[str, Tuple[Tuple[str, ...], np.ndarray]] = {}

    for var in ATM_VARS:
        if var not in atmos_ds:
            raise KeyError(f"Missing required ERA5 atmospheric variable '{var}'.")
        da = atmos_ds[var].astype(np.float32)
        min_p = 1 if allow_incomplete_warmup else window_days
        rolled = da.rolling(time=window_days, min_periods=min_p, center=False).mean()
        rolled_vals = rolled.values.astype(np.float32)

        if not allow_incomplete_warmup and np.any(np.isnan(rolled_vals[check_mask][:, m_bool])):
            nan_dates = times[check_mask & np.isnan(rolled_vals[:, m_bool]).any(axis=1)]
            sample = [d.strftime("%Y-%m-%d") for d in nan_dates[:5]]
            raise ValueError(
                f"Incomplete 7-day trailing atmospheric warmup for '{var}' on dates {sample}. "
                "Provide 6 prior warmup days (e.g., 2014-12-25..31) or set allow_incomplete_warmup=True."
            )

        clim_out[var] = {}
        anom_arr = np.full_like(rolled_vals, np.nan, dtype=np.float32)
        for s in SEASONS:
            s_train_mask = train_time_mask & (seasons == s)
            if not np.any(s_train_mask):
                raise ValueError(f"Season {s} absent in training atmospheric slice.")
            s_mean = np.nanmean(rolled_vals[s_train_mask], axis=0).astype(np.float32)
            clim_out[var][s] = s_mean
            s_all_mask = seasons == s
            anom_arr[s_all_mask] = rolled_vals[s_all_mask] - s_mean[np.newaxis, :, :]

        valid_train_vals = anom_arr[train_time_mask][:, m_bool]
        finite_train = valid_train_vals[np.isfinite(valid_train_vals)]
        if finite_train.size == 0:
            raise ValueError(f"No finite training anomaly samples for ERA5 variable '{var}'.")
        v_min = float(np.min(finite_train))
        v_max = float(np.max(finite_train))
        bounds_out[var] = (v_min, v_max)
        anom_vars[var] = (("time", "lat", "lon"), anom_arr)

    anom_ds = xr.Dataset(
        data_vars=anom_vars,
        coords={"time": atmos_ds["time"], "lat": atmos_ds["lat"], "lon": atmos_ds["lon"]},
    )
    return anom_ds, clim_out, bounds_out


def fit_s2s_training_anomalies(
    s2s_datasets_by_issue: Dict[str, xr.Dataset],
    eval_mask: np.ndarray,
    train_issue_dates: Sequence[str],
) -> Tuple[Dict[int, Dict[str, Dict[str, np.ndarray]]], Dict[int, Dict[str, Tuple[float, float]]]]:
    """
    Fits training-only seasonal climatology and empirical active-cell anomaly extrema
    across all 11 ECMWF members per lead (1, 2) and variable (t2m, d2m, tcw).
    """
    m_bool = np.asarray(eval_mask) == 1
    train_dates_sorted = sorted(train_issue_dates)
    if not train_dates_sorted:
        raise ValueError("train_issue_dates must not be empty.")

    issue_seasons = date_to_season(train_dates_sorted)
    s2s_clim: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {1: {}, 2: {}}
    s2s_bounds: Dict[int, Dict[str, Tuple[float, float]]] = {1: {}, 2: {}}

    for lead in (1, 2):
        for var in S2S_VARS:
            stacked = np.stack(
                [s2s_datasets_by_issue[d][var].sel(lead=lead).values.astype(np.float32) for d in train_dates_sorted],
                axis=0,
            )  # (N_train, M=11, H, W)

            s2s_clim[lead][var] = {}
            anom_stacked = np.empty_like(stacked, dtype=np.float32)
            for s in SEASONS:
                s_mask = issue_seasons == s
                if not np.any(s_mask):
                    raise ValueError(f"Season {s} missing in training S2S issues.")
                # Pool across training cases AND 11 members for seasonal baseline
                s_mean = np.mean(stacked[s_mask], axis=(0, 1)).astype(np.float32)
                s2s_clim[lead][var][s] = s_mean
                anom_stacked[s_mask] = stacked[s_mask] - s_mean[np.newaxis, np.newaxis, :, :]

            act_vals = anom_stacked[:, :, m_bool]
            s2s_bounds[lead][var] = (float(np.min(act_vals)), float(np.max(act_vals)))

    return s2s_clim, s2s_bounds


def assemble_v3_case_tensors(
    issue_date: str,
    rzsm_anom_ds: xr.Dataset,
    atmos_anom_ds: xr.Dataset,
    s2s_ds: xr.Dataset,
    eval_mask: np.ndarray,
    norm_params: V3PredictorNormalizationParams,
    rzsm_var_name: str = "rzsm_0_100_seasonal_anomaly",
    clip_predictors: bool = False,
) -> Tuple[CaseTensorHierarchy, Dict[str, Any]]:
    """
    Assembles Thesis v3.0 multi-lead input/target tensors with:
      - Antecedent RZSM lags at [-1, -7, -14]d (empirical training min-max scaled)
      - 7-day trailing ERA5 atmospheric seasonal anomalies sampled strictly at lag = -1d
      - ECMWF S2S W1/W2 seasonal anomalies dynamically scaled (no ch10 saturation)
      - Ocean/buffer cells strictly zero-filled
    """
    t0 = pd.Timestamp(issue_date)
    t0_str = t0.strftime("%Y-%m-%d")
    lag1_str = (t0 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    m_bool = np.asarray(eval_mask) == 1
    ocean = ~m_bool
    num_members = len(s2s_ds["member"])
    diagnostics: Dict[str, Any] = {"excursions": {}}

    # 1. Antecedent RZSM lags (-1, -7, -14)
    rzsm_lags = []
    for d_off in (1, 7, 14):
        d_str = (t0 - pd.Timedelta(days=d_off)).strftime("%Y-%m-%d")
        arr_2d = rzsm_anom_ds[rzsm_var_name].sel(time=d_str).values.astype(np.float32)
        if arr_2d.ndim == 3:
            arr_2d = arr_2d[0]
        scaled, diag = empirical_min_max_scale(arr_2d, norm_params.rzsm_min, norm_params.rzsm_max, clip=clip_predictors)
        diagnostics["excursions"][f"rzsm_lag_{d_off}"] = diag
        rzsm_lags.append(scaled)
    rzsm_m11 = np.repeat(np.stack(rzsm_lags, axis=-1)[np.newaxis, ...], num_members, axis=0)

    # 2. Trailing 7-day ERA5 atmospheric anomalies strictly at lag = -1d (t0 - 1d)
    atm_chans = []
    for var in ATM_VARS:
        slice_2d = atmos_anom_ds[var].sel(time=lag1_str).values.astype(np.float32)
        if slice_2d.ndim == 3:
            slice_2d = slice_2d[0]
        if not np.all(np.isfinite(slice_2d[m_bool])):
            raise ValueError(f"Non-finite trailing ERA5 anomaly for {var} at lag-1d={lag1_str}.")
        vmin, vmax = norm_params.atm_bounds[var]
        scaled, diag = empirical_min_max_scale(slice_2d, vmin, vmax, clip=clip_predictors)
        diagnostics["excursions"][f"atm_{var}"] = diag
        atm_chans.append(scaled)
    atm_m11 = np.repeat(np.stack(atm_chans, axis=-1)[np.newaxis, ...], num_members, axis=0)

    # 3. ECMWF S2S dynamic triplet anomalies (W1, W2)
    season_t0 = str(date_to_season([t0_str])[0])
    s2s_scaled_by_lead: Dict[int, np.ndarray] = {}
    for lead in (1, 2):
        lead_chans = []
        for var in S2S_VARS:
            raw_m = s2s_ds[var].sel(lead=lead).values.astype(np.float32)
            clim_2d = norm_params.s2s_climatology[lead][var][season_t0]
            anom_m = raw_m - clim_2d[np.newaxis, :, :]
            vmin, vmax = norm_params.s2s_bounds[lead][var]
            scaled, diag = empirical_min_max_scale(anom_m, vmin, vmax, clip=clip_predictors)
            diagnostics["excursions"][f"s2s_w{lead}_{var}"] = diag
            lead_chans.append(scaled)
        s2s_scaled_by_lead[lead] = np.stack(lead_chans, axis=-1)

    # 4. Targets for Leads 1..4 at [+6, +13, +20, +27] days
    targets = []
    target_dates: Dict[int, str] = {}
    for idx, d_off in enumerate((6, 13, 20, 27), start=1):
        tgt_str = (t0 + pd.Timedelta(days=d_off)).strftime("%Y-%m-%d")
        tgt_2d = rzsm_anom_ds[rzsm_var_name].sel(time=tgt_str).values.astype(np.float32)
        if tgt_2d.ndim == 3:
            tgt_2d = tgt_2d[0]
        scaled, _ = empirical_min_max_scale(tgt_2d, norm_params.rzsm_min, norm_params.rzsm_max, clip=False)
        scaled[ocean] = 0.0
        targets.append(scaled[np.newaxis, :, :, np.newaxis])
        target_dates[idx] = tgt_str

    x_w1 = np.concatenate([rzsm_m11, atm_m11, s2s_scaled_by_lead[1]], axis=-1).astype(np.float32)
    x_w2_base = np.concatenate([rzsm_m11, atm_m11, s2s_scaled_by_lead[2]], axis=-1).astype(np.float32)
    x_w3_base = rzsm_m11.copy().astype(np.float32)
    x_w4_base = rzsm_m11.copy().astype(np.float32)

    for arr in (x_w1, x_w2_base, x_w3_base, x_w4_base):
        arr[:, ocean, :] = 0.0

    case = CaseTensorHierarchy(
        issue_date=t0_str,
        hdate=str(s2s_ds.attrs.get("hdate", t0_str)),
        model_version_date=None if s2s_ds.attrs.get("model_version_date") in (None, "unknown") else str(s2s_ds.attrs.get("model_version_date")),
        x_w1=x_w1,
        x_w2_base=x_w2_base,
        x_w3_base=x_w3_base,
        x_w4_base=x_w4_base,
        y_w1=targets[0],
        y_w2=targets[1],
        y_w3=targets[2],
        y_w4=targets[3],
        channel_schedule={1: 11, 2: 12, 3: 5, 4: 6},
        num_members=num_members,
        target_dates=target_dates,
        is_normalized=True,
        normalization_contract_used="thesis_v3_empirical_training_only",
    )
    return case, diagnostics
