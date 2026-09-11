"""
src/data/temporal.py
--------------------
Temporal preprocessing, climatological anomaly transformations, antecedent lag extraction,
and target window assembly for the RISE-UNet Mindanao regional adaptation (Track B).

Authoritative Foundation:
  - Lesinger & Tian (2025), Nature Communications, DOI: 10.1038/s41467-025-62761-3
  - Kyle Lesinger's verified author code: function/preprocessUtils.py

Methodological Principles:
  1. Strict Zero Future Data Leakage:
     - All rolling averages are backward-looking trailing windows (center=False).
     - All climatologies and min-max normalization bounds are fitted exclusively on the
       nominal training period (2015–2021) and then applied out-of-sample to validation
       (2022–2023) and held-out test (2024–2025).
  2. Strict Target Parity:
     - Ground truth targets Y (Weeks 1 to 4) use the exact same temporal rolling mean,
       climatology anomaly subtraction, and standardization as antecedent inputs.
  3. Spatial Domain Masking:
     - Loss and verification are evaluated over the 126 Candidate A evaluation land cells.
     - Out-of-domain and ocean buffer cells are padded strictly to 0.0.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr

from .rzsm import apply_land_mask, compute_backward_rolling_mean


@dataclass(frozen=True)
class TemporalConfig:
    """
    Configuration parameters for Mindanao RISE-UNet temporal preprocessing.
    Reconciled strictly against verified parent EX29 contracts:
      - Target lead offsets: [6, 13, 20, 27] days corresponding to (lead * 7) - 1.
        Ending at Day +6 captures Week 1 [Days 0..6], Day +13 captures Week 2 [Days 7..13],
        Day +20 captures Week 3 [Days 14..20], Day +27 captures Week 4 [Days 21..27].
      - Climatology method: 'season' (DJF, MAM, JJA, SON) locked for Model A0.
    """
    train_start_year: int = 2015
    train_end_year: int = 2021
    val_years: Tuple[int, int] = (2022, 2023)
    test_years: Tuple[int, int] = (2024, 2025)
    rolling_window_days: int = 7
    min_periods: int = 7
    antecedent_lags: Tuple[int, ...] = (-1, -7, -14)
    target_leads_days: Tuple[int, ...] = (6, 13, 20, 27)  # Verified parent EX29: (lead * 7) - 1
    climatology_method: str = "season"  # Locked Model A0 baseline
    fill_value: float = 0.0


def compute_trailing_rolling_mean(
    data: xr.DataArray,
    window: int = 7,
    min_periods: int = 7,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Computes an unweighted backward trailing rolling window mean along the time axis.
    Guarantees center=False to prevent any forward temporal data leakage.

    Parameters
    ----------
    data : xr.DataArray
        Daily spatio-temporal data array.
    window : int, optional
        Window size in days (default: 7).
    min_periods : int, optional
        Minimum number of valid observations required (default: 7).
    time_dim : str, optional
        Time dimension name (default: 'time').

    Returns
    -------
    xr.DataArray
        Smoothed trailing rolling mean data array.
    """
    return compute_backward_rolling_mean(
        data, window=window, min_periods=min_periods, time_dim=time_dim
    )


def compute_training_climatology(
    data: xr.DataArray,
    train_end_year: int = 2021,
    train_start_year: Optional[int] = 2015,
    method: str = "season",
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Computes climatological seasonal means fitted strictly on the training partition
    (year <= train_end_year) to prevent future data leakage.

    Methodological Lock for Model A0:
      - Model A0 strictly locks method='season' (3-month meteorological seasons DJF, MAM, JJA, SON)
        matching Kyle Lesinger's authoritative parent implementation in preprocessUtils.py:L341-387.
      - Exploratory alternatives such as daily 'dayofyear' harmonic smoothing are strictly
        quarantined to future Model A1 (Mindanao Enhancement).

    Parameters
    ----------
    data : xr.DataArray
        Daily rolling-mean data array indexed by datetime coordinates.
    train_end_year : int, optional
        Final calendar year of the training set (default: 2021).
    train_start_year : int, optional
        First calendar year of the training set (default: 2015).
    method : str, optional
        Grouping method (default: 'season' [LOCKED for Model A0]):
          - 'season': 3-month meteorological seasons (DJF, MAM, JJA, SON) matching
            Kyle Lesinger's parent implementation in preprocessUtils.py:L341-387.
          - 'dayofyear': 1 to 365/366 day-of-year climatology (quarantined to Model A1).
          - 'month': 1 to 12 calendar month climatology.
    time_dim : str, optional
        Time coordinate name (default: 'time').

    Returns
    -------
    xr.DataArray
        Climatological mean array indexed by grouping key (e.g. season).
    """
    if time_dim not in data.coords:
        raise KeyError(f"Coordinate '{time_dim}' not found in DataArray coords: {list(data.coords)}")

    # Ensure datetime64 coordinate
    t_coord = pd.to_datetime(data[time_dim].values)

    # Filter strictly to training years
    year_mask = t_coord.year <= train_end_year
    if train_start_year is not None:
        year_mask = year_mask & (t_coord.year >= train_start_year)

    if not np.any(year_mask):
        raise ValueError(
            f"No timesteps found in range [{train_start_year}, {train_end_year}]. "
            f"Dataset spans {t_coord.year.min()} to {t_coord.year.max()}."
        )

    train_subset = data.isel({time_dim: year_mask})

    valid_methods = {"dayofyear", "season", "month"}
    if method not in valid_methods:
        raise ValueError(f"Unknown climatology method '{method}'. Valid options: {valid_methods}")

    groupby_dim = f"{time_dim}.{method}"
    climatology = train_subset.groupby(groupby_dim).mean(dim=time_dim)
    climatology.attrs["climatology_method"] = method
    climatology.attrs["train_end_year"] = train_end_year
    climatology.attrs["train_start_year"] = train_start_year if train_start_year else "all_prior"

    return climatology


def compute_seasonal_anomalies(
    data: xr.DataArray,
    climatology: xr.DataArray,
    method: str = "season",
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Computes seasonal anomalies by subtracting pre-computed training climatology
    from the raw or rolling-mean data array:

        Anomaly(t, lat, lon) = Data(t, lat, lon) - Climatology(key(t), lat, lon)

    Parameters
    ----------
    data : xr.DataArray
        Daily data array (training, validation, or test partition).
    climatology : xr.DataArray
        Pre-computed training climatology from `compute_training_climatology`.
    method : str, optional
        Grouping method matching climatology (default: 'season' [LOCKED for Model A0]).
    time_dim : str, optional
        Time dimension name (default: 'time').

    Returns
    -------
    xr.DataArray
        Spatio-temporal anomaly array with identical coordinates and dimensions as `data`.
    """
    groupby_dim = f"{time_dim}.{method}"

    # Handle leap year day 366 for non-leap years or boundary edge cases
    if method == "dayofyear" and 366 in data[time_dim].dt.dayofyear.values:
        if 366 not in climatology["dayofyear"].values:
            # Map day 366 to day 365 climatology if training set lacked leap day
            clim_365 = climatology.sel(dayofyear=365).copy(deep=True)
            clim_365 = clim_365.assign_coords(dayofyear=366)
            climatology = xr.concat([climatology, clim_365], dim="dayofyear")

    anomalies = data.groupby(groupby_dim) - climatology

    # Drop grouping coordinate if added
    if method in anomalies.coords and method not in data.coords:
        anomalies = anomalies.drop_vars(method)

    anomalies.name = f"{data.name}_anomaly" if data.name else "anomaly"
    anomalies.attrs["long_name"] = f"Seasonal Anomaly ({method})"
    return anomalies


def extract_antecedent_and_target_windows(
    data: xr.DataArray,
    issue_date: Union[str, pd.Timestamp, np.datetime64],
    antecedent_lags: Tuple[int, ...] = (-1, -7, -14),
    target_leads: Tuple[int, ...] = (6, 13, 20, 27),
    time_dim: str = "time",
) -> Tuple[Dict[int, xr.DataArray], Dict[int, xr.DataArray]]:
    """
    Extracts antecedent lagged predictor states and multi-lead ground truth targets
    for a single forecast issuance date with strict mathematical parity.

    Following the verified parent EX29 contract (function/funs.py:L430, L = (lead*7) - 1):
      - Antecedent inputs: Trailing 7-day rolling means sampled at Day -1, Day -7, Day -14.
      - Targets (W1–W4): Trailing 7-day rolling means sampled at Day +6 (Week 1, Days 0..6),
        Day +13 (Week 2, Days 7..13), Day +20 (Week 3, Days 14..20), and Day +27 (Week 4, Days 21..27).


    Parameters
    ----------
    data : xr.DataArray
        Continuous daily spatio-temporal array (e.g. daily rolling RZSM anomalies).
    issue_date : str, pd.Timestamp, or np.datetime64
        Forecast issuance date t0 (Day 0).
    antecedent_lags : tuple of int, optional
        Relative day offsets for inputs (default: (-1, -7, -14)).
    target_leads : tuple of int, optional
        Relative day offsets for targets (default: (6, 13, 20, 27), matching L = (lead*7) - 1).
    time_dim : str, optional
        Time dimension name (default: 'time').

    Returns
    -------
    tuple of (antecedent_dict, target_dict)
        antecedent_dict: Mapping lag offset (e.g. -1, -7, -14) to 2D spatial slice (lat, lon).
        target_dict: Mapping lead offset (e.g. 6, 13, 20, 27) to 2D spatial slice (lat, lon).
    """
    t0 = pd.to_datetime(issue_date)

    antecedent_dict: Dict[int, xr.DataArray] = {}
    for lag in antecedent_lags:
        d_lag = t0 + pd.Timedelta(days=lag)
        d_str = d_lag.strftime("%Y-%m-%d")
        try:
            slice_2d = data.sel({time_dim: d_str})
            if time_dim in slice_2d.dims:
                slice_2d = slice_2d.squeeze(time_dim)
            antecedent_dict[lag] = slice_2d
        except KeyError:
            raise KeyError(
                f"Antecedent date {d_str} (lag={lag}d from issue {t0.strftime('%Y-%m-%d')}) "
                f"not found in {time_dim} coordinates."
            )

    target_dict: Dict[int, xr.DataArray] = {}
    for lead in target_leads:
        d_lead = t0 + pd.Timedelta(days=lead)
        d_str = d_lead.strftime("%Y-%m-%d")
        try:
            slice_2d = data.sel({time_dim: d_str})
            if time_dim in slice_2d.dims:
                slice_2d = slice_2d.squeeze(time_dim)
            target_dict[lead] = slice_2d
        except KeyError:
            raise KeyError(
                f"Target date {d_str} (lead={lead}d from issue {t0.strftime('%Y-%m-%d')}) "
                f"not found in {time_dim} coordinates."
            )

    return antecedent_dict, target_dict


def fit_min_max_bounds(
    data: xr.DataArray,
    mask: Optional[Union[xr.DataArray, np.ndarray]] = None,
    train_end_year: int = 2021,
    train_start_year: Optional[int] = 2015,
    time_dim: str = "time",
) -> Tuple[float, float]:
    """
    Computes scalar minimum and maximum normalization bounds fitted strictly over
    the training fold and active evaluation land cells.

    Methodological Lock for Model A0 (Four-Part Scope):
      1. Training cases only: train_start_year (2015) <= year <= train_end_year (2021).
      2. Evaluation cells only: computed exclusively over cells where mask == 1 (126 active cells).
         Ocean/buffer cells are strictly ignored during bounds fitting.
      3. Per-variable / per-lead: bounds are fitted per variable or per lead channel.
      4. Domain-wide active scalar bounds (NOT pixel-wise per-cell bounds):
         Following Kyle Lesinger's authoritative parent implementation in
         function/preprocessUtils.py:L739-757:
           "We are looking at the mean and standard deviation of all ensemble members
            for the entire time series (not individual grid cells)."
         A single pair of global scalar bounds (min_v, max_v) is extracted across all
         active evaluation cells. This prevents distorting spatial soil moisture gradients
         and avoids extreme local variance sensitivity over heterogeneous tropical topography.

    Parameters
    ----------
    data : xr.DataArray
        Spatio-temporal data or anomaly array.
    mask : xr.DataArray or np.ndarray, optional
        Binary evaluation mask (1 = active land, 0 = ocean/buffer).
    train_end_year : int, optional
        Final training year (default: 2021).
    train_start_year : int, optional
        Starting training year (default: 2015).
    time_dim : str, optional
        Time coordinate name (default: 'time').

    Returns
    -------
    tuple of (min_val, max_val)
        Floating-point minimum and maximum bounds (domain-wide scalars).
    """
    t_coord = pd.to_datetime(data[time_dim].values)
    year_mask = t_coord.year <= train_end_year
    if train_start_year is not None:
        year_mask = year_mask & (t_coord.year >= train_start_year)

    train_subset = data.isel({time_dim: year_mask})

    if mask is not None:
        m_bool = (mask.values if hasattr(mask, "values") else np.asarray(mask)) == 1
        arr = train_subset.values
        valid_vals = arr[..., m_bool]
        min_v = float(np.nanmin(valid_vals))
        max_v = float(np.nanmax(valid_vals))
    else:
        min_v = float(np.nanmin(train_subset.values))
        max_v = float(np.nanmax(train_subset.values))

    if np.isclose(max_v, min_v):
        raise ValueError(f"Degenerate training bounds: max ({max_v}) equals min ({min_v})")

    return min_v, max_v


def standardize_with_training_bounds(
    data: Union[xr.DataArray, np.ndarray],
    train_min: float,
    train_max: float,
    mask: Optional[Union[xr.DataArray, np.ndarray]] = None,
    fill_value: float = 0.0,
    clip: bool = True,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Standardizes data into the unit interval [0, 1] using pre-computed training bounds:

        scaled = (x - train_min) / (train_max - train_min)

    Ocean and out-of-domain cells are explicitly padded with `fill_value`.
    Values exceeding [0, 1] due to out-of-sample extremes in validation/test are optionally clipped.

    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Array to standardize.
    train_min : float
        Minimum value from training fold.
    train_max : float
        Maximum value from training fold.
    mask : xr.DataArray or np.ndarray, optional
        Binary evaluation mask (1 = active land, 0 = ocean/buffer).
    fill_value : float, optional
        Value assigned to masked/ocean cells (default: 0.0).
    clip : bool, optional
        Whether to clip standardized active values to [0.0, 1.0] (default: True).

    Returns
    -------
    xr.DataArray or np.ndarray
        Standardized array conforming to input type.
    """
    denom = train_max - train_min
    if np.isclose(denom, 0.0):
        raise ValueError(f"Degenerate bounds: max ({train_max}) equals min ({train_min})")

    scaled = (data - train_min) / denom

    if clip:
        if isinstance(scaled, xr.DataArray):
            scaled = scaled.clip(0.0, 1.0)
        else:
            scaled = np.clip(scaled, 0.0, 1.0)

    if mask is not None:
        scaled = apply_land_mask(scaled, mask, fill_value=fill_value)

    return scaled
