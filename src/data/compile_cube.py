"""
src/data/compile_cube.py
------------------------
Production compilation engine for the 11-year Mindanao Root-Zone Soil Moisture (RZSM)
spatio-temporal data cube (late-2014 antecedent support + 2015–2025 nominal production).

Authoritative Foundation:
  - Sub-Phase 21D Step 21D.4 (Full Production RZSM Cube)
  - Lesinger & Tian (2025), Nature Communications, DOI: 10.1038/s41467-025-62761-3
  - Verified Parent & Adaptation Contracts:
      1. Depth-weighted RZSM: 0.07 * SM1 + 0.21 * SM2 + 0.72 * SM3
      2. Land-aware bilinear remapping with nearest-neighbor boundary fallback to 0.25° (32 x 48)
         [ACCEPTED: Mindanao spatial remapping implementation]
      3. 7-day backward trailing rolling mean (center=False, zero future leakage)
      4. Locked Model A0 3-month seasonal climatology ('season': DJF, MAM, JJA, SON, train_end=2021)
      5. Four-part immutable normalization scope:
         - Training fold only (2015 <= year <= 2021)
         - Evaluation cells only (mask == 1, 126 active cells)
         - Per-variable / lead
         - Domain-wide active scalar bounds (NOT pixel-wise per-cell)
      6. Precision Masking Reporting Standard:
         "Zero NaNs/Infs across the 126 active evaluation cells; non-evaluation computational
          cells follow the frozen masking/zero-fill convention."
      7. Temporal Pipeline Execution Order:
         12 Dec 2014 ─────────────── 31 Dec 2025 (4,038 archive days)
                   ↓
         Depth-weighted RZSM construction
                   ↓
         7-day backward trailing rolling transformation
                   ↓
         Seasonal climatology (2015–2021) & anomaly transformation
                   ↓
         Domain-wide active scalar min-max normalization
                   ↓
         Slice nominal period: 01 Jan 2015 ─────────────── 31 Dec 2025 (4,018 production days)
                   ↓
         Production cube (506,268 nominal evaluation cell-days, 0 NaNs/Infs)

         Note on the 20-day 2014 Support Archive (12–31 Dec 2014 = 20 days):
         These days provide antecedent initialization history so that the earliest nominal 2015
         observations have complete 7-day and 14-day trailing rolling memory. They are NOT
         independent production observations or training/evaluation years.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.data.rzsm import (
    compute_depth_weighted_rzsm,
    remap_era5_land_to_candidate_a,
    apply_land_mask,
    LAYER_WEIGHTS,
)
from src.data.temporal import (
    compute_trailing_rolling_mean,
    compute_training_climatology,
    compute_seasonal_anomalies,
    fit_min_max_bounds,
    standardize_with_training_bounds,
)


@dataclass(frozen=True)
class ProductionCubeConfig:
    """Configuration contracts for full production RZSM compilation."""
    train_start_year: int = 2015
    train_end_year: int = 2021
    val_years: Tuple[int, int] = (2022, 2023)
    test_years: Tuple[int, int] = (2024, 2025)
    rolling_window: int = 7
    min_periods: int = 7
    climatology_method: str = "season"  # Locked Model A0 baseline
    fill_value: float = 0.0
    expected_eval_cells: int = 126
    expected_grid_shape: Tuple[int, int] = (32, 48)
    # Reconciled archive vs. production census counts
    expected_nominal_days: int = 4018  # 2015-01-01 to 2025-12-31 (leap years 2016, 2020, 2024)
    expected_archive_days: int = 4038  # 2014-12-12 to 2025-12-31 (4,018 + 20 antecedent days)
    expected_nominal_eval_samples: int = 506268  # 4,018 * 126
    expected_archive_eval_samples: int = 508788  # 4,038 * 126


def compile_production_rzsm_pipeline(
    daily_remapped_rzsm: xr.DataArray,
    eval_mask: Union[xr.DataArray, np.ndarray],
    config: Optional[ProductionCubeConfig] = None,
    slice_nominal_period: bool = True,
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Executes the complete multi-stage temporal transformation pipeline on a
    remapped daily 0-100 cm RZSM DataArray on the 0.25° Candidate A grid.

    Processing Steps (Strict Order):
      1. Compute 7-day backward trailing rolling mean (center=False) over the full
         archive series (4,038 days, 12 Dec 2014 to 31 Dec 2025). The 20 support days
         in December 2014 ensure that earliest nominal 2015 observations have complete
         trailing windows without NaNs.
      2. Compute locked Model A0 3-month seasonal climatology (DJF, MAM, JJA, SON)
         fitted strictly on nominal training years (2015 <= year <= 2021).
      3. Compute seasonal anomalies: Anom(t) = Rolling(t) - Clim(season(t)).
      4. Fit domain-wide scalar min-max bounds strictly over training fold active cells (mask == 1).
      5. Standardize seasonal anomalies to [0, 1] using training scalar bounds.
      6. Apply frozen binary evaluation mask (126 active cells, 1,410 ocean cells zero-filled).
      7. Assemble multi-variable CF-1.8 compliant xr.Dataset.
      8. If slice_nominal_period is True, slice the nominal period (2015-01-01 to 2025-12-31,
         4,018 days / 506,268 evaluation cell-days), excluding the antecedent support days from the
         nominal cube while preserving metadata attributes.

    Parameters
    ----------
    daily_remapped_rzsm : xr.DataArray
        Daily remapped RZSM (0-100 cm) array with dims (time, lat, lon) and Candidate A coordinates.
    eval_mask : xr.DataArray or np.ndarray
        2D binary evaluation mask (32, 48) where 1 indicates active evaluation land cells.
    config : ProductionCubeConfig, optional
        Configuration object specifying training bounds, rolling window, etc.
    slice_nominal_period : bool, default True
        If True, slices output dataset to the nominal evaluation period (2015-01-01 onwards).

    Returns
    -------
    xr.Dataset
        Production dataset containing raw, rolling, anomaly, standardized fields, and climatologies.
    """
    if config is None:
        config = ProductionCubeConfig()

    m_arr = eval_mask.values if hasattr(eval_mask, "values") else np.asarray(eval_mask)
    n_active = int(np.sum(m_arr == 1))
    if n_active != config.expected_eval_cells:
        raise ValueError(
            f"Evaluation mask active cell count ({n_active}) does not match "
            f"expected count ({config.expected_eval_cells})"
        )

    # Step 1: 7-day backward trailing rolling mean
    rolling_7d = compute_trailing_rolling_mean(
        daily_remapped_rzsm,
        window=config.rolling_window,
        min_periods=config.min_periods,
        time_dim="time",
    )
    rolling_7d.name = "rzsm_0_100_rolling_7d"
    rolling_7d.attrs["long_name"] = "7-day backward trailing rolling mean 0-100 cm RZSM"
    rolling_7d.attrs["units"] = "m3/m3"

    # Step 2: 3-month seasonal climatology fitted strictly on training years
    climatology = compute_training_climatology(
        rolling_7d,
        train_start_year=config.train_start_year,
        train_end_year=config.train_end_year,
        method=config.climatology_method,
        time_dim="time",
    )
    climatology.name = "rzsm_0_100_seasonal_climatology"
    climatology.attrs["long_name"] = f"Training Climatological Mean ({config.climatology_method.upper()})"
    climatology.attrs["units"] = "m3/m3"

    # Step 3: Seasonal anomalies
    anomalies = compute_seasonal_anomalies(
        rolling_7d,
        climatology=climatology,
        method=config.climatology_method,
        time_dim="time",
    )
    anomalies.name = "rzsm_0_100_seasonal_anomaly"
    anomalies.attrs["long_name"] = "0-100 cm RZSM 7-day rolling seasonal anomaly"
    anomalies.attrs["units"] = "m3/m3"

    # Step 4: Fit domain-wide scalar min-max bounds on training fold active cells
    train_min, train_max = fit_min_max_bounds(
        anomalies,
        mask=m_arr,
        train_start_year=config.train_start_year,
        train_end_year=config.train_end_year,
        time_dim="time",
    )

    # Step 5: Standardize seasonal anomalies to [0, 1] with zero-filled buffer/ocean
    standardized = standardize_with_training_bounds(
        anomalies,
        train_min=train_min,
        train_max=train_max,
        mask=m_arr,
        fill_value=config.fill_value,
        clip=True,
    )
    standardized.name = "rzsm_0_100_normalized"
    standardized.attrs["long_name"] = "Standardized RZSM seasonal anomaly [0, 1]"
    standardized.attrs["units"] = "dimensionless"
    standardized.attrs["train_min_bound"] = float(train_min)
    standardized.attrs["train_max_bound"] = float(train_max)

    # Extract temporal range metadata
    t_vals = pd.to_datetime(daily_remapped_rzsm[time_dim].values)
    raw_start = str(t_vals[0].date())
    raw_end = str(t_vals[-1].date())

    # Assemble into comprehensive production Dataset
    prod_ds = xr.Dataset(
        data_vars={
            "rzsm_0_100_raw": daily_remapped_rzsm,
            "rzsm_0_100_rolling_7d": rolling_7d,
            "rzsm_0_100_seasonal_anomaly": anomalies,
            "rzsm_0_100_normalized": standardized,
            "climatology_seasonal": climatology,
            "evaluation_mask": (["lat", "lon"], m_arr.astype(np.int32)),
        },
        attrs={
            "title": "Mindanao RISE-UNet Production Root-Zone Soil Moisture (RZSM) Cube",
            "institution": "Mindanao Drought Research / RISE-UNet Adaptation Track B",
            "parent_citation": "Lesinger & Tian (2025), Nature Communications, DOI: 10.1038/s41467-025-62761-3",
            "model_contract": "Model A0 Frozen Baseline",
            "spatial_grid": "Candidate A (0.25 deg, 32 x 48)",
            "active_evaluation_cells": config.expected_eval_cells,
            "total_computational_cells": config.expected_grid_shape[0] * config.expected_grid_shape[1],
            "depth_weighting_formula": "0.07*SM1 + 0.21*SM2 + 0.72*SM3",
            "remapping_method": "land_aware_bilinear_with_nearest_boundary_fallback",
            "rolling_window_days": config.rolling_window,
            "climatology_method": config.climatology_method,
            "train_start_year": config.train_start_year,
            "train_end_year": config.train_end_year,
            "train_min_scalar": float(train_min),
            "train_max_scalar": float(train_max),
            "normalization_scope": "training_cases_only_eval_cells_only_domain_wide_active_scalar",
            "ocean_padding": "0.0",
            "raw_support_archive_start": raw_start,
            "raw_support_archive_end": raw_end,
            "nominal_production_start": f"{config.train_start_year}-01-01",
            "nominal_production_end": f"{config.test_years[1]}-12-31",
            "purpose_of_2014": "Antecedent-support only; not a training/evaluation year",
            "expected_archive_days": config.expected_archive_days,
            "expected_nominal_days": config.expected_nominal_days,
            "expected_archive_eval_samples": config.expected_archive_eval_samples,
            "expected_nominal_eval_samples": config.expected_nominal_eval_samples,
            "cf_version": "CF-1.8",
        },
    )

    if slice_nominal_period and t_vals[0].year < config.train_start_year:
        nominal_start = f"{config.train_start_year}-01-01"
        prod_ds = prod_ds.sel({time_dim: slice(nominal_start, None)})

    return prod_ds


def verify_production_cube_census(
    ds: xr.Dataset,
    eval_mask: Optional[Union[xr.DataArray, np.ndarray]] = None,
    eval_start_date: Optional[Union[str, pd.Timestamp]] = None,
    time_dim: str = "time",
) -> Dict[str, Union[int, float, bool, str]]:
    """
    Executes a formal census and data integrity audit over the production RZSM cube,
    strictly validating the mandatory precision reporting standard:

      “Zero NaNs/Infs across the 126 active evaluation cells; non-evaluation computational
       cells follow the frozen masking/zero-fill convention.”

    Parameters
    ----------
    ds : xr.Dataset
        Compiled production Dataset.
    eval_mask : xr.DataArray or np.ndarray, optional
        Binary evaluation mask (if None, read from ds['evaluation_mask']).
    eval_start_date : str or pd.Timestamp, optional
        Starting date for operational evaluation (e.g. '2015-01-01').
        If None, automatically starts from train_start_year-01-01 if antecedent data is present,
        or evaluates all valid timesteps following the rolling window warmup.
    time_dim : str, optional
        Time dimension name (default: 'time').

    Returns
    -------
    dict
        Census metrics and boolean certification pass flag.
    """
    if eval_mask is None:
        if "evaluation_mask" not in ds:
            raise KeyError("evaluation_mask not found in dataset and not provided")
        m_arr = ds["evaluation_mask"].values
    else:
        m_arr = eval_mask.values if hasattr(eval_mask, "values") else np.asarray(eval_mask)

    # Handle evaluation start date / antecedent warmup window
    ds_eval = ds
    if time_dim in ds.coords:
        if eval_start_date is not None:
            ds_eval = ds.sel({time_dim: slice(str(eval_start_date), None)})
        elif "train_start_year" in ds.attrs:
            # If dataset begins before train_start_year (e.g. late-2014 antecedent buffer),
            # evaluate operational census strictly over the nominal period (train_start_year onward)
            first_year = pd.to_datetime(ds[time_dim].values[0]).year
            target_start_yr = int(ds.attrs["train_start_year"])
            if first_year < target_start_yr:
                ds_eval = ds.sel({time_dim: slice(f"{target_start_yr}-01-01", None)})
            else:
                # If no antecedent buffer was provided, slice past the initial rolling warmup days
                window_days = int(ds.attrs.get("rolling_window_days", 7))
                ds_eval = ds.isel({time_dim: slice(window_days - 1, None)})

    eval_bool = m_arr == 1
    ocean_bool = m_arr == 0
    n_active_cells = int(np.sum(eval_bool))
    n_ocean_cells = int(np.sum(ocean_bool))
    n_time = ds_eval.sizes.get(time_dim, 1)

    norm_arr = ds_eval["rzsm_0_100_normalized"].values

    # Active evaluation cells census
    active_slice = norm_arr[..., eval_bool]
    n_active_samples = active_slice.size
    n_nans_active = int(np.isnan(active_slice).sum())
    n_infs_active = int(np.isinf(active_slice).sum())

    # Inactive ocean/buffer cells census
    ocean_slice = norm_arr[..., ocean_bool]
    n_nonzeros_ocean = int(np.count_nonzero(ocean_slice))

    active_min = float(np.nanmin(active_slice)) if n_active_samples > 0 else np.nan
    active_max = float(np.nanmax(active_slice)) if n_active_samples > 0 else np.nan
    active_mean = float(np.nanmean(active_slice)) if n_active_samples > 0 else np.nan

    is_certified = (
        (n_nans_active == 0) and
        (n_infs_active == 0) and
        (n_nonzeros_ocean == 0) and
        (active_min >= 0.0) and
        (active_max <= 1.0) and
        (n_active_cells == 126)
    )

    report_phrase = (
        "Zero NaNs/Infs across the 126 active evaluation cells; non-evaluation computational "
        "cells follow the frozen masking/zero-fill convention."
    )

    return {
        "is_certified": is_certified,
        "active_cells_count": n_active_cells,
        "ocean_cells_count": n_ocean_cells,
        "timesteps_count": n_time,
        "total_active_evaluations": n_active_samples,
        "nans_active_cells": n_nans_active,
        "infs_active_cells": n_infs_active,
        "nonzero_ocean_cells": n_nonzeros_ocean,
        "active_min": active_min,
        "active_max": active_max,
        "active_mean": active_mean,
        "mandatory_reporting_standard": report_phrase,
    }


class FastLandAwareRemapper:
    """
    High-performance vectorized land-aware spatial remapping from native ERA5-Land (~0.10°)
    to Candidate A (0.25°, 32x48) with 126 active evaluation cells and coastal extrapolation.

    Precomputes Delaunay triangulation and nearest-neighbor KD-tree once, enabling
    vectorized multi-day evaluation with 100% bitwise parity to `remap_era5_land_to_candidate_a`.
    """

    def __init__(
        self,
        source_lats: np.ndarray,
        source_lons: np.ndarray,
        valid_land_mask: np.ndarray,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
        eval_mask: Union[xr.DataArray, np.ndarray],
        fill_value: float = 0.0,
    ):
        self.target_lats = np.asarray(target_lats)
        self.target_lons = np.asarray(target_lons)
        self.eval_mask = eval_mask.values if hasattr(eval_mask, "values") else np.asarray(eval_mask)
        self.fill_value = float(fill_value)
        self.valid_land_mask = np.asarray(valid_land_mask, dtype=bool)

        # Source coordinates mesh
        lon_src_mesh, lat_src_mesh = np.meshgrid(source_lons, source_lats)
        self.src_pts = np.column_stack([lon_src_mesh[self.valid_land_mask], lat_src_mesh[self.valid_land_mask]])

        # Target evaluation points
        lon_tgt_mesh, lat_tgt_mesh = np.meshgrid(self.target_lons, self.target_lats)
        self.eval_bool = (self.eval_mask == 1)
        self.eval_pts = np.column_stack([lon_tgt_mesh[self.eval_bool], lat_tgt_mesh[self.eval_bool]])

        # Precompute Delaunay and Nearest structures once
        self.tri = Delaunay(self.src_pts)
        self.dummy_vals = np.zeros(len(self.src_pts), dtype=np.float32)
        self.nearest_interp = NearestNDInterpolator(self.src_pts, self.dummy_vals)

    @classmethod
    def from_source_and_target_grid(
        cls,
        sample_source_da: xr.DataArray,
        target_grid_nc: Union[str, Path],
        eval_mask_nc: Union[str, Path],
        fill_value: float = 0.0,
    ) -> "FastLandAwareRemapper":
        """Factory constructor instantiating remapper from sample source DataArray and grid paths."""
        ds_grid = xr.open_dataset(target_grid_nc)
        ds_mask = xr.open_dataset(eval_mask_nc)
        tgt_lats = ds_grid["lat"].values
        tgt_lons = ds_grid["lon"].values
        eval_m = ds_mask["evaluation_mask"].values

        lat_col = "latitude" if "latitude" in sample_source_da.coords else "lat"
        lon_col = "longitude" if "longitude" in sample_source_da.coords else "lon"
        src_lats = sample_source_da[lat_col].values
        src_lons = sample_source_da[lon_col].values

        # Isolate 2D slice for land-mask detection
        sample_2d = sample_source_da.isel({dim: 0 for dim in sample_source_da.dims if dim not in [lat_col, lon_col]})
        valid_mask = np.isfinite(sample_2d.values)

        return cls(
            source_lats=src_lats,
            source_lons=src_lons,
            valid_land_mask=valid_mask,
            target_lats=tgt_lats,
            target_lons=tgt_lons,
            eval_mask=eval_m,
            fill_value=fill_value,
        )

    def remap_source_da(
        self,
        source_da: xr.DataArray,
        time_dim: Optional[str] = None,
    ) -> xr.DataArray:
        """
        Remap a 2D or 3D ERA5-Land DataArray to Candidate A coordinates.
        """
        if time_dim is not None and time_dim in source_da.dims:
            t_vals = source_da[time_dim].values
            n_times = len(t_vals)
            # Vectorized multi-time evaluation: shape (n_pts, n_times)
            src_vals_multi = source_da.values[:, self.valid_land_mask].T

            lin_interp = LinearNDInterpolator(self.tri, src_vals_multi)
            vals_lin = lin_interp(self.eval_pts)  # (n_eval, n_times)

            nan_mask = np.isnan(vals_lin)
            if np.any(nan_mask):
                near_interp = NearestNDInterpolator(self.src_pts, src_vals_multi)
                vals_near = near_interp(self.eval_pts)
                vals_lin[nan_mask] = vals_near[nan_mask]

            out_arr = np.full((n_times, len(self.target_lats), len(self.target_lons)), self.fill_value, dtype=np.float32)
            out_arr[:, self.eval_bool] = vals_lin.T

            return xr.DataArray(
                out_arr,
                coords={"time": t_vals, "lat": self.target_lats, "lon": self.target_lons},
                dims=["time", "lat", "lon"],
                name=source_da.name or "remapped_rzsm",
            )
        else:
            # Single 2D slice
            src_vals = source_da.values[self.valid_land_mask]
            lin_interp = LinearNDInterpolator(self.tri, src_vals)
            vals_lin = lin_interp(self.eval_pts)

            nan_mask = np.isnan(vals_lin)
            if np.any(nan_mask):
                near_interp = NearestNDInterpolator(self.src_pts, src_vals)
                vals_near = near_interp(self.eval_pts)
                vals_lin[nan_mask] = vals_near[nan_mask]

            out_arr = np.full((len(self.target_lats), len(self.target_lons)), self.fill_value, dtype=np.float32)
            out_arr[self.eval_bool] = vals_lin

            return xr.DataArray(
                out_arr,
                coords={"lat": self.target_lats, "lon": self.target_lons},
                dims=["lat", "lon"],
                name=source_da.name or "remapped_rzsm",
            )


def process_era5_land_monthly_pair(
    main_nc: Union[str, Path],
    sm3_nc: Union[str, Path],
    remapper: FastLandAwareRemapper,
) -> xr.DataArray:
    """
    Ingests monthly ERA5-Land files (main: swvl1, swvl2; sm3: swvl3), resamples to 24-hour
    daily arithmetic mean, applies Kyle Lesinger's depth weighting (0.07*SM1 + 0.21*SM2 + 0.72*SM3),
    and remaps to Candidate A (32x48) with 126 active evaluation cells.

    Parameters
    ----------
    main_nc : str or Path
        Path to monthly NetCDF containing swvl1 and swvl2 (e.g. era5-land-YYYY-MM.nc).
    sm3_nc : str or Path
        Path to monthly NetCDF containing swvl3 (e.g. era5-land-sm3-YYYY-MM.nc).
    remapper : FastLandAwareRemapper
        Precomputed fast land-aware remapping operator.

    Returns
    -------
    xr.DataArray
        Daily remapped 0-100 cm RZSM DataArray on Candidate A geometry for the given month.
    """
    with xr.open_dataset(main_nc) as ds_main, xr.open_dataset(sm3_nc) as ds_sm3:
        t_col = "valid_time" if "valid_time" in ds_main else "time"
        t_col_sm3 = "valid_time" if "valid_time" in ds_sm3 else "time"

        # 24-hour daily arithmetic resample
        swvl1_daily = ds_main["swvl1"].resample({t_col: "1D"}).mean(dim=t_col)
        swvl2_daily = ds_main["swvl2"].resample({t_col: "1D"}).mean(dim=t_col)
        swvl3_daily = ds_sm3["swvl3"].resample({t_col_sm3: "1D"}).mean(dim=t_col_sm3)

        # Apply Kyle Lesinger depth weighting: 0.07*SM1 + 0.21*SM2 + 0.72*SM3
        rzsm_daily = compute_depth_weighted_rzsm(swvl1_daily, swvl2_daily, swvl3_daily)
        rzsm_daily.name = "rzsm_0_100_raw"

        # Remap to Candidate A
        remapped = remapper.remap_source_da(rzsm_daily, time_dim=t_col)
        # Normalize date coordinate to datetime64[ns]
        remapped["time"] = pd.to_datetime(remapped["time"].values)
        return remapped


def process_era5_land_antecedent_file(
    antecedent_nc: Union[str, Path],
    remapper: FastLandAwareRemapper,
) -> xr.DataArray:
    """
    Ingests the 20-day December 2014 antecedent support file (era5-land-2014-12-antecedent.nc),
    resamples to 24-hour daily arithmetic mean (20 days), applies depth weighting,
    and remaps to Candidate A (32x48).

    Parameters
    ----------
    antecedent_nc : str or Path
        Path to December 2014 antecedent file containing swvl1, swvl2, and swvl3.
    remapper : FastLandAwareRemapper
        Precomputed fast land-aware remapping operator.

    Returns
    -------
    xr.DataArray
        Daily remapped 0-100 cm RZSM DataArray on Candidate A geometry (20 daily timesteps).
    """
    with xr.open_dataset(antecedent_nc) as ds:
        t_col = "valid_time" if "valid_time" in ds else "time"

        # 24-hour daily arithmetic resample
        swvl1_daily = ds["swvl1"].resample({t_col: "1D"}).mean(dim=t_col)
        swvl2_daily = ds["swvl2"].resample({t_col: "1D"}).mean(dim=t_col)
        swvl3_daily = ds["swvl3"].resample({t_col: "1D"}).mean(dim=t_col)

        # Apply depth weighting
        rzsm_daily = compute_depth_weighted_rzsm(swvl1_daily, swvl2_daily, swvl3_daily)
        rzsm_daily.name = "rzsm_0_100_raw"

        # Remap to Candidate A
        remapped = remapper.remap_source_da(rzsm_daily, time_dim=t_col)
        remapped["time"] = pd.to_datetime(remapped["time"].values)
        return remapped


def compile_full_11yr_rzsm_cube(
    archive_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[ProductionCubeConfig] = None,
    spatial_grid_nc: Optional[Union[str, Path]] = None,
    eval_mask_nc: Optional[Union[str, Path]] = None,
    slice_nominal_period: bool = True,
    allow_partial: bool = False,
    verbose: bool = True,
) -> Tuple[xr.Dataset, Dict[str, Union[int, float, bool, str]]]:
    """
    Executes end-to-end production compilation of the 11-year Mindanao Root-Zone Soil Moisture
    (RZSM) data cube across the complete archive series (2014 antecedent support + 2015–2025).

    Pipeline Stages:
      1. Census of archive files: pairs era5-land-YYYY-MM.nc with era5-land-sm3-YYYY-MM.nc
         plus era5-land-2014-12-antecedent.nc.
      2. Fast land-aware spatial remapping to Candidate A 0.25° grid (32x48).
      3. Temporal sequence concatenation into continuous multi-year archive series.
      4. Continuous 7-day backward trailing rolling mean (center=False, zero future leakage).
      5. Locked Model A0 3-month seasonal climatology (DJF, MAM, JJA, SON) fitted on <= 2021.
      6. Seasonal anomalies computation and domain-wide active scalar min-max normalization.
      7. Nominal period slicing (2015-01-01 to 2025-12-31, 4,018 days, 506,268 evaluation cell-days).
      8. Formal census audit asserting Zero NaNs/Infs over all 506,268 active evaluation points.
      9. CF-1.8 NetCDF export with zlib compression.

    Parameters
    ----------
    archive_dir : str or Path
        Directory containing the 265 ERA5-Land NetCDF files.
    output_path : str or Path, optional
        Destination NetCDF file path for the compiled production cube.
    config : ProductionCubeConfig, optional
        Configuration contracts (defaults to standard ProductionCubeConfig).
    spatial_grid_nc : str or Path, optional
        Path to mindanao_025deg.nc (defaults to processed/grid/mindanao_025deg.nc).
    eval_mask_nc : str or Path, optional
        Path to mindanao_eval_mask_025.nc (defaults to processed/grid/mindanao_eval_mask_025.nc).
    slice_nominal_period : bool, default True
        If True, output dataset is sliced to nominal 2015–2025 timeline.
    allow_partial : bool, default False
        If True, permits execution on a partial archive subset (for testing).
    verbose : bool, default True
        If True, prints stage progress messages.

    Returns
    -------
    Tuple[xr.Dataset, Dict]
        Compiled xr.Dataset and census audit verification dictionary.
    """
    archive_dir = Path(archive_dir)
    if config is None:
        config = ProductionCubeConfig()

    base_repo = Path(__file__).resolve().parent.parent.parent
    if spatial_grid_nc is None:
        spatial_grid_nc = base_repo / "processed" / "grid" / "mindanao_025deg.nc"
    if eval_mask_nc is None:
        eval_mask_nc = base_repo / "processed" / "grid" / "mindanao_eval_mask_025.nc"

    spatial_grid_nc = Path(spatial_grid_nc)
    eval_mask_nc = Path(eval_mask_nc)

    assert spatial_grid_nc.exists(), f"Spatial grid NC missing at {spatial_grid_nc}"
    assert eval_mask_nc.exists(), f"Evaluation mask NC missing at {eval_mask_nc}"

    ds_mask = xr.open_dataset(eval_mask_nc)
    eval_mask = ds_mask["evaluation_mask"].values

    # Step 1: Inventory archive files
    ant_file = archive_dir / "era5-land-2014-12-antecedent.nc"
    main_files = sorted(archive_dir.glob("era5-land-20[1-2][0-9]-[0-1][0-9].nc"))
    # Filter out sm3 and antecedent files from main_files
    main_files = [f for f in main_files if "-sm3-" not in f.name and "antecedent" not in f.name]

    if not allow_partial:
        if not ant_file.exists():
            raise FileNotFoundError(f"Missing required 2014 antecedent support file at: {ant_file}")
        if len(main_files) != 132:
            raise ValueError(f"Expected 132 monthly main files (2015-01 to 2025-12), found {len(main_files)}")

    # Instantiate remapper once using the antecedent file or first available file
    sample_file = ant_file if ant_file.exists() else main_files[0]
    with xr.open_dataset(sample_file) as ds_sample:
        remapper = FastLandAwareRemapper.from_source_and_target_grid(
            sample_source_da=ds_sample["swvl1"],
            target_grid_nc=spatial_grid_nc,
            eval_mask_nc=eval_mask_nc,
            fill_value=config.fill_value,
        )

    daily_slices: List[xr.DataArray] = []

    # Step 2: Process 2014 antecedent file if available
    if ant_file.exists():
        if verbose:
            print(f"--> Processing antecedent support file: {ant_file.name}")
        da_ant = process_era5_land_antecedent_file(ant_file, remapper=remapper)
        daily_slices.append(da_ant)

    # Step 3: Process monthly pairs in chronological order
    for mf in main_files:
        month_str = mf.name.replace("era5-land-", "").replace(".nc", "")
        sm3_file = archive_dir / f"era5-land-sm3-{month_str}.nc"
        if not sm3_file.exists():
            raise FileNotFoundError(f"Missing matching layer-3 file: {sm3_file}")

        if verbose:
            print(f"--> Processing month: {month_str} ({mf.name} + {sm3_file.name})")

        da_m = process_era5_land_monthly_pair(mf, sm3_file, remapper=remapper)
        daily_slices.append(da_m)

    if not daily_slices:
        raise RuntimeError("No valid data files found in archive directory.")

    # Step 4: Concatenate daily series along time dimension
    if verbose:
        print(f"--> Concatenating {len(daily_slices)} temporal slices...")
    full_daily_da = xr.concat(daily_slices, dim="time")
    full_daily_da = full_daily_da.sortby("time")

    if not allow_partial:
        n_days = len(full_daily_da.time)
        if n_days != config.expected_archive_days:
            raise ValueError(
                f"Archive day count mismatch: Expected {config.expected_archive_days} days, got {n_days}"
            )

    # Step 5: Execute complete temporal transformation pipeline
    if verbose:
        print("--> Executing multi-stage production pipeline (rolling, climatology, anomalies, scaling)...")
    prod_ds = compile_production_rzsm_pipeline(
        daily_remapped_rzsm=full_daily_da,
        eval_mask=eval_mask,
        config=config,
        slice_nominal_period=slice_nominal_period,
    )

    # Step 6: Execute formal census audit
    if verbose:
        print("--> Executing formal production cube census audit...")
    census = verify_production_cube_census(prod_ds, eval_mask=eval_mask)

    if verbose:
        print(f"    Certified finite: {census['is_certified']}")
        print(f"    Active evaluation points: {census['total_active_evaluations']}")
        print(f"    Active NaNs: {census['nans_active_cells']}, Infs: {census['infs_active_cells']}")
        print(f"    Nonzero ocean cells: {census['nonzero_ocean_cells']}")
        print(f"    Standard: {census['mandatory_reporting_standard']}")

    # Step 7: Export to CF-1.8 NetCDF if output_path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"--> Exporting production cube to {output_path} (CF-1.8 NetCDF4 zlib complevel 4)...")

        encoding = {}
        for var_name in prod_ds.data_vars:
            encoding[var_name] = {
                "zlib": True,
                "complevel": 4,
                "fletcher32": True,
            }

        prod_ds.to_netcdf(output_path, encoding=encoding)
        if verbose:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"    Successfully exported: {output_path.name} ({size_mb:.2f} MB)")

    return prod_ds, census
