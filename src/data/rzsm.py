"""
src/data/rzsm.py
----------------
Core Root-Zone Soil Moisture (RZSM) computation and preprocessing transformations
for the RISE-UNet Mindanao regional adaptation (Track B).

Formulation & Methodology:
  - Depth-weighted integration of ERA5-Land volumetric soil water layers (top 100 cm):
      RZSM_0_100 = 0.07 * swvl1 + 0.21 * swvl2 + 0.72 * swvl3
    where:
      swvl1: Layer 1 (0 to 7 cm depth)   -> Weight: 7 / 100 = 0.07
      swvl2: Layer 2 (7 to 28 cm depth)  -> Weight: 21 / 100 = 0.21
      swvl3: Layer 3 (28 to 100 cm depth)-> Weight: 72 / 100 = 0.72
  - Temporal preprocessing:
      - 7-day backward trailing rolling mean (center=False).
      - 3-lag antecedent state extraction (offsets: -1, -7, -14 days).
      - Binary evaluation masking (126 active Mindanao land cells, 1,410 ocean cells zero-filled).
      - Min-max standardization to [0, 1] fitted strictly on training years.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata, NearestNDInterpolator

# Canonical layer thickness weights for top 100 cm RZSM
LAYER_WEIGHTS: Dict[str, float] = {
    "layer1": 0.07,  # 0 to 7 cm
    "layer2": 0.21,  # 7 to 28 cm
    "layer3": 0.72,  # 28 to 100 cm
}

DEFAULT_LAGS: Tuple[int, ...] = (-1, -7, -14)


def compute_depth_weighted_rzsm(
    sm1: Union[xr.DataArray, np.ndarray],
    sm2: Union[xr.DataArray, np.ndarray],
    sm3: Union[xr.DataArray, np.ndarray],
    w1: float = 0.07,
    w2: float = 0.21,
    w3: float = 0.72,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Computes depth-weighted root-zone soil moisture (0 to 100 cm) from ERA5-Land
    volumetric soil water layers 1, 2, and 3.

    Formula:
        RZSM_0_100 = w1 * sm1 + w2 * sm2 + w3 * sm3

    Parameters
    ----------
    sm1 : xr.DataArray or np.ndarray
        Volumetric soil water layer 1 (0-7 cm) [m3/m3].
    sm2 : xr.DataArray or np.ndarray
        Volumetric soil water layer 2 (7-28 cm) [m3/m3].
    sm3 : xr.DataArray or np.ndarray
        Volumetric soil water layer 3 (28-100 cm) [m3/m3].
    w1 : float, optional
        Weight for layer 1 (default: 0.07).
    w2 : float, optional
        Weight for layer 2 (default: 0.21).
    w3 : float, optional
        Weight for layer 3 (default: 0.72).

    Returns
    -------
    xr.DataArray or np.ndarray
        Integrated 0-100 cm RZSM array matching the input type and coordinates.
    """
    weight_sum = w1 + w2 + w3
    if not np.isclose(weight_sum, 1.0, atol=1e-5):
        raise ValueError(f"Layer weights must sum to 1.0, got {weight_sum:.6f}")

    rzsm = w1 * sm1 + w2 * sm2 + w3 * sm3

    if isinstance(rzsm, xr.DataArray):
        rzsm.name = "rzsm_0_100"
        rzsm.attrs["long_name"] = "Root-Zone Soil Moisture (0-100 cm depth-weighted)"
        rzsm.attrs["units"] = "m3/m3"
        rzsm.attrs["formula"] = f"RZSM = {w1}*sm1 + {w2}*sm2 + {w3}*sm3"

    return rzsm


def compute_backward_rolling_mean(
    data: xr.DataArray,
    window: int = 7,
    min_periods: int = 7,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Applies an unweighted backward-looking trailing rolling window mean
    along the time dimension (center=False).

    Parameters
    ----------
    data : xr.DataArray
        Input time-series DataArray.
    window : int, optional
        Rolling window size in timesteps/days (default: 7).
    min_periods : int, optional
        Minimum number of valid observations required in window (default: 7).
    time_dim : str, optional
        Name of the time dimension (default: 'time').

    Returns
    -------
    xr.DataArray
        Smoothed DataArray with trailing rolling mean applied.
    """
    if time_dim not in data.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in DataArray dims: {data.dims}")

    return data.rolling({time_dim: window}, min_periods=min_periods, center=False).mean()


def extract_antecedent_lags(
    data: xr.DataArray,
    target_date: Union[str, pd.Timestamp, np.datetime64],
    lag_days: Tuple[int, ...] = DEFAULT_LAGS,
    time_dim: str = "time",
) -> Dict[int, xr.DataArray]:
    """
    Extracts antecedent lagged states relative to a specific target initiation date.

    Following the verified parent EX29 contract, lags represent trailing 7-day averages
    sampled at relative day offsets [-1, -7, -14] days prior to forecast issue.

    Parameters
    ----------
    data : xr.DataArray
        Daily rolling-mean DataArray indexed by time.
    target_date : str, pd.Timestamp, or np.datetime64
        Forecast initialization date (Day 0).
    lag_days : tuple of int, optional
        Relative day offsets for lags (default: (-1, -7, -14)).
    time_dim : str, optional
        Time dimension name (default: 'time').

    Returns
    -------
    dict
        Dictionary mapping lag offset (e.g. -1, -7, -14) to the 2D spatial slice.
    """
    t0 = pd.to_datetime(target_date)
    lags_dict: Dict[int, xr.DataArray] = {}

    for lag in lag_days:
        offset_date = t0 + pd.Timedelta(days=lag)
        date_str = offset_date.strftime("%Y-%m-%d")
        try:
            slice_2d = data.sel({time_dim: date_str})
            if time_dim in slice_2d.dims:
                slice_2d = slice_2d.squeeze(time_dim)
            lags_dict[lag] = slice_2d
        except KeyError:
            raise KeyError(
                f"Lag date {date_str} (lag={lag}d from {t0.strftime('%Y-%m-%d')}) "
                f"not found in {time_dim} coordinates."
            )

    return lags_dict


def apply_land_mask(
    data: Union[xr.DataArray, np.ndarray],
    mask: Union[xr.DataArray, np.ndarray],
    fill_value: float = 0.0,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Applies the binary land evaluation mask to an RZSM spatial array.
    Active land cells (mask == 1) retain their values; non-evaluation / ocean
    cells (mask == 0) are padded with `fill_value`.

    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Array containing spatial dimensions (lat, lon).
    mask : xr.DataArray or np.ndarray
        Binary mask (1 = active evaluation land, 0 = ocean/buffer).
    fill_value : float, optional
        Value to assign to inactive cells (default: 0.0).

    Returns
    -------
    xr.DataArray or np.ndarray
        Masked array with inactive cells set to `fill_value`.
    """
    if isinstance(data, xr.DataArray) and isinstance(mask, xr.DataArray):
        # Harmonize coordinate names if needed
        m = mask
        if "lat" in data.dims and "latitude" in m.dims:
            m = m.rename({"latitude": "lat", "longitude": "lon"})
        masked = xr.where(m == 1, data, fill_value)
        return masked.transpose(*data.dims)
    elif isinstance(data, np.ndarray):
        m_arr = mask.values if hasattr(mask, "values") else np.asarray(mask)
        out = data.copy()
        out[..., m_arr == 0] = fill_value
        return out
    else:
        # Hybrid case: data is xr.DataArray, mask is ndarray
        m_arr = mask.values if hasattr(mask, "values") else np.asarray(mask)
        masked = xr.where(m_arr == 1, data, fill_value)
        return masked.transpose(*data.dims)


def compute_min_max_scale(
    data: Union[xr.DataArray, np.ndarray],
    mask: Optional[Union[xr.DataArray, np.ndarray]] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    fill_value: float = 0.0,
) -> Tuple[Union[xr.DataArray, np.ndarray], float, float]:
    """
    Min-max standardizes an active land array to the unit interval [0, 1]
    using strictly active evaluation land cell statistics:

        x_scaled = (x - min_val) / (max_val - min_val)

    Ocean / non-evaluation cells are explicitly set to `fill_value`.

    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Input spatial or spatio-temporal data array.
    mask : xr.DataArray or np.ndarray, optional
        Binary mask where 1 indicates valid evaluation cells.
    min_val : float, optional
        Pre-computed minimum value (e.g. from training fold). If None, computed from data.
    max_val : float, optional
        Pre-computed maximum value (e.g. from training fold). If None, computed from data.
    fill_value : float, optional
        Value assigned to masked/ocean cells (default: 0.0).

    Returns
    -------
    tuple of (scaled_data, min_val, max_val)
        The standardized array and the (min_val, max_val) bounds used.
    """
    # Extract values for min/max computation over active evaluation cells
    if min_val is None or max_val is None:
        if mask is not None:
            m_bool = (mask.values if hasattr(mask, "values") else np.asarray(mask)) == 1
            arr = data.values if hasattr(data, "values") else np.asarray(data)
            valid_vals = arr[..., m_bool]
            computed_min = float(np.nanmin(valid_vals))
            computed_max = float(np.nanmax(valid_vals))
        else:
            arr = data.values if hasattr(data, "values") else np.asarray(data)
            computed_min = float(np.nanmin(arr))
            computed_max = float(np.nanmax(arr))

        min_val = computed_min if min_val is None else min_val
        max_val = computed_max if max_val is None else max_val

    denom = max_val - min_val
    if np.isclose(denom, 0.0):
        raise ValueError(f"Degenerate min-max range: max ({max_val}) equals min ({min_val})")

    scaled = (data - min_val) / denom

    # Enforce masking on inactive cells
    if mask is not None:
        scaled = apply_land_mask(scaled, mask, fill_value=fill_value)

    return scaled, min_val, max_val


def remap_era5_land_to_candidate_a(
    source_da: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    eval_mask: Union[xr.DataArray, np.ndarray],
    fill_value: float = 0.0,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Executes land-aware linear spatial interpolation (piecewise-linear Delaunay triangulation
    with nearest-neighbor fallback) from ERA5-Land native grid (~0.10 deg) to the frozen
    Candidate A reference grid (0.25 deg, 32 x 48).

    Methodological Parity & Coastal Behavior:
      1. Source Points: Extracted strictly where source soil moisture observations are finite
         (non-NaN valid land points). Ocean cells in ERA5-Land are NaNs.
      2. Piecewise-Linear Interpolation: Evaluated on target evaluation points (cells where eval_mask == 1)
         via Delaunay simplex barycentric interpolation.
      3. Coastal Extrapolation Fallback: For target coastal edge cells outside the convex hull of valid land,
         NearestNDInterpolator provides fallback extrapolation from adjacent valid land observations,
         completely eliminating peninsula truncation.
      4. Masking & Padding: 126 active evaluation cells receive interpolated values; all 1,410
         computational buffer/ocean cells are padded strictly to `fill_value` (default: 0.0).
      5. Guarantees Zero NaNs/Infs across all 126 active evaluation cells.


    Parameters
    ----------
    source_da : xr.DataArray
        ERA5-Land DataArray with dimensions (time, lat, lon) or (lat, lon).
    target_lats : np.ndarray
        1D array of 32 latitude coordinates (11.75 to 4.00, descending).
    target_lons : np.ndarray
        1D array of 48 longitude coordinates (116.00 to 127.75, ascending).
    eval_mask : xr.DataArray or np.ndarray
        2D binary evaluation mask (32, 48) where 1 indicates active evaluation land cells.
    fill_value : float, optional
        Value assigned to inactive buffer/ocean cells (default: 0.0).
    time_dim : str, optional
        Name of time dimension if 3D (default: 'time').

    Returns
    -------
    xr.DataArray
        Remapped DataArray on Candidate A geometry (time, 32, 48) or (32, 48).
    """
    m_arr = eval_mask.values if hasattr(eval_mask, "values") else np.asarray(eval_mask)
    if m_arr.shape != (len(target_lats), len(target_lons)):
        raise ValueError(
            f"eval_mask shape {m_arr.shape} does not match target grid "
            f"({len(target_lats)}, {len(target_lons)})"
        )

    # Detect source coordinate names
    lat_name = "latitude" if "latitude" in source_da.coords else "lat"
    lon_name = "longitude" if "longitude" in source_da.coords else "lon"

    src_lat = source_da[lat_name].values
    src_lon = source_da[lon_name].values
    lon_src_mesh, lat_src_mesh = np.meshgrid(src_lon, src_lat)
    lon_tgt_mesh, lat_tgt_mesh = np.meshgrid(target_lons, target_lats)

    eval_bool = m_arr == 1
    eval_pts = np.column_stack([lon_tgt_mesh[eval_bool], lat_tgt_mesh[eval_bool]])

    is_3d = time_dim in source_da.dims
    if is_3d:
        n_times = source_da.sizes[time_dim]
        out_arr = np.full((n_times, len(target_lats), len(target_lons)), fill_value, dtype=np.float32)

        for t in range(n_times):
            slice_2d = source_da.isel({time_dim: t}).values
            valid_src = np.isfinite(slice_2d)
            if not np.any(valid_src):
                continue
            src_pts = np.column_stack([lon_src_mesh[valid_src], lat_src_mesh[valid_src]])
            src_vals = slice_2d[valid_src]

            linear_v = griddata(src_pts, src_vals, eval_pts, method="linear")
            if np.isnan(linear_v).any():
                nearest_interp = NearestNDInterpolator(src_pts, src_vals)
                nearest_v = nearest_interp(eval_pts)
                interp_v = np.where(np.isfinite(linear_v), linear_v, nearest_v)
            else:
                interp_v = linear_v

            out_arr[t, eval_bool] = interp_v

        coords = {
            time_dim: source_da[time_dim],
            "lat": target_lats,
            "lon": target_lons,
        }
        dims = [time_dim, "lat", "lon"]
    else:
        out_arr = np.full((len(target_lats), len(target_lons)), fill_value, dtype=np.float32)
        valid_src = np.isfinite(source_da.values)
        src_pts = np.column_stack([lon_src_mesh[valid_src], lat_src_mesh[valid_src]])
        src_vals = source_da.values[valid_src]

        linear_v = griddata(src_pts, src_vals, eval_pts, method="linear")
        if np.isnan(linear_v).any():
            nearest_interp = NearestNDInterpolator(src_pts, src_vals)
            nearest_v = nearest_interp(eval_pts)
            interp_v = np.where(np.isfinite(linear_v), linear_v, nearest_v)
        else:
            interp_v = linear_v

        out_arr[eval_bool] = interp_v
        coords = {"lat": target_lats, "lon": target_lons}
        dims = ["lat", "lon"]

    remapped_da = xr.DataArray(out_arr, coords=coords, dims=dims, name=source_da.name)
    remapped_da.attrs.update(source_da.attrs)
    remapped_da.attrs["remapping_method"] = "land_aware_linear_with_nearest_boundary_fallback"
    remapped_da.attrs["grid"] = "Candidate_A_025deg_32x48"
    return remapped_da

