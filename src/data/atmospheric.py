"""
atmospheric.py
--------------
Derives the 5 RISE-UNet daily surface atmospheric predictor channels from raw ERA5:
1. tmax:      Daily maximum of analyzed hourly 2m temperature [K]
2. diff_temp: Daily diurnal temperature difference (tmax - tmin) [K]
3. spfh:      Daily surface specific humidity derived from hourly d2m & sp via Bolton (1980) [kg/kg]
4. pwat:      Daily mean total column water vapour (tcwv) [kg/m²]
5. hgt_pres:  Daily mean 200 hPa geopotential height (z200 / 9.80665) [gpm]
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

G0 = 9.80665  # WMO standard gravitational acceleration (m/s²)
EPSILON = 0.622

def compute_specific_humidity(d2m_k: xr.DataArray, sp_pa: xr.DataArray) -> xr.DataArray:
    """
    Computes specific humidity q [kg/kg] from 2m dewpoint [K]
    and surface pressure [Pa] using Bolton (1980) / Tetens formulation.
    """
    td_c = d2m_k - 273.15
    e_pa = 611.2 * np.exp(17.67 * td_c / (td_c + 243.5))
    q = (EPSILON * e_pa) / (sp_pa - (1.0 - EPSILON) * e_pa)
    return q

def derive_era5_atmospheric_month(
    year: int,
    month: int,
    raw_atmos_dir: Path,
    grid_path: Path,
    eval_mask_path: Path,
    out_dir: Path,
    prefix: str = "era5_atmospheric",
) -> Path:
    """
    Derives 5 atmospheric channels for a specific month and writes to NetCDF.
    """
    month_str = f"{year}_{month:02d}"
    single_path = raw_atmos_dir / "single" / f"era5_single_levels_{month_str}.nc"
    pressure_path = raw_atmos_dir / "pressure" / f"era5_z200_{month_str}.nc"

    if not single_path.exists():
        raise FileNotFoundError(f"Missing raw single-level ERA5 file: {single_path}")
    if not pressure_path.exists():
        raise FileNotFoundError(f"Missing raw pressure-level ERA5 file: {pressure_path}")

    ds_single = xr.open_dataset(single_path)
    ds_pressure = xr.open_dataset(pressure_path)
    ds_grid = xr.open_dataset(grid_path)
    ds_mask = xr.open_dataset(eval_mask_path)

    # Coordinate standardization
    rename_single = {}
    if "valid_time" in ds_single.coords and "time" not in ds_single.coords:
        rename_single["valid_time"] = "time"
    if "latitude" in ds_single.coords:
        rename_single["latitude"] = "lat"
    if "longitude" in ds_single.coords:
        rename_single["longitude"] = "lon"
    if rename_single:
        ds_single = ds_single.rename(rename_single)

    rename_pressure = {}
    if "valid_time" in ds_pressure.coords and "time" not in ds_pressure.coords:
        rename_pressure["valid_time"] = "time"
    if "latitude" in ds_pressure.coords:
        rename_pressure["latitude"] = "lat"
    if "longitude" in ds_pressure.coords:
        rename_pressure["longitude"] = "lon"
    if rename_pressure:
        ds_pressure = ds_pressure.rename(rename_pressure)

    # Variable naming standardization
    var_rename = {}
    for name in ["2t", "t2m"]:
        if name in ds_single and "2m_temperature" not in ds_single:
            var_rename[name] = "2m_temperature"
    for name in ["2d", "d2m"]:
        if name in ds_single and "2m_dewpoint_temperature" not in ds_single:
            var_rename[name] = "2m_dewpoint_temperature"
    if "sp" in ds_single and "surface_pressure" not in ds_single:
        var_rename["sp"] = "surface_pressure"
    if "tcwv" in ds_single and "total_column_water_vapour" not in ds_single:
        var_rename["tcwv"] = "total_column_water_vapour"
    if var_rename:
        ds_single = ds_single.rename(var_rename)

    if "z" in ds_pressure and "geopotential" not in ds_pressure:
        ds_pressure = ds_pressure.rename({"z": "geopotential"})

    # Channel computation
    tmax = ds_single["2m_temperature"].resample(time="1D").max(dim="time")
    tmin = ds_single["2m_temperature"].resample(time="1D").min(dim="time")
    diff_temp = tmax - tmin

    hourly_q = compute_specific_humidity(ds_single["2m_dewpoint_temperature"], ds_single["surface_pressure"])
    spfh = hourly_q.resample(time="1D").mean(dim="time")
    pwat = ds_single["total_column_water_vapour"].resample(time="1D").mean(dim="time")

    z200 = ds_pressure["geopotential"]
    for dim_name in ["level", "pressure_level", "isobaricInhPa"]:
        if dim_name in z200.dims:
            z200 = z200.squeeze(dim_name)
    if len(z200.dims) > 3:
        z200 = z200.squeeze()
    hgt_pres = (z200 / G0).resample(time="1D").mean(dim="time")

    # Evaluation mask check
    mask_key = "evaluation_mask" if "evaluation_mask" in ds_mask else "eval_mask"
    eval_mask = ds_mask[mask_key].values.astype(bool)
    assert np.sum(eval_mask) == 126, "Expected 126 evaluation cells"

    # Assemble dataset
    out_ds = xr.Dataset(
        data_vars={
            "tmax": (("time", "lat", "lon"), tmax.values, {"units": "K"}),
            "diff_temp": (("time", "lat", "lon"), diff_temp.values, {"units": "K"}),
            "spfh": (("time", "lat", "lon"), spfh.values, {"units": "kg/kg"}),
            "pwat": (("time", "lat", "lon"), pwat.values, {"units": "kg/m**2"}),
            "hgt_pres": (("time", "lat", "lon"), hgt_pres.values, {"units": "gpm"}),
            "tmin_diagnostic": (("time", "lat", "lon"), tmin.values, {"units": "K"}),
        },
        coords={
            "time": tmax.time,
            "lat": ds_grid.lat,
            "lon": ds_grid.lon,
        },
        attrs={
            "title": f"Derived ERA5 Atmospheric Predictors ({year}-{month:02d}) for Mindanao RISE-UNet",
            "spatial_resolution": "0.25 degree Candidate A grid (32 x 48)",
            "evaluation_cells": 126,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / f"{prefix}_{month_str}.nc"
    out_ds.to_netcdf(out_nc)
    return out_nc
