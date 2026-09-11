#!/usr/bin/env python3
"""
verify_and_derive_era5_pilot.py
--------------------------------
Validates raw ERA5 atmospheric fields for January 2015, executes the
mathematical derivation of the 5 RISE-UNet atmospheric predictor channels,
verifies tensor dimensions (31, 32, 48) and mask completeness over
the 126 Mindanao evaluation cells, and exports the verified pilot dataset.

Authoritative Channel Derivations:
  1. tmax:      Daily maximum of analyzed hourly 2m temperature (2m_temperature) [K]
  2. diff_temp: Daily diurnal temperature difference (tmax - tmin) [K]
  3. spfh:      Daily surface specific humidity derived from hourly d2m & sp via Bolton (1980) [kg/kg]
  4. pwat:      Daily mean total column water vapour (tcwv) [kg/m²]
  5. hgt_pres:  Daily mean 200 hPa geopotential height (z / 9.80665) [gpm]
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Gravitational acceleration constant (WMO standard)
G0 = 9.80665

def compute_specific_humidity(d2m_k: xr.DataArray, sp_pa: xr.DataArray) -> xr.DataArray:
    """
    Computes specific humidity q [kg/kg] from 2m dewpoint temperature [K]
    and surface pressure [Pa] using Bolton (1980) / Tetens formulation.
    """
    # Convert dewpoint to Celsius
    td_c = d2m_k - 273.15
    # Actual vapor pressure e in Pa (Bolton 1980, eq. 10)
    e_pa = 611.2 * np.exp(17.67 * td_c / (td_c + 243.5))
    # Specific humidity q = (epsilon * e) / (p - (1 - epsilon) * e)
    epsilon = 0.622
    q = (epsilon * e_pa) / (sp_pa - (1.0 - epsilon) * e_pa)
    return q

def main():
    print("=================================================================")
    print("Sub-Phase 21C: ERA5 Atmospheric Pilot Verification & Derivation")
    print("Target Month: January 2015 (31 days)")
    print("=================================================================\n")

    single_path = Path("Data/raw_downloads/ERA5_atmospheric/single/era5_single_levels_2015_01.nc")
    single_annual_path = Path("Data/raw_downloads/ERA5_atmospheric/single/era5_single_levels_2015.nc")
    pressure_path = Path("Data/raw_downloads/ERA5_atmospheric/pressure/era5_z200_2015_01.nc")
    pressure_annual_path = Path("Data/raw_downloads/ERA5_atmospheric/pressure/era5_z200_2015.nc")
    mask_path = Path("processed/grid/mindanao_eval_mask_025.nc")
    grid_path = Path("processed/grid/mindanao_025deg.nc")

    # Step 1: Check raw file existence
    if single_path.exists():
        actual_single = single_path
    elif single_annual_path.exists():
        actual_single = single_annual_path
    else:
        print(f"[ERROR] Single-level raw file not found: checked {single_path} and {single_annual_path}")
        sys.exit(1)

    if pressure_path.exists():
        actual_pressure = pressure_path
    elif pressure_annual_path.exists():
        actual_pressure = pressure_annual_path
    else:
        print(f"[ERROR] Pressure-level raw file not found: checked {pressure_path} and {pressure_annual_path}")
        sys.exit(1)

    if not mask_path.exists():
        print(f"[ERROR] Evaluation mask not found: {mask_path}")
        sys.exit(1)

    print(f"[RAW FILE CHECK]")
    print(f"  Single-levels   : {actual_single} ({actual_single.stat().st_size / 1e6:.2f} MB)")
    print(f"  Pressure-levels : {actual_pressure} ({actual_pressure.stat().st_size / 1e6:.2f} MB)")

    # Step 2: Open and inspect raw datasets
    ds_single = xr.open_dataset(actual_single)
    ds_pressure = xr.open_dataset(actual_pressure)
    ds_mask = xr.open_dataset(mask_path)
    ds_grid = xr.open_dataset(grid_path)

    # Harmonize coordinate names (valid_time -> time, latitude -> lat, longitude -> lon if needed)
    for ds_name, ds_obj in [("single", ds_single), ("pressure", ds_pressure)]:
        rename_dict = {}
        if "valid_time" in ds_obj.coords and "time" not in ds_obj.coords:
            rename_dict["valid_time"] = "time"
        if "latitude" in ds_obj.coords:
            rename_dict["latitude"] = "lat"
        if "longitude" in ds_obj.coords:
            rename_dict["longitude"] = "lon"
        if rename_dict:
            if ds_name == "single":
                ds_single = ds_single.rename(rename_dict)
            else:
                ds_pressure = ds_pressure.rename(rename_dict)

    # Slice January 2015 if annual file was provided
    if len(ds_single.time) > 744:
        ds_single = ds_single.sel(time=slice("2015-01-01T00:00:00", "2015-01-31T23:00:00"))
    if len(ds_pressure.time) > 744:
        ds_pressure = ds_pressure.sel(time=slice("2015-01-01T00:00:00", "2015-01-31T23:00:00"))

    # Map ECMWF short variable names to canonical names if needed
    var_rename_single = {}
    if "t2m" in ds_single and "2m_temperature" not in ds_single:
        var_rename_single["t2m"] = "2m_temperature"
    if "2t" in ds_single and "2m_temperature" not in ds_single:
        var_rename_single["2t"] = "2m_temperature"
    if "d2m" in ds_single and "2m_dewpoint_temperature" not in ds_single:
        var_rename_single["d2m"] = "2m_dewpoint_temperature"
    if "2d" in ds_single and "2m_dewpoint_temperature" not in ds_single:
        var_rename_single["2d"] = "2m_dewpoint_temperature"
    if "sp" in ds_single and "surface_pressure" not in ds_single:
        var_rename_single["sp"] = "surface_pressure"
    if "tcwv" in ds_single and "total_column_water_vapour" not in ds_single:
        var_rename_single["tcwv"] = "total_column_water_vapour"
    if var_rename_single:
        ds_single = ds_single.rename(var_rename_single)

    if "z" in ds_pressure and "geopotential" not in ds_pressure:
        ds_pressure = ds_pressure.rename({"z": "geopotential"})

    print(f"\n[RAW COORDINATE AUDIT]")
    print(f"  Single-levels time steps : {len(ds_single.time)} (Expected: 744 hours = 31 days * 24h)")
    print(f"  Single-levels spatial    : lat={len(ds_single.lat)}, lon={len(ds_single.lon)} (Expected: 32 x 48)")
    print(f"  Pressure-levels spatial  : lat={len(ds_pressure.lat)}, lon={len(ds_pressure.lon)} (Expected: 32 x 48)")
    
    assert len(ds_single.time) == 744, f"Expected 744 hourly timesteps, got {len(ds_single.time)}"
    assert len(ds_single.lat) == 32, f"Expected 32 latitudes, got {len(ds_single.lat)}"
    assert len(ds_single.lon) == 48, f"Expected 48 longitudes, got {len(ds_single.lon)}"
    np.testing.assert_allclose(ds_single.lat.values, ds_grid.lat.values, err_msg="Latitude mismatch with Candidate A")
    np.testing.assert_allclose(ds_single.lon.values, ds_grid.lon.values, err_msg="Longitude mismatch with Candidate A")
    print("  --> Coordinate parity with Candidate A 0.25° grid: 100% PERFECT MATCH.")

    # Step 3: Check raw fields
    required_single = ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure", "total_column_water_vapour"]
    for v in required_single:
        assert v in ds_single, f"Missing required variable {v} in single-levels (found {list(ds_single.data_vars.keys())})"
    assert "geopotential" in ds_pressure, f"Missing geopotential in pressure-levels (found {list(ds_pressure.data_vars.keys())})"
    print(f"  --> All 5 required raw variables present and valid.")

    # Step 4: Mathematical Derivation of 5 Model Channels
    print(f"\n[MATHEMATICAL CHANNEL DERIVATION]")
    print("  1. Computing daily tmax from hourly 2m_temperature...")
    tmax = ds_single["2m_temperature"].resample(time="1D").max(dim="time")

    print("  2. Computing daily tmin from hourly 2m_temperature...")
    tmin = ds_single["2m_temperature"].resample(time="1D").min(dim="time")

    print("  3. Computing daily diff_temp (tmax - tmin)...")
    diff_temp = tmax - tmin

    print("  4. Computing hourly specific humidity q(d2m, sp) via Bolton (1980)...")
    hourly_q = compute_specific_humidity(ds_single["2m_dewpoint_temperature"], ds_single["surface_pressure"])
    spfh = hourly_q.resample(time="1D").mean(dim="time")

    print("  5. Computing daily pwat from total_column_water_vapour...")
    pwat = ds_single["total_column_water_vapour"].resample(time="1D").mean(dim="time")

    print("  6. Computing daily hgt_pres (z200 / 9.80665 gpm)...")
    z200 = ds_pressure["geopotential"]
    for dim_name in ["level", "pressure_level", "isobaricInhPa"]:
        if dim_name in z200.dims:
            z200 = z200.squeeze(dim_name)
    if len(z200.dims) > 3:
        z200 = z200.squeeze()
    hgt_pres = (z200 / G0).resample(time="1D").mean(dim="time")

    # Step 5: Verify Final Tensor Dimensions (time x 32 x 48)
    channels = {
        "tmax": tmax,
        "tmin": tmin,
        "diff_temp": diff_temp,
        "spfh": spfh,
        "pwat": pwat,
        "hgt_pres": hgt_pres,
    }

    print(f"\n[TENSOR SHAPE VERIFICATION]")
    for name, da in channels.items():
        shape = da.shape
        print(f"  Channel {name:<10}: shape = {shape} (time x lat x lon)")
        assert shape == (31, 32, 48), f"Channel {name} shape {shape} != (31, 32, 48)"
    print("  --> All channels strictly conform to (31, 32, 48) tensor geometry.")

    # Step 6: Evaluation Mask Integrity Audit
    mask_key = "evaluation_mask" if "evaluation_mask" in ds_mask else "eval_mask"
    eval_mask = ds_mask[mask_key].values.astype(bool)
    num_eval_cells = int(np.sum(eval_mask))
    print(f"\n[EVALUATION MASK INTEGRITY AUDIT]")
    print(f"  Active Evaluation Land Cells: {num_eval_cells} (Expected: 126)")
    assert num_eval_cells == 126, f"Expected 126 evaluation cells, got {num_eval_cells}"

    print("\n[PHYSICAL DISTRIBUTION & MISSING VALUE AUDIT OVER 126 CELLS]")
    print(f"{'Channel':<12} {'Min':<10} {'Mean':<10} {'Max':<10} {'Std':<10} {'NaNs':<8} {'Infs':<8} {'Units':<10}")
    print("-" * 75)

    units_dict = {
        "tmax": "K",
        "tmin": "K",
        "diff_temp": "K",
        "spfh": "kg/kg",
        "pwat": "kg/m²",
        "hgt_pres": "gpm",
    }

    # Verify 0 NaNs and physical distributions across the 126 evaluation cells for all 31 days
    for name, da in channels.items():
        arr = da.values  # shape (31, 32, 48)
        # Extract active evaluation cell values across all 31 days: (31 * 126 = 3,906 samples)
        eval_samples = arr[:, eval_mask]
        n_nans = int(np.isnan(eval_samples).sum())
        n_infs = int(np.isinf(eval_samples).sum())
        
        c_min = float(np.nanmin(eval_samples))
        c_mean = float(np.nanmean(eval_samples))
        c_max = float(np.nanmax(eval_samples))
        c_std = float(np.nanstd(eval_samples))
        
        print(f"{name:<12} {c_min:<10.4f} {c_mean:<10.4f} {c_max:<10.4f} {c_std:<10.4f} {n_nans:<8} {n_infs:<8} {units_dict[name]:<10}")
        assert n_nans == 0, f"Detected {n_nans} NaNs in {name} over evaluation cells!"
        assert n_infs == 0, f"Detected {n_infs} Infs in {name} over evaluation cells!"

    print("-" * 75)
    print("  --> ZERO NaNs, ZERO Infs detected across all 3,906 evaluation sample points (31 days * 126 cells)!")

    # Step 7: Construct Verified Pilot NetCDF
    out_dir = Path("processed/atmospheric/pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc_path = out_dir / "era5_atmospheric_pilot_2015_01.nc"

    ds_out = xr.Dataset(
        data_vars={
            "tmax": (("time", "lat", "lon"), tmax.values, {"long_name": "Daily Maximum 2m Temperature", "units": "K", "method": "daily_max(hourly_t2m)"}),
            "diff_temp": (("time", "lat", "lon"), diff_temp.values, {"long_name": "Diurnal Temperature Difference", "units": "K", "method": "tmax - tmin"}),
            "spfh": (("time", "lat", "lon"), spfh.values, {"long_name": "Daily Mean Surface Specific Humidity", "units": "kg/kg", "method": "Bolton (1980) from hourly d2m and sp"}),
            "pwat": (("time", "lat", "lon"), pwat.values, {"long_name": "Daily Mean Precipitable Water", "units": "kg/m**2", "source": "total_column_water_vapour"}),
            "hgt_pres": (("time", "lat", "lon"), hgt_pres.values, {"long_name": "Daily Mean 200 hPa Geopotential Height", "units": "gpm", "method": "daily_mean(z200) / 9.80665"}),
            "tmin_diagnostic": (("time", "lat", "lon"), tmin.values, {"long_name": "Daily Minimum 2m Temperature (Diagnostic)", "units": "K", "method": "daily_min(hourly_t2m)"}),
        },
        coords={
            "time": tmax.time,
            "lat": ds_grid.lat,
            "lon": ds_grid.lon,
        },
        attrs={
            "title": "Verified 21C ERA5 Atmospheric Predictor Pilot (January 2015) for Mindanao RISE-UNet",
            "source_reanalysis": "Copernicus ERA5 (0.25° native resolution)",
            "spatial_domain": "Mindanao Candidate A Grid (32x48)",
            "temporal_resolution": "Daily aggregated from 24 hourly timesteps",
            "evaluation_land_cells": 126,
            "verification_status": "CERTIFIED_PASS",
        }
    )

    ds_out.to_netcdf(str(out_nc_path))
    print(f"\n[EXPORT] Verified pilot NetCDF saved to: {out_nc_path} ({out_nc_path.stat().st_size / 1e3:.2f} KB)")

    # Step 8: Generate 5-panel Publication Verification Figure
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "mindanao_era5_atmospheric_pilot_verification.png"

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    fig.suptitle("Sub-Phase 21C Verified Atmospheric Pilot — Monthly Mean Fields (January 2015)\nCandidate A Grid (32×48, 0.25° Resolution) | 126 Mindanao Evaluation Land Cells", fontsize=13, fontweight="bold")

    panels = [
        ("tmax", "Daily Tmax (K)", "inferno", tmax.mean(dim="time")),
        ("diff_temp", "Diurnal Temp Range: diff_temp (K)", "magma", diff_temp.mean(dim="time")),
        ("spfh", "Specific Humidity: spfh (kg/kg)", "viridis", spfh.mean(dim="time")),
        ("pwat", "Precipitable Water: pwat (kg/m²)", "YlGnBu", pwat.mean(dim="time")),
        ("hgt_pres", "200 hPa Height: hgt_pres (gpm)", "plasma", hgt_pres.mean(dim="time")),
    ]

    for idx, (var_name, title, cmap, mean_da) in enumerate(panels):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Mask non-evaluation cells with transparency or light background for land cells
        im = ax.imshow(
            mean_da.values,
            extent=[ds_grid.lon.values.min(), ds_grid.lon.values.max(), ds_grid.lat.values.min(), ds_grid.lat.values.max()],
            origin="upper",
            cmap=cmap,
            aspect="auto"
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 6th panel: Evaluation mask summary
    ax_mask = axes[1, 2]
    im_mask = ax_mask.imshow(
        eval_mask.astype(int),
        extent=[ds_grid.lon.values.min(), ds_grid.lon.values.max(), ds_grid.lat.values.min(), ds_grid.lat.values.max()],
        origin="upper",
        cmap="Greens",
        aspect="auto"
    )
    ax_mask.set_title("Active Evaluation Mask (126 Land Cells)", fontsize=11, fontweight="bold")
    ax_mask.set_xlabel("Longitude (°E)", fontsize=9)
    ax_mask.set_ylabel("Latitude (°N)", fontsize=9)
    plt.colorbar(im_mask, ax=ax_mask, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(str(fig_path))
    plt.close()
    print(f"[FIGURE] Publication verification composite saved to: {fig_path} ({fig_path.stat().st_size / 1e3:.2f} KB)")
    print("\n=================================================================")
    print("SUB-PHASE 21C PILOT VERIFICATION COMPLETE: ALL CRITERIA PASSED!")
    print("=================================================================")

if __name__ == "__main__":
    main()
