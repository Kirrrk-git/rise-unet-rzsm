#!/usr/bin/env python3
"""
03_derive_era5_atmospheric_daily.py
------------------------------------
Derives the 5 RISE-UNet daily atmospheric predictor channels from raw hourly ERA5
reanalysis fields over Mindanao (Candidate A 32x48 grid):
  1. tmax:      Daily maximum of analyzed hourly 2m temperature [K]
  2. diff_temp: Daily diurnal temperature difference (tmax - tmin) [K]
  3. spfh:      Daily surface specific humidity derived from hourly d2m & sp via Bolton (1980) [kg/kg]
  4. pwat:      Daily mean total column water vapour (tcwv) [kg/m²]
  5. hgt_pres:  Daily mean 200 hPa geopotential height (z / 9.80665) [gpm]

Supports both:
  - Standalone pilot certification: January 2015 with publication figure export.
  - Batch derivation: Multi-month / multi-year processing (e.g. 2022-2023 for validation pipeline).
"""

import argparse
import calendar
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Gravitational acceleration constant (WMO standard)
G0 = 9.80665
EPSILON = 0.622


def compute_specific_humidity(d2m_k: xr.DataArray, sp_pa: xr.DataArray) -> xr.DataArray:
    """
    Computes specific humidity q [kg/kg] from 2m dewpoint temperature [K]
    and surface pressure [Pa] using Bolton (1980) / Tetens formulation.
    """
    td_c = d2m_k - 273.15
    e_pa = 611.2 * np.exp(17.67 * td_c / (td_c + 243.5))
    q = (EPSILON * e_pa) / (sp_pa - (1.0 - EPSILON) * e_pa)
    return q


def upload_to_gcs(local_path: Path, gcs_dest: str):
    """Uploads a local NetCDF file to Google Cloud Storage."""
    target_uri = f"{gcs_dest.rstrip('/')}/{local_path.name}"
    print(f"[GCS UPLOAD] Syncing {local_path.name} -> {target_uri}...")
    cmd = ["gcloud", "storage", "cp", str(local_path), target_uri]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[GCS UPLOAD] Success: {target_uri}")
    else:
        print(f"[GCS ERROR] Upload failed: {res.stderr.strip()}")


def find_raw_files(year: int, month: int, raw_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locates the raw single-level and pressure-level NetCDF files for a given year and month.
    Checks multiple candidate paths including subdirectories and parent study data.
    """
    month_str = f"{year}_{month:02d}"
    candidate_singles = [
        raw_dir / "single" / f"era5_single_levels_{month_str}.nc",
        raw_dir / f"era5_single_levels_{month_str}.nc",
        raw_dir / "single" / f"era5_single_levels_{year}.nc",
        raw_dir / f"era5_single_levels_{year}.nc",
        Path(f"parent_study_ex29/Data/raw_downloads/ERA5_atmospheric/single/era5_single_levels_{month_str}.nc"),
        Path(f"Data/raw_downloads/ERA5_atmospheric/single/era5_single_levels_{month_str}.nc"),
        Path(f"parent_study_ex29/Data/raw_downloads/ERA5_atmospheric/single/era5_single_levels_{year}.nc"),
        Path(f"Data/raw_downloads/ERA5_atmospheric/single/era5_single_levels_{year}.nc"),
    ]
    candidate_pressures = [
        raw_dir / "pressure" / f"era5_z200_{month_str}.nc",
        raw_dir / f"era5_z200_{month_str}.nc",
        raw_dir / "pressure" / f"era5_z200_{year}.nc",
        raw_dir / f"era5_z200_{year}.nc",
        Path(f"parent_study_ex29/Data/raw_downloads/ERA5_atmospheric/pressure/era5_z200_{month_str}.nc"),
        Path(f"Data/raw_downloads/ERA5_atmospheric/pressure/era5_z200_{month_str}.nc"),
        Path(f"parent_study_ex29/Data/raw_downloads/ERA5_atmospheric/pressure/era5_z200_{year}.nc"),
        Path(f"Data/raw_downloads/ERA5_atmospheric/pressure/era5_z200_{year}.nc"),
    ]

    single_path = next((p for p in candidate_singles if p.exists()), None)
    pressure_path = next((p for p in candidate_pressures if p.exists()), None)
    return single_path, pressure_path


def derive_month(
    year: int,
    month: int,
    single_path: Path,
    pressure_path: Path,
    grid_path: Path,
    mask_path: Path,
    out_nc_path: Path,
    is_pilot: bool = False,
) -> Path:
    """
    Executes mathematical channel derivation and validation for a single month.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    expected_hours = days_in_month * 24

    ds_single = xr.open_dataset(single_path)
    ds_pressure = xr.open_dataset(pressure_path)
    ds_grid = xr.open_dataset(grid_path)
    ds_mask = xr.open_dataset(mask_path)

    # Standardize coordinate names
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

    # Slice month if annual bundle provided
    if len(ds_single.time) > expected_hours:
        start_str = f"{year}-{month:02d}-01T00:00:00"
        end_str = f"{year}-{month:02d}-{days_in_month:02d}T23:00:00"
        ds_single = ds_single.sel(time=slice(start_str, end_str))
    if len(ds_pressure.time) > expected_hours:
        start_str = f"{year}-{month:02d}-01T00:00:00"
        end_str = f"{year}-{month:02d}-{days_in_month:02d}T23:00:00"
        ds_pressure = ds_pressure.sel(time=slice(start_str, end_str))

    # Standardize variable names
    var_rename_single = {}
    for name in ["2t", "t2m"]:
        if name in ds_single and "2m_temperature" not in ds_single:
            var_rename_single[name] = "2m_temperature"
    for name in ["2d", "d2m"]:
        if name in ds_single and "2m_dewpoint_temperature" not in ds_single:
            var_rename_single[name] = "2m_dewpoint_temperature"
    if "sp" in ds_single and "surface_pressure" not in ds_single:
        var_rename_single["sp"] = "surface_pressure"
    if "tcwv" in ds_single and "total_column_water_vapour" not in ds_single:
        var_rename_single["tcwv"] = "total_column_water_vapour"
    if var_rename_single:
        ds_single = ds_single.rename(var_rename_single)

    if "z" in ds_pressure and "geopotential" not in ds_pressure:
        ds_pressure = ds_pressure.rename({"z": "geopotential"})

    # Assert coordinate parity
    assert len(ds_single.lat) == 32, f"Expected 32 latitudes, got {len(ds_single.lat)}"
    assert len(ds_single.lon) == 48, f"Expected 48 longitudes, got {len(ds_single.lon)}"
    np.testing.assert_allclose(ds_single.lat.values, ds_grid.lat.values, err_msg="Latitude mismatch with Candidate A")
    np.testing.assert_allclose(ds_single.lon.values, ds_grid.lon.values, err_msg="Longitude mismatch with Candidate A")

    # Mathematical Channel Derivation
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

    channels = {
        "tmax": tmax,
        "tmin": tmin,
        "diff_temp": diff_temp,
        "spfh": spfh,
        "pwat": pwat,
        "hgt_pres": hgt_pres,
    }

    # Shape verification
    for name, da in channels.items():
        assert da.shape == (days_in_month, 32, 48), f"Channel {name} shape {da.shape} != ({days_in_month}, 32, 48)"

    # Evaluation Mask Integrity Audit
    mask_key = "evaluation_mask" if "evaluation_mask" in ds_mask else "eval_mask"
    eval_mask = ds_mask[mask_key].values.astype(bool)
    assert int(np.sum(eval_mask)) == 126, f"Expected 126 evaluation cells, got {np.sum(eval_mask)}"

    # Check 0 NaNs and 0 Infs over 126 evaluation land cells
    for name, da in channels.items():
        eval_samples = da.values[:, eval_mask]
        n_nans = int(np.isnan(eval_samples).sum())
        n_infs = int(np.isinf(eval_samples).sum())
        assert n_nans == 0, f"Detected {n_nans} NaNs in {name} over evaluation cells for {year}-{month:02d}!"
        assert n_infs == 0, f"Detected {n_infs} Infs in {name} over evaluation cells for {year}-{month:02d}!"

    # Export NetCDF
    out_nc_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        data_vars={
            "tmax": (("time", "lat", "lon"), tmax.values, {"long_name": "Daily Maximum 2m Temperature", "units": "K"}),
            "diff_temp": (("time", "lat", "lon"), diff_temp.values, {"long_name": "Diurnal Temperature Difference", "units": "K"}),
            "spfh": (("time", "lat", "lon"), spfh.values, {"long_name": "Daily Mean Surface Specific Humidity", "units": "kg/kg"}),
            "pwat": (("time", "lat", "lon"), pwat.values, {"long_name": "Daily Mean Precipitable Water", "units": "kg/m**2"}),
            "hgt_pres": (("time", "lat", "lon"), hgt_pres.values, {"long_name": "Daily Mean 200 hPa Geopotential Height", "units": "gpm"}),
            "tmin_diagnostic": (("time", "lat", "lon"), tmin.values, {"long_name": "Daily Minimum 2m Temperature", "units": "K"}),
        },
        coords={
            "time": tmax.time,
            "lat": ds_grid.lat,
            "lon": ds_grid.lon,
        },
        attrs={
            "title": f"Derived ERA5 Atmospheric Predictors ({year}-{month:02d}) for Mindanao RISE-UNet",
            "source_reanalysis": "Copernicus ERA5 (0.25° native resolution)",
            "spatial_domain": "Mindanao Candidate A Grid (32x48)",
            "temporal_resolution": "Daily aggregated from 24 hourly timesteps",
            "evaluation_land_cells": 126,
            "derivation_status": "DERIVED_BASIC_CHECKS_PASS",
        },
    )
    ds_out.to_netcdf(str(out_nc_path))
    print(f"  [SUCCESS] Derived {out_nc_path.name} ({out_nc_path.stat().st_size / 1e3:.1f} KB, {days_in_month} days)")

    # Pilot-only verification figure
    if is_pilot:
        fig_dir = Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / "era5_atmospheric_pilot_quickcheck.png"
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
        fig.suptitle(
            "Sub-Phase 21C Verified Atmospheric Pilot — Monthly Mean Fields (January 2015)\nCandidate A Grid (32×48, 0.25° Resolution) | 126 Mindanao Evaluation Land Cells",
            fontsize=13,
            fontweight="bold",
        )
        panels = [
            ("tmax", "Daily Tmax (K)", "inferno", tmax.mean(dim="time")),
            ("diff_temp", "Diurnal Temp Range: diff_temp (K)", "magma", diff_temp.mean(dim="time")),
            ("spfh", "Specific Humidity: spfh (kg/kg)", "viridis", spfh.mean(dim="time")),
            ("pwat", "Precipitable Water: pwat (kg/m²)", "YlGnBu", pwat.mean(dim="time")),
            ("hgt_pres", "200 hPa Height: hgt_pres (gpm)", "plasma", hgt_pres.mean(dim="time")),
        ]
        for idx, (var_name, title, cmap, mean_da) in enumerate(panels):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            im = ax.imshow(
                mean_da.values,
                extent=[ds_grid.lon.values.min(), ds_grid.lon.values.max(), ds_grid.lat.values.min(), ds_grid.lat.values.max()],
                origin="upper",
                cmap=cmap,
                aspect="auto",
            )
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Longitude (°E)", fontsize=9)
            ax.set_ylabel("Latitude (°N)", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax_mask = axes[1, 2]
        im_mask = ax_mask.imshow(
            eval_mask.astype(int),
            extent=[ds_grid.lon.values.min(), ds_grid.lon.values.max(), ds_grid.lat.values.min(), ds_grid.lat.values.max()],
            origin="upper",
            cmap="Greens",
            aspect="auto",
        )
        ax_mask.set_title("Active Evaluation Mask (126 Land Cells)", fontsize=11, fontweight="bold")
        ax_mask.set_xlabel("Longitude (°E)", fontsize=9)
        ax_mask.set_ylabel("Latitude (°N)", fontsize=9)
        plt.colorbar(im_mask, ax=ax_mask, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(str(fig_path))
        plt.close()
        print(f"[FIGURE] Publication verification composite saved to: {fig_path}")

    return out_nc_path


def main():
    parser = argparse.ArgumentParser(
        description="Derive the 5 daily atmospheric predictor channels for Mindanao RISE-UNet forecasting."
    )
    parser.add_argument("--pilot", action="store_true", help="Run January 2015 pilot derivation with publication figure export.")
    parser.add_argument("--start-year", type=int, default=None, help="Start year for batch derivation (e.g. 2022).")
    parser.add_argument("--end-year", type=int, default=None, help="End year for batch derivation (e.g. 2023).")
    parser.add_argument("--start-month", type=int, default=1, help="Start month (1-12, default: 1).")
    parser.add_argument("--end-month", type=int, default=12, help="End month (1-12, default: 12).")
    parser.add_argument("--raw-dir", type=str, default="Data/raw_downloads/ERA5_atmospheric", help="Raw NetCDF directory.")
    parser.add_argument("--out-dir", type=str, default="processed/atmospheric", help="Output directory for derived NetCDFs.")
    parser.add_argument("--grid-path", type=str, default="processed/grid/mindanao_025deg.nc", help="Candidate A grid path.")
    parser.add_argument("--mask-path", type=str, default="processed/grid/mindanao_eval_mask_025.nc", help="Evaluation mask path.")
    parser.add_argument("--upload-gcs", action="store_true", help="Sync derived NetCDFs to Google Cloud Storage.")
    parser.add_argument("--gcs-prefix", type=str, default="gs://rise-unet-rzsm/processed/atmospheric", help="GCS target prefix.")
    parser.add_argument("--no-skip", action="store_true", help="Re-derive even if output NetCDF already exists.")
    args = parser.parse_args()

    # Determine execution mode: if no start-year or --pilot given, default to pilot
    is_pilot_mode = args.pilot or (args.start_year is None and args.end_year is None)

    grid_path = Path(args.grid_path)
    mask_path = Path(args.mask_path)
    raw_dir = Path(args.raw_dir)

    if not grid_path.exists():
        print(f"[ERROR] Grid file not found: {grid_path}")
        sys.exit(1)
    if not mask_path.exists():
        print(f"[ERROR] Mask file not found: {mask_path}")
        sys.exit(1)

    if is_pilot_mode:
        print("=================================================================")
        print("Sub-Phase 21C: ERA5 Atmospheric Pilot Verification & Derivation")
        print("Target Month: January 2015 (31 days)")
        print("=================================================================\n")
        single_path, pressure_path = find_raw_files(2015, 1, raw_dir)
        if single_path is None or pressure_path is None:
            print(f"[ERROR] Could not find raw pilot files for 2015-01 in {raw_dir}")
            sys.exit(1)
        out_dir = Path(args.out_dir) / "pilot" if "pilot" not in args.out_dir else Path(args.out_dir)
        out_path = out_dir / "era5_atmospheric_pilot_2015_01.nc"
        derive_month(2015, 1, single_path, pressure_path, grid_path, mask_path, out_path, is_pilot=True)
        if args.upload_gcs:
            upload_to_gcs(out_path, f"{args.gcs_prefix.rstrip('/')}/pilot")
        print("\n=================================================================")
        print("SUB-PHASE 21C PILOT VERIFICATION COMPLETE: ALL CRITERIA PASSED!")
        print("=================================================================")
        return

    # Batch derivation mode (e.g. 2022-2023 for validation pipeline)
    start_y = args.start_year
    end_y = args.end_year or start_y
    out_dir = Path(args.out_dir)

    print("=================================================================")
    print("Batch ERA5 Atmospheric Predictor Derivation")
    print(f"Target Period: {start_y}-{args.start_month:02d} to {end_y}-{args.end_month:02d}")
    print(f"Target Output: {out_dir}")
    print("=================================================================\n")

    derived_count = 0
    skipped_count = 0

    for year in range(start_y, end_y + 1):
        m_start = args.start_month if year == start_y else 1
        m_end = args.end_month if year == end_y else 12
        for month in range(m_start, m_end + 1):
            out_nc = out_dir / f"era5_atmospheric_{year}_{month:02d}.nc"
            if out_nc.exists() and not args.no_skip:
                print(f"[SKIP] Already exists: {out_nc.name} ({out_nc.stat().st_size / 1e3:.1f} KB)")
                skipped_count += 1
                if args.upload_gcs:
                    upload_to_gcs(out_nc, args.gcs_prefix)
                continue

            single_path, pressure_path = find_raw_files(year, month, raw_dir)
            if single_path is None or pressure_path is None:
                print(f"[ERROR] Missing raw files for {year}-{month:02d} in {raw_dir}")
                print(f"        Single: {single_path} | Pressure: {pressure_path}")
                sys.exit(1)

            print(f"Deriving {year}-{month:02d}...")
            derive_month(year, month, single_path, pressure_path, grid_path, mask_path, out_nc, is_pilot=False)
            derived_count += 1
            if args.upload_gcs:
                upload_to_gcs(out_nc, args.gcs_prefix)

    print("\n=================================================================")
    print(f"BATCH DERIVATION COMPLETE: {derived_count} derived, {skipped_count} skipped.")
    print("=================================================================")


if __name__ == "__main__":
    main()
