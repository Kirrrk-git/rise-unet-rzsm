#!/usr/bin/env python3
"""
download_era5_atmospheric_mindanao.py
------------------------------------
Authoritative acquisition engine for ERA5 Atmospheric Reanalysis variables over Mindanao,
supporting the RISE-UNet subseasonal forecasting architecture (Candidate A domain).

Scientific Source of Record:
  ECMWF / Copernicus Climate Change Service: ERA5 Global Reanalysis (0.25° grid)
  - Dataset 1: reanalysis-era5-single-levels (surface atmospheric drivers)
  - Dataset 2: reanalysis-era5-pressure-levels (200 hPa upper-troposphere dynamics)

Variables Extracted:
  1. total_column_water_vapour (tcwv, kg/m²) -> Precipitable Water (pwat)
  2. 2m_dewpoint_temperature (d2m, K)        -> Surface Specific Humidity (spfh) via Bolton/Tetens
  3. surface_pressure (sp, Pa)               -> Surface Specific Humidity (spfh)
  4. maximum_2m_temperature_since_previous_post_processing (mx2t, K) -> Daily Tmax (tmax)
  5. minimum_2m_temperature_since_previous_post_processing (mn2t, K) -> Diurnal Temp Range (diff_temp = tmax - tmin)
  6. geopotential (z, m²/s²) at 200 hPa      -> 200 hPa Geopotential Height (z200 = z / 9.80665 m/s²)

Access Methods:
  1. 'arco-gcp' (Google Cloud ARCO-ERA5, Default for Cloud/Colab):
     Reads the Analysis-Ready Cloud-Optimized Zarr representation directly from
     gs://gcp-public-data-arco-era5/. Avoids CDS extraction-job queueing.
  2. 'cds' (Copernicus Climate Data Store API, Official Fallback):
     Submits asynchronous extraction requests to the ECMWF CDS API queue.

Spatial Domain:
  Bounding Box [North: 11.75°N, West: 116.0°E, South: 4.0°N, East: 127.75°E]
  (Produces exact 32x48 grid on native 0.25° resolution matching Candidate A and RISE-UNet architecture).

Temporal Domain:
  Start: 2014-12 (Antecedent 20-day support window for January 2015 forecasts)
  End:   2025-12 (Full baseline: Train 2015-2021, Val 2022-2023, Test 2024-2025)
  Timesteps: All 24 hours (00:00 to 23:00) for every day of each month.
"""

import argparse
import calendar
import os
import subprocess
import sys
import time
from pathlib import Path

# Spatial bounding box [North, West, South, East] matching Candidate A 0.25° grid centers (32x48)
DEFAULT_BBOX = [11.75, 116.0, 4.0, 127.75]

ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

SINGLE_LEVEL_VARS = [
    "2m_temperature",            # Hourly analyzed field (ECMWF standard for daily tmax, tmin, diff_temp)
    "2m_dewpoint_temperature",   # Basis for surface specific humidity (spfh) via Bolton/Tetens
    "surface_pressure",          # Basis for surface specific humidity (spfh)
    "total_column_water_vapour", # Precipitable water (pwat = tcwv)
]

FORECAST_EXTREMA_VARS = [
    "maximum_2m_temperature_since_previous_post_processing",
    "minimum_2m_temperature_since_previous_post_processing",
]

PRESSURE_LEVEL_VARS = ["geopotential"]
PRESSURE_LEVELS = ["200"]


def get_cds_client():
    """Initializes and returns the Copernicus CDS API client."""
    try:
        import cdsapi
        return cdsapi.Client()
    except ImportError:
        print("[ERROR] 'cdsapi' is not installed in this environment.")
        print("Install it via: pip install cdsapi")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to initialize CDS API client: {e}")
        print("Please verify ~/.cdsapirc or CDSAPI_URL / CDSAPI_KEY environment variables.")
        sys.exit(1)


def check_gcs_exists(gcs_uri: str) -> bool:
    """Checks whether an object already exists in Google Cloud Storage."""
    cmd = ["gcloud", "storage", "ls", gcs_uri]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0 and gcs_uri in res.stdout


def upload_file_to_gcs(local_path: Path, gcs_dest_dir: str):
    """Uploads a local NetCDF file to GCS."""
    gcs_target = f"{gcs_dest_dir.rstrip('/')}/{local_path.name}"
    print(f"[GCS UPLOAD] Syncing {local_path.name} -> {gcs_target}...")
    cmd = ["gcloud", "storage", "cp", str(local_path), gcs_target]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[GCS UPLOAD] Success: {gcs_target}")
    else:
        print(f"[GCS ERROR] Upload failed: {res.stderr}")


def download_single_month_cds(
    client,
    dataset_type: str,
    year: int,
    month: int,
    output_dir: Path,
    bbox: list,
    upload_gcs: bool = False,
    gcs_bucket_prefix: str = "gs://rise-unet-rzsm/raw/era5",
    skip_if_present: bool = True,
    include_forecast_extrema: bool = False,
):
    """
    Retrieves one month of atmospheric variables via the Copernicus CDS API (queued).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    month_str = f"{month:02d}"

    if dataset_type == "single":
        dataset_name = "reanalysis-era5-single-levels"
        filename = f"era5_single_levels_{year}_{month_str}.nc"
        vars_to_request = list(SINGLE_LEVEL_VARS)
        if include_forecast_extrema:
            vars_to_request.extend(FORECAST_EXTREMA_VARS)
        request = {
            "product_type": ["reanalysis"],
            "variable": vars_to_request,
            "year": [str(year)],
            "month": [month_str],
            "day": ALL_DAYS,
            "time": ALL_HOURS,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": bbox,
        }
    elif dataset_type == "pressure":
        dataset_name = "reanalysis-era5-pressure-levels"
        filename = f"era5_z200_{year}_{month_str}.nc"
        request = {
            "product_type": ["reanalysis"],
            "variable": PRESSURE_LEVEL_VARS,
            "pressure_level": PRESSURE_LEVELS,
            "year": [str(year)],
            "month": [month_str],
            "day": ALL_DAYS,
            "time": ALL_HOURS,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": bbox,
        }
    else:
        raise ValueError(f"Unknown atmospheric dataset_type: {dataset_type}")

    local_path = output_dir / filename
    gcs_target = f"{gcs_bucket_prefix.rstrip('/')}/{dataset_type}/{filename}"

    if skip_if_present:
        if local_path.exists() and local_path.stat().st_size > 1024:
            print(f"[SKIP] Local file exists: {local_path} ({local_path.stat().st_size / 1e6:.2f} MB)")
            if upload_gcs and not check_gcs_exists(gcs_target):
                upload_file_to_gcs(local_path, f"{gcs_bucket_prefix.rstrip('/')}/{dataset_type}")
            return local_path
        if upload_gcs and check_gcs_exists(gcs_target):
            print(f"[SKIP] Already present in GCS: {gcs_target}")
            return None

    print(f"\n=======================================================")
    print(f"[CDS API RETRIEVAL] Dataset: {dataset_name}")
    print(f"  Target Month : {year}-{month_str}")
    print(f"  Bounding Box : North={bbox[0]}, West={bbox[1]}, South={bbox[2]}, East={bbox[3]}")
    print(f"  Local Output : {local_path}")
    print(f"=======================================================")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            client.retrieve(dataset_name, request, str(local_path))
            print(f"[SUCCESS] Retrieved {filename} ({local_path.stat().st_size / 1e6:.2f} MB)")
            break
        except Exception as e:
            print(f"[WARNING] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait_sec = 30 * attempt
                print(f"Retrying in {wait_sec} seconds...")
                time.sleep(wait_sec)
            else:
                print(f"[ERROR] Failed to retrieve {filename} after {max_retries} attempts.")
                raise e

    if upload_gcs and local_path.exists():
        upload_file_to_gcs(local_path, f"{gcs_bucket_prefix.rstrip('/')}/{dataset_type}")

    return local_path


def download_single_year_cds(
    client,
    dataset_type: str,
    year: int,
    output_dir: Path,
    bbox: list,
    upload_gcs: bool = False,
    gcs_bucket_prefix: str = "gs://rise-unet-rzsm/raw/era5",
    skip_if_present: bool = True,
    include_forecast_extrema: bool = False,
):
    """
    Retrieves one full calendar year (all 12 months, all 24 hours, all days)
    of atmospheric variables via the Copernicus CDS API in a single bundled request.
    Reduces 12 monthly queue waits into 1 efficient extraction job.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_months = [f"{m:02d}" for m in range(1, 13)]

    if dataset_type == "single":
        dataset_name = "reanalysis-era5-single-levels"
        filename = f"era5_single_levels_{year}.nc"
        vars_to_request = list(SINGLE_LEVEL_VARS)
        if include_forecast_extrema:
            vars_to_request.extend(FORECAST_EXTREMA_VARS)
        request = {
            "product_type": ["reanalysis"],
            "variable": vars_to_request,
            "year": [str(year)],
            "month": all_months,
            "day": ALL_DAYS,
            "time": ALL_HOURS,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": bbox,
        }
    elif dataset_type == "pressure":
        dataset_name = "reanalysis-era5-pressure-levels"
        filename = f"era5_z200_{year}.nc"
        request = {
            "product_type": ["reanalysis"],
            "variable": PRESSURE_LEVEL_VARS,
            "pressure_level": PRESSURE_LEVELS,
            "year": [str(year)],
            "month": all_months,
            "day": ALL_DAYS,
            "time": ALL_HOURS,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": bbox,
        }
    else:
        raise ValueError(f"Unknown atmospheric dataset_type: {dataset_type}")

    local_path = output_dir / filename
    gcs_target = f"{gcs_bucket_prefix.rstrip('/')}/{dataset_type}/{filename}"

    if skip_if_present:
        if local_path.exists() and local_path.stat().st_size > 1024:
            print(f"[SKIP] Local file exists: {local_path} ({local_path.stat().st_size / 1e6:.2f} MB)")
            if upload_gcs and not check_gcs_exists(gcs_target):
                upload_file_to_gcs(local_path, f"{gcs_bucket_prefix.rstrip('/')}/{dataset_type}")
            return local_path
        if upload_gcs and check_gcs_exists(gcs_target):
            print(f"[SKIP] Already present in GCS: {gcs_target}")
            return None

    print(f"\n=======================================================")
    print(f"[CDS API ANNUAL RETRIEVAL] Dataset: {dataset_name}")
    print(f"  Target Year  : {year} (All 12 months, 24 hours, all days)")
    print(f"  Bounding Box : North={bbox[0]}, West={bbox[1]}, South={bbox[2]}, East={bbox[3]}")
    print(f"  Local Output : {local_path}")
    print(f"=======================================================")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            client.retrieve(dataset_name, request, str(local_path))
            print(f"[SUCCESS] Retrieved {filename} ({local_path.stat().st_size / 1e6:.2f} MB)")
            break
        except Exception as e:
            print(f"[WARNING] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait_sec = 30 * attempt
                print(f"Retrying in {wait_sec} seconds...")
                time.sleep(wait_sec)
            else:
                print(f"[ERROR] Failed to retrieve {filename} after {max_retries} attempts.")
                raise e

    if upload_gcs and local_path.exists():
        upload_file_to_gcs(local_path, f"{gcs_bucket_prefix.rstrip('/')}/{dataset_type}")

    return local_path


def download_single_month_arco(
    dataset_type: str,
    year: int,
    month: int,
    output_dir: Path,
    bbox: list,
    upload_gcs: bool = False,
    gcs_bucket_prefix: str = "gs://rise-unet-rzsm/raw/era5",
    skip_if_present: bool = True,
    include_forecast_extrema: bool = False,
):
    """
    Directly extracts the sliced Mindanao bounding box from Google Cloud Public Dataset
    ARCO-ERA5 (gs://gcp-public-data-arco-era5/) using xarray + zarr.
    Avoids CDS extraction-job queueing by reading the cloud-optimized Zarr representation directly.
    """
    try:
        import xarray as xr
    except ImportError:
        print("[ERROR] xarray is required. Install via: pip install xarray netCDF4 zarr gcsfs")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    month_str = f"{month:02d}"
    last_day = calendar.monthrange(year, month)[1]

    if dataset_type == "single":
        filename = f"era5_single_levels_{year}_{month_str}.nc"
    elif dataset_type == "pressure":
        filename = f"era5_z200_{year}_{month_str}.nc"
    else:
        raise ValueError(f"Unknown atmospheric dataset_type: {dataset_type}")

    local_path = output_dir / filename
    gcs_target = f"{gcs_bucket_prefix.rstrip('/')}/{dataset_type}/{filename}"

    if skip_if_present:
        if local_path.exists() and local_path.stat().st_size > 1024:
            print(f"[SKIP] Local file exists: {local_path} ({local_path.stat().st_size / 1e6:.2f} MB)")
            if upload_gcs and not check_gcs_exists(gcs_target):
                upload_file_to_gcs(local_path, f"{gcs_bucket_prefix.rstrip('/')}/{dataset_type}")
            return local_path
        if upload_gcs and check_gcs_exists(gcs_target):
            print(f"[SKIP] Already present in GCS: {gcs_target}")
            return None

    print(f"\n=======================================================")
    print(f"[GOOGLE ARCO-ERA5 SLICE] Access: gs://gcp-public-data-arco-era5/")
    print(f"  Dataset Type : {dataset_type}")
    print(f"  Target Month : {year}-{month_str} (Days 01 to {last_day:02d})")
    print(f"  Bounding Box : North={bbox[0]}, West={bbox[1]}, South={bbox[2]}, East={bbox[3]}")
    print(f"  Local Output : {local_path}")
    print(f"=======================================================")

    time_slice = slice(f"{year}-{month_str}-01T00:00:00", f"{year}-{month_str}-{last_day:02d}T23:00:00")
    lat_slice = slice(bbox[0], bbox[2])  # North down to South (ERA5 latitude is descending)
    lon_slice = slice(bbox[1], bbox[3])  # West to East

    zarr_store = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

    try:
        import gcsfs
        import dask
        dask.config.set(num_workers=16)
        fs = gcsfs.GCSFileSystem(token="anon")
        store = fs.get_mapper(zarr_store)
        ds = xr.open_zarr(store, consolidated=True, chunks={"time": 48})
    except Exception as e:
        print(f"[ERROR] Failed to open Google ARCO-ERA5 Zarr store: {e}")
        print("Ensure 'zarr', 'gcsfs', and 'dask' are installed: pip install zarr gcsfs dask")
        raise e

    if dataset_type == "single":
        var_candidates = [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "surface_pressure",
            "total_column_water_vapour",
        ]
        if include_forecast_extrema:
            var_candidates.extend(FORECAST_EXTREMA_VARS)
        var_subset = [v for v in var_candidates if v in ds]
        print(f"  Streaming {len(var_subset)} single-level variables to {local_path.name}...")
        ds_sub = ds[var_subset].sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)
        ds_sub.to_netcdf(str(local_path))
    elif dataset_type == "pressure":
        print(f"  Streaming geopotential (200 hPa) to {local_path.name}...")
        ds_sub = ds[["geopotential"]].sel(time=time_slice, level=200, latitude=lat_slice, longitude=lon_slice)
        ds_sub.to_netcdf(str(local_path))

    print(f"[SUCCESS] Extracted {filename} ({local_path.stat().st_size / 1e6:.2f} MB)")
    if upload_gcs and local_path.exists():
        upload_file_to_gcs(local_path, f"{gcs_bucket_prefix.rstrip('/')}/{dataset_type}")
    return local_path


def main():
    parser = argparse.ArgumentParser(
        description="Authoritative ERA5 atmospheric reanalysis acquisition engine for Mindanao RISE-UNet forecasting."
    )
    parser.add_argument("--pilot", action="store_true", help="Acquire only 1 benchmark month (January 2015).")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year (default: 2015; 2014 for antecedent window).")
    parser.add_argument("--start-month", type=int, default=1, help="Start month (1-12, default: 1).")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (default: 2025).")
    parser.add_argument("--end-month", type=int, default=12, help="End month (1-12, default: 12).")
    parser.add_argument("--year", type=int, default=None, help="Acquire a specific single calendar year.")
    parser.add_argument("--month", type=int, default=None, help="Acquire a specific single month.")
    parser.add_argument(
        "--by-year",
        action="store_true",
        help="Attempt annual bundling (note: ECMWF CDS-Beta enforces strict cost limits on multi-variable requests; monthly is the recommended standard).",
    )
    parser.add_argument(
        "--dataset",
        choices=["both", "single", "pressure"],
        default="both",
        help="Atmospheric dataset: 'both' (single-levels + z200), 'single', or 'pressure' (default: both).",
    )
    parser.add_argument(
        "--access",
        choices=["cds", "arco-gcp"],
        default="cds",
        help="Access method: 'cds' (Copernicus CDS API, default & recommended) or 'arco-gcp' (Google Cloud ARCO-ERA5 for Cloud VMs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Data/raw_downloads/ERA5_atmospheric",
        help="Local directory for retrieved NetCDF files (default: Data/raw_downloads/ERA5_atmospheric).",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=DEFAULT_BBOX,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help=f"Spatial bounding box [N, W, S, E] (default: {DEFAULT_BBOX} matching soil moisture archive).",
    )
    parser.add_argument(
        "--upload-gcs",
        action="store_true",
        help="Automatically upload completed NetCDF files to Google Cloud Storage.",
    )
    parser.add_argument(
        "--gcs-prefix",
        type=str,
        default="gs://rise-unet-rzsm/raw/era5",
        help="Target GCS bucket prefix (default: gs://rise-unet-rzsm/raw/era5).",
    )
    parser.add_argument(
        "--include-forecast-extrema",
        action="store_true",
        help="Also retrieve forecast mx2t and mn2t alongside analyzed hourly 2m_temperature.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Force re-acquisition even if file already exists locally or in GCS.",
    )

    args = parser.parse_args()

    client = None
    if args.access == "cds":
        client = get_cds_client()

    base_out = Path(args.output_dir)

    datasets_to_run = []
    if args.dataset in ["both", "single"]:
        datasets_to_run.append("single")
    if args.dataset in ["both", "pressure"]:
        datasets_to_run.append("pressure")

    # Determine execution mode: monthly is default for CDS-Beta cost limit compliance
    use_annual = (
        args.by_year
        and args.access == "cds"
        and not args.pilot
        and args.month is None
    )

    start_time = time.time()

    if args.pilot:
        print("[MODE] PILOT RUN: Acquiring 1-month atmospheric benchmark (January 2015).")
        total_jobs = len(datasets_to_run)
        current_job = 0
        for ds_type in datasets_to_run:
            current_job += 1
            print(f"\n>>> Progress: Job {current_job}/{total_jobs} ({current_job/total_jobs*100:.1f}%) <<<")
            ds_dir = base_out / ds_type
            if args.access == "cds":
                download_single_month_cds(
                    client=client,
                    dataset_type=ds_type,
                    year=2015,
                    month=1,
                    output_dir=ds_dir,
                    bbox=args.bbox,
                    upload_gcs=args.upload_gcs,
                    gcs_bucket_prefix=args.gcs_prefix,
                    skip_if_present=not args.no_skip,
                    include_forecast_extrema=args.include_forecast_extrema,
                )
            else:
                download_single_month_arco(
                    dataset_type=ds_type,
                    year=2015,
                    month=1,
                    output_dir=ds_dir,
                    bbox=args.bbox,
                    upload_gcs=args.upload_gcs,
                    gcs_bucket_prefix=args.gcs_prefix,
                    skip_if_present=not args.no_skip,
                    include_forecast_extrema=args.include_forecast_extrema,
                )
    elif use_annual:
        # Annual bundled mode (CDS API only): 1 request per year per dataset
        if args.year is not None:
            years = [args.year]
        else:
            years = list(range(args.start_year, args.end_year + 1))

        total_jobs = len(years) * len(datasets_to_run)
        print(f"[MODE] ANNUAL BUNDLE: {len(years)} calendar years ({years[0]} to {years[-1]}), {total_jobs} total requests.")
        print(f"       Each annual request bundles all 12 months, 24 hours, and days into 1 NetCDF file.")
        print(f"[SPATIAL] Bounding Box: {args.bbox} (Exact 32x48 Candidate A grid parity)")
        print(f"[ACCESS METHOD] CDS API (Copernicus Climate Data Store)")
        print(f"[DATASETS] Selected: {args.dataset}")
        print(f"[GCS SYNC] Active: {args.upload_gcs} (Target: {args.gcs_prefix})")

        current_job = 0
        for y in years:
            for ds_type in datasets_to_run:
                current_job += 1
                print(f"\n>>> Progress: Job {current_job}/{total_jobs} ({current_job/total_jobs*100:.1f}%) <<<")
                ds_dir = base_out / ds_type
                download_single_year_cds(
                    client=client,
                    dataset_type=ds_type,
                    year=y,
                    output_dir=ds_dir,
                    bbox=args.bbox,
                    upload_gcs=args.upload_gcs,
                    gcs_bucket_prefix=args.gcs_prefix,
                    skip_if_present=not args.no_skip,
                    include_forecast_extrema=args.include_forecast_extrema,
                )
    else:
        # Monthly mode
        if args.year is not None:
            if args.month is not None:
                schedule = [(args.year, args.month)]
            else:
                schedule = [(args.year, m) for m in range(1, 13)]
        else:
            schedule = []
            for y in range(args.start_year, args.end_year + 1):
                m_start = args.start_month if y == args.start_year else 1
                m_end = args.end_month if y == args.end_year else 12
                for m in range(m_start, m_end + 1):
                    schedule.append((y, m))

        total_jobs = len(schedule) * len(datasets_to_run)
        print(f"[MODE] MONTHLY RUN: {len(schedule)} months, {total_jobs} total requests.")
        print(f"[SPATIAL] Bounding Box: {args.bbox} (Exact 32x48 Candidate A grid parity)")
        print(f"[ACCESS METHOD] {args.access.upper()}")
        print(f"[DATASETS] Selected: {args.dataset}")
        print(f"[GCS SYNC] Active: {args.upload_gcs} (Target: {args.gcs_prefix})")

        current_job = 0
        for year, month in schedule:
            for ds_type in datasets_to_run:
                current_job += 1
                print(f"\n>>> Progress: Job {current_job}/{total_jobs} ({current_job/total_jobs*100:.1f}%) <<<")
                ds_dir = base_out / ds_type
                if args.access == "arco-gcp":
                    download_single_month_arco(
                        dataset_type=ds_type,
                        year=year,
                        month=month,
                        output_dir=ds_dir,
                        bbox=args.bbox,
                        upload_gcs=args.upload_gcs,
                        gcs_bucket_prefix=args.gcs_prefix,
                        skip_if_present=not args.no_skip,
                        include_forecast_extrema=args.include_forecast_extrema,
                    )
                else:
                    download_single_month_cds(
                        client=client,
                        dataset_type=ds_type,
                        year=year,
                        month=month,
                        output_dir=ds_dir,
                        bbox=args.bbox,
                        upload_gcs=args.upload_gcs,
                        gcs_bucket_prefix=args.gcs_prefix,
                        skip_if_present=not args.no_skip,
                        include_forecast_extrema=args.include_forecast_extrema,
                    )

    elapsed = time.time() - start_time
    print(f"\n=======================================================")
    print(f"[COMPLETED] All {total_jobs} atmospheric jobs processed in {elapsed / 60:.2f} minutes.")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
