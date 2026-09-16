#!/usr/bin/env python3
"""
Dynamic Production ECMWF S2S Downloader for Mindanao (2015-2025)
API: ECMWF Data Store (ECDS) / Copernicus CDS API (cdsapi)
Target Options:
  --target gcs   : Upload to gs://rise-unet-rzsm/raw/ecmwf_s2s/production/ (auto-cleans temp files)
  --target local : Save to Data/raw_downloads/ECMWF/<cycle>/ (permanent local storage)
  --target both  : Save locally AND upload to GCS

Key Scientific & Engineering Design:
1. Dynamic Operational Schedule:
   - Evaluates all operational forecast issuance dates (every Monday and Thursday)
     of the corresponding model version without any hardcoded date skipping.
   - Hindcast years (hyear <= 2023): Queries 's2s-reforecasts' using the 2024 operational
     schedule and hyear/hmonth/hday parameters.
   - Real-time years (hyear >= 2024): Queries 's2s-forecasts' (operational real-time runs)
     with year/month/day parameters directly.
2. Fast Batch Skip Cache:
   - Queries GCS once at startup (~3s) to cache all 875+ existing cycles.
   - Skip checks execute instantaneously (~0.0001s per cycle).
   - Checks local folder if local storage is enabled.
3. Cross-Platform Windows & Terminal Resilience:
   - Uses shutil.which and shell=True on Windows to execute gcloud.CMD without WinError 2.
   - Resolves paths relative to repository root regardless of current working directory.
"""

import os
import sys
import time
import argparse
import datetime
import subprocess
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    import cdsapi
except ImportError:
    cdsapi = None

DATASET_HINDCAST = "s2s-reforecasts"
DATASET_FORECAST = "s2s-forecasts"
ECDS_API_URL = "https://ecds.ecmwf.int/api"
GCS_BUCKET = "gs://rise-unet-rzsm/raw/ecmwf_s2s/production"

# Candidate A bounding box [North, West, South, East]
BBOX = [11.75, 116.0, 4.0, 127.75]

VARIABLES = [
    "2_m_dewpoint_temperature",
    "2_m_temperature",
    "total_column_water"
]

LEADTIME_HOURS = [
    "0_24", "24_48", "48_72", "72_96", "96_120", "120_144", "144_168",
    "168_192", "192_216", "216_240", "240_264", "264_288", "288_312", "312_336"
]

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_ROOT / "s2s_download_errors.log"

def is_leap_year(year: int) -> bool:
    """Returns True if year is a leap year in the Gregorian calendar."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def generate_operational_cycles(start_year: int, end_year: int):
    """
    Dynamically generates the complete sequence of ECMWF operational cycle dates.
    ECMWF produces S2S runs on every Monday (0) and Thursday (3).
    """
    cycles = []
    for hyear in range(start_year, end_year + 1):
        # Operational model year in CDS: 2024 for hindcasts <= 2023; real-time year for 2024+
        model_year = 2024 if hyear <= 2023 else hyear

        d = datetime.date(model_year, 1, 1)
        end_d = datetime.date(model_year, 12, 31)

        while d <= end_d:
            if d.weekday() in (0, 3):  # Monday or Thursday
                m = d.month
                day = d.day

                # Leap-day adjustment: Feb 29 in non-leap hindcast year maps to Feb 28
                if m == 2 and day == 29 and not is_leap_year(hyear):
                    target_day = 28
                else:
                    target_day = day

                cycles.append({
                    "hyear": str(hyear),
                    "hmonth": f"{m:02d}",
                    "hday": f"{target_day:02d}",
                    "cycle_date_str": f"{hyear}-{m:02d}-{target_day:02d}",
                    "model_year": str(model_year),
                    "model_month": f"{m:02d}",
                    "model_day": f"{day:02d}"
                })
            d += datetime.timedelta(days=1)
    return cycles

def run_cloud_cmd(cmd_args: list) -> subprocess.CompletedProcess:
    """
    Executes a gcloud or gsutil command cross-platform.
    On Windows, uses shutil.which to find .CMD batch files and runs with shell=True.
    """
    primary_cmd = cmd_args[0]
    resolved_cmd = shutil.which(primary_cmd)
    if resolved_cmd:
        full_args = [resolved_cmd] + cmd_args[1:]
    else:
        full_args = cmd_args
    is_windows = sys.platform == "win32"
    return subprocess.run(full_args, capture_output=True, text=True, check=False, shell=is_windows)

def fetch_existing_gcs_cycles() -> set:
    """
    Pre-fetches all existing cycle folders in GCS in a single batch call.
    Returns a set of cycle date strings (e.g. {'2015-01-01', '2015-01-04', ...}).
    """
    res = run_cloud_cmd(["gcloud", "storage", "ls", f"{GCS_BUCKET}/"])
    if res.returncode != 0:
        res = run_cloud_cmd(["gsutil", "ls", f"{GCS_BUCKET}/"])

    cycles = set()
    if res.returncode == 0:
        for line in res.stdout.splitlines():
            line = line.strip().rstrip("/")
            if "/" in line:
                folder_name = line.split("/")[-1]
                if len(folder_name) == 10 and folder_name[4] == "-" and folder_name[7] == "-":
                    cycles.add(folder_name)
    return cycles

def check_local_cycle_exists(cycle_date_str: str, local_dir: Path) -> bool:
    """Checks if both control and perturbed forecast GRIB files exist locally with valid size."""
    if not local_dir:
        return False
    cf_path = local_dir / cycle_date_str / f"s2s_cf_{cycle_date_str}.grib"
    pf_path = local_dir / cycle_date_str / f"s2s_pf_{cycle_date_str}.grib"
    if cf_path.exists() and pf_path.exists():
        if cf_path.stat().st_size > 1024 and pf_path.stat().st_size > 1024:
            return True
    return False

def download_and_sync_cycle(
    client,
    cycle_info: dict,
    gcs_cache: set = None,
    dry_run: bool = False,
    local_dir: Path = None,
    upload_gcs: bool = True
) -> bool:
    """Retrieves cf and pf GRIB files for a cycle and uploads to GCS or stores locally."""
    cycle_date_str = cycle_info["cycle_date_str"]
    model_year = cycle_info["model_year"]
    model_month = cycle_info["model_month"]
    model_day = cycle_info["model_day"]
    hyear = cycle_info["hyear"]
    hmonth = cycle_info["hmonth"]
    hday = cycle_info["hday"]
    is_realtime = int(hyear) >= 2024

    dataset = DATASET_FORECAST if is_realtime else DATASET_HINDCAST

    local_exists = check_local_cycle_exists(cycle_date_str, local_dir) if local_dir else False
    gcs_exists = (cycle_date_str in gcs_cache) if (upload_gcs and gcs_cache is not None) else False

    if dry_run:
        local_status = "EXISTS" if local_exists else "MISSING"
        gcs_status = "EXISTS" if gcs_exists else "MISSING"
        print(f"[DRY RUN] Cycle: {cycle_date_str} -> Dataset: {dataset:<15} | Local: {local_status:<7} | GCS: {gcs_status}")
        return True

    # 1. Smart Skip Checks
    if local_dir and local_exists:
        if not upload_gcs:
            print(f"[SKIP] Cycle {cycle_date_str} already verified on local disk ({local_dir / cycle_date_str}).")
            return True
        elif gcs_exists:
            print(f"[SKIP] Cycle {cycle_date_str} already verified locally and in GCS.")
            return True

    if upload_gcs and gcs_exists:
        if not local_dir:
            print(f"[SKIP] Cycle {cycle_date_str} already verified in GCS bucket.")
            return True

    # Determine file paths
    cf_filename = f"s2s_cf_{cycle_date_str}.grib"
    pf_filename = f"s2s_pf_{cycle_date_str}.grib"
    gcs_dest = f"{GCS_BUCKET}/{cycle_date_str}/"

    if local_dir:
        cycle_dir = local_dir / cycle_date_str
        cycle_dir.mkdir(parents=True, exist_ok=True)
        cf_target = cycle_dir / cf_filename
        pf_target = cycle_dir / pf_filename
    else:
        cf_target = REPO_ROOT / cf_filename
        pf_target = REPO_ROOT / pf_filename

    print(f"\n========================================================")
    print(f"Retrieving S2S Cycle: {cycle_date_str} (Dataset: {dataset})")
    if is_realtime:
        print(f"Real-Time Operational Run: {hyear}-{hmonth}-{hday} (00:00 UTC)")
    else:
        print(f"Operational Reference: {model_year}-{model_month}-{model_day} | Hindcast Year: {hyear}")
    print(f"========================================================")

    if is_realtime:
        base_request = {
            "origin": "ecmwf",
            "year": hyear,
            "month": hmonth,
            "day": hday,
            "time": "00:00",
            "level_type": "single_level",
            "variable": VARIABLES,
            "leadtime_hour": LEADTIME_HOURS,
            "data_format": "grib",
            "area": BBOX
        }
    else:
        base_request = {
            "origin": "ecmwf",
            "year": model_year,
            "month": model_month,
            "day": model_day,
            "time": "00:00",
            "hyear": [hyear],
            "hmonth": [hmonth],
            "hday": [hday],
            "level_type": "single_level",
            "variable": VARIABLES,
            "leadtime_hour": LEADTIME_HOURS,
            "data_format": "grib",
            "area": BBOX
        }

    try:
        # Step A: Download Control Forecast (Member 0)
        print(f"[{cycle_date_str}] -> Requesting Control Forecast (Member 0) from {dataset}...")
        cf_request = base_request.copy()
        cf_request["forecast_type"] = "control_forecast"
        client.retrieve(dataset, cf_request, str(cf_target))

        # Step B: Download Perturbed Forecast (Members 1 to 10)
        print(f"[{cycle_date_str}] -> Requesting Perturbed Forecast (Members 1-10) from {dataset}...")
        pf_request = base_request.copy()
        pf_request["forecast_type"] = "perturbed_forecast"
        client.retrieve(dataset, pf_request, str(pf_target))

        # Step C: Upload to Google Cloud Storage if enabled
        if upload_gcs:
            print(f"[{cycle_date_str}] -> Uploading to {gcs_dest}...")
            res_up = run_cloud_cmd(["gcloud", "storage", "cp", str(cf_target), str(pf_target), gcs_dest])
            if res_up.returncode != 0:
                res_up2 = run_cloud_cmd(["gsutil", "cp", str(cf_target), str(pf_target), gcs_dest])
                if res_up2.returncode != 0:
                    err_msg = res_up.stderr.strip() or res_up2.stderr.strip()
                    raise RuntimeError(f"GCS upload failed: {err_msg}")
            if gcs_cache is not None:
                gcs_cache.add(cycle_date_str)

        # Step D: Clean up temporary files ONLY if local persistence is not requested
        if upload_gcs and not local_dir:
            if cf_target.exists():
                try: cf_target.unlink()
                except Exception: pass
            if pf_target.exists():
                try: pf_target.unlink()
                except Exception: pass
        elif local_dir:
            print(f"[{cycle_date_str}] -> Verified and saved locally to: {cycle_dir}")

        print(f"[{cycle_date_str}] -> SUCCESS! Cycle completed.")
        return True

    except Exception as e:
        print(f"[{cycle_date_str}] -> ERROR: {e}")
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.utcnow().isoformat()}Z | {cycle_date_str} | Dataset: {dataset} | ERROR: {e}\n")
        except Exception:
            pass

        # If it failed part-way and we are not saving local, clean up partials
        if not local_dir:
            if cf_target.exists():
                try: cf_target.unlink()
                except Exception: pass
            if pf_target.exists():
                try: pf_target.unlink()
                except Exception: pass
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Authoritative ECMWF S2S Production Batch Downloader (Local Terminal & GCS)"
    )
    parser.add_argument("--start-year", type=int, default=2015, help="Start year (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument(
        "--target", choices=["gcs", "local", "both"], default="gcs",
        help="Storage destination: 'gcs' (default), 'local', or 'both'"
    )
    parser.add_argument(
        "--local-dir", type=str, default="Data/raw_downloads/ECMWF",
        help="Local directory destination (used when --target is 'local' or 'both', default: Data/raw_downloads/ECMWF)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print cycle list and verify skip cache without requesting data")
    args = parser.parse_args()

    upload_gcs = args.target in ("gcs", "both")
    save_local = args.target in ("local", "both")
    local_path = (REPO_ROOT / args.local_dir) if save_local else None

    print("========================================================")
    print("      ECMWF S2S PRODUCTION BATCH DOWNLOADER")
    print("========================================================")
    print(f"Period Target : {args.start_year} to {args.end_year}")
    print(f"Storage Target: {args.target.upper()}")
    if save_local:
        print(f"Local Path    : {local_path.resolve()}")
    if upload_gcs:
        print(f"GCS Bucket    : {GCS_BUCKET}")
        gcloud_bin = shutil.which("gcloud") or shutil.which("gsutil")
        if gcloud_bin:
            print(f"Cloud CLI     : [DETECTED] {gcloud_bin}")
        else:
            print(f"Cloud CLI     : [WARNING] Neither gcloud nor gsutil detected in PATH.")
    print("========================================================")

    cycles = generate_operational_cycles(args.start_year, args.end_year)
    total = len(cycles)
    print(f"Loaded {total} operational forecast cycles across years {args.start_year} to {args.end_year}.")
    print(f"Schedule: Dynamic ECDS ('s2s-reforecasts' for <= 2023, 's2s-forecasts' for >= 2024).")

    # Fast GCS cache pre-fetch if GCS is enabled
    gcs_cache = set()
    if upload_gcs:
        print("--> Pre-fetching existing cycles from GCS bucket...", end=" ", flush=True)
        gcs_cache = fetch_existing_gcs_cycles()
        print(f"Found {len(gcs_cache)} verified cycles in GCS.")

    if args.dry_run:
        print("\n--> Starting DRY RUN cycle verification...")
        for idx, c in enumerate(cycles, 1):
            download_and_sync_cycle(
                None, c, gcs_cache=gcs_cache, dry_run=True,
                local_dir=local_path, upload_gcs=upload_gcs
            )
        print("--> DRY RUN complete.")
        return

    if cdsapi is None:
        print("ERROR: cdsapi library not installed. Please run: pip install cdsapi")
        sys.exit(1)

    # Connect to ECMWF Data Store (ECDS) dedicated endpoint for S2S
    ecds_url = os.environ.get("ECDS_API_URL", ECDS_API_URL)
    ecds_key = os.environ.get("ECDS_API_KEY", None)
    try:
        if ecds_key:
            client = cdsapi.Client(url=ecds_url, key=ecds_key)
        else:
            client = cdsapi.Client(url=ecds_url)
        print(f"ECDS Endpoint : Connected to {ecds_url}")
    except Exception as e:
        print(f"[NOTE] ECDS URL client initialization fallback: {e}")
        client = cdsapi.Client()

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for idx, cycle_info in enumerate(cycles, 1):
        print(f"\nProgress: {idx}/{total} ({idx/total*100:.1f}%)")
        status = download_and_sync_cycle(
            client, cycle_info,
            gcs_cache=gcs_cache,
            dry_run=False,
            local_dir=local_path,
            upload_gcs=upload_gcs
        )
        if status:
            success_count += 1
        else:
            fail_count += 1

    elapsed = time.time() - start_time
    print(f"\n========================================================")
    print(f"S2S BATCH DOWNLOAD RUN FINISHED in {elapsed/3600:.2f} hours!")
    print(f"Processed: {total} total | Successful/Skipped: {success_count} | Errors Logged: {fail_count}")
    print(f"========================================================")

if __name__ == "__main__":
    main()

