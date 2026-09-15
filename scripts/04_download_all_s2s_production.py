#!/usr/bin/env python3
"""
Dynamic Production ECMWF S2S Reforecast Downloader for Mindanao (2015-2025)
API: Copernicus Climate Data Store (cdsapi)
Target: gs://rise-unet-rzsm/raw/ecmwf_s2s/production/

Key Scientific & Engineering Design:
1. Dynamic Operational Schedule:
   - Evaluates all operational forecast issuance dates (every Monday and Thursday)
     of the corresponding model version without any hardcoded date skipping.
   - For hindcast years (hyear <= 2023), ECMWF runs the CY48R1 reforecasts during the
     2024 operational schedule (105 runs across all 12 months, including late December).
   - For real-time years (hyear >= 2024), it queries the operational runs of that year.
2. Smart Resumption & GCS Parity:
   - Pre-checks GCS bucket (gs://rise-unet-rzsm/raw/ecmwf_s2s/production/<cycle>/).
   - Skips already-verified cycles in ~0.1s.
   - Automatically backfills any previously omitted cycles.
3. Resilient Error Handling:
   - Never crashes on an isolated server error or unissued cycle.
   - Logs skipped/failed dates to 's2s_download_errors.log' and proceeds autonomously.
"""

import os
import sys
import time
import argparse
import datetime
import subprocess

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

def check_gcs_cycle_exists(cycle_date_str: str) -> bool:
    """Checks if both control and perturbed forecast GRIB files exist in GCS."""
    try:
        res = subprocess.run(
            ["gcloud", "storage", "ls", f"{GCS_BUCKET}/{cycle_date_str}/*.grib"],
            capture_output=True, text=True, check=False
        )
        lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            return True
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["gsutil", "ls", f"{GCS_BUCKET}/{cycle_date_str}/*.grib"],
            capture_output=True, text=True, check=False
        )
        lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
        return len(lines) >= 2
    except Exception:
        return False

def download_and_sync_cycle(
    client,
    cycle_info: dict,
    dry_run: bool = False,
    local_dir: str = None,
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

    if dry_run:
        print(f"[DRY RUN] Cycle: {cycle_date_str} -> Dataset: {dataset} (Model: {model_year}-{model_month}-{model_day})")
        return True

    if upload_gcs and check_gcs_cycle_exists(cycle_date_str):
        print(f"[SKIP] Cycle {cycle_date_str} already verified in GCS bucket.")
        return True

    cf_filename = f"s2s_cf_{cycle_date_str}.grib"
    pf_filename = f"s2s_pf_{cycle_date_str}.grib"
    gcs_dest = f"{GCS_BUCKET}/{cycle_date_str}/"

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
        # 1. Download Control Forecast (Member 0)
        print(f"[{cycle_date_str}] -> Requesting Control Forecast (Member 0) from {dataset}...")
        cf_request = base_request.copy()
        cf_request["forecast_type"] = "control_forecast"
        client.retrieve(dataset, cf_request, cf_filename)

        # 2. Download Perturbed Forecast (Members 1 to 10)
        print(f"[{cycle_date_str}] -> Requesting Perturbed Forecast (Members 1-10) from {dataset}...")
        pf_request = base_request.copy()
        pf_request["forecast_type"] = "perturbed_forecast"
        client.retrieve(dataset, pf_request, pf_filename)

        # 3. Direct upload to GCS bucket if enabled
        if upload_gcs:
            print(f"[{cycle_date_str}] -> Uploading to {gcs_dest}...")
            try:
                subprocess.run(["gcloud", "storage", "cp", cf_filename, pf_filename, gcs_dest], check=True)
            except Exception:
                subprocess.run(["gsutil", "cp", cf_filename, pf_filename, gcs_dest], check=True)

        # 4. Handle local persistence if requested
        if local_dir:
            import shutil
            from pathlib import Path
            target_dir = Path(local_dir) / cycle_date_str
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(cf_filename, str(target_dir / cf_filename))
            shutil.copy(pf_filename, str(target_dir / pf_filename))
            print(f"[{cycle_date_str}] -> Saved locally to: {target_dir}")

        # Clean up temporary working directory files if uploaded or copied
        if upload_gcs or local_dir:
            if os.path.exists(cf_filename): os.remove(cf_filename)
            if os.path.exists(pf_filename): os.remove(pf_filename)

        print(f"[{cycle_date_str}] -> SUCCESS! Cycle completed.")
        return True

    except Exception as e:
        print(f"[{cycle_date_str}] -> ERROR: {e}")
        with open("s2s_download_errors.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.utcnow().isoformat()}Z | {cycle_date_str} | Dataset: {dataset} | ERROR: {e}\n")
        if os.path.exists(cf_filename): os.remove(cf_filename)
        if os.path.exists(pf_filename): os.remove(pf_filename)
        return False

def main():
    parser = argparse.ArgumentParser(description="Authoritative ECMWF S2S Production Batch Downloader to GCS & Local Terminal")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument("--local-dir", type=str, default=None, help="Optional local directory to store GRIB files")
    parser.add_argument("--no-gcs-upload", action="store_true", help="Skip uploading to Google Cloud Storage")
    parser.add_argument("--dry-run", action="store_true", help="Print cycle list without requesting data")
    args = parser.parse_args()

    cycles = generate_operational_cycles(args.start_year, args.end_year)
    total = len(cycles)
    print(f"Loaded {total} operational forecast cycles across years {args.start_year} to {args.end_year}.")
    print(f"Schedule: Fully dynamic (ECDS s2s-reforecasts for <= 2023, s2s-forecasts for >= 2024).")

    upload_gcs = not args.no_gcs_upload

    if args.dry_run:
        for idx, c in enumerate(cycles, 1):
            download_and_sync_cycle(None, c, dry_run=True, local_dir=args.local_dir, upload_gcs=upload_gcs)
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
            dry_run=False,
            local_dir=args.local_dir,
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
