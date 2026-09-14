"""
src/data/calendar.py
--------------------
Authoritative Production Case Calendar & Operational Cycle Engine for Mindanao RISE-UNet.

Scientific and Operational Reference:
- Calendar Type: ECMWF CY48R1 Operational Schedule-Referenced Forecast Origin Calendar
- Total Coverage: 2015-01-01 through 2025-12-29 (1,154 total cycles)
- Partitioning:
    - Train (2015-2021): 735 cycles (105/yr)
    - Validation (2022-2023): 210 cycles (105/yr)
    - Sealed Test (2024-2025): 209 cycles (2024: 105, 2025: 104)

4-Way Data Intersection per Case:
    Usable Case = Antecedent RZSM Lags (t0 - 1d, 7d, 14d)
                x Future Target RZSM (W1-W4: t0 + 6d, 13d, 20d, 27d)
                x ERA5 Atmospheric Lags (t0 - 1d)
                x ECMWF S2S Forecast Issue & Lead Availability (W1-W2 dynamic forcings)
"""

import csv
import datetime
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# RZSM temporal boundary constraints
RZSM_CUBE_START = datetime.date(2015, 1, 1)
RZSM_CUBE_END = datetime.date(2025, 12, 31)
RZSM_ANTECEDENT_SUPPORT_START = datetime.date(2014, 12, 12)

# ERA5 atmospheric monthly archive covers all 264 months (2015-2025)
ERA5_ATM_START = datetime.date(2015, 1, 1)
ERA5_ATM_END = datetime.date(2025, 12, 31)

# Default GCS Lake Bucket for S2S
GCS_S2S_BUCKET = "gs://rise-unet-rzsm/raw/ecmwf_s2s/production"


def is_leap_year(year: int) -> bool:
    """Returns True if year is a leap year in the Gregorian calendar."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def generate_operational_cycles(start_year: int = 2015, end_year: int = 2025) -> List[Dict[str, Any]]:
    """
    Generates the complete sequence of ECMWF operational cycle dates.
    
    Under ECMWF CY48R1, reforecasts are archived on the month/day pairs matching
    the operational reference model runs (Mondays and Thursdays in 2024 for hindcasts,
    and actual operational days in 2024-2025).
    """
    cycles = []
    for hyear in range(start_year, end_year + 1):
        # Operational model year in CDS: 2024 for hindcasts <= 2023; real-time year for 2024+
        model_year = 2024 if hyear <= 2023 else hyear

        d = datetime.date(model_year, 1, 1)
        end_d = datetime.date(model_year, 12, 31)

        while d <= end_d:
            if d.weekday() in (0, 3):  # Monday or Thursday in operational schedule
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
                    "model_day": f"{day:02d}",
                })
            d += datetime.timedelta(days=1)
    return cycles


def get_available_s2s_cycles(
    repo_root: Optional[Path] = None,
    gcs_bucket: str = GCS_S2S_BUCKET,
) -> Set[str]:
    """Queries GCS or local cache to retrieve all verified S2S cycle date strings."""
    available_dates: Set[str] = set()
    root = repo_root or REPO_ROOT

    # Check if a cached listing exists from previous run in workspace scratch
    cache_path = root.parent / "scratch" / "current_gcs_s2s_list.txt"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-16le", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
            for l in lines:
                parts = l.rstrip("/").split("/")
                if len(parts) >= 2:
                    d_str = parts[-1]
                    if len(d_str) == 10 and d_str[4] == "-" and d_str[7] == "-":
                        available_dates.add(d_str)
        except Exception:
            pass

    if available_dates:
        return available_dates

    # Fallback: query gcloud storage directly
    try:
        res = subprocess.run(
            ["gcloud", "storage", "ls", f"{gcs_bucket}/"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            for l in res.stdout.strip().split("\n"):
                l = l.strip().rstrip("/")
                if not l:
                    continue
                parts = l.split("/")
                if len(parts) >= 2:
                    d_str = parts[-1]
                    if len(d_str) == 10 and d_str[4] == "-" and d_str[7] == "-":
                        available_dates.add(d_str)
    except Exception:
        pass

    return available_dates


def generate_production_calendar(
    start_year: int = 2015,
    end_year: int = 2025,
    repo_root: Optional[Path] = None,
    gcs_bucket: str = GCS_S2S_BUCKET,
) -> List[Dict[str, Any]]:
    """
    Generates all 1,154 ECMWF S2S operational forecast issuance cycles from 2015 to 2025,
    evaluating the 4-way intersection per cycle.
    """
    s2s_available = get_available_s2s_cycles(repo_root=repo_root, gcs_bucket=gcs_bucket)
    raw_cycles = generate_operational_cycles(start_year, end_year)

    calendar_rows = []

    for idx, c_info in enumerate(raw_cycles, start=1):
        cycle_date_str = c_info["cycle_date_str"]
        hyear = int(c_info["hyear"])
        model_year = int(c_info["model_year"])
        model_month = int(c_info["model_month"])
        model_day = int(c_info["model_day"])

        t0 = datetime.date.fromisoformat(cycle_date_str)
        model_date = datetime.date(model_year, model_month, model_day)

        # 1. Deterministic Antecedent Lags
        lag_1d = t0 - datetime.timedelta(days=1)
        lag_7d = t0 - datetime.timedelta(days=7)
        lag_14d = t0 - datetime.timedelta(days=14)

        # 2. Deterministic Verification Targets (W1..W4)
        target_w1 = t0 + datetime.timedelta(days=6)
        target_w2 = t0 + datetime.timedelta(days=13)
        target_w3 = t0 + datetime.timedelta(days=20)
        target_w4 = t0 + datetime.timedelta(days=27)

        # 3. Partition Split Assignment
        if hyear <= 2021:
            split = "TRAIN"
        elif hyear <= 2023:
            split = "VAL"
        else:
            split = "SEALED_TEST"

        # 4. Component Availability Checks
        antecedent_rzsm_ok = (lag_14d >= RZSM_ANTECEDENT_SUPPORT_START)
        target_rzsm_ok = (target_w4 <= RZSM_CUBE_END)
        era5_atm_ok = (t0 >= ERA5_ATM_START and t0 <= ERA5_ATM_END)
        s2s_ok = (cycle_date_str in s2s_available)

        # 5. Usability Classification
        if not target_rzsm_ok:
            usable_status = "TARGET_OUT_OF_BOUNDS"
            notes = f"W4 target date ({target_w4}) exceeds 2025-12-31 RZSM cube boundary"
        elif not antecedent_rzsm_ok:
            usable_status = "ANTECEDENT_OUT_OF_BOUNDS"
            notes = f"Lag-14d ({lag_14d}) precedes antecedent support (2014-12-12)"
        elif not era5_atm_ok:
            usable_status = "ATMOSPHERE_UNAVAILABLE"
            notes = "ERA5 atmospheric data unavailable"
        elif s2s_ok:
            usable_status = "USABLE_READY"
            notes = "All 4 data components verified and available in GCS lake"
        else:
            usable_status = "QUEUED_S2S_DOWNLOAD"
            notes = "Antecedent RZSM, future target RZSM, and ERA5 atm verified; S2S download in progress"

        case_id = f"CASE_{t0.strftime('%Y%m%d')}_{idx:04d}"

        row = {
            "case_id": case_id,
            "cycle_index": idx,
            "issue_date": cycle_date_str,
            "operational_run_date": model_date.isoformat(),
            "operational_weekday": model_date.strftime("%A"),
            "hindcast_weekday": t0.strftime("%A"),
            "split": split,
            "lag_1d_date": lag_1d.isoformat(),
            "lag_7d_date": lag_7d.isoformat(),
            "lag_14d_date": lag_14d.isoformat(),
            "target_w1_date": target_w1.isoformat(),
            "target_w2_date": target_w2.isoformat(),
            "target_w3_date": target_w3.isoformat(),
            "target_w4_date": target_w4.isoformat(),
            "antecedent_rzsm_available": antecedent_rzsm_ok,
            "future_target_rzsm_available": target_rzsm_ok,
            "era5_atmospheric_available": era5_atm_ok,
            "s2s_available": s2s_ok,
            "usable_status": usable_status,
            "notes": notes,
        }
        calendar_rows.append(row)

    return calendar_rows


def export_case_calendar(
    output_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generates and writes the production case calendar to CSV."""
    root = repo_root or REPO_ROOT
    dest = Path(output_path) if output_path else root / "manifests" / "production_case_calendar.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_production_calendar(repo_root=root)

    fieldnames = list(rows[0].keys())
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    census: Dict[str, Any] = {
        "total_cycles": len(rows),
        "by_split": {},
        "by_status": {},
        "by_year": {},
    }

    for r in rows:
        sp = r["split"]
        st = r["usable_status"]
        yr = r["issue_date"].split("-")[0]
        census["by_split"][sp] = census["by_split"].get(sp, 0) + 1
        census["by_status"][st] = census["by_status"].get(st, 0) + 1
        census["by_year"][yr] = census["by_year"].get(yr, 0) + 1

    return census


def load_case_calendar(calendar_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads production case calendar from CSV into list of dictionaries."""
    path = Path(calendar_path) if calendar_path else REPO_ROOT / "manifests" / "production_case_calendar.csv"
    if not path.exists():
        export_case_calendar(output_path=path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
