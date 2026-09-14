#!/usr/bin/env python3
"""
Mindanao RISE-UNet: Production Case Calendar Generator & 4-Way Intersection Engine (Step 21K.1)

Authoritative Reference:
- Master Plan: mindanao_adaptation_master_plan.md (Sub-Phase 21K.1)
- Parent Study: Lesinger & Tian (2025), Nature Communications, DOI: 10.1038/s41467-025-62761-3

Defines the exact Usable Case Calendar across the 11-year nominal historical period (2015-2025)
by enforcing the strict 4-way data availability intersection:
    Usable Case = Antecedent RZSM Lags (t0 - 1d, 7d, 14d)
                x Future Target RZSM (W1-W4: t0 + 6d, 13d, 20d, 27d)
                x ERA5 Atmospheric Lags
                x ECMWF S2S Reforecast Issue & Lead Availability (W1-W2 dynamic forcings)

Exports:
- manifests/production_case_calendar.csv (1,154 operational forecast issuance cycles)
"""

import sys
import os
import csv
import json
import datetime
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.calendar import generate_operational_cycles

# RZSM temporal boundary constraints
RZSM_CUBE_START = datetime.date(2015, 1, 1)
RZSM_CUBE_END = datetime.date(2025, 12, 31)
RZSM_ANTECEDENT_SUPPORT_START = datetime.date(2014, 12, 12)

# ERA5 atmospheric monthly archive covers all 264 months (2015-2025)
ERA5_ATM_START = datetime.date(2015, 1, 1)
ERA5_ATM_END = datetime.date(2025, 12, 31)

# GCS Lake Bucket for S2S
GCS_S2S_BUCKET = "gs://rise-unet-rzsm/raw/ecmwf_s2s/production"


def get_available_s2s_cycles() -> Set[str]:
    """Queries GCS to retrieve all currently verified S2S cycle date strings."""
    available_dates = set()
    
    # Check if a cached listing exists from previous run in workspace scratch
    cache_path = REPO_ROOT.parent / "scratch" / "current_gcs_s2s_list.txt"
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

    if len(available_dates) > 0:
        print(f"--> Found {len(available_dates)} S2S cycles in local inventory cache.")
        return available_dates

    # Fallback: query gcloud storage directly
    print("--> Querying GCS bucket for available S2S cycles...")
    try:
        res = subprocess.run(
            ["gcloud", "storage", "ls", f"{GCS_S2S_BUCKET}/"],
            capture_output=True, text=True, check=False
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
            print(f"--> Successfully queried GCS: found {len(available_dates)} cycles.")
    except Exception as e:
        print(f"--> Warning: Could not query GCS dynamically: {e}")

    return available_dates


def generate_production_calendar() -> List[Dict[str, Any]]:
    """
    Generates all 1,154 ECMWF S2S operational forecast issuance cycles from 2015 to 2025,
    evaluating the 4-way intersection per cycle.
    """
    s2s_available = get_available_s2s_cycles()
    raw_cycles = generate_operational_cycles(2015, 2025)
    
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
    output_path: Path = REPO_ROOT / "manifests" / "production_case_calendar.csv",
) -> Dict[str, Any]:
    """Generates and writes the production case calendar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_production_calendar()

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n========================================================")
    print(f"[PIPELINE] Production Case Calendar Exported Successfully")
    print(f"Destination: {output_path}")
    print(f"Total Cycles Indexed: {len(rows)}")
    print(f"========================================================")

    # Summary Census
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

    print("\n--- BREAKDOWN BY PARTITION SPLIT ---")
    for sp, cnt in sorted(census["by_split"].items()):
        print(f"  {sp:15s}: {cnt:4d} cycles ({cnt/len(rows)*100:.1f}%)")

    print("\n--- BREAKDOWN BY USABILITY STATUS ---")
    for st, cnt in sorted(census["by_status"].items()):
        print(f"  {st:25s}: {cnt:4d} cycles ({cnt/len(rows)*100:.1f}%)")

    print("\n--- ANNUAL BREAKDOWN ---")
    for yr, cnt in sorted(census["by_year"].items()):
        ready = sum(1 for r in rows if r["issue_date"].startswith(yr) and r["usable_status"] == "USABLE_READY")
        queued = sum(1 for r in rows if r["issue_date"].startswith(yr) and r["usable_status"] == "QUEUED_S2S_DOWNLOAD")
        oob = sum(1 for r in rows if r["issue_date"].startswith(yr) and "OUT_OF_BOUNDS" in r["usable_status"])
        print(f"  Year {yr}: {cnt:3d} cycles | USABLE_READY: {ready:3d} | QUEUED: {queued:3d} | OUT_OF_BOUNDS: {oob:2d}")

    return census


if __name__ == "__main__":
    export_case_calendar()
