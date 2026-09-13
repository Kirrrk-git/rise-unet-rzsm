"""
src/data/s2s.py
---------------
Authoritative ECMWF S2S subseasonal reforecast harmonization engine for Mindanao RISE-UNet.
Implements pure-Python vectorized GRIB2 decoding (Template 0 Simple Packing),
weekly forecast step aggregation (Lead 1: days 1-7, Lead 2: days 8-14),
11-member ensemble structuring (Control + 10 Perturbed), and spatial remapping
to the frozen Candidate A 0.25° grid (32x48).

Scientific Governance:
  - Parent Model Parity: Strictly ingests the verified EX29 dynamic triplet
    (t2m, d2m, tcw) for Leads 1 and 2. Precipitation (tp) is explicitly excluded.
  - Coordinate Contract: Remaps native S2S grid (1.5° regular lat/lon) to
    Candidate A (32 lats descending from 11.75°N to 4.00°N, 48 lons ascending from 116.00°E to 127.75°E).
  - Masking Contract: Ocean/buffer non-evaluation cells are zero-filled per the frozen evaluation mask.
"""

import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Coordinate specifications for Candidate A domain
CANDIDATE_A_LATS = np.linspace(11.75, 4.00, 32, dtype=np.float32)
CANDIDATE_A_LONS = np.linspace(116.00, 127.75, 48, dtype=np.float32)

# ECMWF S2S Dynamic Triplet parameter identification in GRIB2
# Product Definition Template 4.60 / 4.61:
# Discipline 0 (Meteorological):
# - Cat 0, Param 0: 2m Temperature (t2m, K) - surf1 type 103 (2m above ground)
# - Cat 0, Param 6 (or surf1 height variant): 2m Dewpoint Temperature (d2m, K)
# - Cat 1, Param 52: Total Column Water Vapour (tcw, kg/m²)
TRIPLET_PARAMS = {
    "tcw": {"disc": 0, "cat": 1, "pnum": 52},
    "t2m": {"disc": 0, "cat": 0, "pnum": 0, "order": 1},
    "d2m": {"disc": 0, "cat": 0, "pnum": 0, "order": 2},
}


def decode_grib2_signed_int16(raw_val: int) -> int:
    """
    Decodes a 16-bit signed integer using WMO GRIB2 sign-magnitude encoding.
    (Bit 15 is sign bit: 1 = negative; Bits 0-14 represent magnitude).
    """
    if raw_val & 0x8000:
        return -(raw_val & 0x7FFF)
    return raw_val


def decode_grib2_signed_int32(raw_val: int) -> int:
    """
    Decodes a 32-bit signed integer using WMO GRIB2 sign-magnitude encoding.
    (Bit 31 is sign bit: 1 = negative; Bits 0-30 represent magnitude).
    """
    if raw_val & 0x80000000:
        return -(raw_val & 0x7FFFFFFF)
    return raw_val


def unpack_grib2_section7_simple(
    sec7_bytes: bytes,
    shape: Tuple[int, int],
    r: float,
    e: int,
    d: int,
    nbits: int,
) -> np.ndarray:
    """
    Decodes GRIB2 Section 7 packed data using Data Representation Template 5.0 (Simple Packing).

    Formula:
        Y = (R + X * 2^E) / (10^D)
    where:
        R = Reference value (IEEE 32-bit float)
        E = Binary scale factor (int16)
        D = Decimal scale factor (int16)
        nbits = Number of bits per packed integer X
    """
    nj, ni = shape
    total_pts = nj * ni

    if nbits == 0:
        # Constant field across the entire grid
        val = (r) / (10.0**d) if d != 0 else r
        return np.full((nj, ni), val, dtype=np.float32)

    raw_bytes = np.frombuffer(sec7_bytes, dtype=np.uint8)
    bits = np.unpackbits(raw_bytes)
    bits_needed = total_pts * nbits
    if len(bits) < bits_needed:
        raise ValueError(
            f"Bitstream length ({len(bits)}) is smaller than required ({bits_needed}) for shape {shape} at {nbits} bits."
        )

    bits_reshaped = bits[:bits_needed].reshape(total_pts, nbits)
    powers = 2 ** np.arange(nbits, dtype=np.uint32)[::-1]
    X = bits_reshaped.dot(powers)

    # Vectorized scale evaluation
    scale_bin = 2.0**e
    scale_dec = 10.0**d
    Y = (r + X * scale_bin) / scale_dec
    return Y.reshape(nj, ni).astype(np.float32)


def parse_ecmwf_s2s_grib_messages(filepath: Union[str, Path]) -> List[Dict]:
    """
    Parses an ECMWF S2S GRIB2 file in pure Python and decodes all contained messages.

    Returns:
        List of message dictionaries with keys:
            'date': issue/hindcast date (YYYY-MM-DD)
            'step': forecast lead step in hours (e.g. 0, 6, 12, ..., 330)
            'member': ensemble member (0 for CF, 1-10 for PF)
            'var': variable name ('t2m', 'd2m', 'tcw')
            'grid': 2D NumPy array of decoded physical values (lat, lon)
            'lats': 1D NumPy array of latitudes
            'lons': 1D NumPy array of longitudes
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GRIB file not found: {filepath}")

    with open(filepath, "rb") as f:
        data = f.read()

    messages = []
    pos = 0
    t_order_toggle = 0

    while pos < len(data):
        idx = data.find(b"GRIB", pos)
        if idx == -1:
            break

        total_len = struct.unpack(">Q", data[idx + 8 : idx + 16])[0]
        end_idx = idx + total_len
        msg_bytes = data[idx:end_idx]

        disc = msg_bytes[6]
        sec_pos = 16
        msg_meta = {"disc": disc}
        grid_data = None
        lats_1d = None
        lons_1d = None
        shape = None
        packing = None

        while sec_pos < len(msg_bytes) - 4:
            sec_len = struct.unpack(">I", msg_bytes[sec_pos : sec_pos + 4])[0]
            sec_num = msg_bytes[sec_pos + 4]

            if sec_num == 1:
                # Identification Section
                ref_year = struct.unpack(">H", msg_bytes[sec_pos + 12 : sec_pos + 14])[0]
                ref_mon = msg_bytes[sec_pos + 14]
                ref_day = msg_bytes[sec_pos + 15]
                hdate_str = f"{ref_year:04d}-{ref_mon:02d}-{ref_day:02d}"
                msg_meta["hdate"] = hdate_str
                msg_meta["date"] = hdate_str  # Maintained as primary historical forecast issue date

            elif sec_num == 3:
                # Grid Definition Section
                ni, nj = struct.unpack(">II", msg_bytes[sec_pos + 30 : sec_pos + 38])
                la1_raw = struct.unpack(">I", msg_bytes[sec_pos + 46 : sec_pos + 50])[0]
                lo1_raw = struct.unpack(">I", msg_bytes[sec_pos + 50 : sec_pos + 54])[0]
                la2_raw = struct.unpack(">I", msg_bytes[sec_pos + 55 : sec_pos + 59])[0]
                lo2_raw = struct.unpack(">I", msg_bytes[sec_pos + 59 : sec_pos + 63])[0]
                la1 = decode_grib2_signed_int32(la1_raw) / 1e6
                lo1 = decode_grib2_signed_int32(lo1_raw) / 1e6
                la2 = decode_grib2_signed_int32(la2_raw) / 1e6
                lo2 = decode_grib2_signed_int32(lo2_raw) / 1e6
                shape = (nj, ni)
                lats_1d = np.linspace(la1, la2, nj, dtype=np.float32)
                lons_1d = np.linspace(lo1, lo2, ni, dtype=np.float32)

            elif sec_num == 4:
                # Product Definition Section
                pdt = struct.unpack(">H", msg_bytes[sec_pos + 7 : sec_pos + 9])[0]
                cat = msg_bytes[sec_pos + 9]
                pnum = msg_bytes[sec_pos + 10]
                step = struct.unpack(">I", msg_bytes[sec_pos + 18 : sec_pos + 22])[0]
                msg_meta["pdt"] = pdt
                msg_meta["cat"] = cat
                msg_meta["pnum"] = pnum
                msg_meta["step"] = step

                # Perturbation number (byte 28 offset in PDT 61)
                if len(msg_bytes) >= sec_pos + 36:
                    # In PDT 61, perturbation number is at sec_pos + 7 + 28 = sec_pos + 35
                    pert_num = msg_bytes[sec_pos + 35]
                    msg_meta["member"] = pert_num
                else:
                    msg_meta["member"] = 0

                # Operational model version date in PDT 60/61 (Octets 39-45 in Sec 4)
                if pdt in (60, 61) and sec_len >= 42:
                    model_year = struct.unpack(">H", msg_bytes[sec_pos + 37 : sec_pos + 39])[0]
                    model_mon = msg_bytes[sec_pos + 39]
                    model_day = msg_bytes[sec_pos + 40]
                    msg_meta["model_version_date"] = f"{model_year:04d}-{model_mon:02d}-{model_day:02d}"
                else:
                    msg_meta["model_version_date"] = None

            elif sec_num == 5:
                # Data Representation Section
                drt = struct.unpack(">H", msg_bytes[sec_pos + 9 : sec_pos + 11])[0]
                r = struct.unpack(">f", msg_bytes[sec_pos + 11 : sec_pos + 15])[0]
                e_raw = struct.unpack(">H", msg_bytes[sec_pos + 15 : sec_pos + 17])[0]
                d_raw = struct.unpack(">H", msg_bytes[sec_pos + 17 : sec_pos + 19])[0]
                e = decode_grib2_signed_int16(e_raw)
                d = decode_grib2_signed_int16(d_raw)
                nbits = msg_bytes[sec_pos + 19]
                packing = (drt, r, e, d, nbits)

            elif sec_num == 7:
                # Data Section
                sec7_data = msg_bytes[sec_pos + 5 : sec_pos + sec_len]
                if packing and shape:
                    drt, r, e, d, nbits = packing
                    if drt == 0:
                        grid_data = unpack_grib2_section7_simple(sec7_data, shape, r, e, d, nbits)
                    else:
                        raise NotImplementedError(f"Unsupported GRIB2 DRT: {drt}")
                break

            sec_pos += sec_len
            if msg_bytes[sec_pos : sec_pos + 4] == b"7777":
                break

        msg_meta["grid"] = grid_data
        msg_meta["lats"] = lats_1d
        msg_meta["lons"] = lons_1d
        messages.append(msg_meta)
        pos = end_idx

    # Assign variable identity: clean ECDS files use pnum 6 for d2m and pnum 0 for t2m;
    # legacy pilot files lack pnum 6 and interleave t2m/d2m under pnum 0.
    has_pnum_6 = any(m.get("pnum") == 6 for m in messages)
    t_order_toggle = 0
    for m in messages:
        cat = m.get("cat")
        pnum = m.get("pnum")
        if cat == 1 and pnum in (51, 52):
            var_name = "tcw"
        elif cat == 0 and pnum == 6:
            var_name = "d2m"
        elif cat == 0 and pnum == 0:
            if has_pnum_6:
                var_name = "t2m"
            else:
                t_order_toggle += 1
                var_name = "t2m" if (t_order_toggle % 2 == 1) else "d2m"
        else:
            var_name = f"var_{cat}_{pnum}"
        m["var"] = var_name

    return messages


def remap_s2s_grid_to_candidate_a(
    src_grid: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    target_lats: np.ndarray = CANDIDATE_A_LATS,
    target_lons: np.ndarray = CANDIDATE_A_LONS,
) -> np.ndarray:
    """
    Remaps a 2D S2S grid to Candidate A (32x48) using bilinear interpolation
    with nearest-boundary extrapolation for outer buffer cells.
    """
    # RegularGridInterpolator requires strictly ascending coordinate axes
    lat_ascending = src_lats[0] < src_lats[-1]
    lon_ascending = src_lons[0] < src_lons[-1]

    s_lats = src_lats if lat_ascending else src_lats[::-1]
    s_lons = src_lons if lon_ascending else src_lons[::-1]
    s_grid = src_grid.copy()
    if not lat_ascending:
        s_grid = s_grid[::-1, :]
    if not lon_ascending:
        s_grid = s_grid[:, ::-1]

    interp = RegularGridInterpolator(
        (s_lats, s_lons),
        s_grid,
        bounds_error=False,
        fill_value=None,  # Nearest boundary / linear extrapolation
    )

    t_grid = np.meshgrid(target_lats, target_lons, indexing="ij")
    pts = np.stack([t_grid[0].ravel(), t_grid[1].ravel()], axis=-1)
    remapped = interp(pts).reshape((len(target_lats), len(target_lons)))
    return remapped.astype(np.float32)


def harmonize_s2s_cycle(
    cf_path: Union[str, Path],
    pf_path: Optional[Union[str, Path]] = None,
    target_date: Optional[str] = None,
    eval_mask_path: Optional[Union[str, Path]] = None,
    allow_step0_fallback: bool = False,
) -> xr.Dataset:
    """
    Harmonizes an ECMWF S2S reforecast cycle (Control + Perturbed forecasts)
    into a standardized xarray Dataset conforming to RISE-UNet EX29 specifications.

    Parameters:
        cf_path: Path to Control Forecast GRIB2 file (Member 0).
        pf_path: Optional path to Perturbed Forecast GRIB2 file (Members 1-10).
        target_date: Forecast issuance date (hdate, YYYY-MM-DD). If None, uses earliest available date.
        eval_mask_path: Optional path to binary evaluation mask NetCDF (for ocean zero-filling).
        allow_step0_fallback: If False (DEFAULT/PRODUCTION), missing required forecast steps raise a hard
                              ValueError, rejecting defective cases. If True (DEBUG/PILOT ONLY), permits fallback
                              to step 0 with an explicit warning and logs a diagnostic flag in dataset attributes.

    Output Dimensions:
        (lead=2, member=11, lat=32, lon=48)
    Data Variables:
        t2m: 2m Temperature (K)
        d2m: 2m Dewpoint Temperature (K)
        tcw: Total Column Water (kg/m²)
    """
    cf_msgs = parse_ecmwf_s2s_grib_messages(cf_path)
    all_msgs = list(cf_msgs)

    if pf_path is not None:
        pf_msgs = parse_ecmwf_s2s_grib_messages(pf_path)
        all_msgs.extend(pf_msgs)

    # Determine target issue date (hdate)
    available_dates = sorted(list(set(m["hdate"] for m in all_msgs if "hdate" in m)))
    if not available_dates:
        available_dates = sorted(list(set(m["date"] for m in all_msgs)))
    if target_date is None:
        target_date = available_dates[0]
    elif target_date not in available_dates:
        raise ValueError(f"Requested date {target_date} not in file. Available: {available_dates}")

    # Extract operational model version date if present in PDT 60/61
    model_version_dates = sorted(list(set(m["model_version_date"] for m in all_msgs if m.get("model_version_date") is not None)))
    model_version_date = model_version_dates[0] if model_version_dates else None

    # Filter messages for the target date and variables
    cycle_msgs = [
        m for m in all_msgs
        if (m.get("hdate") == target_date or m.get("date") == target_date)
        and m["var"] in ["t2m", "d2m", "tcw"]
    ]

    # Members present
    members = sorted(list(set(m["member"] for m in cycle_msgs)))
    num_members = len(members)

    # Weekly lead aggregation bins:
    # Under the author's daily lead index contract (00_min_max_...:Cell 13, 29):
    # Week 1: daily leads L=0..6, corresponding to forecast steps 0 <= step < 168 hours
    #         (hours 0, 24, 48, 72, 96, 120, 144 for daily resolution; hours 0..162 for 6-hourly)
    # Week 2: daily leads L=7..13, corresponding to forecast steps 168 <= step < 336 hours
    #         (hours 168, 192, 216, 240, 264, 288, 312 for daily resolution; hours 168..330 for 6-hourly)
    lead_bins = {
        1: (0, 168),
        2: (168, 336),
    }

    src_lats = cycle_msgs[0]["lats"]
    src_lons = cycle_msgs[0]["lons"]

    # Storage arrays: (lead=2, member=M, lat=32, lon=48)
    compiled_vars = {}
    for var in ["t2m", "d2m", "tcw"]:
        compiled_vars[var] = np.zeros(
            (2, num_members, len(CANDIDATE_A_LATS), len(CANDIDATE_A_LONS)),
            dtype=np.float32,
        )

    contains_fallback_detected = False
    for m_idx, mem in enumerate(members):
        for lead_idx, lead in enumerate([1, 2]):
            step_min, step_max = lead_bins[lead]
            for var in ["t2m", "d2m", "tcw"]:
                matching = [
                    m["grid"]
                    for m in cycle_msgs
                    if m["member"] == mem
                    and m["var"] == var
                    and step_min <= m["step"] < step_max
                ]
                if not matching:
                    if allow_step0_fallback:
                        # Fallback to step=0 permitted ONLY in explicit debug/pilot mode
                        matching = [
                            m["grid"]
                            for m in cycle_msgs
                            if m["member"] == mem and m["var"] == var and m["step"] == 0
                        ]
                        if matching:
                            contains_fallback_detected = True
                            print(
                                f"[WARNING: STEP-0 FALLBACK APPLIED] Member {mem}, var '{var}', lead {lead} "
                                f"missing forecast steps in [{step_min}, {step_max})h; fell back to step 0."
                            )
                    else:
                        raise ValueError(
                            f"[CRITICAL S2S ERROR] Missing required forecast steps for member {mem}, "
                            f"var '{var}', lead {lead} (expected in window [{step_min}, {step_max})h). "
                            f"Step-0 fallback is strictly forbidden in production mode (allow_step0_fallback=False). Case rejected."
                        )

                if matching:
                    # Average over steps within the lead week
                    mean_grid = np.mean(matching, axis=0)
                    remapped_grid = remap_s2s_grid_to_candidate_a(
                        mean_grid, src_lats, src_lons, CANDIDATE_A_LATS, CANDIDATE_A_LONS
                    )
                    compiled_vars[var][lead_idx, m_idx, :, :] = remapped_grid
                else:
                    raise ValueError(
                        f"Missing S2S steps for member {mem}, var '{var}', lead {lead} ({step_min}-{step_max}h)"
                    )

    # Optional masking
    if eval_mask_path is not None and Path(eval_mask_path).exists():
        mask_ds = xr.open_dataset(eval_mask_path)
        eval_mask = mask_ds["evaluation_mask"].values
        # Zero-fill ocean/buffer cells outside evaluation mask
        for var in ["t2m", "d2m", "tcw"]:
            for l in range(2):
                for m in range(num_members):
                    compiled_vars[var][l, m, :, :] = np.where(
                        eval_mask == 1, compiled_vars[var][l, m, :, :], 0.0
                    )

    # Assemble xarray Dataset with explicit hdate and model_version_date
    ds = xr.Dataset(
        data_vars={
            "t2m": (("lead", "member", "lat", "lon"), compiled_vars["t2m"]),
            "d2m": (("lead", "member", "lat", "lon"), compiled_vars["d2m"]),
            "tcw": (("lead", "member", "lat", "lon"), compiled_vars["tcw"]),
        },
        coords={
            "lead": [1, 2],
            "member": members,
            "lat": CANDIDATE_A_LATS,
            "lon": CANDIDATE_A_LONS,
        },
        attrs={
            "description": "Harmonized ECMWF S2S subseasonal reforecast dataset for Mindanao RISE-UNet",
            "issue_date": target_date,
            "hdate": target_date,
            "model_version_date": model_version_date if model_version_date else "unknown",
            "variables": "t2m, d2m, tcw",
            "leads": "Week 1 (daily leads L=0..6, hours 0-144h), Week 2 (daily leads L=7..13, hours 168-312h)",
            "ensemble_members": num_members,
            "contains_step0_fallback": contains_fallback_detected,
            "remapping_method": "bilinear_with_nearest_boundary_fallback",
        },
    )

    return ds
