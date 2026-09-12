<!-- markdownlint-disable -->
# ERA5 Atmospheric Reanalysis Archive (Mindanao Domain)

**Authoritative Cloud Storage of Record**: `gs://rise-unet-rzsm/raw/era5/`  
**Spatial Domain**: Mindanao Candidate A Grid (`[11.75°N, 116.00°E, 4.00°N, 127.75°E]`, $32 \times 48$ regular $0.25^\circ \times 0.25^\circ$)  
**Temporal Domain**: 2014-12 (antecedent window) through 2025-12 (11-year baseline)  
**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Sub-Phase**: 21C (Atmospheric Reanalysis Acquisition)  

---

## 1. Purpose & Data Policy

This directory serves as the **local acquisition landing zone** for raw ERA5 atmospheric reanalysis variables over Mindanao.

### Git Storage & Cloud Lake Architecture:
- **Raw NetCDF files (`*.nc`) are excluded from Git** via `.gitignore` to prevent repository bloat and comply with GitHub's file size limits.
- **Google Cloud Storage (`gs://rise-unet-rzsm/raw/era5/`)** serves as the authoritative, permanent cloud data lake for all raw multi-year NetCDF archives.
- **This README is tracked in Git** to preserve the directory structure across clones and provide immediate, actionable retrieval instructions for collaborators.

---

## 2. Directory Structure

```text
Data/raw_downloads/ERA5_atmospheric/
├── README.md               <-- Tracked in Git (this file)
├── single/                 <-- Local landing zone for surface variables (*.nc, git-ignored)
│   ├── era5_single_levels_2015_01.nc
│   └── ...
└── pressure/               <-- Local landing zone for 200 hPa dynamics (*.nc, git-ignored)
    ├── era5_z200_2015_01.nc
    └── ...
```

---

## 3. Extracted Variables & Model Channel Mappings

| Directory | ERA5 Variable Name | ECMWF Short Name | Resolution | Target RISE-UNet Model Channel |
| :--- | :--- | :--- | :--- | :--- |
| `single/` | `2m_temperature` | `2t` / `t2m` | Hourly, 0.25° | `tmax` (daily max) & `diff_temp` ($T_{max} - T_{min}$) |
| `single/` | `2m_dewpoint_temperature` | `2d` / `d2m` | Hourly, 0.25° | `spfh` (surface specific humidity via Bolton 1980) |
| `single/` | `surface_pressure` | `sp` | Hourly, 0.25° | `spfh` (barometric pressure input to Bolton 1980) |
| `single/` | `total_column_water_vapour` | `tcwv` | Hourly, 0.25° | `pwat` (precipitable water daily mean) |
| `pressure/` | `geopotential` (at 200 hPa) | `z` | Hourly, 0.25° | `hgt_pres` (200 hPa geopotential height: $z / 9.80665$ gpm) |

---

## 4. How to Retrieve & Synchronize Data

### Option A: Pull the Archive from Google Cloud Storage
If the data is already extracted and synchronized to GCS, pull it directly to this folder:
```powershell
gcloud storage rsync -r gs://rise-unet-rzsm/raw/era5/ Data/raw_downloads/ERA5_atmospheric/
```

### Option B: Acquire from Copernicus CDS API
To download raw NetCDF files directly from the ECMWF Copernicus Climate Data Store:
```powershell
# 1-month benchmark pilot (January 2015)
python scripts/download_era5_atmospheric_mindanao.py --pilot

# Full 11-year baseline archive (2015–2025)
python scripts/download_era5_atmospheric_mindanao.py --start-year 2015 --end-year 2025

# Antecedent December 2014 window
python scripts/download_era5_atmospheric_mindanao.py --year 2014 --month 12
```

### Option C: Synchronize Local Downloads to GCS
To back up all completed local downloads to the cloud data lake:
```powershell
gcloud storage rsync -r Data/raw_downloads/ERA5_atmospheric/ gs://rise-unet-rzsm/raw/era5/
```

---

## 5. Verification & Quality Assurance

After downloading any month, run the verification and derivation suite:
```powershell
python scripts/verify_and_derive_era5_pilot.py
```
This certifies:
1. Exact $(31, 32, 48)$ tensor geometry matching Candidate A grid coordinates.
2. **0 NaNs** and **0 Infs** across the 126 active evaluation land cells of Mindanao.
3. Proper Bolton (1980) thermodynamic specific humidity derivation.
4. Export of the derived 5-channel pilot dataset and 300 DPI publication composite.
