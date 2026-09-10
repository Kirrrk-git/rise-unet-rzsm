# ERA5-Land Mindanao RZSM Pilot Reference Dataset

**Filename**: `era5_land_rzsm_pilot_2014_2015.nc`  
**File Size**: 133,762 bytes (~130.6 KB)  
**Conventions**: CF-1.8  
**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Sub-Phase**: 21B (Steps 21B.3 & 21B.4)  
**Audit Dossier**: [`reproduction_audit/ERA5_LAND_PILOT_RZSM_PREPROCESSING_AUDIT.md`](../../../reproduction_audit/ERA5_LAND_PILOT_RZSM_PREPROCESSING_AUDIT.md)  
**Pipeline Notebook**: [`notebooks/04_mindanao_rzsm_pilot_preprocessing.ipynb`](../../../notebooks/04_mindanao_rzsm_pilot_preprocessing.ipynb)  

---

## 1. Purpose & Architecture of This Pilot File

This dataset is committed to the Git repository as a **lightweight, offline reference artifact**. 

### Why is this file in Git?
1. **Immediate Offline Reproducibility**: Anyone cloning this repository can immediately inspect, visualize, and verify the model's preprocessed input tensors without needing Google Cloud Platform (GCP) authentication, service account keys, or cloud storage downloads.
2. **Unit Testing & CI/CD Pipeline Verification**: PyTorch `Dataset` and `DataLoader` classes, model forward passes, and training graph validations can run instantaneous regression tests against real, preprocessed regional tensors in automated GitHub Actions.
3. **Data Lake Boundary Enforcement**: Full-scale 11-year datasets (2015–2025, multi-gigabyte arrays) are stored in Google Cloud Storage (`gs://rise-unet-rzsm/processed/rzsm/production/`) to prevent Git repository bloat. This pilot file serves as the definitive structural template for all production datasets.

---

## 2. Dataset Specifications

| Attribute | Specification |
| :--- | :--- |
| **Source Data** | ERA5-Land Reanalysis (ECMWF Copernicus Climate Change Service) |
| **Temporal Span** | 12 December 2014 to 31 December 2014 (20 calendar days) |
| **Hourly Aggregation** | 24-hour unweighted daily arithmetic mean ($480\text{ hours} \to 20\text{ daily timesteps}$) |
| **Depth Integration** | Top 100 cm: $\text{RZSM}_{0-100} = 0.07\,\text{swvl}_1 + 0.21\,\text{swvl}_2 + 0.72\,\text{swvl}_3$ |
| **Spatial Grid** | Candidate A Mindanao Grid ($32 \times 48$, regular $0.25^\circ \times 0.25^\circ$) |
| **Coordinate Bounds** | Longitude: $116.00^\circ\text{E} \to 127.75^\circ\text{E}$ (48 points) <br> Latitude: $11.75^\circ\text{N} \to 4.00^\circ\text{N}$ (32 points, descending) |
| **Active Evaluation Cells** | **126 cells** (Majority land rule: fractional coverage $f \ge 0.50$, covering $96,085.57\text{ km}^2$) |
| **Zero-Padded Buffer** | **1,410 cells** (buffer/ocean cells strictly padded with constant `0.0`) |
| **Numerical Integrity** | **0 NaNs, 0 Infs** across all $20 \times 126 = 2,520$ valid evaluation samples |
| **Physical Value Range** | $[0.2488, 0.5087]\text{ m}^3/\text{m}^3$ (Mean: $0.4156\text{ m}^3/\text{m}^3$) |

---

## 3. Data Variables & Coordinate Structure

```text
Dimensions:
  time : 20       (daily dates: 2014-12-12 to 2014-12-31)
  lat  : 32       (11.75, 11.50, ..., 4.00)
  lon  : 48       (116.00, 116.25, ..., 127.75)

Variables:
  float32 rzsm_0_100_masked(time, lat, lon)
      long_name: Daily Depth-Weighted Root-Zone Soil Moisture (0-100 cm)
      units: m3/m3
      note: Active Mindanao cells (N=126) contain volumetric soil water; ocean/buffer cells (N=1,410) are padded with 0.0.

  int8 evaluation_mask(lat, lon)
      long_name: Frozen Candidate A Binary Evaluation Mask (f >= 0.50)
      values: 0 = Ocean/Buffer, 1 = Active Mindanao Evaluation Domain
```

---

## 4. How to Use This File in Python

### Quick Inspection with `xarray`
```python
import xarray as xr
import numpy as np

# Open the pilot reference dataset
ds = xr.open_dataset("processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc")
print(ds)

# Extract model input tensor: shape (20, 32, 48)
tensor = ds['rzsm_0_100_masked'].values
mask = ds['evaluation_mask'].values

# Verify active evaluation cells
active_values = tensor[:, mask == 1]
print(f"Evaluated samples: {active_values.size} (20 days x 126 cells)")
print(f"NaN count: {np.isnan(active_values).sum()}")
print(f"Soil moisture range: [{active_values.min():.4f}, {active_values.max():.4f}] m3/m3")

# Basic integrity assertion check
assert tensor.shape == (20, 32, 48), f"Unexpected shape: {tensor.shape}"
assert np.isnan(active_values).sum() == 0, "Found unexpected NaNs in evaluation domain!"
assert (mask == 1).sum() == 126, f"Expected 126 active cells, found {(mask == 1).sum()}"
print("Data integrity check: PASSED!")
```

### Quick Spatial Visualization with `matplotlib`
```python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

ds = xr.open_dataset("processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc")

# Select day 1 (2014-12-12)
sample_day = ds['rzsm_0_100_masked'].isel(time=0).values
mask = ds['evaluation_mask'].values

# Mask out background ocean (0.0) for clean visualization
viz_data = np.where(mask == 1, sample_day, np.nan)

plt.figure(figsize=(8, 5))
im = plt.imshow(viz_data, extent=[116.0, 127.75, 4.0, 11.75], cmap="YlGnBu", origin="upper")
plt.colorbar(im, label="RZSM (m³/m³)")
plt.title("Mindanao Pilot RZSM (2014-12-12) — Candidate A Grid (32x48)")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("processed/rzsm/pilot/sample_pilot_viz.png", dpi=200)
plt.show()
```

### PyTorch Dataset / DataLoader Integration
```python
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

class MindanaoRZSMDataset(Dataset):
    """
    Lightweight PyTorch Dataset for Mindanao RZSM tensors.
    Yields input tensors of shape (1, 32, 48) representing daily root-zone soil moisture.
    """
    def __init__(self, nc_path):
        ds = xr.open_dataset(nc_path)
        self.data = torch.from_numpy(ds['rzsm_0_100_masked'].values).float()
        self.mask = torch.from_numpy(ds['evaluation_mask'].values).byte()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Shape: (channels=1, lat=32, lon=48)
        return self.data[idx].unsqueeze(0)

# Instantiate PyTorch DataLoader
dataset = MindanaoRZSMDataset("processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc")
loader = DataLoader(dataset, batch_size=4, shuffle=False)

sample_batch = next(iter(loader))
print(f"Batch shape for UNet: {sample_batch.shape}")  # Output: torch.Size([4, 1, 32, 48])
```

---

## 5. Full Production Data Access & Cloud Scaling

While this 130 KB pilot reference file is tracked in Git, the **full 11-year daily production dataset (2015–2025, ~1.2 GB)** is stored in Google Cloud Storage:

- **GCS Bucket**: `gs://rise-unet-rzsm/processed/rzsm/production/`
- **Git Policy**: As defined in [`.gitignore`](../../../.gitignore), `processed/rzsm/production/` and `processed/**/*.nc` are excluded from version control to prevent Git repository bloat, while `!processed/rzsm/pilot/*.nc` is explicitly whitelisted.

### Downloading Production Data (Cloud Lake Access)
```bash
# Requires gcloud CLI and authenticated GCP project access:
gcloud storage cp -r gs://rise-unet-rzsm/processed/rzsm/production/ processed/rzsm/production/
```

### Generating Full Production Data Locally / on Colab
To run the full 11-year data extraction pipeline:
1. Open [`notebooks/04_mindanao_rzsm_pilot_preprocessing.ipynb`](../../../notebooks/04_mindanao_rzsm_pilot_preprocessing.ipynb).
2. Set `PROCESS_ALL_YEARS = True` and specify the output path to `processed/rzsm/production/`.
3. The script will iterate through all 265 monthly raw ERA5-Land NetCDFs, aggregate daily means, depth-weight, remap to the $32 \times 48$ grid, and save the full time series.

