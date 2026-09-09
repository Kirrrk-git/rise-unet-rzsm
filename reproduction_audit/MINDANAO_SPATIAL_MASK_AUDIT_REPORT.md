<!-- markdownlint-disable -->
# Mindanao 0.25° Spatial Mask Construction & Geodetic Validation Report

**Domain**: Mindanao Regional Spatial Masking & Evaluation Domain Boundary  
**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Git Branch**: `mindanao-adaptation`  
**Evaluation Status**: **CERTIFIED PASS (Geodesic Quadrature, Census & Topology)**  
**Date of Certification**: 2026-09-10  

---

## 1. Executive Summary & Audit Verdict

| Audit Dimension | Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **Authoritative Boundary** | PSA-NAMRIA certified unified regional boundary | `mindanao_analysis_boundary.gpkg` (SHA-256: `9c7478ec...`) | ✅ **PASS** |
| **Frozen Computational Grid** | Candidate A $32 \times 48$ domain ($0.25^\circ$ spacing) | `mindanao_025deg.nc` (SHA-256: `51a69199...`) | ✅ **PASS** |
| **Mask Nature Specification** | Regional administrative boundary coverage (not physical land-cover) | Formally declared as boundary-coverage mask | ✅ **PASS** |
| **Geodetic Quadrature Rigor** | WGS 84 ellipsoidal geodesic quadrature (`pyproj.Geod`) | Discrepancy between integrated cell area and boundary $= 0.0000\%$ | ✅ **PASS** |
| **Fractional Mask Generation** | Continuous boundary-coverage values $f \in [0.0, 1.0]$ | `mindanao_fraction_025.nc` (SHA-256: `9294a1c7...`) | ✅ **PASS** |
| **Binary Evaluation Mask** | Derived via explicit inclusion threshold $f \ge 0.50$ | `mindanao_eval_mask_025.nc` (SHA-256: `d7fd80e1...`) | ✅ **PASS** |
| **Mask Preservation Contract** | Retain BOTH fractional and binary mask files simultaneously | Both NetCDF4 files written and verified in `processed/grid/` | ✅ **PASS** |
| **Benchmark Inspection** | Manual validation of interior, coastal, exterior, and island cells | 5 benchmark cells evaluated with exact geodetic coordinates | ✅ **PASS** |

$$\Large\boxed{\textbf{FINAL VERDICT: PASS}}$$

---

## 2. Cryptographic Artifact Provenance

| Artifact Role | File Path | Format | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :---: | :---: | :--- |
| **Input Boundary** | `processed/boundary/mindanao_analysis_boundary.gpkg` | GeoPackage | 63,459,328 | `9c7478ec02718153c124e8f528211951c6a855c0221dfd8e443b5c430bd623d2` |
| **Input Coordinate Grid** | `processed/grid/mindanao_025deg.nc` | NetCDF4 | 16,024 | `51a691994a16b75b3a103efdb539febce2695179dbb89f5cb346a477e694548f` |
| **Output Fractional Mask** | `processed/grid/mindanao_fraction_025.nc` | NetCDF4 (CF-1.8) | 16,024 | `9294a1c74ec5aa2288c9e9bf54fc6763f905dbda1c4ff0ecda0a9b9ae9ff6325` |
| **Output Binary Mask** | `processed/grid/mindanao_eval_mask_025.nc` | NetCDF4 (CF-1.8) | 15,965 | `d7fd80e1f95cdd840daded176b06b2a0d23557df555acfaf0c01d2d1964621f6` |
| **Generator Script** | `scripts/generate_mindanao_masks.py` | Python 3 | 5,820 | `b8565a5cb33a7e583c27e0256860d5b4d707b227e85c2c77f0a6d23467610660` |

---

## 3. Mathematical Formulation & Geodetic Quadrature

### 3.1 Boundary-Coverage Definition
The fractional mask represents the precise geometric proportion of each $0.25^\circ \times 0.25^\circ$ grid cell covered by the authoritative Philippine administrative boundary for Mindanao:

$$f_{i,j} = \frac{\text{Area}_{\text{WGS84}}\left( \mathcal{C}_{i,j} \cap \mathcal{B} \right)}{\text{Area}_{\text{WGS84}}\left( \mathcal{C}_{i,j} \right)} \in [0.0, 1.0]$$

where:
* $\mathcal{C}_{i,j} = \left[ \text{lon}_j - 0.125^\circ, \text{lon}_j + 0.125^\circ \right] \times \left[ \text{lat}_i - 0.125^\circ, \text{lat}_i + 0.125^\circ \right]$ is the spatial bounding box of cell $(i, j)$ centered at $(\text{lat}_i, \text{lon}_j)$.
* $\mathcal{B}$ is the validated PSA-NAMRIA MultiPolygon regional boundary (`mindanao_analysis_boundary.gpkg`).
* $\text{Area}_{\text{WGS84}}(\cdot)$ denotes the true geodesic surface area computed on the WGS 84 reference ellipsoid ($a = 6,378,137.0\text{ m}$, $f = 1/298.257223563$) via `pyproj.Geod`.

### 3.2 Area Integration Parity
Because the grid completely encloses the Mindanao boundary with zero clipping or truncation, the discrete summation of fractional areas across all $H \times W = 32 \times 48 = 1,536$ cells must equal the total continuous geodetic area of the boundary polygon:

$$\sum_{i=1}^{32} \sum_{j=1}^{48} f_{i,j} \cdot \text{Area}_{\text{WGS84}}(\mathcal{C}_{i,j}) = \text{Area}_{\text{WGS84}}(\mathcal{B})$$

**Empirical Quadrature Accounting**:
* **Authoritative Boundary Surface Area**: $99,948.76\text{ km}^2$
* **Discrete Integrated Fractional Cell Area**: $99,948.76\text{ km}^2$
* **Numerical Discrepancy**: $\mathbf{0.0000\%}$ ($\Delta A < 0.001\text{ km}^2$)

---

## 4. Comprehensive Cell Census & Coverage Distribution

The frozen computational domain consists of $32 \times 48 = 1,536$ cells.

| Fractional Coverage Tier | Coverage Interval | Cell Count | Percentage of Grid | Evaluation Role |
| :--- | :---: | :---: | :---: | :--- |
| **Pure Ocean / Outer Buffer** | $f = 0.000$ | 1,283 | 83.53% | Zero-filled convolutional context; excluded from loss |
| **Trace Boundary / Minor Islets** | $0.000 < f < 0.100$ | 71 | 4.62% | Sub-threshold boundary buffer; excluded from loss |
| **Low Coastal Coverage** | $0.100 \le f < 0.250$ | 27 | 1.76% | Sub-threshold boundary buffer; excluded from loss |
| **Moderate Coastal Transition** | $0.250 \le f < 0.500$ | 29 | 1.89% | Sub-threshold boundary buffer; excluded from loss |
| **Substantial Land Dominance** | $0.500 \le f < 0.750$ | 25 | 1.63% | **Included in Binary Evaluation Mask ($M=1$)** |
| **Near-Complete Coverage** | $0.750 \le f < 1.000$ | 38 | 2.47% | **Included in Binary Evaluation Mask ($M=1$)** |
| **100% Solid Interior Land** | $f = 1.000$ | 63 | 4.10% | **Included in Binary Evaluation Mask ($M=1$)** |
| **Total Active Boundary Cells** | $f > 0.000$ | **253** | **16.47%** | Complete geographic footprint of Mindanao |
| **Total Binary Evaluation Mask** | $f \ge 0.500$ | **126** | **8.20%** | **Core verification domain ($96,085.57\text{ km}^2$)** |

---

## 5. Benchmark Cells Manual Inspection

To verify the numerical precision, cell-centering alignment, and edge behavior, five geographically diverse benchmark cells were extracted and verified:

```text
                                 10.00°N [Dinagat Island] (f = 0.2282, M = 0)
                                    ▲
                                    │
                                    │      8.00°N, 125.00°E [Bukidnon Interior] (f = 1.0000, M = 1)
                                    │         ▲
                                    │         │
  6.00°N, 121.00°E [Jolo Island] ───┼─────────┼─── 6.25°N, 126.25°E [Davao Oriental Coast] (f = 0.0289, M = 0)
  (f = 0.5969, M = 1)               │         │
                                    ▼         ▼
                                 4.00°N, 116.00°E [Celebes Sea Ocean Buffer] (f = 0.0000, M = 0)
```

### Benchmark Cell 1: Fully Interior Cell (Central Bukidnon / Malaybalay)
* **Center Coordinates**: `lat = 8.00°N` (index 15), `lon = 125.00°E` (index 36)
* **Cell Extent**: $[124.875^\circ\text{E}, 7.875^\circ\text{N}] \to [125.125^\circ\text{E}, 8.125^\circ\text{N}]$
* **Geodetic Cell Area**: $762.03\text{ km}^2$
* **Fractional Coverage**: $\mathbf{1.000000}$
* **Binary Evaluation Mask**: $\mathbf{1}$ (INCLUDED)
* **Audit Verdict**: Cell is 100% enclosed within the Mindanao landmass. Zero edge clipping.

### Benchmark Cell 2: Coastal Boundary Cell (Cape San Agustin / Davao Oriental)
* **Center Coordinates**: `lat = 6.25°N` (index 22), `lon = 126.25°E` (index 41)
* **Cell Extent**: $[126.125^\circ\text{E}, 6.125^\circ\text{N}] \to [126.375^\circ\text{E}, 6.375^\circ\text{N}]$
* **Geodetic Cell Area**: $765.23\text{ km}^2$
* **Fractional Coverage**: $\mathbf{0.028912}$ ($\approx 2.89\%$ land tip)
* **Binary Evaluation Mask**: $\mathbf{0}$ (MASKED / BUFFER)
* **Audit Verdict**: Correctly captures the narrow spit of Cape San Agustin. Excluded from binary evaluation because $f < 0.50$, preventing severe coastal water contamination in point verification metrics.

### Benchmark Cell 2b: Coastal Inclusion Cell (Northern Surigao Coast)
* **Center Coordinates**: `lat = 9.75°N` (index 8), `lon = 125.50°E` (index 38)
* **Cell Extent**: $[125.375^\circ\text{E}, 9.625^\circ\text{N}] \to [125.625^\circ\text{E}, 9.875^\circ\text{N}]$
* **Geodetic Cell Area**: $758.11\text{ km}^2$
* **Fractional Coverage**: $\mathbf{0.556811}$ ($\approx 55.68\%$ land)
* **Binary Evaluation Mask**: $\mathbf{1}$ (INCLUDED)
* **Audit Verdict**: Correctly identifies land dominance along the Surigao coastline. Included in evaluation because $f \ge 0.50$.

### Benchmark Cell 3: Exterior Ocean Buffer Cell (Celebes Sea / South Buffer)
* **Center Coordinates**: `lat = 4.00°N` (index 31), `lon = 116.00°E` (index 0)
* **Cell Extent**: $[115.875^\circ\text{E}, 3.875^\circ\text{N}] \to [116.125^\circ\text{E}, 4.125^\circ\text{N}]$
* **Geodetic Cell Area**: $768.61\text{ km}^2$
* **Fractional Coverage**: $\mathbf{0.000000}$
* **Binary Evaluation Mask**: $\mathbf{0}$ (MASKED / BUFFER)
* **Audit Verdict**: Strict zero boundary overlap. Acts as pure zero-padded receptive field buffer during U-Net convolutions.

### Benchmark Cell 4: Fragmented Island Area (Dinagat Islands)
* **Center Coordinates**: `lat = 10.00°N` (index 7), `lon = 125.50°E` (index 38)
* **Cell Extent**: $[125.375^\circ\text{E}, 9.875^\circ\text{N}] \to [125.625^\circ\text{E}, 10.125^\circ\text{N}]$
* **Geodetic Cell Area**: $757.51\text{ km}^2$
* **Fractional Coverage**: $\mathbf{0.228202}$ ($\approx 22.82\%$ archipelagic land)
* **Binary Evaluation Mask**: $\mathbf{0}$ (MASKED / BUFFER)
* **Audit Verdict**: Correctly quantifies the archipelagic island chain of Dinagat. Preserved accurately in the fractional mask for area-weighted sensitivity studies; excluded from binary evaluation mask under the $f \ge 0.50$ rule.

### Benchmark Cell 5: Major Island Area (Jolo Island, Sulu Archipelago)
* **Center Coordinates**: `lat = 6.00°N` (index 23), `lon = 121.00°E` (index 20)
* **Cell Extent**: $[120.875^\circ\text{E}, 5.875^\circ\text{N}] \to [121.125^\circ\text{E}, 6.125^\circ\text{N}]$
* **Geodetic Cell Area**: $765.65\text{ km}^2$
* **Fractional Coverage**: $\mathbf{0.596937}$ ($\approx 59.69\%$ land)
* **Binary Evaluation Mask**: $\mathbf{1}$ (INCLUDED)
* **Audit Verdict**: Validates that major populated islands of BARMM / Sulu Archipelago with substantial landmass are properly included in the binary evaluation mask ($f \ge 0.50$).

---

## 6. Visual Verification Guide for QGIS

To inspect the generated masks and boundary overlay in QGIS:

1. **Launch QGIS** (e.g. QGIS 3.44 Solothurn).
2. **Load Authoritative Boundary**:
   * Menu: `Layer` $\to$ `Add Layer` $\to$ `Add Vector Layer...`
   * Select: `processed/boundary/mindanao_analysis_boundary.gpkg`
   * Style: Transparent fill with a high-visibility stroke (e.g., Red or Cyan, width 0.4 mm).
3. **Load NetCDF Masks as Mesh / Raster Layers**:
   * Menu: `Layer` $\to$ `Add Layer` $\to$ `Add Raster Layer...`
   * Open: `processed/grid/mindanao_fraction_025.nc`
   * Open: `processed/grid/mindanao_eval_mask_025.nc`
4. **Symbology Configuration**:
   * For `mindanao_eval_mask_025.nc`: Set symbology to `Paletted/Unique values`:
     * Value `0`: Fully transparent (opacity 0%).
     * Value `1`: Semi-transparent green or orange (opacity 50%).
   * For `mindanao_fraction_025.nc`: Set symbology to `Singleband pseudocolor` (`Viridis` or `YlGnBu` ramp from 0.0 to 1.0).
5. **Expected Visual Findings**:
   * The vector boundary should fall strictly within the grid domain, with positive water buffers on all four margins ($0.71^\circ\text{--}2.19^\circ$).
   * The binary mask (`Value 1`) should form the contiguous footprint of the Mindanao mainland and major archipelagic centers, perfectly matching the interior of the vector polygon.
   * Coastal cells will display graduated fractional values aligning with coastal peninsulas, bays (e.g., Davao Gulf, Iligan Bay), and island chains.

---

## 7. Mask Preservation Contract Sign-Off

* Both `processed/grid/mindanao_fraction_025.nc` and `processed/grid/mindanao_eval_mask_025.nc` are permanently retained in the project repository.
* The binary mask shall be utilized for:
  1. Deep learning loss computation (zeroing out gradients on exterior cells).
  2. Subseasonal forecast verification metrics (ACC, CRPSS, RMSE, KGE).
* The fractional mask shall be preserved for:
  1. Area-weighted regional average soil-moisture index aggregation.
  2. Downstream spatial sensitivity analyses and coastal edge evaluations.

**Certification Sign-Off**:
* Step 21A.7A (Fractional Boundary-Coverage Mask): **CERTIFIED PASS**
* Step 21A.7B (Binary Evaluation Mask $f \ge 0.50$): **CERTIFIED PASS**
