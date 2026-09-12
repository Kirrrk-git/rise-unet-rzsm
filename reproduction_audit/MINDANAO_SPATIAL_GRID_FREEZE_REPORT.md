<!-- markdownlint-disable -->
# Step 21A.6 Audit Dossier: Authoritative Mindanao 0.25° Spatial Grid Freeze Report

**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Repository Branch**: `mindanao-adaptation`  
**Execution Date**: September 10, 2026  
**Audited Reference Geometry**: `mindanao_analysis_boundary.gpkg` (SHA-256: `9c7478ec02718153c124e8f528211951c6a855c0221dfd8e443b5c430bd623d2`)  
**Status**: **CERTIFIED PASS (Grid Frozen, Mathematically Audited & Bound Checked)**

---

## 1. Executive Summary & Grid Freeze Decision

Following the geometric enumeration in Step 21A.4 and the recursive cascade gate in Step 21A.5, **Candidate A ($32 \times 48$)** is formally selected and frozen as the authoritative 0.25° spatial computational domain for the Mindanao RISE-UNet adaptation:

$$\mathbf{H \times W = 32 \times 48 \quad (1,536\text{ Total Spatial Grid Cells})}$$

### Rationale for Freezing Candidate A ($32 \times 48$):
1. **Complete Boundary Containment**: Encompasses 100% of the validated PSA-NAMRIA archipelagic boundary (`mindanao_analysis_boundary.gpkg`) with positive protective buffers on all four margins ($0.71^\circ\text{--}2.19^\circ$).
2. **Architectural Compatibility**: Exactly satisfies the $2^4 = 16$ divisibility constraint required by the 4-level U-Net backbone ($H/16 = 2, W/16 = 3$), certified via empirical shape tracing.
3. **Computational Efficiency**: Contains 1,536 cells, avoiding the 33% spatial overhead of Candidate B ($32 \times 64 = 2,048$ cells) and the 50%–100% overhead of $48 \times 48$ / $48 \times 64$ which needlessly ingest vast expanses of the western Pacific Ocean and Celebes Sea.
4. **Adequate Receptive Field Buffer**: Preserves sufficient surrounding maritime buffer (at least 2.8 to 8.7 grid cells) to support convolutional spatial receptive fields without border boundary edge effects.

---

## 2. Explicit Coordinate Convention: Cell Centers vs. Cell Edges

To prevent systematic coordinate offsets during regridding and rasterization, the coordinate convention is mathematically decoupled into cell centers and outer cell edges:

### A. Cell-Center Convention (Coordinate Arrays in NetCDF)
The 1D coordinate variables `lat` and `lon` stored in [`processed/grid/mindanao_025deg.nc`](../processed/grid/mindanao_025deg.nc) represent the **exact geometric centers** of each $0.25^\circ \times 0.25^\circ$ cell:

* **Latitude (`lat`)**: 32 points, sorted descending (North $\to$ South) matching ERA5 / CDO conventions:
  $$\text{lat}_i = 11.75^\circ - i \cdot 0.25^\circ \quad \text{for } i = 0, 1, \dots, 31$$
  $$\text{lat} = [11.75, 11.50, 11.25, \dots, 4.25, 4.00]^\circ\text{N}$$
* **Longitude (`lon`)**: 48 points, sorted ascending (West $\to$ East):
  $$\text{lon}_j = 116.00^\circ + j \cdot 0.25^\circ \quad \text{for } j = 0, 1, \dots, 47$$
  $$\text{lon} = [116.00, 116.25, 116.50, \dots, 127.50, 127.75]^\circ\text{E}$$

### B. Cell-Edge Bounding Box (Physical Footprint Envelope)
Because each grid cell has a spatial span of $\Delta\phi = 0.25^\circ$ and $\Delta\lambda = 0.25^\circ$, each cell $(i, j)$ centered at $(\text{lat}_i, \text{lon}_j)$ physically covers:

$$[\text{lat}_i - 0.125^\circ, \text{lat}_i + 0.125^\circ] \times [\text{lon}_j - 0.125^\circ, \text{lon}_j + 0.125^\circ]$$

The overall outer bounding envelope of the frozen grid is therefore:
* **South Edge**: $4.00^\circ - 0.125^\circ = \mathbf{3.875^\circ\text{N}}$
* **North Edge**: $11.75^\circ + 0.125^\circ = \mathbf{11.875^\circ\text{N}}$
* **West Edge**: $116.00^\circ - 0.125^\circ = \mathbf{115.875^\circ\text{E}}$
* **East Edge**: $127.75^\circ + 0.125^\circ = \mathbf{127.875^\circ\text{E}}$

---

## 3. Boundary Containment & Buffer Audit

Cross-referencing the official GIS boundary bounds against the frozen cell-edge envelope:

| Boundary Extreme | Mindanao Boundary Coordinate | Outer Grid Edge | Buffer Distance | Buffer in Grid Cells ($\Delta = 0.25^\circ$) | Containment Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **South Latitude** (Sitangkai, Tawi-Tawi) | $4.587294^\circ\text{N}$ | $3.875000^\circ\text{N}$ | $+0.712294^\circ$ | **2.85 cells** | **PASS (Clean Buffer)** |
| **North Latitude** (Dinagat Islands) | $10.471595^\circ\text{N}$ | $11.875000^\circ\text{N}$ | $+1.403405^\circ$ | **5.61 cells** | **PASS (Clean Buffer)** |
| **West Longitude** (Turtle Islands) | $118.064236^\circ\text{E}$ | $115.875000^\circ\text{E}$ | $+2.189236^\circ$ | **8.76 cells** | **PASS (Clean Buffer)** |
| **East Longitude** (Cape San Agustin) | $126.604966^\circ\text{E}$ | $127.875000^\circ\text{E}$ | $+1.270034^\circ$ | **5.08 cells** | **PASS (Clean Buffer)** |

**Zero boundary truncation**. All 1,378 regional polygon fragments are strictly internal to the computational grid.

---

## 4. Frozen Artifacts & Cryptographic Checksums

The spatial grid has been generated and validated with zero warnings:

1. **Authoritative Coordinate NetCDF**:
   * Path: [`processed/grid/mindanao_025deg.nc`](../processed/grid/mindanao_025deg.nc)
   * Format: CF-1.8 Compliant NetCDF4
   * Coordinate Dimensions: `lat` (32), `lon` (48)
   * Boundary Variables: `lat_bnds` (32, 2), `lon_bnds` (48, 2)
   * Coordinate System: `EPSG:4326` (WGS 84)
   * **SHA-256**: `51a691994a16b75b3a103efdb539febce2695179dbb89f5cb346a477e694548f`

2. **CDO Standard Grid Specification**:
   * Path: [`processed/grid/mindanao_0.25_grid.grd`](../processed/grid/mindanao_0.25_grid.grd)
   * Format: CDO lonlat grid description file (matching parent `Data/masks/conus_0.5_grid.grd`)
   * **SHA-256**: `b8fabe5c21a6b5beda2df8b9ef4438d8b63d74427e86223137a55c3afff40049`

---

## 5. Certification Sign-Off & Transition

* **Step 21A.6 Status**: **CERTIFIED PASS**.
* **Next Sequence**:
  * **Step 21A.7A**: Fractional Boundary-Coverage Mask (`processed/grid/mindanao_fraction_025.nc`).
  * **Step 21A.7B**: Binary Evaluation Mask (`processed/grid/mindanao_eval_mask_025.nc`, [`reproduction_audit/MINDANAO_SPATIAL_MASK_AUDIT_REPORT.md`](MINDANAO_SPATIAL_MASK_AUDIT_REPORT.md)).
  * **Step 21A.8**: Freeze Machine-Readable Spatial Contract ([`contracts/spatial/spatial_grid_contract.yaml`](../contracts/spatial/spatial_grid_contract.yaml)).
  * **Step 21A.9**: ERA5-Land RZSM Target Acceptance & Independent Cross-Reference ([`reproduction_audit/ERA5_LAND_TARGET_ACCEPTANCE_REPORT.md`](ERA5_LAND_TARGET_ACCEPTANCE_REPORT.md)).

