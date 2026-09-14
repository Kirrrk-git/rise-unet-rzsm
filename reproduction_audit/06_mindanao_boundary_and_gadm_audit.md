<!-- markdownlint-disable -->
# Mindanao Regional Boundary Construction & Geometry Verification Report

**Domain**: Mindanao Regional Administrative Boundary & Geometry Validation  
**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Repository**: `dl_dm_rzsm_subseasonal_forecast` (`https://github.com/Kirrrk-git/rise-unet-rzsm.git`)  
**Active Git Branch**: `mindanao-adaptation`  
**Milestone**: Sub-Phase 21A Step 21A.2  
**Evaluation Status**: **CERTIFIED PASS (Methodological, Administrative & Geometric)**  
**Date of Certification**: 2026-09-10  

---

## 1. Executive Summary & Audit Verdict

| Audit Dimension | Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **Administrative Authority** | Official PSA PSGC 6 Mindanao administrative regions | `PSGC_2Q2026_Publication_Datafile.xlsx` verified (Regions IX, X, XI, XII, XIII, BARMM) | ✅ **PASS** |
| **GIS Geometry Source** | Official Philippine Government regional GIS boundary | NAMRIA / Philippine Geoportal `regionalboundary_20210504` (496 MiB export) | ✅ **PASS** |
| **Region Code Extraction** | Parse embedded PSGC metadata & exclude non-polygons | 1,384 selected $\to$ 1,378 polygons written, 6 label points skipped, 0 unmatched | ✅ **PASS** |
| **Code Verification** | Automated check of expected vs observed region codes | All 6 expected regions present, 0 unexpected codes, 0 missing codes | ✅ **PASS** |
| **Spatial Dissolve** | Merge 1,378 fragments into unified archipelagic boundary | Single MultiPolygon feature (1 feature, CRS: EPSG:4326) | ✅ **PASS** |
| **Topology & Validity** | Native GEOS topological validity check | `VALID_COUNT: 1, INVALID_COUNT: 0, ERROR_COUNT: 0` (No "Fix Geometries" used) | ✅ **PASS** |
| **Reproducibility** | Immutable manifests, scripts, and cryptographic hashes | SHA-256 recorded for all artifacts; `boundary_manifest.yaml` generated | ✅ **PASS** |

$$\Large\boxed{\textbf{FINAL VERDICT: PASS}}$$

---

## 2. Source Provenance & Data Separation

The boundary construction strictly separated administrative naming/hierarchy from GIS geometry:

1. **Current Administrative Reference**:
   * **Source**: Philippine Statistics Authority (PSA) Philippine Standard Geographic Code (PSGC).
   * **Dataset**: `PSGC_2Q2026_Publication_Datafile.xlsx`.
   * **SHA-256**: `31892bc2bdde3ea0682562d9412b5bab4d45a0be5e5a5b4f6c9d7714b94bca5d`.
   * **Role**: Defines the official administrative classification for the 6 Mindanao regions:
     * Region IX (Zamboanga Peninsula)
     * Region X (Northern Mindanao)
     * Region XI (Davao Region)
     * Region XII (SOCCSKSARGEN)
     * Region XIII (Caraga)
     * BARMM (Bangsamoro Autonomous Region in Muslim Mindanao)

2. **Official Regional Polygon Geometry Source**:
   * **Source**: National Mapping and Resource Information Authority (NAMRIA) via Philippine Geoportal.
   * **Layer**: `geoportal:regionalboundary_20210504`.
   * **Export Artifact**: `geoportal-regionalboundary_20210504_export.geojson`.
   * **Size**: 519,138,911 bytes (~496 MiB).
   * **SHA-256**: `62847e530c0d46df9ef6dc31ef64e372032ef7c0defc05d4758f4bf8c429ae9c`.
   * **CRS**: `EPSG:4326` (WGS 84).

3. **Analysis of Discarded Alternatives**:
   * *Geoportal WFS Feature Query*: Rejected due to upstream server gateway timeouts (`Could not access any server machines`).
   * *Geoportal WMS-generated KML (470 MiB)*: Rejected because WMS is a raster portrayer; KML conversion produced polygon slivers and attribute loss.
   * *MAPOG Regions Level 01*: Rejected because underlying source was third-party GeoBoundaries and retained obsolete pre-BARMM "ARMM" geometry.
   * *PSA-Municipal GeoJSON (99 MiB)*: Rejected due to fatal omission of the entire BARMM territory.

---

## 3. Extraction & Numerical Feature Accounting

GDAL/OGR (GDAL 3.13.3) was executed with `OGR_GEOJSON_MAX_OBJ_SIZE=0` to parse the embedded HTML attribute payload in the GeoJSON `description` field (`reg_psgc20`):

```text
Source features in raw export:      3,745
Selected Mindanao records:          1,384
Polygon features written:           1,378
Non-polygon features skipped:           6 (Regional centroid/label points)
Empty geometries skipped:               0
Unmatched records:                      0
```

### Regional Polygon Fragment Census:
| Regional Code (`reg_psgc20`) | Region Name | Polygon Fragments | Status |
| :--- | :--- | :---: | :---: |
| `090000000` | Region IX (Zamboanga Peninsula) | 121 | Verified |
| `100000000` | Region X (Northern Mindanao) | 10 | Verified |
| `110000000` | Region XI (Davao Region) | 17 | Verified |
| `120000000` | Region XII (SOCCSKSARGEN) | 2 | Verified |
| `150000000` | BARMM (Bangsamoro Autonomous Region) | 942 | Verified |
| `160000000` | Region XIII (Caraga) | 286 | Verified |
| **Total Polygons** | **All 6 Mindanao Regions** | **1,378** | **Exact Sum** |

* Note on BARMM coding: In the 2020-era 9-digit PSGC system, BARMM was indexed as `150000000`. In the current 10-digit PSGC system, BARMM is indexed as `1900000000`. The extraction script correctly utilized `150000000` matching `reg_psgc20`.
* Filtered Output: `PSA_Geoportal_Regional_20210504_Mindanao.gpkg` (Layer: `mindanao_regions`).
* Filtered GeoPackage SHA-256: `a10204e1efe8f2ef11d15c17cce8291dfd74b21f4d2ca03badc238e8937f413f` (65,658,880 bytes).
* Automated verification script `verify_mindanao_codes.py` returned: `PASS` (`Unexpected codes: []`, `Missing expected codes: []`).

---

## 4. Dissolve & Geometric Topology Validation

* **Tool**: QGIS 3.44.14-Solothurn native Vector Geometry Dissolve (GEOS 3.14.1).
* **Settings**:
  * Input layer: `mindanao_regions` (1,378 features).
  * Dissolve fields: None (dissolve all).
  * `Keep disjoint features separate`: **False** (mandatory setting to merge all archipelagic fragments into exactly 1 study domain feature).
* **Output Artifact**: `mindanao_analysis_boundary.gpkg` (Layer: `mindanao_analysis_boundary`).
* **Output Size**: 63,459,328 bytes.
* **Output SHA-256**: `9c7478ec02718153c124e8f528211951c6a855c0221dfd8e443b5c430bd623d2`.

### Structural & Spatial Properties:
* **Feature Count**: Exactly 1 feature (`MultiPolygon`).
* **Coordinate Reference System**: `EPSG:4326` (WGS 84 geographic coordinates).
* **Geographic Bounding Box**:
  $$\text{West Longitude: } 118.064236^\circ\text{E} \quad \text{(Turtle Islands / Sitangkai, Tawi-Tawi)}$$
  $$\text{East Longitude: } 126.604966^\circ\text{E} \quad \text{(Cape San Agustin / Caraga Coast)}$$
  $$\text{South Latitude: } 4.587294^\circ\text{N} \quad \text{(Sitangkai / Frances Reef, Tawi-Tawi)}$$
  $$\text{North Latitude: } 10.471595^\circ\text{N} \quad \text{(Dinagat Islands / Surigao Strait)}$$
* **Topological Validity**: Evaluated via QGIS GEOS validity check (`Ignore ring self-intersections = False`).
  * `VALID_COUNT: 1`
  * `INVALID_COUNT: 0`
  * `ERROR_COUNT: 0`
  * Zero heuristic geometry repair ("Fix Geometries") was needed; original NAMRIA survey boundary geometry was preserved without artifact introduction.
