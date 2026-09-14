<!-- markdownlint-disable -->
# Production Package Architecture (`src/`)

This directory contains the core Python source package for the **Mindanao Tropical RISE-UNet Adaptation (Track B)**.

Unlike standalone CLI scripts or interactive notebooks, files within `src/` are **modular, reusable library components** (classes, functions, data transformers, and neural network architectures). As such, they follow standard Python packaging conventions (PEP 8 valid module identifiers without numeric prefixes) and are organized into a clear **6-Layer Architectural Hierarchy**.

---

## 1. Package Layering & Dependency Hierarchy

Dependencies flow strictly in one direction from foundation to deep learning models:

```text
  Layer 5: Deep Learning Architectures
  └── src.models.a0_unet                     <── Kyle Lesinger UNET_RZSM with 3 deep supervision heads
        ▲
  Layer 4: TensorFlow Dataset & Batching
  └── src.data.tf_dataset                    <── tf.data pipeline, B mod 11 ensemble grouping, target broadcasting
        ▲
  Layer 3: Feature Synthesis & Case Construction
  └── src.data.case_builder                  <── Assembles multi-source tensors (W1-W4), normalization & masking
        ▲
  Layer 2: Temporal Aggregations & Datacubes
  ├── src.data.temporal                      <── Rolling 7-day target windows & antecedent lags (t0-1d, 7d, 14d)
  ├── src.data.compile_cube                  <── 11-Year continuous daily NetCDF target cube compilation
  └── src.data.preflight                     <── 11-Point preflight census & contract verification gate
        ▲
  Layer 1: Raw Observation & Forecast Decoders
  ├── src.data.rzsm                          <── Layer-depth weighted integration (0-100 cm) & bilinear remap
  ├── src.data.atmospheric                   <── Daily diurnal aggregations (T2m, Tmax, D2m, P, Z200)
  └── src.data.s2s                           <── Pure-Python GRIB2 Section 7 decoder & 1.5° -> 0.25° remap
        ▲
  Layer 0: Infrastructure & Domain Foundations
  ├── src.data.cloud_lake                    <── Local-first priority with on-demand GCS stream
  └── src.data.calendar                      <── ECMWF CY48R1 bi-weekly Mon/Thu operational calendar
```

---

## 2. Detailed Module Catalog by Layer

### Layer 0: Infrastructure & Domain Foundations
- **[`src.data.cloud_lake`](data/cloud_lake.py)**
  - **Role**: Cloud lake resolution engine. Automatically prioritizes committed local repository artifacts (allowing full offline development with zero network overhead), while fetching remote datasets on-demand from Google Cloud Storage (`gs://rise-unet-rzsm/`).
- **[`src.data.calendar`](data/calendar.py)**
  - **Role**: Authoritative date math and cycle calendar definitions. Generates the 1,154 bi-weekly Monday/Thursday forecast-origin cycles matching the ECMWF CY48R1 reference schedule across 2015–2025.

---

### Layer 1: Raw Observation & Forecast Decoders
- **[`src.data.rzsm`](data/rzsm.py)**
  - **Role**: Geospatial processing for ERA5-Land soil moisture: depth-weighted integration across Layers 1 ($0\text{--}7\,\text{cm}$), 2 ($7\text{--}28\,\text{cm}$), and 3 ($28\text{--}100\,\text{cm}$), bilinear remapping to Candidate A grid, and spatial cropping.
- **[`src.data.atmospheric`](data/atmospheric.py)**
  - **Role**: Daily diurnal aggregation of ERA5 hourly surface and pressure-level reanalysis fields, extracting daily mean $T_{2m}$, maximum $T_{max}$, mean dewpoint $D_{2m}$, total precipitation $P$, and $200\,\text{hPa}$ geopotential height $Z_{200}$.
- **[`src.data.s2s`](data/s2s.py)**
  - **Role**: High-performance, pure-Python GRIB2 Section 7 decoder and spatial remapper for ECMWF CY48R1 subseasonal reforecast triplets (`t2m`, `d2m`, `tcw`), converting 11 ensemble members from native $1.5^\circ$ grid to $0.25^\circ$ Candidate A grid.

---

### Layer 2: Temporal Aggregations & Datacubes
- **[`src.data.temporal`](data/temporal.py)**
  - **Role**: Temporal rolling window utilities. Computes 7-day trailing rolling means for target lead windows ($W_1$: days 0–6, $W_2$: days 7–13, $W_3$: days 14–20, $W_4$: days 21–27) and slices antecedent memory lags ($t_0-1\text{d}, t_0-7\text{d}, t_0-14\text{d}$).
- **[`src.data.compile_cube`](data/compile_cube.py)**
  - **Role**: Production compilation engine assembling monthly ERA5-Land files into a single, continuous 11-year daily CF-1.8 NetCDF datacube (`era5_land_rzsm_production_2015_2025.nc`, 4,018 nominal days, 126 active cells).
- **[`src.data.preflight`](data/preflight.py)**
  - **Role**: Deterministic 11-point preflight verification engine certifying archive completeness, timestamp hygiene, and spatial contract hashes before data cube compilation.

---

### Layer 3: Feature Synthesis & Case Construction
- **[`src.data.case_builder`](data/case_builder.py)**
  - **Role**: Master feature synthesizer. Ingests RZSM lags, atmospheric observations, and S2S reforecasts to build structured input tensors:
    - Lead 1 ($W_1$): 11 channels (3 RZSM lags + 5 atmospheric + 3 S2S Lead 1)
    - Lead 2 ($W_2$): 12 channels (3 RZSM lags + 5 atmospheric + 3 S2S Lead 2 + 1 Recursive $\hat{Y}_{W1}$)
    - Lead 3 ($W_3$): 5 channels (3 RZSM lags + 1 Recursive $\hat{Y}_{W1}$ + 1 Recursive $\hat{Y}_{W2}$)
    - Lead 4 ($W_4$): 6 channels (3 RZSM lags + 1 Recursive $\hat{Y}_{W1}$ + 1 Recursive $\hat{Y}_{W2}$ + 1 Recursive $\hat{Y}_{W3}$)
  - Actively enforces frozen normalization parameters (`contracts/A0/normalization_parameters.yaml`) and zeroes ocean buffer cells using `mindanao_eval_mask_025.nc`.

---

### Layer 4: TensorFlow Dataset & Batching Pipeline
- **[`src.data.tf_dataset`](data/tf_dataset.py)**
  - **Role**: Production TensorFlow batch generator. Enforces the 11-member ensemble grouping constraint ($B \in \{11, 22, 33, \dots\}$), broadcasts single-sample ground truth targets to all 11 ensemble members ($Y_{Wk} (1, 32, 48, 1) \to (11, 32, 48, 1)$), formats 3 deep supervision heads, and manages checkpoint save/restore parity.

---

### Layer 5: Deep Learning Architectures
- **[`src.models.a0_unet`](models/a0_unet.py)**
  - **Role**: Factory for instantiating the peer-reviewed RISE-UNet architecture from Kyle Lesinger & Di Tian (2025). Features:
    - 4 encoder downsampling stages and 4 decoder upsampling stages with skip connections.
    - 3 deep supervision output heads ($RZSM\_output\_1$, $RZSM\_output\_2$, $RZSM\_output\_3$).
    - Parameter count: exactly 1,630,307 trainable weights for Lead 2.
    - Native multi-head Continuous Ranked Probability Score (CRPS) loss computation.

---

## 3. Why `src/` Modules Are Not Numbered

In Python, source files within library packages must adhere to the following architectural requirements:
1. **Valid Python Identifiers**: Python module names cannot begin with a number. Writing `from src.data.01_calendar import ...` is a syntax error (`SyntaxError: invalid decimal literal`).
2. **Library vs. Pipeline Distinction**: Files in `src/` are not executed linearly from first to last. They are imported dynamically as dependencies by CLI scripts, tests, and notebooks.
3. **Semantic Discoverability**: Standard naming (e.g. `case_builder.py`, `tf_dataset.py`) enables clean imports throughout the application and aligns with standard static analysis and IDE autocomplete tools.
