<!-- markdownlint-disable -->
# Step 21A.3 Audit Dossier: Authoritative EX29 Channel, Lag & Variable Contract Reconciliation

**Parent Study**: Lesinger, K., & Tian, D. (2025). *Subseasonal root-zone soil moisture drought forecasting using a deep learning-dynamic model hybrid approach*. **Nature Communications**, 16, 62761. DOI: [10.1038/s41467-025-62761-3](https://doi.org/10.1038/s41467-025-62761-3)  
**Parent Source Commit**: [`4af8e8c869b7df6a398bf12e122a8e2af3f30eeb`](https://github.com/kyle-lesinger/dl_dm_rzsm_subseasonal_forecast/commit/4af8e8c869b7df6a398bf12e122a8e2af3f30eeb)  
**Repository Branch**: `mindanao-adaptation`  
**Execution Date**: September 10, 2026  
**Status**: **PASSED (Audited & Reconciled Against Author Code and Published Literature)**

---

## 1. Executive Summary & Audit Objective

The objective of **Step 21A.3** is to reconcile the apparent discrepancies between:
1. **The Published Parent Paper Literature**: The general descriptions and illustrative tables in the paper (e.g., Supplementary Tables S1 and S6) presenting generic or illustrative configurations (such as six antecedent lag weeks).
2. **The Author's Implementation-Specific Codebase**: The actual empirical implementation of **Experiment EX29**—the author's top-performing hybrid recursive model across CONUS—as defined in [`function/experimentType.py`](../function/experimentType.py), [`function/channelExperiment.py`](../function/channelExperiment.py), [`function/loadDataAllWeeks.py`](../function/loadDataAllWeeks.py), and [`function/preprocessUtils.py`](../function/preprocessUtils.py).

This audit establishes the frozen mathematical and dimensional data contract for the **Mindanao Model A0** baseline prior to constructing synthetic candidate tensors (Step 21A.5) and acquiring regional climate data.

---

## 2. Comprehensive Reconciliation Findings

### Finding 1: Antecedent RZSM Lag Schedule (3 Lags vs. 6 Lags)
* **Literature Context**: In the main paper text, the authors state that RISE-UNet accepts multiple antecedent lag weeks, providing six lag weeks as an illustrative example of long-term soil memory ingestion.
* **Code-Level Reality for EX29**:
  * In [`function/experimentType.py:L57`](../function/experimentType.py#L57):
    ```python
    EX29 = {
        'region_name': region_name,
        'num_lags_obs_RZSM': 3,
        'include_lags_obs_pwat_spfh_tmax': True,
        'include_reforecast_or_not': True,
        'addtl_experiment': False,
        'experiment_test': 2
    }
    ```
  * In [`function/channelExperiment.py:L85-91`](../function/channelExperiment.py#L85-L91):
    ```python
    if num_lags_obs_RZSM == 3:
        if experiment_test == 2:
            experiment_name = 'EX29'
    ```
  * In [`function/channelExperiment.py:L107-124`](../function/channelExperiment.py#L107-L124):
    `return_num_day_lags_from_weekly_lags(3)` produces `[-1, -7, -14]`.
  * **Conclusion**: Experiment EX29 strictly specifies **3 antecedent RZSM weekly lags** (`num_lags_obs_RZSM = 3`), mapped to daily indices `[-1, -7, -14]`.

---

### Finding 2: Antecedent Lag Temporal Nature (7-Day Rolling Windows vs. Discrete Daily Samples)
* **Question**: Do the lag indices `[-1, -7, -14]` represent instantaneous daily snapshots or multi-day rolling averages?
* **Code Verification**:
  * In [`function/preprocessUtils.py:L88`](../function/preprocessUtils.py#L88):
    ```python
    file = file.rolling(time=7, min_periods=7, center=False).mean()
    ```
  * Every input dataset (RZSM observations, atmospheric variables, and target RZSM) undergoes a **7-day backward trailing rolling mean** (`center=False`) before any lag indexing or extraction occurs.
  * Therefore, relative to the forecast issue date $t$:
    * `Lag -1`: Represents the 7-day backward rolling average over $[t - 7\text{ d}, t - 1\text{ d}]$, which corresponds to **Antecedent Week 1**.
    * `Lag -7`: Represents the 7-day backward rolling average over $[t - 13\text{ d}, t - 7\text{ d}]$, which corresponds to **Antecedent Week 2**.
    * `Lag -14`: Represents the 7-day backward rolling average over $[t - 20\text{ d}, t - 14\text{ d}]$, which corresponds to **Antecedent Week 3**.
  * In [`04d_Plot_permutation_test_results_UPDATE.ipynb:L403`](../04d_Plot_permutation_test_results_UPDATE.ipynb#L403), the author explicitly confirms this correspondence by renaming the dictionary keys:
    ```python
    sorted_columns.rename(index={
        'RZSM_obs_lag-1':  'RZSM_obs_lag_Wk_1',
        'RZSM_obs_lag-7':  'RZSM_obs_lag_Wk_2',
        'RZSM_obs_lag-14': 'RZSM_obs_lag_Wk_3',
    })
    ```
  * **Conclusion**: Lags in EX29 are **7-day backward rolling window averages**, sampled at 7-day intervals preceding forecast issuance.

---

### Finding 3: Lead-Dependent Channel Schedule ($W_1 \to W_4$)
* **Code Verification**:
  * Extracted directly from [`function/loadDataAllWeeks.py:L702-721`](../function/loadDataAllWeeks.py#L702-L721):
    ```python
    var_list = ['pwat_eatm', 'spfh_2m', 'tmax_2m', 'diff_temp_2m', 'hgt_pres']
    if ref_source == 'ECMWF':
        var_list_ref = ['t2m', 'd2m', 'tcw']
        
    if (experiment_test == 2) and (lead == 1):
        add_channels = len(lag_integer_list) + len(var_list) + len(var_list_ref) # 3 + 5 + 3 = 11
    elif (experiment_test == 2) and (lead == 2):
        add_channels = len(lag_integer_list) + len(var_list) + len(var_list_ref) + (lead - 1) # 3 + 5 + 3 + 1 = 12
    elif (experiment_test == 2) and (lead > 2):
        add_channels = len(lag_integer_list) + (lead - 1) # Lead 3: 3 + 2 = 5; Lead 4: 3 + 3 = 6
    ```
  * Dynamic Predictor Omission Rule:
    * In lines 737 and 830, atmospheric observations (`var_list`) and dynamic S2S reforecasts (`var_list_ref`) are ingested **only for Leads 1 and 2** (`if lead <= 2:` and `if (lead in [1, 2]):`).
    * For Leads 3 and 4, atmospheric reanalysis and dynamic numerical reforecasts are completely omitted due to dynamic model skill decay at longer horizons. The model relies strictly on the 3 antecedent RZSM lags plus recursive autoregressive predictions from preceding weeks ($\hat{y}_{W1}, \dots, \hat{y}_{W(k-1)}$).

| Forecast Lead | Total Channels | Channel Decomposition | Explicit Input Variables Ingested | Source Code Citation |
| :---: | :---: | :--- | :--- | :--- |
| **Lead 1 ($W_1$)** | **11** | 3 RZSM + 5 ERA5 + 3 S2S | `RZSM_lag-1`, `RZSM_lag-7`, `RZSM_lag-14`, `pwat`, `spfh`, `tmax`, `diff_temp`, `hgt_pres`, `t2m_ref`, `d2m_ref`, `tcw_ref` | [`loadDataAllWeeks.py:L710`](../function/loadDataAllWeeks.py#L710) |
| **Lead 2 ($W_2$)** | **12** | 3 RZSM + 5 ERA5 + 3 S2S + 1 Rec. | Same 11 as $W_1$ (with $W_2$ S2S) + Prior prediction $\hat{y}_{W1}$ appended as 12th channel | [`loadDataAllWeeks.py:L716`](../function/loadDataAllWeeks.py#L716) |
| **Lead 3 ($W_3$)** | **5** | 3 RZSM + 2 Rec. | `RZSM_lag-1`, `RZSM_lag-7`, `RZSM_lag-14`, Prior predictions $\hat{y}_{W1}$, $\hat{y}_{W2}$ | [`loadDataAllWeeks.py:L719`](../function/loadDataAllWeeks.py#L719) |
| **Lead 4 ($W_4$)** | **6** | 3 RZSM + 3 Rec. | `RZSM_lag-1`, `RZSM_lag-7`, `RZSM_lag-14`, Prior predictions $\hat{y}_{W1}$, $\hat{y}_{W2}$, $\hat{y}_{W3}$ | [`loadDataAllWeeks.py:L719`](../function/loadDataAllWeeks.py#L719) |

* **Conclusion**: The lead-dependent channel schedule $[11, 12, 5, 6]$ is verified to 100% fidelity against the author's primary EX29 execution pathway.

---

### Finding 4: Atmospheric Predictor Mapping (`pwat` $\leftrightarrow$ `tcwv`)
* **Author Ingestion File**:
  * In [`00_min_max_each_region_&reforecast.ipynb:L383`](../00_min_max_each_region_&reforecast.ipynb#L383):
    `f'{era5_dir}/total_column_water_merged.nc4'` is loaded with internal variable identifier `ERA_precipitable_water_obs`.
  * In [`function/addPredictors.py:L62-77`](../function/addPredictors.py#L62-L77):
    `pwat_eatm` is mapped to channel name `'pwat'`, and dynamic variable `'tcw'` is mapped to `'pwat_eatm'`.
  * In the Copernicus Climate Data Store (CDS) for ERA5:
    * `total_column_water_vapour` (short name `tcwv`, units $\text{kg/m}^2$) represents the vertically integrated water vapour, physically identical to Precipitable Water (`pwat`).
* **Conclusion**: Parent `pwat` / `pwat_eatm` is strictly mapped from ERA5 Total Column Water Vapour (`tcwv` / `tcw`, $\text{kg/m}^2$).

---

### Finding 5: Specific Humidity Derivation Formula (`spfh_2m`)
* **Author Code Implementation**:
  * In [`function/preprocessUtils.py:L186`](../function/preprocessUtils.py#L186):
    ```python
    spfh_val = specific_humidity_from_dewpoint(
        pressure[xarray_varname(pressure)].values * units.Pa,
        dewpoint[xarray_varname(dewpoint)].values * units.K
    ).to('kg/kg')
    ```
* **Thermodynamic Formulation**:
  The MetPy calculation evaluates the actual vapor pressure $e$ from dewpoint temperature $T_d$ (in Kelvin), then determines specific humidity $q$:
  1. Vapor pressure $e$ via Bolton (1980) / Tetens relation:
     $$e = 611.2 \cdot \exp\left( \frac{17.67 \cdot (T_d - 273.15)}{T_d - 29.65} \right) \quad [\text{Pa}]$$
  2. Specific humidity $q$:
     $$q = \frac{\epsilon \cdot e}{p - (1 - \epsilon) \cdot e} \quad [\text{kg/kg}]$$
     where:
     * $p$: Surface pressure in Pascals ($\text{Pa}$).
     * $\epsilon = \frac{R_d}{R_v} \approx 0.62198$ (ratio of specific gas constants for dry air and water vapor).
* **Conclusion**: Mindanao A0 data pipelines can either call `metpy.calc.specific_humidity_from_dewpoint` directly or implement the vectorized analytical formula above with identical double-precision constants.

---

### Finding 6: 200-hPa Geopotential Mapping (`hgt_pres`)
* **Author CDS Retrieval**:
  * In [`Data/raw_downloads/ERA5_real/multi_month_china_download_ERA.py:L119-124`](../Data/raw_downloads/ERA5_real/multi_month_china_download_ERA.py#L119-L124):
    Dataset: `reanalysis-era5-pressure-levels`, `pressure_level: '200'`, `variable: 'geopotential'`.
  * In [`00_min_max_each_region_&reforecast.ipynb:L387`](../00_min_max_each_region_&reforecast.ipynb#L387):
    Loaded from `f'{era5_dir}/geopotential_merged.nc4'` under variable `ERA_geopotential_z200_obs`.
  * In [`function/addPredictors.py:L70`](../function/addPredictors.py#L70):
    Mapped to channel name `z200`.
* **Physical Units**:
  * In ERA5, `geopotential` ($z$) has units $\text{m}^2/\text{s}^2$.
  * Geopotential height ($Z$, in geopotential meters $\text{gpm}$) is obtained via $Z = \frac{z}{g_0}$ with $g_0 = 9.80665\text{ m/s}^2$.
  * Because min-max scaling to $[0, 1]$ is applied per grid cell across the training climatology, dividing by constant $g_0$ is mathematically invariant under min-max normalization. For physical clarity in Mindanao A0 pipelines, $Z$ will be stored in standard geopotential meters ($\text{gpm}$).

---

### Finding 7: ECMWF S2S Dynamic Triplet Contract
* **Dynamic Variable Set**:
  * In [`function/loadDataAllWeeks.py:L708`](../function/loadDataAllWeeks.py#L708):
    `var_list_ref = ['t2m', 'd2m', 'tcw']`.
  * Total precipitation (`tp` / `precip`) is **NOT** included in the ECMWF S2S reforecast input channels for EX29.
  * Ingestion is restricted to **Weeks 1 and 2 only**; dynamic reforecasts are dropped for Weeks 3 and 4.

---

## 3. Authoritative EX29 Machine Contract Specification

```yaml
# =============================================================================
# AUTHORITATIVE RECONCILED EX29 CONTRACT SPECIFICATION (STEP 21A.3)
# =============================================================================
experiment_name: "EX29"
primary_task: "Subseasonal Recursive Hybrid RZSM Forecasting"
forecast_horizon_weeks: [1, 2, 3, 4]

channel_schedule:
  week_1: 11
  week_2: 12
  week_3: 5
  week_4: 6

antecedent_rzsm_lags:
  count: 3
  lag_offsets_days: [-1, -7, -14]
  temporal_filter: "7-day backward trailing rolling mean (center=False)"
  physical_mapping:
    lag_-1: "Antecedent Week 1 ([t-7d, t-1d])"
    lag_-7: "Antecedent Week 2 ([t-13d, t-7d])"
    lag_-14: "Antecedent Week 3 ([t-20d, t-14d])"

atmospheric_reanalysis_predictors:
  source: "ERA5"
  temporal_filter: "7-day backward trailing rolling mean"
  active_leads: [1, 2] # Omitted for leads 3 and 4
  lag_offset_days: -1
  variables:
    - name: "pwat_eatm"
      raw_source_variable: "tcwv (total_column_water_vapour)"
      channel_name: "pwat"
      units: "kg/m^2"
    - name: "spfh_2m"
      raw_source_variables: ["surface_pressure (sp)", "2m_dewpoint_temperature (d2m)"]
      derivation_method: "Bolton (1980) / MetPy specific_humidity_from_dewpoint"
      channel_name: "spfh"
      units: "kg/kg"
    - name: "tmax_2m"
      raw_source_variable: "maximum_2m_temperature_since_previous_post_processing"
      channel_name: "tmax"
      units: "Kelvin"
    - name: "diff_temp_2m"
      raw_source_variables: ["tmax", "tmin"]
      formula: "tmax - tmin"
      channel_name: "diff_temp"
      units: "Kelvin"
    - name: "hgt_pres"
      raw_source_variable: "geopotential (z) at 200 hPa"
      formula: "z / 9.80665"
      channel_name: "z200"
      units: "gpm"

dynamic_s2s_predictors:
  source: "ECMWF S2S Reforecasts"
  active_leads: [1, 2] # Omitted for leads 3 and 4
  ensemble_members: 11
  variables:
    - name: "t2m"
      channel_name: "2m_temp_ref"
    - name: "d2m"
      channel_name: "2m_dewpoint_ref"
    - name: "tcw"
      channel_name: "pwat_eatm_ref"

recursive_autoregressive_feedback:
  lead_1: null
  lead_2: ["RZSM_prediction_lead1"]
  lead_3: ["RZSM_prediction_lead1", "RZSM_prediction_lead2"]
  lead_4: ["RZSM_prediction_lead1", "RZSM_prediction_lead2", "RZSM_prediction_lead3"]
  masking_rule: "Zero-fill non-land pixels (np.where(mask==0, 0.0, prediction))"
```

---

## 4. Formal Step 21A.3 Certification & Next Steps

* **Certification**: Step 21A.3 is formally declared **PASSED**.
* **Transition Clearance**: With the channel schedule formally locked at $[11, 12, 5, 6]$ and all predictor mappings verified, the project proceeds to:
  * **Step 21A.4**: Enumerate candidate rectangular computational domains $H \times W$ around the Mindanao GIS boundary ($[118.06^\circ\text{E}, 4.59^\circ\text{N}] \to [126.60^\circ\text{E}, 10.47^\circ\text{N}]$) satisfying 16-pooling divisibility.
  * **Step 21A.5**: Execute synthetic 4-week cascade compatibility tests across candidate dimensions through frozen `UNET_RZSM`.
