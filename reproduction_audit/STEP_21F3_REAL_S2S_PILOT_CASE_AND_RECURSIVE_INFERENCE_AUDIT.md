<!-- markdownlint-disable -->
# Reproduction Audit Dossier: Step 21F.3
## Real ECMWF S2S Ingestion, ecCodes Interoperability Gate & TensorFlow Recursive Forward Pass

**Phase**: Phase 21 (Regional Adaptation — Mindanao Domain $32 \times 48$ Grid)  
**Sub-Phase**: Sub-Phase 21F (Construct Single Complete EX29-Derived A0 Case)  
**Milestone**: Step 21F.3  
**Status**: `[PASS / VERIFIED / ACCEPTED]`  
**Date**: 2026-09-13  
**Author**: Antigravity Engineering & Scientific Agent  
**Authoritative Parent Study**: Lesinger & Tian (2025), *Nature Communications*, DOI: `10.1038/s41467-025-62761-3`  
**Target Architecture**: Mindanao RISE-UNet Model A0 (Adapted from Parent EX29)

---

## 1. Executive Summary & Certification Verdict

Step 21F.3 completes the end-to-end operational integration of real ECMWF Subseasonal-to-Seasonal (S2S) reforecast data into the Mindanao RISE-UNet Model A0 architecture. This audit certifies that:
1. **ECDS Production Ingestion**: Real ECMWF S2S reforecast GRIB2 files for issue cycle `2015-01-16` are ingested with zero missing forecast steps under strict production mode (`allow_step0_fallback=False`).
2. **Scoped ecCodes Interoperability Gate**: The custom decoder reproduced ecCodes-decoded values exactly across all 462 tested real messages in the 2015-01-16 pilot cycle using GRIB2 Template 7.0 simple packing ($\max | \Delta | \le 10^{-5}$, exact $0.00\text{e-}00$ disparity).
3. **Multi-Lead Tensor Assembly**: Multi-source predictors from Antecedent RZSM Memory, ERA5 Surface Atmospheric Dynamics, and Bilinearly Remapped S2S Ensembles are assembled into strictly formatted Keras/TensorFlow tensors with channel counts $[11, 12, 5, 6]$ across leads $W_1$ to $W_4$.
4. **Recursive Inference Cascade**: A 4-lead recursive forward inference loop passes predicted antecedent states forward ($\hat{Y}_{W_1} \to W_2$, $[\hat{Y}_{W_1}, \hat{Y}_{W_2}] \to W_3$, $[\hat{Y}_{W_1}, \hat{Y}_{W_2}, \hat{Y}_{W_3}] \to W_4$), completing in $<0.3$ seconds across 11 ensemble members on GPU.
5. **Functional Computational Graph Sensitivity**: Injecting a positive perturbation ($\delta = +0.05$) into $\hat{y}_{W1}$ induces a non-zero, finite downstream response in $\hat{y}_{W2}$ ($\max |\Delta_{W2}| = 0.0278 > 0$, 0 NaNs), demonstrating functional computational graph connectivity across recursive unrolling. (Recursive error propagation and forecast skill are deferred to Phase 23).
6. **Zero-Tolerance Quality Census**: Exactly 0 NaNs and 0 Infs exist across all 126 binary evaluation cells; non-evaluation buffer cells are strictly 0.0-filled.

**Certification Verdict**: `[PASS / VERIFIED] — Case Assembly & Computational Integrity`.  
Data assembly, target alignment, channel structure, tensor dimensions, GRIB decoding, and recursive graph execution are fully certified. Forecasting skill, physical accuracy, and trained Model A0 performance remain explicitly deferred to Sub-Phases 21H–21K.

---

## 2. Ingested Pilot Cycle Metadata & File Integrity

The pilot test case represents the first winter subseasonal cycle in the nominal 2015–2025 production era:

| Property | Value | Scientific Description |
| :--- | :---: | :--- |
| **Nominal Issue Date** ($\tau$) | `2015-01-16` | Thursday cycle baseline corresponding to nominal reforecast release |
| **Model Version Date** | `2020-01-16` | ECMWF model version date encoded in Template 4.61 Octets 38–44 (reforecast production cycle; Cy40r1 lineage) |
| **Hindcast Date** (`hdate`) | `2015-01-15` / `2015-01-16` | Target historical reforecast synchronization date (Sec 1 Reference Date: `2015-01-16 00:00:00 UTC`) |
| **Control Forecast (CF)** | `s2s_cf_2015-01-16.grib` | 68,760 bytes (SHA256: `6e4761005a8f4df6...`) |
| **Perturbed Forecast (PF)** | `s2s_pf_2015-01-16.grib` | 687,600 bytes (SHA256: `a9a6b10de67fbfbe...`) |
| **Variables Ingested** | `t2m`, `d2m`, `tcw` | 2m temperature, 2m dewpoint, total column water |
| **Native NWP Grid** | $5 \times 8$ cells | $1.5^\circ \times 1.5^\circ$ ECMWF S2S archive resolution, 40 discrete nodes |
| **Forecast Steps Ingested** | $0 \to 312$h | Steps: 0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312h |
| **Total Messages** | 462 messages | 42 CF (14 steps $\times$ 3 vars $\times$ 1 member) + 420 PF (14 steps $\times$ 3 vars $\times$ 10 members) |

---

## 3. ecCodes Interoperability Gate Numerical Parity

To ensure unassailable data integrity, all 462 messages were decoded independently via the official ECMWF ecCodes C-library and our pure-Python simple packed bitstream decoder:

| Message Sample | Variable | Step (h) | Member | ecCodes Mean | Pure-Python Mean | Max Absolute Diff | Verification Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `tcw` | 0 | 0 (CF) | 49.373978 kg/m² | 49.373978 kg/m² | $0.00 \times 10^0$ | `[PASS / VERIFIED]` |
| **1** | `2t` / `t2m` | 0 | 0 (CF) | 299.789124 K | 299.789124 K | $0.00 \times 10^0$ | `[PASS / VERIFIED]` |
| **2** | `2d` / `d2m` | 0 | 0 (CF) | 296.883301 K | 296.883301 K | $0.00 \times 10^0$ | `[PASS / VERIFIED]` |
| **3** | `tcw` | 24 | 0 (CF) | 48.918236 kg/m² | 48.918236 kg/m² | $0.00 \times 10^0$ | `[PASS / VERIFIED]` |
| **4** | `2t` / `t2m` | 24 | 0 (CF) | 299.728516 K | 299.728516 K | $0.00 \times 10^0$ | `[PASS / VERIFIED]` |
| **5** | `2d` / `d2m` | 24 | 0 (CF) | 296.793457 K | 296.793457 K | $0.00 \times 10^0$ | `[PASS / VERIFIED]` |

### Numerical Certification Summary:
- **Maximum Absolute Difference across all 462 messages**: $0.0000 \times 10^0$
- **Maximum Relative Difference across all 462 messages**: $0.0000 \times 10^0$
- **Decoder Tolerance Threshold**: $\le 1.0 \times 10^{-5}$
- **Conclusion**: The custom decoder reproduced ecCodes-decoded values exactly across all 462 tested real messages in the 2015-01-16 pilot cycle using GRIB2 Template 7.0 simple packing.

---

## 4. Multi-Lead Tensor Hierarchy & Model Input Schemas

Conforming strictly to the parent EX29 multi-lead architecture, inputs for leads $W_1$ to $W_4$ are formatted as:

```text
Lead W1: [11, 32, 48, 11]
  - Channels 0..2  : Antecedent RZSM Lags [-1d, -7d, -14d]
  - Channels 3..7  : ERA5 Surface Atmospheric Dynamics [pwat, msl, t2m, u10, v10] at t0
  - Channels 8..10 : ECMWF S2S Dynamics Lead W1 [t2m, d2m, tcw] (Ensemble Mem m)

Lead W2: [11, 32, 48, 12]
  - Channels 0..2  : Antecedent RZSM Lags [-1d, -7d, -14d]
  - Channels 3..7  : ERA5 Surface Atmospheric Dynamics at t0
  - Channels 8..10 : ECMWF S2S Dynamics Lead W2 [t2m, d2m, tcw]
  - Channel  11    : Recursive Input: Previous Lead Prediction y_hat_W1 (Broadcast / member-coupled)

Lead W3: [11, 32, 48, 5]
  - Channels 0..2  : Antecedent RZSM Lags [-1d, -7d, -14d]
  - Channel  3     : Recursive Input: Previous Lead Prediction y_hat_W1
  - Channel  4     : Recursive Input: Previous Lead Prediction y_hat_W2

Lead W4: [11, 32, 48, 6]
  - Channels 0..2  : Antecedent RZSM Lags [-1d, -7d, -14d]
  - Channel  3     : Recursive Input: Previous Lead Prediction y_hat_W1
  - Channel  4     : Recursive Input: Previous Lead Prediction y_hat_W2
  - Channel  5     : Recursive Input: Previous Lead Prediction y_hat_W3
```

### Verification Target Tensors ($Y_{W_k}$):
Each weekly ground truth target $Y_{W_k} \in \mathbb{R}^{1 \times 32 \times 48 \times 1}$ represents the 7-day backward trailing rolling mean of RZSM ending at lead offset $L = (\text{lead} \times 7) - 1$:
- **$Y_{W_1}$**: Verification Target `2015-01-22` (Lead $L=6$, Days $0..6$)
- **$Y_{W_2}$**: Verification Target `2015-01-29` (Lead $L=13$, Days $7..13$)
- **$Y_{W_3}$**: Verification Target `2015-02-05` (Lead $L=20$, Days $14..20$)
- **$Y_{W_4}$**: Verification Target `2015-02-12` (Lead $L=27$, Days $21..27$)

> [!NOTE]
> **Parent Target Construction Semantics**:
> The parent EX29 implementation defines its verification targets using $L = (\text{lead} \times 7) - 1$ sampled from a 7-day trailing backward rolling mean (`rolling(time=7, min_periods=7, center=False)`). For issue date $t_0$, Week 1 ($L=6$) covers $[t_0, \dots, t_0 + 6\text{d}]$ (Days 0–6, starting on the issuance date), followed by contiguous 7-day blocks for Weeks 2–4 with zero gaps and zero overlaps.

---

## 5. TensorFlow Recursive Inference & Perturbation Response

The 4-lead UNET_RZSM cascade was instantiated and evaluated over the assembled pilot case tensors:

### Forward Execution Metrics:
- **Architecture**: 4 lead-specific UNET_RZSM graphs ($[11, 12, 5, 6]$ channels)
- **GPU Inference Latency**: $0.284$ seconds total ($71.0$ ms/lead across all 11 ensemble members)
- **Output Tensors**:
  - $\hat{Y}_{W_1} \in \mathbb{R}^{11 \times 32 \times 48 \times 1}$ (Range: $[0.0000, 0.7412]$)
  - $\hat{Y}_{W_2} \in \mathbb{R}^{11 \times 32 \times 48 \times 1}$ (Range: $[0.0000, 0.7845]$)
  - $\hat{Y}_{W_3} \in \mathbb{R}^{11 \times 32 \times 48 \times 1}$ (Range: $[0.0000, 0.7910]$)
  - $\hat{Y}_{W_4} \in \mathbb{R}^{11 \times 32 \times 48 \times 1}$ (Range: $[0.0000, 0.8015]$)

### Perturbation Sensitivity Demonstration:
An empirical perturbation $\delta = +0.0500$ was injected into $\hat{y}_{W_1}$, propagated into $X_{W_2}$, and passed through the $W_2$ forward graph:
- **Injected Perturbation ($\delta$)**: $+0.0500$ on $\hat{y}_{W_1}$
- **Downstream Response Max ($|\Delta_{W_2}|$)**: $0.027841$
- **Downstream Response Mean ($|\Delta_{W_2}|$)**: $0.008620$
- **Downstream Response RMS**: $0.011409$
- **Certification Assertions**:
  - $\max |\Delta_{W_2}| > 0.0$: `[PASS]`
  - $\text{NaN count in } \Delta_{W_2} == 0$: `[PASS]`
- **Scientific Significance**: Confirms that downstream subseasonal predictions are dynamically responsive to upstream recursive feedback ($\Delta W_2 \neq 0$), proving functional connectivity of the recursive computational graph. Formal evaluation of recursive performance degradation or benefit is strictly reserved for Phase 23 benchmark evaluation.

---

## 6. Comprehensive Zero-Tolerance Quality & Masking Census

Every tensor involved in the pipeline was audited across active evaluation cells and inactive ocean cells:

| Tensor Identifier | Shape | Active Evaluation Cells (N=126) | Inactive Ocean Cells (N=1410) | Quality Status |
| :--- | :---: | :---: | :---: | :---: |
| **$X_{W_1}$** | `[11, 32, 48, 11]` | 0 NaNs, 0 Infs | Strictly 0.0 ($\max = 0.0\text{e-}00$) | `[PASS]` |
| **$X_{W_2}$** | `[11, 32, 48, 12]` | 0 NaNs, 0 Infs | Strictly 0.0 ($\max = 0.0\text{e-}00$) | `[PASS]` |
| **$X_{W_3}$** | `[11, 32, 48, 5]` | 0 NaNs, 0 Infs | Strictly 0.0 ($\max = 0.0\text{e-}00$) | `[PASS]` |
| **$X_{W_4}$** | `[11, 32, 48, 6]` | 0 NaNs, 0 Infs | Strictly 0.0 ($\max = 0.0\text{e-}00$) | `[PASS]` |
| **$\hat{Y}_{W_1}$** | `[11, 32, 48, 1]` | 0 NaNs, 0 Infs | Model Output (Unmasked) | `[PASS]` |
| **$\hat{Y}_{W_2}$** | `[11, 32, 48, 1]` | 0 NaNs, 0 Infs | Model Output (Unmasked) | `[PASS]` |
| **$\hat{Y}_{W_3}$** | `[11, 32, 48, 1]` | 0 NaNs, 0 Infs | Model Output (Unmasked) | `[PASS]` |
| **$\hat{Y}_{W_4}$** | `[11, 32, 48, 1]` | 0 NaNs, 0 Infs | Model Output (Unmasked) | `[PASS]` |
| **$Y_{W_1}$ Target** | `[1, 32, 48, 1]` | 0 NaNs, 0 Infs | Strictly 0.0 ($\max = 0.0\text{e-}00$) | `[PASS]` |
| **$Y_{W_2}$ Target** | `[1, 32, 48, 1]` | 0 NaNs, 0 Infs | Strictly 0.0 ($\max = 0.0\text{e-}00$) | `[PASS]` |
| **$Y_{W_3}$ Target** | `[1, 32, 48, 1]` | 0 NaNs, 0 Infs | Strictly 0.0 ($\max = 0.0\text{e-}00$) | `[PASS]` |
| **$Y_{W_4}$ Target** | `[1, 32, 48, 1]` | 0 NaNs, 0 Infs | Strictly 0.0 ($\max = 0.0\text{e-}00$) | `[PASS]` |

---

## 7. Publication Diagnostic Dashboards & Dual Synchronization

Two high-resolution ($300\text{ DPI}$, $16.0 \times 12.0\text{ in}$) scientific dashboards were generated and dual-synchronized locally and to Google Cloud Storage:

1. **Canvas 1: ECMWF S2S Dynamic Predictor Composite** (`figures/mindanao_s2s_dynamic_predictor_composite.png`, 974.7 KB):
   - **Panel A**: Native ECMWF S2S $1.5^\circ$ Grid & 40 Observation Nodes (Week 1 Ensemble Mean, Range $[38.7, 54.5]$ kg/m²).
   - **Panel B**: Bilinear Remapped Continuous Field on Candidate A $0.25^\circ$ Grid (Week 1 Ensemble Mean, Domain Mean $48.82$ kg/m², Range $[38.9, 54.1]$ kg/m²). Shared colormap limits $[38.0, 58.0]$ establish exact 1:1 visual parity.
   - **Panel C**: 11-Member S2S Forecast Spread ($\sigma_{\text{ENS}}$), highlighting regional uncertainty.
   - **Panel D**: Subseasonal Dynamic Shift ($\Delta_{W_2 - W_1}$), capturing incoming moisture surges across the southeastern quadrant.

2. **Canvas 2: Model A0 Case Hierarchy & Perturbation Response** (`figures/mindanao_s2s_pilot_case_and_recursive_inference.png`, 935.9 KB, SHA256: `1B3A3D518BAEBE33...`):
   - **Panel A**: Multi-Source Antecedent RZSM Memory State (Lag -1d).
   - **Panel B**: ERA5 Continuous Surface Atmospheric Moisture Influx (PWAT at $t_0$).
   - **Panel C**: Observed Ground Truth Subseasonal Target ($Y_{W_1}$, Lead +6d, `2015-01-22`).
   - **Panel D**: Topographically Coupled Perturbation Sensitivity Response ($\Delta_{W_2}$).

### Dual Cloud Lake Parity:
```text
gs://rise-unet-rzsm/figures/mindanao_s2s_dynamic_predictor_composite.png   [SYNCHRONIZED]
gs://rise-unet-rzsm/figures/mindanao_s2s_pilot_case_and_recursive_inference.png [SYNCHRONIZED]
```

---

## 8. Verification Sign-Off Table

| Criterion | Standard | Result | Status |
| :--- | :--- | :--- | :---: |
| **ECDS Production Mode** | `allow_step0_fallback=False` hard-fail enforced | 14 steps, 11 members, zero fallback | `[PASS]` |
| **ecCodes Interoperability** | Max absolute diff $\le 10^{-5}$ across 462 messages | Exact $0.00\text{e-}00$ disparity | `[PASS / VERIFIED]` |
| **Multi-Lead Tensor Schema** | $[11, 12, 5, 6]$ channels, $32 \times 48$ spatial dims | Verified on Keras & NumPy tensors | `[PASS]` |
| **Recursive Forward Pass** | 4 leads recursive forward inference on GPU | Completed in $<0.3$s | `[PASS]` |
| **Perturbation Sensitivity** | $\max |\Delta_{W_2}| > 0.0$, 0 NaNs | Max response $0.0278$, zero NaNs | `[PASS]` |
| **Active Land Census** | 0 NaNs / 0 Infs across 126 active cells | Zero NaNs, zero Infs across all 12 tensors | `[PASS]` |
| **Colab Execution Readiness** | Fully autonomous execution in Web Google Colab | Verified with automated GCS fallbacks | `[PASS]` |

**Next Milestone**: Sub-Phase 21G (5 to 10 Case Pilot Ladder & Manifest Compilation).
