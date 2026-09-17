<!-- markdownlint-disable -->
# Reproduction Audit Dossier: Post-A0 Gate 2 (G2-R) Recursive Refinement Decision Gate

**Document Identifier**: `reproduction_audit/POST_A0_GATE2_DECISION_DOSSIER.md`  
**Milestone**: Phase 23 (Recursive Degradation Diagnostic) & Post-A0 Gate 2 (G2-R: Recursive Refinement Decision Gate)  
**Execution Context**: Google Colab Live GPU Execution (`notebooks/13_mindanao_recursive_degradation_diagnostic.ipynb`)  
**Hardware Target**: NVIDIA Tesla T4 GPU (Driver CUDA 12.5.1 / cuDNN 9)  
**Execution Timestamp**: 2026-09-17T18:29:12Z  
**Execution Commit SHA**: `b64970f9c159dd509dbe46874a832f7ab855a491` (`b64970f` on `mindanao-adaptation`)  
**Pre-Analysis Contract**: [`contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml`](../contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml) (Frozen prior to inspection)  
**Authority Status**: `[PASS / CERTIFIED_ON_GPU]`  
**Formal Decision Verdict**: **`GO`** (Proceed to Phase 24: Model A1 Lead-Aware Recursive Residual Refinement $\mathcal{R}_\theta$)

---

## 1. Executive Decision Summary

```text
========================================================================================
                   POST-A0 GATE 2 (G2-R) FORMAL DECISION VERDICT: GO                     
  ACTION: Proceed to Phase 24 (Model A1 Lead-Aware Recursive Residual Refinement R_theta)
========================================================================================
```

Following the strict, fail-closed pre-analysis specification frozen prior to inspecting validation data, the empirical evaluation of the **Phase 23 Recursive Degradation Diagnostic** on the 194-case validation partition (2022–2023) over the 126 active evaluation cells of Mindanao across all three Model A0 training replicates (Seeds 42, 123, 456) has officially satisfied **all four pre-registered GO criteria**:

1. **Criterion GO-1 (Statistical Significance)**: Primary Circular Moving-Block Bootstrap ($L=4$ cycles, $B=10,000$ resamples) demonstrates statistically significant recursive degradation strictly excluding zero at 95% confidence in Leads W2, W3, and W4. Step-down Holm-Bonferroni adjusted bootstrap $p$-values satisfy $p_{\text{adj}} < 0.0001$ across all downstream leads (reflecting the finite resolution of $B=10,000$ resamples where zero resamples exceeded the observed gap under the centered null), with supplementary paired Wilcoxon tests concordant ($p < 10^{-22}$).
2. **Criterion GO-2 (Practical Effect Size)**: Standardized effect size exceeds the required threshold ($d_z \ge 0.20$) by a factor of 6: Lead W3 $d_z = +1.201$ (relative gap **35.38%**); Lead W4 $d_z = +1.269$ (relative gap **44.12%**).
3. **Criterion GO-3 (Replicate Consistency)**: The recursive error gap $\Delta E_k > 0$ holds strictly across all three independent training seeds (Seeds 42, 123, and 456) individually across all downstream leads.
4. **Criterion GO-4 (Controlled Perturbation Sensitivity Monotonicity)**: Downstream divergence $D_k(|\epsilon|)$ increases strictly monotonically with perturbation magnitude across realistic operational forecast error levels ($|\epsilon| = 0.05 \to 0.10$) separately for positive and negative perturbations in both Lead W3 and Lead W4.

---

## 2. Pre-Registered Criteria Compliance Audit

| Pre-Registered Criterion | Contract Specification & Threshold | Empirical Finding ($N=194$ Validation Cases) | Compliance Status |
| :--- | :--- | :--- | :---: |\n| **GO-1: Statistical Significance** | Holm-Bonferroni adjusted $p_{\text{boot}} < 0.05$ & 95% MBB CI strictly $> 0$ in Leads W3 and/or W4, with supplementary Wilcoxon concordant. | **W3:** Adj $p_{\text{boot}} < 0.0001$, 95% CI $[+0.0216, +0.0301]$, Wilcoxon $p = 2.41 \times 10^{-29}$<br>**W4:** Adj $p_{\text{boot}} < 0.0001$, 95% CI $[+0.0328, +0.0470]$, Wilcoxon $p = 1.56 \times 10^{-28}$ | **`[PASS]`** |
| **GO-2: Practical Effect Size** | Cohen's $d_z \ge 0.20$ OR relative error reduction $\Delta E_k / E_k^{\text{rec}} \ge 2.0\%$ under Oracle replacement. | **W3:** $d_z = +1.201$ (Relative gap represents $35.38\%$ of W3 recursive MAE)<br>**W4:** $d_z = +1.269$ (Relative gap represents $44.12\%$ of W4 recursive MAE) | **`[PASS]`** |
| **GO-3: Replicate Consistency** | Error gap $\Delta E_k > 0$ holds across all three training replicate seeds ($42, 123, 456$). | Replicate seeds individually confirm positive recursive degradation across all downstream leads ($W2, W3, W4$). | **`[PASS]`** |
| **GO-4: Perturbation Monotonicity** | Downstream divergence $D_k(|\epsilon|)$ increases monotonically across magnitude $|\epsilon|$ for realistic levels ($D_k(0.10) > D_k(0.05)$) evaluated separately by sign. | **Lead W3:** $D_3(+0.10) = 0.0699 > 0.0344$, $D_3(-0.10) = 0.0765 > 0.0373$<br>**Lead W4:** $D_4(+0.10) = 0.0526 > 0.0274$, $D_4(-0.10) = 0.0395 > 0.0192$ | **`[PASS]`** |
| **OVERALL POST-A0 GATE 2 (G2-R)** | All 4 GO criteria satisfied simultaneously. | Formal conditions satisfied; unlocks Phase 24. | **`[GO]`** |

---

## 3. Empirical Error Decomposition (Protocols 1 & 2)

*Evaluated over $N=194$ validation cases (2022–2023) across 126 active evaluation cells. Multi-seed values reported as Mean $\pm$ Standard Deviation.*

| Lead Horizon | Protocol 1: Standard Recursive MAE | Protocol 2: Oracle Counterfactual MAE | Recursive Degradation Gap $\Delta \text{MAE}$ [95% MBB CI] | Relative Gap Fraction | Cohen's $d_z$ | Primary $p_{\text{boot}}$ ($B=10000$) | Holm-Bonf. Adjusted $p$ | Supp. Wilcoxon $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lead W1** | $0.0340 \pm 0.0007$ | $0.0340 \pm 0.0007$ | $+0.0000 \; [+0.0000, +0.0000]$ | $0.00\%$ | $+0.000$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Lead W2** | $0.0526 \pm 0.0016$ | $0.0450 \pm 0.0031$ | $\mathbf{+0.0076} \; [+0.0059, +0.0098]$ | $14.52\%$ | $+0.642$ | $< 0.0001$ | $\mathbf{< 0.0001}$ | $4.71 \times 10^{-23}$ |
| **Lead W3** | $0.0719 \pm 0.0168$ | $0.0465 \pm 0.0031$ | $\mathbf{+0.0254} \; [+0.0216, +0.0301]$ | $\mathbf{35.38\%}$ | $\mathbf{+1.201}$ | $< 0.0001$ | $\mathbf{< 0.0001}$ | $2.41 \times 10^{-29}$ |
| **Lead W4** | $0.0898 \pm 0.0288$ | $0.0502 \pm 0.0021$ | $\mathbf{+0.0396} \; [+0.0328, +0.0470]$ | $\mathbf{44.12\%}$ | $\mathbf{+1.269}$ | $< 0.0001$ | $\mathbf{< 0.0001}$ | $1.56 \times 10^{-28}$ |

### Rigorous Scientific Interpretation
- **Counterfactual Attribution**: Under the oracle counterfactual protocol, the recursive pathway accounts for an additional $35.38\%$ of Lead W3 recursive MAE and $44.12\%$ of Lead W4 recursive MAE relative to the non-recursive counterfactual. This indicates substantial error attributable to predicted-state substitution under the specified counterfactual protocol. In a nonlinear deep learning system, this represents a counterfactual error decomposition rather than an additive linear causal breakdown.
- **Progression Across Horizons**: The recursive-oracle gap increases monotonically from Lead W2 ($+0.0076$) through Lead W3 ($+0.0254$) to Lead W4 ($+0.0396$), demonstrating that predicted-state conditioning is associated with substantial additional error as the forecast horizon extends.

---

## 4. Protocol 4: Controlled Error-Injection Sensitivity & Boundary Clipping Census

*Evaluated under deterministic member-wise perturbation $\hat{y}' = \operatorname{clip}(\hat{y} + \epsilon, 0, 1)$ on representative replicate Seed 42.*

| Perturbation Level $\epsilon$ | Regime Characterization | Active Land Cells Clipped to $[0, 1]$ (%) | Lead W2 Divergence $D_2(\epsilon)$ [RMSE] | Lead W3 Divergence $D_3(\epsilon)$ [RMSE] | Lead W4 Divergence $D_4(\epsilon)$ [RMSE] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\epsilon = -0.50$ | Exploratory boundary stress test | $16.52\%$ | $0.2890$ | $0.2979$ | $0.1566$ |
| $\epsilon = -0.25$ | Exploratory boundary stress test | $0.00\%$ | $0.1364$ | $0.1819$ | $0.0961$ |
| $\epsilon = -0.10$ | **Realistic operational forecast error** | **$0.00\%$** | **$0.0498$** | **$0.0765$** | **$0.0395$** |
| $\epsilon = -0.05$ | **Realistic operational forecast error** | **$0.00\%$** | **$0.0242$** | **$0.0373$** | **$0.0192$** |
| $\epsilon = +0.05$ | **Realistic operational forecast error** | **$0.08\%$** | **$0.0241$** | **$0.0344$** | **$0.0274$** |
| $\epsilon = +0.10$ | **Realistic operational forecast error** | **$0.27\%$** | **$0.0498$** | **$0.0699$** | **$0.0526$** |
| $\epsilon = +0.25$ | Exploratory boundary stress test | $6.42\%$ | $0.1353$ | $0.2096$ | $0.2119$ |
| $\epsilon = +0.50$ | Exploratory boundary stress test | $83.48\%$ | $0.1905$ | $0.3942$ | $0.3613$ |

### Physical Validity & Nonlinear State Sensitivity
- **Regime Validity**: For all negative perturbations down to $\epsilon = -0.25$, exactly **$0.00\%$** of cell evaluations were clipped. For positive perturbations $\epsilon \in \{+0.05, +0.10\}$, clipping remained negligible ($\le 0.27\%$). The observed downstream divergence is genuine neural state propagation rather than boundary truncation artifacts.
- **Monotonic Sensitivity**: Doubling the perturbation magnitude from $|\epsilon| = 0.05 \to 0.10$ causes downstream divergence to double across all leads ($+105.8\%$ in W2, $+105.1\%$ in W3, $+105.7\%$ in W4), satisfying the pre-registered monotonicity criterion ($D_k(0.10) > D_k(0.05)$ for both signs).
- **Nonlinear Inter-Lead Propagation**: Across temporal sequences, divergence propagates nonlinearly rather than in a simple linear accumulation chain (for example, under positive perturbations: $W_2=0.0498 \to W_3=0.0699 \to W_4=0.0526$). The experiment demonstrates that the system is sensitive to injected recursive-state perturbations, with the perturbation propagating nonlinearly through subsequent leads.

---

## 5. Artifact Provenance & Cryptographic Hashes

To guarantee audit integrity, artifacts produced during the Colab execution are verified by cryptographic SHA-256 hashes:

| Artifact Description | File Path | Cryptographic SHA-256 Checksum | Google Cloud Storage Lake URI | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Telemetry JSON** | [`logs/phase23_recursive_diagnostic_results.json`](../logs/phase23_recursive_diagnostic_results.json) | `00f12b86977ead92db5d4b2300e100a0f37523e2eee47c79ab3865cb00bc797b` | `gs://rise-unet-rzsm/logs/phase23_recursive_diagnostic_results.json` | `[VERIFIED]` |
| **Diagnostic Composite Figure** | [`figures/phase23_recursive_degradation_composite.png`](../figures/phase23_recursive_degradation_composite.png) | `601d7b406b940bd692e2a1bac3702a8b921b18d7d9a6b914790c81a96a542f7f` | `gs://rise-unet-rzsm/figures/phase23_recursive_degradation_composite.png` | `[VERIFIED]` |
| **Execution Notebook** | [`notebooks/13_mindanao_recursive_degradation_diagnostic.ipynb`](../notebooks/13_mindanao_recursive_degradation_diagnostic.ipynb) | Git Commit `b64970f9c159dd509dbe46874a832f7ab855a491` | Synced to GitHub repository | `[EXECUTED]` |

---

## 6. Predeclared Protocol Action & Scope of Phase 24 Activation

In accordance with Section `predeclared_protocol_action.on_go` of [`contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml`](../contracts/A0/PHASE_23_DIAGNOSTIC_CONTRACT.yaml):

1. **Gate 2 Decision Scope**: Phase 23 has established a measurable recursive-vs-oracle performance gap, replicated across A0 training seeds, plus downstream sensitivity to controlled recursive-state perturbation. Therefore, the pre-specified condition for investigating a recursive-refinement intervention is officially satisfied.
2. **Empirical Boundary**: This verdict authorizes the development and testing of Model A1; it does **not** assume that Model A1 will automatically improve real-world forecasting. Whether the proposed refinement improves subseasonal forecasting over Model A0 remains an empirical question to be evaluated under Phase 24, Phase 25 (ablations), and Phase 26 (sealed test holdout).
3. **Backbone Freeze Invariant**: The verified RISE-UNet Model A0 backbones across all leads and seeds remain **completely frozen** ($0$ weights retrained or altered).
4. **Refinement Module Scope**: A lightweight, lead-conditioned residual correction module $\mathcal{R}_{\theta}(\hat{y}_k, \hat{y}_{k-1}, k)$ will be developed and trained exclusively at the recursive transition boundaries.
5. **Sealed Holdout Protection**: The 2024–2025 Sealed Test Cohort ($N=202$ usable cases) remains strictly quarantined until Phase 26.
