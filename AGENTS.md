<!-- markdownlint-disable -->
# Antigravity Workspace Guidelines: Mindanao RISE-UNet Adaptation

## Document Authority Hierarchy
```text
contracts/
    ↓
Machine-Readable Scientific & Technical Truth

master plans (mindanao_adaptation_master_plan.md
            + mindanao_remediation_and_optimal_model_masterplan.md)
    ↓
Project Execution Roadmap & Before/During/After Optimality Playbook

artifact registry (reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md)
    ↓
Inventory, Directory Map & Status Index
```

## Mandatory Master Plan Execution & Optimal Model Engineering Rule
All contributors and AI agents **MUST strictly follow** [`mindanao_adaptation_master_plan.md`](mindanao_adaptation_master_plan.md) and [`mindanao_remediation_and_optimal_model_masterplan.md`](mindanao_remediation_and_optimal_model_masterplan.md) (grounded in [`reproduction_audit/27_end_to_end_verification_and_redo_verdict.md`](reproduction_audit/27_end_to_end_verification_and_redo_verdict.md)) across every stage **before, during, and after** baseline adaptation and `SCPW–MRGR` enhancement to deliver the most accurate, rigorously validated, and academically defendable models:
1. **Before Training (`Phase 21R`)**: Clear all verified code blockers (`scpw_mrgr.py` Keras 3 `keras.ops` + differentiable joint TF MRGR graph + `eval_mask` zeroing + volumetric `m³/m³` drought unscaling; `cloud_lake.py` remote paths; `preprocessing_v3.py` default v3 anomaly input), complete missing `2014-12` warmup & `26` atmospheric months, build unsaturated `1,059`-case v3 tensors (`<= 2023-10-20` fit), profile `10.24M ExtraConv` `B=66` VRAM (`mixed_float16` vs FP32), pass all TF-gated unit tests, and pre-register the `183`-fit OOF inner stopping rule with **zero peeking** into Held-Out OOF years or `Sealed Test`.
2. **During Training (`Phases 23R–24R`)**: Train authentic `modelRzsmReluExtraConv` (`10.24M`) on `[11, 12, 5, 6]` channels with matched initial weights `w0(seed)` across seeds `[42, 123, 456]`, enforce strict `eval_mask` zero-filling (`~eval_mask = 0.0`) on **every** recursive/MRGR handoff, keep `BN` frozen (`training=False`) with backbone dropout active (`training=True`) for `D=10` (`110`-draw) trajectories, verify `RiskNet` Huber & Spearman rank quality before freezing, keep `r*` unclipped while logging active-cell excursions, and enforce zero synthetic diagnostic plots or MAE-as-CRPS mislabeling.
3. **After Training (`Phases 25R–26R`)**: Run external benchmarks (`Persistence`, `U-Net`, `UNet++`) identically on `851/100`, freeze all checkpoint/hyperparameter/decile selections on Validation (`N=100`), unseal `Sealed Test` (`N=108`) **once** for single-pass `110`-draw Eq. (15) CRPS, `< tau(s, k)` volumetric drought Brier/reliability, and `28d`/`56d` calendar MBB + Holm–Bonferroni inference.

## Independent Verification of User Feedback
User feedback is valuable advisory guidance, but **never unverified ground truth**. All agents must independently cross-check, mathematically calculate, empirically test, and scientifically validate all feedback against actual code, raw data, and published literature before adopting claims or numbers.

## Maintenance of Operational Registry & Master Plans
Keep the following documents synchronized upon remarkable changes (excluding trivial edits, typos, or temporary scratch scripts):
- [`reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`](reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md)
- [`mindanao_adaptation_master_plan.md`](mindanao_adaptation_master_plan.md)
- [`mindanao_remediation_and_optimal_model_masterplan.md`](mindanao_remediation_and_optimal_model_masterplan.md)

## Three-Tier Scientific Certification Standard
- `[PASS]`: Software, mathematical, and numerical assertions pass (zero failures, zero errors).
- `[VERIFIED]`: Direct parent EX29 parity established.
- `[ACCEPTED]`: Deliberate, documented regional adaptation for Mindanao.

## Dual Artifact & Figure Synchronization Standard (Codebase & Cloud Lake)
Whenever pipeline figures, diagnostic composites, or processed data artifacts are produced or refined (e.g., via notebooks, Colab runs, or pipeline scripts), both the local codebase (in designated directories like `figures/` and `processed/`) and Google Cloud Storage (`gs://rise-unet-rzsm/`) / Kaggle (`kirkishere/rise-unet-rzsm`) must contain the exact, latest up-to-date versions. If an artifact or figure is beneficial for reproducibility, documentation, or model training, proactively add, update, and maintain full parity across both the repository and cloud storage.

## Autonomous Execution & Destructive Action Safety Boundaries
1. **Continuous Autonomous Execution**:
   - The agent is empowered and expected to perform iterative, multi-step engineering tasks autonomously (e.g., executing test suites, running data preprocessing scripts, generating dashboards, assembling tensors, fixing lints, verifying checksums) without pausing for manual step-by-step confirmation on routine actions.
2. **Strictly Prohibited Actions (Require Explicit User Authorization)**:
   - **Destructive Git Commands**: Never execute `git reset`, `git clean`, `git rm`, `git checkout -- .`, `git restore .`, `git branch -D`, `git push --force`, or unprompted `git commit`.
   - **Local File & Dataset Deletions**: Never delete or drop raw data (`.grib`, `.nc`), processed cubes, trained checkpoints, or repository source files.
   - **Cloud Lake Deletions**: Never execute `gsutil rm` or delete objects/buckets in `gs://rise-unet-rzsm/` or Kaggle datasets.
   - **Authorization Protocol**: Any action that removes data or rewrites git history must be preceded by an explicit explanation, command preview, and pause for user approval.

## Markdown Documentation & Audit File Standards
1. **Markdownlint Header Directive**: Every markdown (`.md`) and audit dossier file across the repository (e.g., in `reproduction_audit/`, `freeze/`, and root) MUST begin with `<!-- markdownlint-disable -->` on line 1 to prevent linter conflicts with scientific tables, LaTeX mathematical equations, and raw execution logs.
2. **Thorough & Continuous Documentation**: Every action, architectural decision, numerical check, and pipeline milestone must be documented thoroughly, with all audit files, operational matrices, and master plans consistently maintained and updated upon any remarkable change.
