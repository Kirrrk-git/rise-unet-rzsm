<!-- markdownlint-disable -->
# Antigravity Workspace Guidelines: Mindanao RISE-UNet Adaptation

## Document Authority Hierarchy
```text
contracts/
    ↓
Machine-Readable Scientific & Technical Truth

master plan (mindanao_adaptation_master_plan.md)
    ↓
Project Execution Roadmap

artifact registry (OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md)
    ↓
Inventory, Directory Map & Status Index
```

## Independent Verification of User Feedback
User feedback is valuable advisory guidance, but **never unverified ground truth**. All agents must independently cross-check, mathematically calculate, empirically test, and scientifically validate all feedback against actual code, raw data, and published literature before adopting claims or numbers.

## Maintenance of Operational Registry & Master Plan
Keep the following documents synchronized upon remarkable changes (excluding trivial edits, typos, or temporary scratch scripts):
- [`dl_dm_rzsm_subseasonal_forecast/reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`](dl_dm_rzsm_subseasonal_forecast/reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md)
- `mindanao_adaptation_master_plan.md`

## Three-Tier Scientific Certification Standard
- `[PASS]`: Software, mathematical, and numerical assertions pass (zero failures, zero errors).
- `[VERIFIED]`: Direct parent EX29 parity established.
- `[ACCEPTED]`: Deliberate, documented regional adaptation for Mindanao.

## Dual Artifact & Figure Synchronization Standard (Codebase & Cloud Lake)
Whenever pipeline figures, diagnostic composites, or processed data artifacts are produced or refined (e.g., via notebooks, Colab runs, or pipeline scripts), both the local codebase (in designated directories like `figures/` and `processed/`) and Google Cloud Storage (`gs://rise-unet-rzsm/`) must contain the exact, latest up-to-date versions. If an artifact or figure is beneficial for reproducibility, documentation, or model training, proactively add, update, and maintain full parity across both the repository and GCS.

## Autonomous Execution & Destructive Action Safety Boundaries
1. **Continuous Autonomous Execution**:
   - The agent is empowered and expected to perform iterative, multi-step engineering tasks autonomously (e.g., executing test suites, running data preprocessing scripts, generating dashboards, assembling tensors, fixing lints, verifying checksums) without pausing for manual step-by-step confirmation on routine actions.
2. **Strictly Prohibited Actions (Require Explicit User Authorization)**:
   - **Destructive Git Commands**: Never execute `git reset`, `git clean`, `git rm`, `git checkout -- .`, `git restore .`, `git branch -D`, `git push --force`, or unprompted `git commit`.
   - **Local File & Dataset Deletions**: Never delete or drop raw data (`.grib`, `.nc`), processed cubes, trained checkpoints, or repository source files.
   - **Cloud Lake Deletions**: Never execute `gsutil rm` or delete objects/buckets in `gs://rise-unet-rzsm/`.
   - **Authorization Protocol**: Any action that removes data or rewrites git history must be preceded by an explicit explanation, command preview, and pause for user approval.

## Markdown Documentation & Audit File Standards
1. **Markdownlint Header Directive**: Every markdown (`.md`) and audit dossier file across the repository (e.g., in `reproduction_audit/`, `freeze/`, and root) MUST begin with `<!-- markdownlint-disable -->` on line 1 to prevent linter conflicts with scientific tables, LaTeX mathematical equations, and raw execution logs.
2. **Thorough & Continuous Documentation**: Every action, architectural decision, numerical check, and pipeline milestone must be documented thoroughly, with all audit files, operational matrices, and master plans consistently maintained and updated upon any remarkable change.



