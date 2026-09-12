<!-- markdownlint-disable -->
# Antigravity Sub-Project Guidelines: dl_dm_rzsm_subseasonal_forecast

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
- [`reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`](reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md)
- `mindanao_adaptation_master_plan.md`

## Three-Tier Scientific Certification Standard
- `[PASS]`: Software, mathematical, and numerical assertions pass (zero failures, zero errors).
- `[VERIFIED]`: Direct parent EX29 parity established.
- `[ACCEPTED]`: Deliberate, documented regional adaptation for Mindanao.

## Dual Artifact & Figure Synchronization Standard (Codebase & Cloud Lake)
Whenever pipeline figures, diagnostic composites, or processed data artifacts are produced or refined (e.g., via notebooks, Colab runs, or pipeline scripts), both the local codebase (in designated directories like `figures/` and `processed/`) and Google Cloud Storage (`gs://rise-unet-rzsm/`) must contain the exact, latest up-to-date versions. If an artifact or figure is beneficial for reproducibility, documentation, or model training, proactively add, update, and maintain full parity across both the repository and GCS.

## Markdown Documentation & Audit File Standards
1. **Markdownlint Header Directive**: Every markdown (`.md`) and audit dossier file across the repository (e.g., in `reproduction_audit/`, `freeze/`, and root) MUST begin with `<!-- markdownlint-disable -->` on line 1 to prevent linter conflicts with scientific tables, LaTeX mathematical equations, and raw execution logs.
2. **Thorough & Continuous Documentation**: Every action, architectural decision, numerical check, and pipeline milestone must be documented thoroughly, with all audit files, operational matrices, and master plans consistently maintained and updated upon any remarkable change.

