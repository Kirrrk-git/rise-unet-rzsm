<!-- markdownlint-disable -->
# Workspace Scientific Governance & Registry Maintenance Rule

## 1. Document Authority Hierarchy

To prevent conflicting sources of truth, all agents must strictly observe the following document hierarchy:

```text
contracts/
    ↓
Machine-Readable Scientific & Technical Truth (Authoritative specifications, schemas, bounds)

master plans (mindanao_adaptation_master_plan.md
            + mindanao_remediation_and_optimal_model_masterplan.md)
    ↓
Project Execution Roadmap & Before/During/After Optimality Playbook

artifact registry (reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md)
    ↓
Inventory, Directory Map & Status Index (Catalog of non-author files, audit index, verification records)
```

The artifact registry is an index and navigation map; it must never contradict or override `contracts/` or the Master Plans.

---

## 2. Independent Empirical Verification of User Feedback

> [!CRITICAL]
> **User Feedback is Advisory, NOT Ground Truth**:
> All suggestions, critique, arithmetic, and feedback provided by the user must be treated as advisory review and expert guidance—**never as unverified ground truth**.
>
> Before incorporating any user claim, critique, or suggested number into the codebase, contracts, or audit records:
> 1. Independently calculate and verify the arithmetic (e.g., date ranges, cell counts, tensor shapes).
> 2. Cross-reference the claim against actual source code, raw data files, and published literature (Lesinger & Tian, 2025).
> 3. Execute unit tests or empirical scripts to confirm software and mathematical behavior.
> 4. Explicitly state the verification findings before committing changes.

---

## 3. Scope of Mandatory Registry Synchronization

All agents working in this workspace must synchronize the following documents upon **remarkable changes**:
- `reproduction_audit/OPERATIONAL_EXECUTION_MATRIX_AND_ARTIFACT_REGISTRY.md`
- `mindanao_adaptation_master_plan.md`
- `mindanao_remediation_and_optimal_model_masterplan.md`

### Trigger Conditions:
An update is required ONLY after significant/remarkable changes:
1. **Structural Additions/Removals**: Creating, modifying, or deleting production modules (`src/`), unit tests (`tests/`), automation scripts (`scripts/`), interactive pipelines (`notebooks/`), or formal audit reports (`reproduction_audit/`).
2. **Scientific Contract Changes**: Altering or adding normalization scopes, target lead indices ($L=[6,13,20,27]$), climatology definitions, depth-weighting formulas, or spatial remapping algorithms.
3. **Execution Milestones**: Running multi-year tensor compilations, NetCDF data cube generation, data lake mirrors, or model training/evaluation runs.
4. **Census & Verification Audits**: Producing numerical census outputs, checksum logs, or CF-1.8 metadata validations.

### Non-Triggers (Do NOT update registry for trivial changes):
- Minor formatting, comment edits, or typo corrections in existing documentation.
- Scratch scripts or temporary diagnostic files in `scratch/` or `.tempmediaStorage/`.
- Transient debugging commands and terminal log inspections.

---

## 4. Three-Tier Scientific Certification Standard

Every finding, audit claim, and milestone must strictly use the three-tier classification:
- `[PASS]`: Software, mathematical, and numerical assertions pass with zero failures and zero errors.
- `[VERIFIED]`: Direct parent EX29 parity established against published code (Lesinger & Tian, 2025).
- `[ACCEPTED]`: Deliberate, documented regional adaptation specifically for the Mindanao domain.
