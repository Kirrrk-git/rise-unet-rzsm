<!-- markdownlint-disable -->
# Autonomous Execution & Destructive Action Safety Policy

## 1. Autonomous Continuous Execution Directive
The agent is explicitly authorized and expected to execute non-destructive tasks autonomously and continuously from start to finish without pausing for approval between routine steps. This includes:
- Writing, updating, and modularizing application and pipeline code.
- Running automated test suites (`python -m unittest`, `pytest`).
- Running non-destructive data processing, tensor assembly, and diagnostic scripts.
- Querying read-only git states (`git status`, `git diff`, `git log`).
- Querying and reading cloud storage (`gsutil ls`, `kaggle datasets status`).
- Generating publication dashboards, rendering diagnostic figures, and updating documentation/audit matrices.
- Correcting code bugs, fixing lint errors, and updating notebook JSON cells safely via Python.

## 2. Strictly Prohibited Destructive Actions (User Authorization Required)
The agent must **NEVER** perform any of the following destructive or irreversible actions without first pausing, explaining the rationale, and receiving explicit written authorization from the user:

### A. Destructive Git Commands
- `git reset` (especially `--hard` or `--mixed`)
- `git clean` (`-f`, `-d`, `-x`)
- `git rm`
- `git checkout -- .` or `git restore .` (wiping working tree changes)
- `git branch -D` (force branch deletion)
- `git push --force` or `--force-with-lease`
- `git commit` without explicit user instruction.

### B. Local File & Data Deletions
- Calling `rm`, `del`, `Remove-Item`, `shutil.rmtree`, or `os.remove` on repository code, processed datasets (`.nc`, `.zarr`, `.npy`), or raw observation lakes (`.grib`, `.nc`).
- Overwriting or dropping database tables, caches, or schema structures without backup.
- Modifying Jupyter notebooks (`.ipynb`) using ad-hoc text/regex replacements (must use strict Python JSON object manipulation).

### C. Cloud Storage & Infrastructure Deletions
- `gsutil rm` or `gcloud storage rm` (deleting files, prefixes, or buckets in `gs://rise-unet-rzsm/` or any cloud lake).
- Modifying IAM policies, service accounts, or deleting cloud compute instances/buckets/Kaggle datasets.

## 3. Mandatory Authorization Protocol
If an operation genuinely requires removing an obsolete artifact or executing a destructive change:
1. Explain **why** the removal or reset is required.
2. Provide the **exact list of files or command** that would be affected.
3. Confirm that no irreversible data loss or unrecoverable git history loss will occur.
4. Stop and request explicit user confirmation before executing.
