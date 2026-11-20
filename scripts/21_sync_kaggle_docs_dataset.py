#!/usr/bin/env python3
"""
scripts/21_sync_kaggle_docs_dataset.py
--------------------------------------
Packages the lightweight living codebase, contracts, manifests, audits, and figures
from this local repository (~16 MB) into a clean temporary directory and creates or
updates the private Kaggle Dataset `<username>/rise-unet-rzsm-docs` via Kaggle CLI.

Strict Safety Invariants:
  - Never includes secrets (.env, *.key, .kaggle/, service_account*.json) or data_cache/.
  - Never includes heavy checkpoints/, raw/, processed/*.nc production cubes > pilot/grid.
  - Creates/updates strictly as PRIVATE (never uses --public).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent

INCLUDE_DIRS = (
    "src",
    "scripts",
    "tests",
    "contracts",
    "manifests",
    "freeze",
    "reproduction_audit",
    "figures",
)

INCLUDE_ROOT_FILES = (
    "README.md",
    "AGENTS.md",
    "mindanao_adaptation_master_plan.md",
    "mindanao_remediation_and_optimal_model_masterplan.md",
    "pyproject.toml",
    "requirements.txt",
)


def _ignore_filter(_src: str, names: list[str]) -> set[str]:
    ignored = set()
    for n in names:
        if n in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"} or n.endswith((".pyc", ".key", ".pem", ".env")):
            ignored.add(n)
    return ignored


def get_kaggle_username() -> str:
    api = KaggleApi()
    api.authenticate()
    username = api.get_config_value("username")
    if not username:
        raise RuntimeError("Could not determine Kaggle username from authenticated API config.")
    return str(username)


def build_stage_tree(stage_dir: Path) -> dict[str, int]:
    pkg_dir = stage_dir / REPO_ROOT.name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    for rel_dir in INCLUDE_DIRS:
        src = REPO_ROOT / rel_dir
        if src.is_dir():
            shutil.copytree(src, pkg_dir / rel_dir, ignore=_ignore_filter, dirs_exist_ok=True)

    # Copy lightweight parent function modules needed by modelRzsmReluExtraConv imports
    p29_func_src = REPO_ROOT / "parent_study_ex29" / "function"
    if p29_func_src.is_dir():
        shutil.copytree(p29_func_src, pkg_dir / "parent_study_ex29" / "function", ignore=_ignore_filter, dirs_exist_ok=True)
    p29_init = REPO_ROOT / "parent_study_ex29" / "__init__.py"
    if p29_init.is_file():
        shutil.copy2(p29_init, pkg_dir / "parent_study_ex29" / "__init__.py")

    for fname in INCLUDE_ROOT_FILES:
        src_f = REPO_ROOT / fname
        if src_f.is_file():
            shutil.copy2(src_f, pkg_dir / fname)

    for root_doc in ("AGENTS.md", "mindanao_adaptation_master_plan.md", "mindanao_remediation_and_optimal_model_masterplan.md"):
        repo_doc = REPO_ROOT / root_doc
        ws_doc = WORKSPACE_ROOT / root_doc
        if not repo_doc.is_file() and ws_doc.is_file():
            shutil.copy2(ws_doc, stage_dir / root_doc)
        elif repo_doc.is_file():
            shutil.copy2(repo_doc, stage_dir / root_doc)

    files = [p for p in stage_dir.rglob("*") if p.is_file()]
    return {"file_count": len(files), "total_bytes": sum(p.stat().st_size for p in files)}


def sync_docs_dataset(slug: str = "rise-unet-rzsm-docs", message: str = "Sync living code, contracts, audits, and figures") -> None:
    username = get_kaggle_username()
    dataset_id = f"{username}/{slug}"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        stats = build_stage_tree(stage)
        print(f"Staged {stats['file_count']} lightweight files ({stats['total_bytes'] / 1e6:.2f} MB) for {dataset_id}")

        meta = {
            "title": "RISE-UNet RZSM Code and Docs",
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}],
            "isPrivate": True,
        }
        (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Check if dataset already exists under user account
        exists = False
        check = subprocess.run(["kaggle", "datasets", "files", dataset_id], capture_output=True, text=True)
        if check.returncode == 0:
            exists = True

        cmd = (
            ["kaggle", "datasets", "version", "-p", str(stage), "-m", message, "--dir-mode", "zip"]
            if exists
            else ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "zip"]
        )
        print("Running:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            raise RuntimeError(f"Kaggle dataset sync failed with exit code {res.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync lightweight code, contracts, and docs to Private Kaggle Dataset.")
    parser.add_argument("--slug", default="rise-unet-rzsm-docs", help="Target Kaggle dataset slug.")
    parser.add_argument("--message", default="Initial clean Thesis v3.0 sync from repository", help="Version commit message.")
    args = parser.parse_args()
    sync_docs_dataset(slug=args.slug, message=args.message)


if __name__ == "__main__":
    main()
