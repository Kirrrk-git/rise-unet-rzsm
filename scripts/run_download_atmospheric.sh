
#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_download_atmospheric.sh
# -----------------------------------------------------------------------------
# Thin convenience wrapper around the authoritative Python downloader:
# scripts/download_era5_atmospheric_mindanao.py
#
# Usage:
#   bash scripts/run_download_atmospheric.sh --pilot
#   bash scripts/run_download_atmospheric.sh --year 2015
#   bash scripts/run_download_atmospheric.sh --start-year 2015 --end-year 2025 --upload-gcs
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${REPO_ROOT}/scripts/download_era5_atmospheric_mindanao.py" "$@"
