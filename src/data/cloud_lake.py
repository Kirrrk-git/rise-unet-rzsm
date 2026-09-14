"""
src/data/cloud_lake.py
----------------------
Cloud-first data lake resolver and local caching interface for Mindanao RISE-UNet.

Policy:
1. Local Repository First:
   If the requested artifact exists locally in the repository (e.g., certified pilot
   artifacts, spatial grids, 11-year RZSM target cube), it is returned immediately
   with zero network overhead and zero GCP credential dependencies.
2. Automatic GCS Retrieval:
   If an artifact is not found locally (e.g., full 1,154 production S2S cycles, raw
   multi-year atmospheric archives, model checkpoints), it is retrieved on-demand from
   Google Cloud Storage (default bucket: `rise-unet-rzsm`) and cached locally.
3. Transparent Offline Diagnostics:
   If remote retrieval fails or credentials are unavailable, descriptive guidance
   is provided with exact GCS paths and remediation steps.
"""

from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_GCS_BUCKET = os.environ.get("RISE_UNET_GCS_BUCKET", "rise-unet-rzsm")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class CloudLakeDataNotFoundError(FileNotFoundError):
    """Raised when an artifact cannot be found locally or retrieved from GCS."""
    pass


def get_gcs_client():
    """
    Attempts to initialize a google.cloud.storage Client.
    Returns None if google-cloud-storage is not installed or unauthenticated.
    """
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception as e:
        logger.debug(f"GCS client initialization unavailable: {e}")
        return None


def download_from_gcs(
    gcs_uri: str,
    target_local_path: Union[str, Path],
    client=None,
) -> Path:
    """
    Downloads a single object from GCS to a local path.
    Supports gs://bucket/path/to/blob or (bucket, blob_name).
    """
    target = Path(target_local_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    cli = client or get_gcs_client()
    if cli is not None:
        try:
            bucket = cli.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if not blob.exists():
                raise CloudLakeDataNotFoundError(
                    f"GCS object not found in cloud lake: {gcs_uri}"
                )
            logger.info(f"Downloading {gcs_uri} -> {target}...")
            blob.download_to_filename(str(target))
            return target
        except Exception as e:
            if isinstance(e, CloudLakeDataNotFoundError):
                raise
            logger.warning(f"Python GCS client download failed ({e}), attempting subprocess fallback...")

    # Subprocess fallback via gsutil / gcloud storage
    import subprocess
    cmd = ["gcloud", "storage", "cp", gcs_uri, str(target)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return target
    except Exception:
        # Try gsutil
        try:
            cmd2 = ["gsutil", "cp", gcs_uri, str(target)]
            res2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
            return target
        except Exception as err:
            raise CloudLakeDataNotFoundError(
                f"Failed to retrieve {gcs_uri} from Google Cloud Storage.\n"
                f"Local target: {target}\n"
                f"Remediation:\n"
                f"  1. Run: gcloud auth application-default login\n"
                f"  2. Or manually download: gsutil cp {gcs_uri} {target}\n"
                f"Error detail: {err}"
            ) from err


def resolve_s2s_cycle(
    issue_date: str,
    local_cache_dir: Optional[Union[str, Path]] = None,
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
) -> Path:
    """
    Resolves the ECMWF S2S reforecast NetCDF for a given issue date (YYYY-MM-DD).

    Order of resolution:
    1. Certified pilot file (processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc) if date matches pilot
    2. Local cache in `data_cache/s2s/<issue_date>/`
    3. On-demand download from `gs://{gcs_bucket}/raw/ecmwf_s2s/production/<issue_date>/`
    """
    cache_dir = Path(local_cache_dir) if local_cache_dir else REPO_ROOT / "data_cache" / "s2s"

    # Check pilot reference
    pilot_path = REPO_ROOT / "processed" / "s2s" / "pilot" / "s2s_pilot_reforecast_w1_w2.nc"
    if issue_date in ("2015-01-15", "pilot") and pilot_path.is_file():
        return pilot_path

    # Check local cache
    candidate_nc = cache_dir / issue_date / f"s2s_mindanao_{issue_date.replace('-', '')}.nc"
    if candidate_nc.is_file():
        return candidate_nc

    # Also check parent directory if stored directly by issue_date
    direct_nc = cache_dir / f"s2s_mindanao_{issue_date.replace('-', '')}.nc"
    if direct_nc.is_file():
        return direct_nc

    # GCS download
    gcs_uri = f"gs://{gcs_bucket}/raw/ecmwf_s2s/production/{issue_date}/s2s_mindanao_{issue_date.replace('-', '')}.nc"
    return download_from_gcs(gcs_uri, candidate_nc)


def resolve_atmospheric_month(
    year: int,
    month: int,
    local_cache_dir: Optional[Union[str, Path]] = None,
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
) -> Path:
    """
    Resolves the derived surface atmospheric NetCDF for a given year and month.

    Order of resolution:
    1. Certified pilot files in `processed/atmospheric/pilot/era5_atmospheric_pilot_<YYYY>_<MM>.nc`
    2. Local cache in `data_cache/atmospheric/`
    3. On-demand download from `gs://{gcs_bucket}/processed/atmospheric/`
    """
    month_str = f"{year}_{month:02d}"
    pilot_file = REPO_ROOT / "processed" / "atmospheric" / "pilot" / f"era5_atmospheric_pilot_{month_str}.nc"
    if pilot_file.is_file():
        return pilot_file

    cache_dir = Path(local_cache_dir) if local_cache_dir else REPO_ROOT / "data_cache" / "atmospheric"
    candidate_file = cache_dir / f"era5_atmospheric_{month_str}.nc"
    if candidate_file.is_file():
        return candidate_file

    # Attempt download of derived monthly NetCDF from cloud lake
    gcs_uri = f"gs://{gcs_bucket}/processed/atmospheric/era5_atmospheric_{month_str}.nc"
    return download_from_gcs(gcs_uri, candidate_file)


def resolve_rzsm_target_cube(
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
) -> Path:
    """
    Resolves the 11-year ERA5-Land RZSM production target cube (2015-2025).

    Order of resolution:
    1. Local codebase artifact: `processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc`
    2. Certified pilot fallback: `processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc`
    3. On-demand retrieval from `gs://{gcs_bucket}/processed/rzsm/`
    """
    prod_path = REPO_ROOT / "processed" / "rzsm" / "production" / "era5_land_rzsm_production_2015_2025.nc"
    if prod_path.is_file():
        return prod_path

    # Check GCS
    gcs_uri = f"gs://{gcs_bucket}/processed/rzsm/era5_land_rzsm_production_2015_2025.nc"
    return download_from_gcs(gcs_uri, prod_path)


def resolve_spatial_mask(
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
) -> Path:
    """
    Resolves the Candidate A evaluation mask NetCDF.
    """
    mask_path = REPO_ROOT / "processed" / "grid" / "mindanao_eval_mask_025.nc"
    if mask_path.is_file():
        return mask_path

    gcs_uri = f"gs://{gcs_bucket}/processed/grid/mindanao_eval_mask_025.nc"
    return download_from_gcs(gcs_uri, mask_path)
