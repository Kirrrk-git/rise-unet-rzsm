"""
src/data/cloud_lake.py
----------------------
Local-first and Kaggle-aware data lake resolver for Mindanao RISE-UNet (v3.0).

Resolution Policy:
1. Local Repository & Cache First:
   Checks committed repository artifacts (`processed/`, `pilot_raw/`, `manifests/`) and
   `data_cache/` with zero network overhead.
2. Kaggle Backup Store (Primary Off-Cloud Archive):
   When running inside a Kaggle Notebook (under `/kaggle/input/...`) or with a locally
   staged Kaggle migration index (`RISE_UNET_KAGGLE_INDEX_DIR` / `data_cache/kaggle_index`),
   uses `object-locations.json` / `kaggle_restore.py` logic (`aaronjalapon02/mindanao-gcs-migration-index`)
   to stream/verify files from attached backup datasets with MD5 & CRC32C checks.
3. Optional Legacy GCS Retrieval (`gs://rise-unet-rzsm/`):
   Attempts GCS only if explicitly permitted via `download_from_gcs` / env fallback; if
   billing is disabled (`HTTP 403`), emits actionable guidance pointing to the Kaggle backup.
"""

from __future__ import annotations
import base64
import hashlib
import json
import logging
import os
import shutil
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_GCS_BUCKET = os.environ.get("RISE_UNET_GCS_BUCKET", "rise-unet-rzsm")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
KAGGLE_INDEX_SLUG = "mindanao-gcs-migration-index"
KAGGLE_INDEX_OWNER = "aaronjalapon02"


class CloudLakeDataNotFoundError(FileNotFoundError):
    """Raised when an artifact cannot be found locally, in Kaggle inputs, or in GCS."""
    pass


def find_kaggle_index_dir(index_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Locates the Kaggle migration index directory containing `object-locations.json`.

    Checks (in priority order):
    1. Explicit `index_dir` argument
    2. `RISE_UNET_KAGGLE_INDEX_DIR` environment variable
    3. Standard Kaggle Notebook mount paths under `/kaggle/input/`
    4. Local staged folder `data_cache/kaggle_index`
    """
    candidates = []
    if index_dir is not None:
        candidates.append(Path(index_dir))
    env_dir = os.environ.get("RISE_UNET_KAGGLE_INDEX_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend([
            kaggle_input / KAGGLE_INDEX_SLUG,
            kaggle_input / "datasets" / KAGGLE_INDEX_OWNER / KAGGLE_INDEX_SLUG,
            kaggle_input / KAGGLE_INDEX_OWNER / KAGGLE_INDEX_SLUG,
        ])
    candidates.append(REPO_ROOT / "data_cache" / "kaggle_index")

    for cand in candidates:
        if (cand / "object-locations.json").is_file():
            return cand
    return None


def _resolve_kaggle_mounted_source(
    destination: Dict[str, Any],
    kaggle_input_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolves the mounted file or `.zip.bin` archive path for a Kaggle index destination entry."""
    input_root = Path(
        kaggle_input_root
        or os.environ.get("RISE_UNET_KAGGLE_INPUT_ROOT")
        or "/kaggle/input"
    )
    if not input_root.is_dir():
        raise CloudLakeDataNotFoundError(
            f"Kaggle input directory not mounted: {input_root}. "
            "Attach `aaronjalapon02/mindanao-gcs-migration-index` v2 and required data datasets in Kaggle."
        )

    owner, slug = destination["dataset_ref"].split("/", 1)
    mounts = [
        p
        for p in (input_root / slug, input_root / "datasets" / owner / slug, input_root / owner / slug)
        if p.is_dir()
    ]
    if not mounts:
        raise CloudLakeDataNotFoundError(
            f"Required private Kaggle dataset '{destination['dataset_ref']}' "
            f"(version {destination.get('dataset_version')}) is not mounted under {input_root}."
        )

    if destination["kind"] == "loose":
        folders = []
        for mount in mounts:
            folders.extend([
                mount / destination["root_folder"],
                mount / "rise-unet-rzsm-part-1" / destination["root_folder"],
            ])
        matches = list({
            folder / destination["relative_file"]
            for folder in folders
            if (folder / destination["relative_file"]).is_file()
        })
    else:
        matches = [
            mount / destination["archive_file"]
            for mount in mounts
            if (mount / destination["archive_file"]).is_file()
        ]

    if len(matches) != 1:
        raise CloudLakeDataNotFoundError(
            f"Expected 1 mounted source file for destination {destination['dataset_ref']} "
            f"(version {destination.get('dataset_version')}), found {len(matches)}: {matches}."
        )
    return matches[0]


@contextmanager
def open_kaggle_backup_object(
    blob_name: str,
    bucket: str = DEFAULT_GCS_BUCKET,
    index_dir: Optional[Union[str, Path]] = None,
    kaggle_input_root: Optional[Union[str, Path]] = None,
) -> Iterator[Tuple[Any, Dict[str, Any]]]:
    """Opens a binary read stream for a backed-up object using the Kaggle migration index."""
    idx = find_kaggle_index_dir(index_dir)
    if idx is None:
        raise CloudLakeDataNotFoundError(
            "Kaggle migration index (`object-locations.json`) not found locally or under `/kaggle/input`."
        )
    rows = json.loads((idx / "object-locations.json").read_text(encoding="utf-8"))
    lookup = {(r["source"]["bucket"], r["source"]["name"]): r for r in rows}
    if (bucket, blob_name) not in lookup:
        raise CloudLakeDataNotFoundError(
            f"Object '{bucket}/{blob_name}' is not recorded in {idx / 'object-locations.json'}."
        )

    row = lookup[(bucket, blob_name)]
    dest = row["destination"]
    src_path = _resolve_kaggle_mounted_source(dest, kaggle_input_root=kaggle_input_root)

    if dest["kind"] == "loose":
        with src_path.open("rb") as fh:
            yield fh, row["source"]
    else:
        with zipfile.ZipFile(src_path) as archive:
            with archive.open(dest["archive_member"]) as fh:
                yield fh, row["source"]


def restore_from_kaggle_backup(
    blob_name: str,
    target_local_path: Union[str, Path],
    bucket: str = DEFAULT_GCS_BUCKET,
    index_dir: Optional[Union[str, Path]] = None,
    kaggle_input_root: Optional[Union[str, Path]] = None,
    verify_checksums: bool = True,
) -> Path:
    """Restores a backed-up file from Kaggle datasets to `target_local_path` with integrity verification."""
    target = Path(target_local_path)
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f".{target.name}.tmp_{os.getpid()}")

    crc_mod = None
    if verify_checksums:
        try:
            import google_crc32c  # type: ignore
            crc_mod = google_crc32c
        except ImportError:
            logger.debug("google_crc32c not available; verifying byte size and MD5.")

    try:
        with open_kaggle_backup_object(
            blob_name=blob_name,
            bucket=bucket,
            index_dir=index_dir,
            kaggle_input_root=kaggle_input_root,
        ) as (source, meta):
            md5 = hashlib.md5()
            crc = crc_mod.Checksum() if (crc_mod is not None and meta.get("crc32c")) else None
            total_bytes = 0
            with tmp_target.open("wb") as dst:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    total_bytes += len(chunk)
                    if verify_checksums:
                        md5.update(chunk)
                        if crc is not None:
                            crc.update(chunk)

            if total_bytes != int(meta["size"]):
                raise IOError(
                    f"Size mismatch restoring {bucket}/{blob_name}: expected {meta['size']} B, got {total_bytes} B"
                )
            if verify_checksums and meta.get("md5"):
                got_md5 = base64.b64encode(md5.digest()).decode("ascii")
                if got_md5 != meta["md5"]:
                    raise IOError(
                        f"MD5 mismatch restoring {bucket}/{blob_name}: expected {meta['md5']}, got {got_md5}"
                    )
            if crc is not None:
                got_crc = base64.b64encode(crc.digest()).decode("ascii")
                if got_crc != meta["crc32c"]:
                    raise IOError(
                        f"CRC32C mismatch restoring {bucket}/{blob_name}: expected {meta['crc32c']}, got {got_crc}"
                    )

        shutil.move(str(tmp_target), str(target))
        return target
    except Exception:
        tmp_target.unlink(missing_ok=True)
        raise


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
                f"Failed to retrieve {gcs_uri} from Google Cloud Storage (billing disabled / HTTP 403 or offline).\n"
                f"Local target: {target}\n"
                f"Remediation:\n"
                f"  1. Recommended: Mount/restore from Kaggle backup (`aaronjalapon02/mindanao-gcs-migration-index` v2 + data datasets) or set RISE_UNET_KAGGLE_INDEX_DIR.\n"
                f"  2. Or if GCS billing is re-enabled: gsutil cp {gcs_uri} {target}\n"
                f"Error detail: {err}"
            ) from err


CONSOLIDATED_BUNDLE_NAMES = (
    "raw_era5_land.zip",
    "raw_era5.zip",
    "raw_ecmwf_s2s.zip",
    "raw_boundaries_and_other.zip",
    "processed_all.zip",
    "checkpoints_legacy_a0_best12.zip",
    "meta_docs_and_audits.zip",
)

_CONSOLIDATED_META_PREFIXES = (
    "contracts/",
    "manifests/",
    "documentation/",
    "figures/",
    "logs/",
    "reproduction_audit/",
    "support/",
)


def _consolidated_bundle_for_blob(blob_name: str) -> "str | None":
    """Return the consolidated topic ZIP filename owning *blob_name*, if any."""
    rel = blob_name.lstrip("/")
    if rel.startswith("raw/era5_land/"):
        return "raw_era5_land.zip"
    if rel.startswith("raw/era5/"):
        return "raw_era5.zip"
    if rel.startswith("raw/ecmwf_s2s/"):
        return "raw_ecmwf_s2s.zip"
    if rel.startswith("raw/"):
        return "raw_boundaries_and_other.zip"
    if rel.startswith("processed/"):
        return "processed_all.zip"
    if rel.startswith("checkpoints/"):
        return "checkpoints_legacy_a0_best12.zip"
    if rel.startswith(_CONSOLIDATED_META_PREFIXES):
        return "meta_docs_and_audits.zip"
    return None


def _is_consolidated_mount(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    if (directory / "manifest_active_files.json").is_file():
        return True
    if any((directory / b).is_file() or (directory / Path(b).stem).is_dir() for b in CONSOLIDATED_BUNDLE_NAMES):
        return True
    return any((directory / top).is_dir() for top in ("raw", "processed", "checkpoints"))


def _atomic_copy_stream(src_fh, target_local_path: Path, expected_size: "int | None" = None) -> Path:
    import shutil

    target_local_path = Path(target_local_path)
    target_local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_local_path.with_name(f"{target_local_path.name}.part")
    try:
        with tmp_path.open("wb") as dst_fh:
            shutil.copyfileobj(src_fh, dst_fh, length=4 * 1024 * 1024)
        actual_size = tmp_path.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise IOError(f"Copied size mismatch for {target_local_path.name}: {actual_size} != {expected_size}")
        tmp_path.replace(target_local_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target_local_path


def find_consolidated_kaggle_dir(
    consolidated_dir: "Path | None" = None,
    kaggle_input_root: "Path | None" = None,
) -> "Path | None":
    """Locate mounted consolidated Kaggle dataset or kernel output directory."""
    candidates: list[Path] = []
    if consolidated_dir is not None:
        candidates.append(Path(consolidated_dir))
    env_dir = os.environ.get("RISE_UNET_KAGGLE_CONSOLIDATED_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    root = Path(kaggle_input_root) if kaggle_input_root is not None else Path("/kaggle/input")
    for slug in ("rise-unet-rzsm", "rise-unet-rzsm-verify-and-consolidate"):
        candidates.extend(
            [
                root / slug,
                root / "datasets" / "kirkishere" / slug,
            ]
        )
    for cand in candidates:
        if _is_consolidated_mount(cand):
            return cand
    return None


def restore_from_consolidated_kaggle(
    blob_name: str,
    target_local_path: Path,
    consolidated_dir: "Path | None" = None,
    kaggle_input_root: "Path | None" = None,
) -> Path:
    """Restore *blob_name* from consolidated `kirkishere/rise-unet-rzsm` bundles or unpacked tree."""
    import zipfile

    rel = blob_name.lstrip("/")
    mount_dir = find_consolidated_kaggle_dir(
        consolidated_dir=consolidated_dir,
        kaggle_input_root=kaggle_input_root,
    )
    if mount_dir is None:
        raise CloudLakeDataNotFoundError(
            f"Consolidated Kaggle dataset not found for '{rel}' under {kaggle_input_root or '/kaggle/input'}."
        )

    bundle_name = _consolidated_bundle_for_blob(rel)

    # Case 1: Unpacked directly under mount or mount/<bundle_stem>/
    unpacked_candidates = [mount_dir / rel]
    if bundle_name is not None:
        unpacked_candidates.append(mount_dir / Path(bundle_name).stem / rel)
    for src in unpacked_candidates:
        if src.is_file():
            with src.open("rb") as src_fh:
                return _atomic_copy_stream(src_fh, target_local_path, expected_size=src.stat().st_size)

    # Case 2: Consolidated topic ZIP_STORED bundle under mount_dir
    bundle_candidates = [bundle_name] if bundle_name is not None else []
    bundle_candidates.extend(b for b in CONSOLIDATED_BUNDLE_NAMES if b not in bundle_candidates)
    for bname in bundle_candidates:
        zpath = mount_dir / bname
        if not zpath.is_file():
            continue
        with zipfile.ZipFile(zpath, "r") as zf:
            try:
                zinfo = zf.getinfo(rel)
            except KeyError:
                continue
            with zf.open(zinfo, "r") as src_fh:
                return _atomic_copy_stream(src_fh, target_local_path, expected_size=zinfo.file_size)

    raise CloudLakeDataNotFoundError(
        f"Blob '{rel}' not found inside consolidated Kaggle mount {mount_dir}."
    )


def _resolve_remote_artifact(
    blob_name: str,
    target_local_path: Path,
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
) -> Path:
    """
    Resolves an uncached remote artifact cleanly: prefers Kaggle Backup Store if mounted,
    then falls back to GCS only when explicitly permitted or when no Kaggle index exists.
    """
    if find_consolidated_kaggle_dir() is not None:
        try:
            return restore_from_consolidated_kaggle(
                blob_name=blob_name,
                target_local_path=target_local_path,
            )
        except CloudLakeDataNotFoundError:
            if find_kaggle_index_dir() is None and os.environ.get("RISE_UNET_ALLOW_GCS_FALLBACK", "1") != "1":
                raise
    if find_kaggle_index_dir() is not None:
        return restore_from_kaggle_backup(
            blob_name=blob_name,
            target_local_path=target_local_path,
            bucket=gcs_bucket,
        )
    if os.environ.get("RISE_UNET_ALLOW_GCS_FALLBACK", "1") == "1":
        return download_from_gcs(f"gs://{gcs_bucket}/{blob_name}", target_local_path)
    raise CloudLakeDataNotFoundError(
        f"Artifact '{gcs_bucket}/{blob_name}' not cached locally ({target_local_path}) and Kaggle index not mounted."
    )


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

    # Kaggle backup or GCS fallback
    blob_name = f"raw/ecmwf_s2s/production/{issue_date}/s2s_mindanao_{issue_date.replace('-', '')}.nc"
    return _resolve_remote_artifact(blob_name, candidate_nc, gcs_bucket=gcs_bucket)


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

    # Attempt Kaggle backup or GCS fallback for derived monthly NetCDF
    blob_name = f"processed/atmospheric/era5_atmospheric_{month_str}.nc"
    return _resolve_remote_artifact(blob_name, candidate_file, gcs_bucket=gcs_bucket)


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

    # Kaggle or GCS fallback
    return _resolve_remote_artifact(
        "processed/rzsm/era5_land_rzsm_production_2015_2025.nc",
        prod_path,
        gcs_bucket=gcs_bucket,
    )


def resolve_spatial_mask(
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
) -> Path:
    """
    Resolves the Candidate A evaluation mask NetCDF.
    """
    mask_path = REPO_ROOT / "processed" / "grid" / "mindanao_eval_mask_025.nc"
    if mask_path.is_file():
        return mask_path

    return _resolve_remote_artifact(
        "processed/grid/mindanao_eval_mask_025.nc",
        mask_path,
        gcs_bucket=gcs_bucket,
    )
