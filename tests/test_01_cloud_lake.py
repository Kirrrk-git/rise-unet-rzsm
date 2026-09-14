"""
tests/test_cloud_lake.py
-------------------------
Verifies cloud lake resolution policies:
1. Instant zero-network resolution for committed codebase artifacts.
2. Formatted GCS URI construction matching cloud lake directory tree.
3. Informative error diagnostics when artifacts are missing and GCS is unreachable.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.cloud_lake import (
    resolve_spatial_mask,
    resolve_rzsm_target_cube,
    resolve_atmospheric_month,
    resolve_s2s_cycle,
    download_from_gcs,
    CloudLakeDataNotFoundError,
)


class TestCloudLakeResolver(unittest.TestCase):
    """Tests for cloud lake data resolution and local codebase prioritization."""

    def test_local_spatial_mask_resolution(self):
        """Confirms Candidate A evaluation mask resolves immediately from codebase."""
        path = resolve_spatial_mask()
        self.assertTrue(path.is_file(), f"Mask file missing: {path}")
        self.assertEqual(path.name, "mindanao_eval_mask_025.nc")

    def test_local_rzsm_target_cube_resolution(self):
        """Confirms 11-year RZSM production cube resolves immediately from codebase."""
        path = resolve_rzsm_target_cube()
        self.assertTrue(path.is_file(), f"RZSM target cube missing: {path}")
        self.assertEqual(path.name, "era5_land_rzsm_production_2015_2025.nc")

    def test_local_atmospheric_pilot_resolution(self):
        """Confirms Jan 2015 pilot atmospheric NetCDF resolves immediately from codebase."""
        path = resolve_atmospheric_month(2015, 1)
        self.assertTrue(path.is_file(), f"Atmospheric pilot file missing: {path}")
        self.assertEqual(path.name, "era5_atmospheric_pilot_2015_01.nc")

    def test_local_s2s_pilot_resolution(self):
        """Confirms Jan 15 2015 pilot S2S reforecast resolves immediately from codebase."""
        path = resolve_s2s_cycle("2015-01-15")
        self.assertTrue(path.is_file(), f"S2S pilot file missing: {path}")
        self.assertEqual(path.name, "s2s_pilot_reforecast_w1_w2.nc")

    def test_missing_gcs_artifact_raises_informative_error(self):
        """Confirms missing remote blob raises CloudLakeDataNotFoundError with actionable text."""
        mock_cli = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob
        mock_cli.bucket.return_value = mock_bucket

        with self.assertRaises(CloudLakeDataNotFoundError) as ctx:
            download_from_gcs(
                "gs://rise-unet-rzsm/raw/nonexistent/missing_file.nc",
                "data_cache/test_target.nc",
                client=mock_cli,
            )
        self.assertIn("GCS object not found in cloud lake", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
