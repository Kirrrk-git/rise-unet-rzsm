"""
tests/test_s2s.py
-----------------
Automated unit test suite verifying GRIB2 Section 7 decoding, S2S spatial remapping,
11-member ensemble aggregation, and EX29 dynamic triplet compliance for Mindanao RISE-UNet (Sub-Phase 21E).
"""

import unittest
from pathlib import Path
import numpy as np
import xarray as xr

from src.data.s2s import (
    unpack_grib2_section7_simple,
    remap_s2s_grid_to_candidate_a,
    parse_ecmwf_s2s_grib_messages,
    harmonize_s2s_cycle,
    CANDIDATE_A_LATS,
    CANDIDATE_A_LONS,
)


class TestGRIB2SimpleUnpacking(unittest.TestCase):
    """Verifies pure-Python GRIB2 Section 7 Template 0 decoding."""

    def test_constant_field_zero_bits(self):
        """Constant field (nbits=0) must return uniform array equal to R / 10^D."""
        shape = (5, 8)
        decoded = unpack_grib2_section7_simple(
            sec7_bytes=b"",
            shape=shape,
            r=300.25,
            e=0,
            d=0,
            nbits=0,
        )
        self.assertEqual(decoded.shape, shape)
        np.testing.assert_allclose(decoded, 300.25, atol=1e-5)

    def test_known_synthetic_bitstream(self):
        """Verifies formula Y = (R + X * 2^E) / 10^D on known 8-bit integers."""
        # 4 values: [0, 1, 2, 3] packed in 8 bits each
        raw_bytes = bytes([0, 1, 2, 3])
        shape = (2, 2)
        r = 10.0
        e = 1  # 2^1 = 2
        d = 0  # 10^0 = 1
        # Expected:
        # X=0 -> 10 + 0*2 = 10
        # X=1 -> 10 + 1*2 = 12
        # X=2 -> 10 + 2*2 = 14
        # X=3 -> 10 + 3*2 = 16
        decoded = unpack_grib2_section7_simple(
            sec7_bytes=raw_bytes,
            shape=shape,
            r=r,
            e=e,
            d=d,
            nbits=8,
        )
        expected = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=np.float32)
        np.testing.assert_allclose(decoded, expected, atol=1e-5)


class TestS2SRemapping(unittest.TestCase):
    """Verifies bilinear spatial remapping from native S2S to Candidate A 0.25° grid."""

    def test_remapping_geometry_and_finite(self):
        """Remapping (5, 8) S2S grid must produce (32, 48) grid with zero NaNs."""
        src_lats = np.array([10.0, 8.5, 7.0, 5.5, 4.0], dtype=np.float32)
        src_lons = np.array([116.5, 118.0, 119.5, 121.0, 122.5, 124.0, 125.5, 127.0], dtype=np.float32)
        src_grid = np.full((5, 8), 300.0, dtype=np.float32)

        remapped = remap_s2s_grid_to_candidate_a(src_grid, src_lats, src_lons)
        self.assertEqual(remapped.shape, (32, 48))
        self.assertFalse(np.isnan(remapped).any())
        np.testing.assert_allclose(remapped, 300.0, atol=1e-5)

    def test_gradient_preservation(self):
        """Monotonic latitude gradient must remain monotonic after remapping."""
        src_lats = np.array([10.0, 8.5, 7.0, 5.5, 4.0], dtype=np.float32)
        src_lons = np.linspace(116.5, 127.0, 8, dtype=np.float32)
        # Gradient along latitude (rows): 310 at 10N down to 290 at 4N
        src_grid = np.repeat(np.linspace(310.0, 290.0, 5)[:, None], 8, axis=1).astype(np.float32)

        remapped = remap_s2s_grid_to_candidate_a(src_grid, src_lats, src_lons)
        # Column slice through interior
        col = remapped[:, 24]
        # Check that temperature decreases monotonically southward (indices 0 to 31)
        self.assertTrue(np.all(np.diff(col) <= 0.001))


class TestS2SPilotData(unittest.TestCase):
    """Verifies harmonization on actual sample GRIB files in scratch/ directory."""

    def setUp(self):
        self.cf_file = Path("../scratch/sample_forcing_cf.grib")
        self.pf_file = Path("../scratch/sample_forcing_pf.grib")
        self.mask_file = Path("processed/grid/mindanao_eval_mask_025.nc")

    def test_pilot_cycle_harmonization(self):
        """Full harmonization must yield (lead=2, member=11, lat=32, lon=48) with physical bounds."""
        if not (self.cf_file.exists() and self.pf_file.exists() and self.mask_file.exists()):
            self.skipTest("Pilot GRIB sample files not present in scratch/ directory.")

        ds = harmonize_s2s_cycle(
            cf_path=self.cf_file,
            pf_path=self.pf_file,
            eval_mask_path=self.mask_file,
        )

        # Coordinate assertions
        self.assertEqual(list(ds.sizes.keys()), ["lead", "member", "lat", "lon"])
        self.assertEqual(ds.sizes["lead"], 2)
        self.assertEqual(ds.sizes["member"], 11)
        self.assertEqual(ds.sizes["lat"], 32)
        self.assertEqual(ds.sizes["lon"], 48)

        # Member list assertion: 0 (CF) + 1..10 (PF)
        self.assertEqual(list(ds["member"].values), list(range(11)))

        # Load mask to check physical evaluation values
        mask_ds = xr.open_dataset(self.mask_file)
        eval_mask = mask_ds["evaluation_mask"].values

        for var in ["t2m", "d2m", "tcw"]:
            self.assertIn(var, ds.data_vars)
            arr = ds[var].values  # (2, 11, 32, 48)

            # Check zero-filling outside evaluation mask
            ocean_vals = arr[:, :, eval_mask == 0]
            np.testing.assert_allclose(ocean_vals, 0.0, atol=1e-7)

            # Check evaluation cell values are non-zero and finite
            eval_vals = arr[:, :, eval_mask == 1]
            self.assertFalse(np.isnan(eval_vals).any())
            self.assertTrue(np.all(eval_vals != 0.0))

            # Physical ranges over Mindanao
            if var == "t2m":
                # Temperature: 285 K (12 C) to 315 K (42 C)
                self.assertTrue(np.all(eval_vals >= 285.0))
                self.assertTrue(np.all(eval_vals <= 315.0))
            elif var == "d2m":
                # Dewpoint: 275 K (2 C) to 310 K (37 C)
                self.assertTrue(np.all(eval_vals >= 275.0))
                self.assertTrue(np.all(eval_vals <= 310.0))
            elif var == "tcw":
                # Total column water: > 0 kg/m2
                self.assertTrue(np.all(eval_vals > 0.0))


if __name__ == "__main__":
    unittest.main()
