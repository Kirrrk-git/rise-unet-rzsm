"""
tests/test_rzsm.py
------------------
Automated unit test suite verifying mathematical precision, rolling window mechanics,
antecedent lag slicing, evaluation masking, and min-max standardization for the
RISE-UNet Mindanao RZSM pipeline (Sub-Phase 21D).
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.data.rzsm import (
    compute_depth_weighted_rzsm,
    compute_backward_rolling_mean,
    extract_antecedent_lags,
    apply_land_mask,
    compute_min_max_scale,
    remap_era5_land_to_candidate_a,
    LAYER_WEIGHTS,
)


class TestRZSMFormulation(unittest.TestCase):
    """Verifies depth-weighted RZSM calculation to machine precision."""

    def test_layer_weights_sum_to_one(self):
        """Weights must sum exactly to 1.0 (0.07 + 0.21 + 0.72)."""
        w_sum = sum(LAYER_WEIGHTS.values())
        self.assertAlmostEqual(w_sum, 1.0, places=7)

    def test_depth_weighted_rzsm_constant_field(self):
        """Uniform soil moisture c across all 3 layers must yield RZSM = c."""
        for c in [0.15, 0.32, 0.45]:
            sm1 = np.full((10, 32, 48), c, dtype=np.float32)
            sm2 = np.full((10, 32, 48), c, dtype=np.float32)
            sm3 = np.full((10, 32, 48), c, dtype=np.float32)
            rzsm = compute_depth_weighted_rzsm(sm1, sm2, sm3)
            np.testing.assert_allclose(rzsm, c, atol=1e-6)

    def test_depth_weighted_rzsm_analytical_solution(self):
        """Known combination: sm1=1.0, sm2=2.0, sm3=3.0 -> 0.07(1) + 0.21(2) + 0.72(3) = 2.65."""
        sm1 = np.ones((5, 5)) * 1.0
        sm2 = np.ones((5, 5)) * 2.0
        sm3 = np.ones((5, 5)) * 3.0
        rzsm = compute_depth_weighted_rzsm(sm1, sm2, sm3)
        expected = 0.07 * 1.0 + 0.21 * 2.0 + 0.72 * 3.0  # 2.65
        np.testing.assert_allclose(rzsm, expected, atol=1e-7)

    def test_invalid_weights_raise_error(self):
        """Weights that do not sum to 1.0 must raise ValueError."""
        sm = np.ones((2, 2))
        with self.assertRaises(ValueError):
            compute_depth_weighted_rzsm(sm, sm, sm, w1=0.1, w2=0.2, w3=0.3)


class TestTemporalPreprocessing(unittest.TestCase):
    """Verifies trailing rolling mean and antecedent lag extractions."""

    def setUp(self):
        # 30-day synthetic daily time series
        dates = pd.date_range("2015-01-01", periods=30, freq="D")
        vals = np.arange(30, dtype=np.float32).reshape(30, 1, 1)
        self.da = xr.DataArray(
            vals,
            coords={"time": dates, "lat": [10.0], "lon": [120.0]},
            dims=["time", "lat", "lon"],
        )

    def test_backward_rolling_mean_mechanics(self):
        """Trailing rolling mean with window=7 must have 6 NaNs followed by trailing means."""
        rolled = compute_backward_rolling_mean(self.da, window=7, min_periods=7)
        # First 6 timesteps (days 0-5) must be NaN
        self.assertTrue(np.isnan(rolled.isel(time=slice(0, 6)).values).all())
        # 7th timestep (day 6, values 0..6) mean = 21/7 = 3.0
        self.assertAlmostEqual(float(rolled.isel(time=6).values[0, 0]), 3.0, places=6)
        # 8th timestep (day 7, values 1..7) mean = 28/7 = 4.0
        self.assertAlmostEqual(float(rolled.isel(time=7).values[0, 0]), 4.0, places=6)

    def test_extract_antecedent_lags(self):
        """Lags at [-1, -7, -14] days must extract the exact preceding dates."""
        target_date = "2015-01-20"
        lags = extract_antecedent_lags(self.da, target_date=target_date, lag_days=(-1, -7, -14))
        self.assertEqual(set(lags.keys()), {-1, -7, -14})
        # Day 20 is index 19 (val=19).
        # lag -1 -> 2015-01-19 (val=18)
        self.assertAlmostEqual(float(lags[-1].values[0, 0]), 18.0)
        # lag -7 -> 2015-01-13 (val=12)
        self.assertAlmostEqual(float(lags[-7].values[0, 0]), 12.0)
        # lag -14 -> 2015-01-06 (val=5)
        self.assertAlmostEqual(float(lags[-14].values[0, 0]), 5.0)


class TestMaskingAndScaling(unittest.TestCase):
    """Verifies binary evaluation masking and min-max standardization."""

    def setUp(self):
        # 4x4 spatial grid with 4 active cells (2x2 center)
        self.mask = np.zeros((4, 4), dtype=np.int8)
        self.mask[1:3, 1:3] = 1

        self.data = np.full((4, 4), 999.0, dtype=np.float32)
        self.data[1:3, 1:3] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    def test_apply_land_mask(self):
        """Active land cells must retain values; inactive cells set to 0.0."""
        masked = apply_land_mask(self.data, self.mask, fill_value=0.0)
        # Active cells
        np.testing.assert_allclose(masked[1:3, 1:3], [[10.0, 20.0], [30.0, 40.0]])
        # Inactive perimeter
        self.assertTrue((masked[0, :] == 0.0).all())
        self.assertTrue((masked[3, :] == 0.0).all())
        self.assertTrue((masked[:, 0] == 0.0).all())
        self.assertTrue((masked[:, 3] == 0.0).all())

    def test_compute_min_max_scale(self):
        """Active cells must scale to [0, 1] based on active min (10.0) and max (40.0)."""
        scaled, min_v, max_v = compute_min_max_scale(self.data, mask=self.mask)
        self.assertAlmostEqual(min_v, 10.0)
        self.assertAlmostEqual(max_v, 40.0)
        # Min cell (10.0) -> 0.0, Max cell (40.0) -> 1.0
        self.assertAlmostEqual(float(scaled[1, 1]), 0.0)
        self.assertAlmostEqual(float(scaled[2, 2]), 1.0)
        self.assertAlmostEqual(float(scaled[1, 2]), (20.0 - 10.0) / 30.0)
        # Inactive cells must be 0.0
        self.assertEqual(float(scaled[0, 0]), 0.0)


class TestRealPilotDataIntegrity(unittest.TestCase):
    """Verifies that functions cleanly integrate with the real verified pilot NetCDF."""

    def test_pilot_rzsm_file_exists_and_passes_physics(self):
        pilot_path = Path("processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc")
        self.assertTrue(pilot_path.exists(), f"Pilot file not found at {pilot_path}")

        ds = xr.open_dataset(pilot_path)
        self.assertEqual(ds.rzsm_0_100_masked.shape, (20, 32, 48))
        self.assertEqual(int(ds.evaluation_mask.sum()), 126)

        # Extract active evaluation samples
        mask_bool = ds.evaluation_mask.values == 1
        active_vals = ds.rzsm_0_100_masked.values[:, mask_bool]

        # Assert zero NaNs / Infs
        self.assertEqual(np.isnan(active_vals).sum(), 0)
        self.assertEqual(np.isinf(active_vals).sum(), 0)

        # Physical volumetric soil moisture bounds (0.20 to 0.60 m3/m3 in tropical soils)
        self.assertGreaterEqual(float(active_vals.min()), 0.20)
        self.assertLessEqual(float(active_vals.max()), 0.60)


class TestLandAwareRemapping(unittest.TestCase):
    """Verifies land-aware linear spatial remapping with boundary extrapolation fallback."""

    def test_remap_synthetic_grid_zero_nan_and_zero_fill(self):
        # Source grid: 0.10 deg resolution covering Mindanao bounding box
        src_lats = np.arange(11.0, 3.9, -0.10)
        src_lons = np.arange(116.5, 127.6, 0.10)
        src_arr = np.full((len(src_lats), len(src_lons)), np.nan, dtype=np.float32)

        # Populate land region (e.g. Mindanao central area) with realistic values
        land_mask = (
            (src_lats[:, None] >= 5.5) & (src_lats[:, None] <= 9.5) &
            (src_lons[None, :] >= 121.5) & (src_lons[None, :] <= 126.5)
        )
        src_arr[land_mask] = 0.38

        src_da = xr.DataArray(
            src_arr,
            coords={"latitude": src_lats, "longitude": src_lons},
            dims=["latitude", "longitude"],
            name="rzsm",
        )

        # Target Candidate A grid: 0.25 deg (32 x 48)
        tgt_lats = np.linspace(11.75, 4.0, 32)
        tgt_lons = np.linspace(116.0, 127.75, 48)

        # Mock evaluation mask: 126 active cells inside Mindanao domain
        eval_mask = np.zeros((32, 48), dtype=np.int32)
        eval_mask[12:26, 22:36] = 1  # 14 * 14 = 196 -> adjust to some active subset
        eval_mask[0:10, :] = 0  # ensure non-evaluation cells exist

        remapped = remap_era5_land_to_candidate_a(
            src_da,
            target_lats=tgt_lats,
            target_lons=tgt_lons,
            eval_mask=eval_mask,
        )

        self.assertEqual(remapped.shape, (32, 48))
        # Zero NaNs/Infs over active evaluation cells
        active_vals = remapped.values[eval_mask == 1]
        self.assertTrue(np.all(np.isfinite(active_vals)))
        # Non-evaluation cells strictly 0.0
        inactive_vals = remapped.values[eval_mask == 0]
        np.testing.assert_array_equal(inactive_vals, 0.0)


if __name__ == "__main__":
    unittest.main()

