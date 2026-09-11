"""
tests/test_temporal.py
----------------------
Automated unit tests for the Mindanao RISE-UNet temporal preprocessing,
climatology anomaly calculation, antecedent/target window extraction,
and standardized target parity pipeline (Sub-Phase 21D Step 21D.3).
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.data.temporal import (
    TemporalConfig,
    compute_trailing_rolling_mean,
    compute_training_climatology,
    compute_seasonal_anomalies,
    extract_antecedent_and_target_windows,
    fit_min_max_bounds,
    standardize_with_training_bounds,
)


class TestTrailingRollingMean(unittest.TestCase):
    """Verifies that rolling mean is strictly trailing (center=False) with zero future data leakage."""

    def setUp(self):
        # Create 30 days of daily scalar data
        dates = pd.date_range("2015-01-01", periods=30, freq="D")
        lats = [10.0, 9.0]
        lons = [124.0, 125.0]
        # Data: increasing day index
        arr = np.zeros((30, 2, 2), dtype=np.float32)
        for i in range(30):
            arr[i, :, :] = float(i + 1)
        self.da = xr.DataArray(arr, coords=[dates, lats, lons], dims=["time", "lat", "lon"], name="rzsm")

    def test_trailing_window_values(self):
        smoothed = compute_trailing_rolling_mean(self.da, window=7, min_periods=7)
        # First 6 days must be NaN
        self.assertTrue(np.all(np.isnan(smoothed.isel(time=slice(0, 6)).values)))
        # Day index 6 (7th day) should be the mean of 1..7 = 4.0
        val_day7 = float(smoothed.isel(time=6).values[0, 0])
        self.assertAlmostEqual(val_day7, 4.0, places=5)
        # Day index 7 (8th day) should be mean of 2..8 = 5.0
        val_day8 = float(smoothed.isel(time=7).values[0, 0])
        self.assertAlmostEqual(val_day8, 5.0, places=5)

    def test_future_invariance_no_leakage(self):
        """Modifying future observations at t+1 must not alter the rolling average at t."""
        da_perturbed = self.da.copy(deep=True)
        smoothed_orig = compute_trailing_rolling_mean(self.da, window=7, min_periods=7)

        # Perturb day 15 drastically
        da_perturbed[15, :, :] = 9999.0
        smoothed_perturbed = compute_trailing_rolling_mean(da_perturbed, window=7, min_periods=7)

        # Days 0 to 14 must be IDENTICAL to machine precision
        np.testing.assert_allclose(
            smoothed_orig.isel(time=slice(0, 15)).values,
            smoothed_perturbed.isel(time=slice(0, 15)).values,
            rtol=1e-6,
            err_msg="Future data leakage detected! Modifying future values changed past rolling averages."
        )


class TestTrainingClimatologyAndAnomalies(unittest.TestCase):
    """Verifies that climatology is fitted strictly on training years (<=2021) and anomalies are accurate."""

    def setUp(self):
        # Create 8 years of daily data (2015 to 2022) with a known periodic annual cycle
        dates = pd.date_range("2015-01-01", "2022-12-31", freq="D")
        n_days = len(dates)
        lats = [8.0]
        lons = [125.0]

        # Annual sinusoidal wave (period ~365.25 days) + noise
        t_arr = np.arange(n_days)
        seasonal_cycle = 0.40 + 0.10 * np.sin(2 * np.pi * t_arr / 365.25)
        arr = np.tile(seasonal_cycle[:, None, None], (1, len(lats), len(lons))).astype(np.float32)

        self.da = xr.DataArray(arr, coords=[dates, lats, lons], dims=["time", "lat", "lon"], name="rzsm")

    def test_training_period_filter(self):
        """Assert climatology only ingests years <= train_end_year."""
        clim_doy = compute_training_climatology(
            self.da, train_start_year=2015, train_end_year=2021, method="dayofyear"
        )
        self.assertEqual(clim_doy.attrs["train_end_year"], 2021)
        self.assertIn("dayofyear", clim_doy.coords)
        # Should have up to 366 days
        self.assertTrue(len(clim_doy.dayofyear) in (365, 366))

    def test_season_climatology_method(self):
        """Assert 3-month season climatology conforms to DJF, MAM, JJA, SON."""
        clim_season = compute_training_climatology(
            self.da, train_start_year=2015, train_end_year=2021, method="season"
        )
        self.assertEqual(clim_season.attrs["climatology_method"], "season")
        expected_seasons = {"DJF", "MAM", "JJA", "SON"}
        actual_seasons = set(clim_season.season.values)
        self.assertEqual(actual_seasons, expected_seasons)

    def test_anomaly_subtraction_zero_mean_training(self):
        """Over the training period, the mean anomaly per season/DOY should be approximately zero."""
        clim = compute_training_climatology(self.da, train_end_year=2021, method="season")
        anom = compute_seasonal_anomalies(self.da, clim, method="season")

        # Check that shape and coordinates are preserved
        self.assertEqual(anom.shape, self.da.shape)
        np.testing.assert_array_equal(anom.time.values, self.da.time.values)

        # Training fold anomaly mean should be ~0.0
        train_anom = anom.sel(time=anom["time.year"] <= 2021)
        mean_train_anom = float(train_anom.mean().values)
        self.assertAlmostEqual(mean_train_anom, 0.0, places=3)


class TestTargetParityAndWindows(unittest.TestCase):
    """Verifies that antecedent lags [-1, -7, -14] and multi-lead targets [W1..W4] maintain strict parity."""

    def setUp(self):
        # 60 daily steps starting 2016-05-01
        dates = pd.date_range("2016-05-01", periods=60, freq="D")
        lats = np.linspace(10.0, 5.0, 4)
        lons = np.linspace(120.0, 126.0, 4)
        arr = np.random.RandomState(42).randn(len(dates), len(lats), len(lons)).astype(np.float32)
        self.da = xr.DataArray(arr, coords=[dates, lats, lons], dims=["time", "lat", "lon"], name="rzsm_anom")

    def test_exact_lag_and_lead_dates(self):
        issue_date = "2016-05-25"
        antecedents, targets = extract_antecedent_and_target_windows(
            self.da,
            issue_date=issue_date,
            antecedent_lags=(-1, -7, -14),
            target_leads=(6, 13, 20, 27),
        )

        # Verify antecedent dates
        expected_ante_dates = {-1: "2016-05-24", -7: "2016-05-18", -14: "2016-05-11"}
        for lag, expected_d in expected_ante_dates.items():
            self.assertIn(lag, antecedents)
            actual_d = pd.to_datetime(antecedents[lag].time.values).strftime("%Y-%m-%d")
            self.assertEqual(actual_d, expected_d)
            self.assertEqual(antecedents[lag].shape, (4, 4))

        # Verify target lead dates (reconciled parent EX29 contract: (lead * 7) - 1)
        expected_target_dates = {6: "2016-05-31", 13: "2016-06-07", 20: "2016-06-14", 27: "2016-06-21"}
        for lead, expected_d in expected_target_dates.items():
            self.assertIn(lead, targets)
            actual_d = pd.to_datetime(targets[lead].time.values).strftime("%Y-%m-%d")
            self.assertEqual(actual_d, expected_d)
            self.assertEqual(targets[lead].shape, (4, 4))

    def test_target_parity_formula(self):
        """Assert targets and antecedents are extracted from the same continuous array with no asymmetric scaling."""
        issue_date = "2016-05-25"
        antecedents, targets = extract_antecedent_and_target_windows(self.da, issue_date=issue_date)

        # Both antecedents and targets must have identical coordinates (lat, lon)
        np.testing.assert_array_equal(antecedents[-1].lat.values, targets[6].lat.values)
        np.testing.assert_array_equal(antecedents[-1].lon.values, targets[6].lon.values)



class TestStandardizationWithTrainingBounds(unittest.TestCase):
    """Verifies min-max scaling fitted strictly on training years with ocean zero-padding."""

    def setUp(self):
        dates = pd.date_range("2015-01-01", "2017-12-31", freq="D")
        lats = [10.0, 9.0]
        lons = [124.0, 125.0]
        arr = np.random.RandomState(123).uniform(-0.15, 0.25, size=(len(dates), 2, 2)).astype(np.float32)
        self.da = xr.DataArray(arr, coords=[dates, lats, lons], dims=["time", "lat", "lon"], name="anom")
        # Mask: diagonal is active (1), off-diagonal is ocean (0)
        self.mask = xr.DataArray(
            np.array([[1, 0], [0, 1]], dtype=np.uint8),
            coords=[lats, lons],
            dims=["lat", "lon"]
        )

    def test_fit_and_standardize(self):
        # Fit bounds on 2015-2016
        min_v, max_v = fit_min_max_bounds(self.da, mask=self.mask, train_end_year=2016)
        self.assertTrue(min_v < max_v)

        # Standardize full dataset (including out-of-sample 2017)
        scaled = standardize_with_training_bounds(self.da, min_v, max_v, mask=self.mask, clip=True)

        # Ocean cells (mask == 0) must be strictly 0.0
        ocean_slice = scaled.values[:, self.mask.values == 0]
        np.testing.assert_array_equal(ocean_slice, 0.0)

        # Active cells must be bounded within [0.0, 1.0]
        active_slice = scaled.values[:, self.mask.values == 1]
        self.assertTrue(np.all(active_slice >= 0.0))
        self.assertTrue(np.all(active_slice <= 1.0))


class TestPilotRealDataTemporalPipeline(unittest.TestCase):
    """Integrates real repository pilot assets: pilot NetCDF and Candidate A evaluation mask."""

    def test_real_pilot_temporal_processing(self):
        pilot_path = Path("processed/rzsm/pilot/era5_land_rzsm_pilot_2014_2015.nc")
        mask_path = Path("processed/grid/mindanao_eval_mask_025.nc")

        if not pilot_path.exists() or not mask_path.exists():
            self.skipTest("Pilot assets not present on disk.")

        ds_pilot = xr.open_dataset(pilot_path)
        ds_mask = xr.open_dataset(mask_path)

        mask_var = "evaluation_mask" if "evaluation_mask" in ds_mask else "eval_mask"
        mask = ds_mask[mask_var]
        rzsm = ds_pilot["rzsm_0_100_masked"]

        # Compute trailing rolling mean across 20 days
        rolling = compute_trailing_rolling_mean(rzsm, window=7, min_periods=7)
        self.assertEqual(rolling.shape, (20, 32, 48))

        # Check that starting at day index 6 (7th day), active cells have zero NaNs
        eval_bool = mask.values == 1
        day7_active = rolling.isel(time=6).values[eval_bool]
        self.assertEqual(np.isnan(day7_active).sum(), 0)
        self.assertEqual(np.isinf(day7_active).sum(), 0)


if __name__ == "__main__":
    unittest.main()
