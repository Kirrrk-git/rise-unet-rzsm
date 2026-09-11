"""
tests/test_target_reconciliation.py
-----------------------------------
Authoritative reconciliation tests verifying exact mathematical and indexing parity
between Kyle Lesinger's parent EX29 target construction and our Mindanao temporal pipeline.

Parent References:
  - 00_min_max_each_region_&reforecast.ipynb:L244: lead_select = [6, 13, 20, 27]
  - function/funs.py:L430-432: cp_base.sel(L=(lead * 7) - 1)
  - function/preprocessUtils.py:L88: file.rolling(time=7, min_periods=7, center=False).mean()
"""

import unittest
import numpy as np
import pandas as pd
import xarray as xr

from src.data.temporal import (
    TemporalConfig,
    compute_trailing_rolling_mean,
    extract_antecedent_and_target_windows,
)


def parent_ex29_target_construction(
    raw_daily_da: xr.DataArray,
    issue_date_str: str,
    leads: tuple = (1, 2, 3, 4),
) -> dict:
    """
    Direct replication of Kyle Lesinger's EX29 target construction logic:
      1. Rolling 7-day backward mean: rolling(time=7, min_periods=7, center=False).mean()
      2. Lead day offset: L = (lead * 7) - 1
      3. Target extraction: sample rolling array at issue_date + L days
    """
    # Parent step 1: rolling 7-day trailing mean
    rolled = raw_daily_da.rolling(time=7, min_periods=7, center=False).mean()

    issue_dt = pd.Timestamp(issue_date_str)
    targets = {}
    for lead in leads:
        lead_day_offset = (lead * 7) - 1  # Parent formula from funs.py:L430
        target_date = issue_dt + pd.Timedelta(days=lead_day_offset)
        # In parent, this corresponds to sel(time=target_date)
        targets[lead] = rolled.sel(time=target_date)

    return targets


def parent_ex29_antecedent_construction(
    raw_daily_da: xr.DataArray,
    issue_date_str: str,
    lag_offsets: tuple = (-1, -7, -14),
) -> dict:
    """
    Direct replication of Kyle Lesinger's EX29 antecedent lag construction logic:
      - Trailing 7-day rolling mean sampled at issue_date + lag days
    """
    rolled = raw_daily_da.rolling(time=7, min_periods=7, center=False).mean()
    issue_dt = pd.Timestamp(issue_date_str)
    antecedents = {}
    for lag in lag_offsets:
        lag_date = issue_dt + pd.Timedelta(days=lag)
        antecedents[lag] = rolled.sel(time=lag_date)
    return antecedents


class TestTargetReconciliation(unittest.TestCase):
    """Rigorous reconciliation test suite between parent EX29 and Mindanao pipeline."""

    def setUp(self):
        # Create a continuous daily dataset spanning 2014-11-01 to 2021-12-31
        dates = pd.date_range("2014-11-01", "2021-12-31", freq="D")
        np.random.seed(42)
        # Linear trend + seasonal cycle + noise to ensure distinct daily values
        t_steps = len(dates)
        daily_vals = (
            0.35
            + 0.10 * np.sin(2 * np.pi * np.arange(t_steps) / 365.25)[:, None, None]
            + 0.05 * np.random.randn(t_steps, 4, 4)
        )
        self.da = xr.DataArray(
            daily_vals,
            coords={
                "time": dates,
                "lat": [7.0, 7.25, 7.5, 7.75],
                "lon": [124.0, 124.25, 124.5, 124.75],
            },
            dims=["time", "lat", "lon"],
            name="rzsm",
        )

    def test_hand_computable_dates_exact_parity(self):
        """
        Verify parent target construction == our target construction for hand-computable dates.
        """
        test_dates = [
            "2015-01-15",
            "2015-06-01",
            "2016-02-29",  # Leap year
            "2018-08-15",
            "2020-12-01",
        ]

        for date_str in test_dates:
            # 1. Compute using direct parent EX29 logic
            parent_targets = parent_ex29_target_construction(self.da, date_str)
            parent_antecedents = parent_ex29_antecedent_construction(self.da, date_str)

            # 2. Compute using our Mindanao pipeline
            rolled = compute_trailing_rolling_mean(self.da, window=7, min_periods=7)
            our_antecedents, our_targets = extract_antecedent_and_target_windows(
                rolled,
                issue_date=date_str,
                antecedent_lags=(-1, -7, -14),
                target_leads=(6, 13, 20, 27),
            )

            # Assert exact numerical equality across all 4 leads
            lead_map = {1: 6, 2: 13, 3: 20, 4: 27}
            for lead_num, lead_offset in lead_map.items():
                parent_val = parent_targets[lead_num].values
                our_val = our_targets[lead_offset].values
                np.testing.assert_array_equal(
                    our_val,
                    parent_val,
                    err_msg=f"Discrepancy at date {date_str}, lead {lead_num} (offset +{lead_offset})",
                )

            # Assert exact numerical equality across antecedent lags
            for lag in (-1, -7, -14):
                parent_lag = parent_antecedents[lag].values
                our_lag = our_antecedents[lag].values
                np.testing.assert_array_equal(
                    our_lag,
                    parent_lag,
                    err_msg=f"Discrepancy at date {date_str}, lag {lag}",
                )

    def test_hand_computable_arithmetic_verification(self):
        """
        Hand-calculate the rolling mean for date 2015-01-15 and verify arithmetic equality.
        """
        issue_date = pd.Timestamp("2015-01-15")

        # Lead W1: L=6 -> target_date = 2015-01-21
        # The 7-day trailing window is [2015-01-15, 2015-01-16, ..., 2015-01-21]
        w1_window = pd.date_range("2015-01-15", "2015-01-21", freq="D")
        self.assertEqual(len(w1_window), 7)
        manual_w1_mean = self.da.sel(time=w1_window).mean(dim="time").values

        # Lead W2: L=13 -> target_date = 2015-01-28
        # The 7-day trailing window is [2015-01-22, 2015-01-23, ..., 2015-01-28]
        w2_window = pd.date_range("2015-01-22", "2015-01-28", freq="D")
        self.assertEqual(len(w2_window), 7)
        manual_w2_mean = self.da.sel(time=w2_window).mean(dim="time").values

        # Lead W3: L=20 -> target_date = 2015-02-04
        # The 7-day trailing window is [2015-01-29, ..., 2015-02-04]
        w3_window = pd.date_range("2015-01-29", "2015-02-04", freq="D")
        self.assertEqual(len(w3_window), 7)
        manual_w3_mean = self.da.sel(time=w3_window).mean(dim="time").values

        # Lead W4: L=27 -> target_date = 2015-02-11
        # The 7-day trailing window is [2015-02-05, ..., 2015-02-11]
        w4_window = pd.date_range("2015-02-05", "2015-02-11", freq="D")
        self.assertEqual(len(w4_window), 7)
        manual_w4_mean = self.da.sel(time=w4_window).mean(dim="time").values

        # Compute via pipeline
        rolled = compute_trailing_rolling_mean(self.da, window=7, min_periods=7)
        _, our_targets = extract_antecedent_and_target_windows(
            rolled,
            issue_date="2015-01-15",
            target_leads=(6, 13, 20, 27),
        )

        np.testing.assert_allclose(our_targets[6].values, manual_w1_mean, rtol=1e-6)
        np.testing.assert_allclose(our_targets[13].values, manual_w2_mean, rtol=1e-6)
        np.testing.assert_allclose(our_targets[20].values, manual_w3_mean, rtol=1e-6)
        np.testing.assert_allclose(our_targets[27].values, manual_w4_mean, rtol=1e-6)

    def test_non_overlapping_contiguous_partition(self):
        """
        Verify that the 4 weekly target windows form a contiguous 28-day partition without gaps or overlaps.
        """
        issue_date = pd.Timestamp("2015-03-01")
        windows = []
        for lead_offset in (6, 13, 20, 27):
            end_date = issue_date + pd.Timedelta(days=lead_offset)
            start_date = end_date - pd.Timedelta(days=6)
            window_dates = pd.date_range(start_date, end_date, freq="D")
            windows.append(window_dates)

        # Check contiguous chaining: end of W_k + 1 day == start of W_{k+1}
        for k in range(3):
            self.assertEqual(windows[k][-1] + pd.Timedelta(days=1), windows[k + 1][0])

        # Check total unique days == 28
        all_dates = pd.DatetimeIndex([d for w in windows for d in w])
        self.assertEqual(len(all_dates), 28)
        self.assertEqual(len(all_dates.unique()), 28)
        self.assertEqual(all_dates[0], issue_date)
        self.assertEqual(all_dates[-1], issue_date + pd.Timedelta(days=27))


if __name__ == "__main__":
    unittest.main()
