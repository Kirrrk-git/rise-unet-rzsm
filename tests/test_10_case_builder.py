"""
tests/test_case_builder.py
--------------------------
Automated unit test suite verifying single-case multi-lead tensor hierarchy assembly,
EX29 channel schedules [11, 12, 5, 6], zero-leakage isolation, and recursive concatenation (Sub-Phase 21F).
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.data.case_builder import (
    assemble_single_a0_case,
    simulate_recursive_cascade_step,
    CaseTensorHierarchy,
)
from src.data.s2s import CANDIDATE_A_LATS, CANDIDATE_A_LONS


class TestCaseBuilderSynthetic(unittest.TestCase):
    """Verifies tensor hierarchy assembly using clean synthetic datasets."""

    def setUp(self):
        # Create synthetic 60-day RZSM cube (t0 - 15d to t0 + 35d)
        dates = pd.date_range("2015-01-01", "2015-02-28", freq="D")
        lats = CANDIDATE_A_LATS
        lons = CANDIDATE_A_LONS

        rzsm_data = np.full((len(dates), len(lats), len(lons)), 0.35, dtype=np.float32)
        self.rzsm_cube = xr.Dataset(
            data_vars={
                "rzsm_rolling_7d": (("time", "lat", "lon"), rzsm_data),
            },
            coords={"time": dates, "lat": lats, "lon": lons},
        )

        # Create synthetic atmospheric dataset at t0
        atm_dates = pd.date_range("2015-01-01", "2015-01-31", freq="D")
        shape_atm = (len(atm_dates), len(lats), len(lons))
        self.atm_ds = xr.Dataset(
            data_vars={
                "pwat": (("time", "lat", "lon"), np.full(shape_atm, 45.0, dtype=np.float32)),
                "spfh": (("time", "lat", "lon"), np.full(shape_atm, 0.018, dtype=np.float32)),
                "tmax": (("time", "lat", "lon"), np.full(shape_atm, 302.0, dtype=np.float32)),
                "diff_temp": (("time", "lat", "lon"), np.full(shape_atm, 6.0, dtype=np.float32)),
                "hgt_pres": (("time", "lat", "lon"), np.full(shape_atm, 12400.0, dtype=np.float32)),
            },
            coords={"time": atm_dates, "lat": lats, "lon": lons},
        )

        # Create synthetic S2S dataset (lead=2, member=11, lat=32, lon=48)
        shape_s2s = (2, 11, len(lats), len(lons))
        self.s2s_ds = xr.Dataset(
            data_vars={
                "t2m": (("lead", "member", "lat", "lon"), np.full(shape_s2s, 301.0, dtype=np.float32)),
                "d2m": (("lead", "member", "lat", "lon"), np.full(shape_s2s, 296.0, dtype=np.float32)),
                "tcw": (("lead", "member", "lat", "lon"), np.full(shape_s2s, 0.02, dtype=np.float32)),
            },
            coords={
                "lead": [1, 2],
                "member": list(range(11)),
                "lat": lats,
                "lon": lons,
            },
        )

        # Synthetic mask (center cells=1, edges=0)
        self.mask = np.zeros((len(lats), len(lons)), dtype=np.int8)
        self.mask[10:25, 15:35] = 1

    def test_case_hierarchy_shapes_and_channels(self):
        """Assembled case tensors must strictly match [11, 12, 5, 6] channel counts across 11 members."""
        t0 = "2015-01-16"
        case = assemble_single_a0_case(
            issue_date=t0,
            rzsm_cube_ds=self.rzsm_cube,
            atmospheric_ds=self.atm_ds,
            s2s_ds=self.s2s_ds,
            eval_mask=self.mask,
        )

        # Lead 1: 11 channels
        self.assertEqual(case.x_w1.shape, (11, 32, 48, 11))
        self.assertEqual(case.y_w1.shape, (1, 32, 48, 1))

        # Lead 2: 11 base channels + 1 recursive -> 12 channels
        y_hat_w1 = np.full((11, 32, 48, 1), 0.32, dtype=np.float32)
        x_w2_full = simulate_recursive_cascade_step(case.x_w2_base, [y_hat_w1])
        self.assertEqual(x_w2_full.shape, (11, 32, 48, 12))
        self.assertEqual(case.y_w2.shape, (1, 32, 48, 1))

        # Lead 3: 3 base lags + 2 recursive -> 5 channels
        y_hat_w2 = np.full((11, 32, 48, 1), 0.31, dtype=np.float32)
        x_w3_full = simulate_recursive_cascade_step(case.x_w3_base, [y_hat_w1, y_hat_w2])
        self.assertEqual(x_w3_full.shape, (11, 32, 48, 5))
        self.assertEqual(case.y_w3.shape, (1, 32, 48, 1))

        # Lead 4: 3 base lags + 3 recursive -> 6 channels
        y_hat_w3 = np.full((11, 32, 48, 1), 0.30, dtype=np.float32)
        x_w4_full = simulate_recursive_cascade_step(case.x_w4_base, [y_hat_w1, y_hat_w2, y_hat_w3])
        self.assertEqual(x_w4_full.shape, (11, 32, 48, 6))
        # Target dates check (Parent EX29 L = (lead * 7) - 1: +6d, +13d, +20d, +27d)
        expected_targets = {
            1: "2015-01-22",
            2: "2015-01-29",
            3: "2015-02-05",
            4: "2015-02-12",
        }
        self.assertEqual(case.target_dates, expected_targets)

    def test_ocean_zero_filling(self):
        """All non-evaluation cells must be strictly 0.0 across all channels and targets."""
        t0 = "2015-01-16"
        case = assemble_single_a0_case(
            issue_date=t0,
            rzsm_cube_ds=self.rzsm_cube,
            atmospheric_ds=self.atm_ds,
            s2s_ds=self.s2s_ds,
            eval_mask=self.mask,
        )

        ocean = self.mask == 0
        self.assertTrue(np.all(case.x_w1[:, ocean, :] == 0.0))
        self.assertTrue(np.all(case.x_w2_base[:, ocean, :] == 0.0))
        self.assertTrue(np.all(case.x_w3_base[:, ocean, :] == 0.0))
        self.assertTrue(np.all(case.x_w4_base[:, ocean, :] == 0.0))
        self.assertTrue(np.all(case.y_w1[:, ocean, :] == 0.0))
        self.assertTrue(np.all(case.y_w4[:, ocean, :] == 0.0))

    def test_missing_antecedent_raises_error(self):
        """Issuance date with incomplete antecedent window must raise KeyError."""
        too_early = "2015-01-05"  # -14d is 2014-12-22, not in synthetic rzsm_cube
        with self.assertRaises(KeyError):
            assemble_single_a0_case(
                issue_date=too_early,
                rzsm_cube_ds=self.rzsm_cube,
                atmospheric_ds=self.atm_ds,
                s2s_ds=self.s2s_ds,
            )


class TestCaseBuilderRealData(unittest.TestCase):
    """Verifies single-case tensor assembly on real project pilot artifacts."""

    def test_real_pilot_case_assembly(self):
        """Assembles real Jan 15, 2015 case using certified production RZSM and pilot files."""
        rzsm_path = Path("processed/rzsm/production/era5_land_rzsm_production_2015_2025.nc")
        atm_path = Path("processed/atmospheric/pilot/era5_atmospheric_pilot_2015_01.nc")
        s2s_path = Path("processed/s2s/pilot/s2s_pilot_reforecast_w1_w2.nc")
        mask_path = Path("processed/grid/mindanao_eval_mask_025.nc")

        if not (rzsm_path.exists() and atm_path.exists() and s2s_path.exists() and mask_path.exists()):
            self.skipTest("Required production or pilot artifacts missing for real data assembly test.")

        rzsm_ds = xr.open_dataset(rzsm_path)
        atm_ds = xr.open_dataset(atm_path)
        s2s_ds = xr.open_dataset(s2s_path)
        mask_ds = xr.open_dataset(mask_path)
        eval_mask = mask_ds["evaluation_mask"].values

        case = assemble_single_a0_case(
            issue_date="2015-01-15",
            rzsm_cube_ds=rzsm_ds,
            atmospheric_ds=atm_ds,
            s2s_ds=s2s_ds,
            target_var_name="rzsm_rolling_7d",
            eval_mask=eval_mask,
        )

        # Assert shapes
        self.assertEqual(case.x_w1.shape, (11, 32, 48, 11))
        self.assertEqual(case.x_w2_base.shape, (11, 32, 48, 11))
        self.assertEqual(case.x_w3_base.shape, (11, 32, 48, 3))
        self.assertEqual(case.x_w4_base.shape, (11, 32, 48, 3))
        self.assertEqual(case.y_w1.shape, (1, 32, 48, 1))
        self.assertEqual(case.y_w2.shape, (1, 32, 48, 1))
        self.assertEqual(case.y_w3.shape, (1, 32, 48, 1))
        self.assertEqual(case.y_w4.shape, (1, 32, 48, 1))

        # Assert active evaluation cells are non-zero and finite
        active = eval_mask == 1
        self.assertFalse(np.isnan(case.x_w1[:, active, :]).any())
        self.assertFalse(np.isnan(case.y_w1[:, active, :]).any())
        self.assertTrue(np.all(case.x_w1[:, active, :] != 0.0))
        self.assertTrue(np.all(case.y_w1[:, active, :] != 0.0))

        # Assert hdate preservation
        self.assertEqual(case.hdate, "2015-01-15")

        # Assert parent EX29 target dates: +6d, +13d, +20d, +27d
        expected_targets = {
            1: "2015-01-21",
            2: "2015-01-28",
            3: "2015-02-04",
            4: "2015-02-11",
        }
        self.assertEqual(case.target_dates, expected_targets)


if __name__ == "__main__":
    unittest.main()
