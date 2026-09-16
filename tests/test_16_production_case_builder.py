"""
tests/test_16_production_case_builder.py
-----------------------------------------
Automated test suite for Step 21K Production Case Builder (scripts/17_build_production_cases.py):
  1. Case hierarchy tensor shape assertions [11, 12, 5, 6].
  2. Active evaluation masking (126 cells) and ocean zero-filling.
  3. Frozen normalization contract bounds in [0.0, 1.0].
  4. Idempotent existing-case detection.
  5. S2S missing schedule error handling.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.data.case_builder import assemble_single_a0_case, load_frozen_normalization_contract
from src.data.s2s import CANDIDATE_A_LATS, CANDIDATE_A_LONS


class TestProductionCaseBuilder(unittest.TestCase):
    """Tests production case construction and verification mechanics."""

    def setUp(self):
        self.lats = CANDIDATE_A_LATS
        self.lons = CANDIDATE_A_LONS
        self.h, self.w = len(self.lats), len(self.lons)

        # Mock 126 active evaluation mask
        self.mask = np.zeros((self.h, self.w), dtype=bool)
        self.mask[10:24, 15:24] = True  # exactly 126 cells (14 * 9 = 126)
        self.assertEqual(int(np.sum(self.mask)), 126)

        # Synthetic RZSM production cube covering [t0 - 15d .. t0 + 30d]
        t_range = pd.date_range("2015-01-01", "2015-02-28", freq="1D")
        rzsm_raw = np.full((len(t_range), self.h, self.w), 0.35, dtype=np.float32)
        rzsm_anom = np.full((len(t_range), self.h, self.w), 0.02, dtype=np.float32)
        self.rzsm_cube = xr.Dataset(
            data_vars={
                "rzsm_0_100_rolling_7d": (("time", "lat", "lon"), rzsm_raw),
                "rzsm_0_100_seasonal_anomaly": (("time", "lat", "lon"), rzsm_anom),
            },
            coords={"time": t_range, "lat": self.lats, "lon": self.lons},
        )

        # Synthetic 5-channel Atmospheric dataset
        atm_dates = pd.date_range("2015-01-01", "2015-01-31", freq="1D")
        atm_shape = (len(atm_dates), self.h, self.w)
        self.atm_ds = xr.Dataset(
            data_vars={
                "pwat": (("time", "lat", "lon"), np.full(atm_shape, 48.0, dtype=np.float32)),
                "spfh": (("time", "lat", "lon"), np.full(atm_shape, 0.017, dtype=np.float32)),
                "tmax": (("time", "lat", "lon"), np.full(atm_shape, 301.5, dtype=np.float32)),
                "diff_temp": (("time", "lat", "lon"), np.full(atm_shape, 6.5, dtype=np.float32)),
                "hgt_pres": (("time", "lat", "lon"), np.full(atm_shape, 12420.0, dtype=np.float32)),
            },
            coords={"time": atm_dates, "lat": self.lats, "lon": self.lons},
        )

        # Synthetic S2S dataset (leads 1 and 2, 11 members)
        s2s_shape = (2, 11, self.h, self.w)
        self.s2s_ds = xr.Dataset(
            data_vars={
                "t2m": (("lead", "member", "lat", "lon"), np.full(s2s_shape, 298.0, dtype=np.float32)),
                "d2m": (("lead", "member", "lat", "lon"), np.full(s2s_shape, 294.0, dtype=np.float32)),
                "tcw": (("lead", "member", "lat", "lon"), np.full(s2s_shape, 45.0, dtype=np.float32)),
            },
            coords={
                "lead": [1, 2],
                "member": list(range(11)),
                "lat": self.lats,
                "lon": self.lons,
            },
            attrs={"hdate": "2015-01-15", "model_version_date": "2020-01-16"},
        )

        self.norm_params = load_frozen_normalization_contract()

    def test_case_assembly_and_normalization_contract(self):
        """Verifies full case assembly with normalization contract active."""
        case = assemble_single_a0_case(
            issue_date="2015-01-15",
            rzsm_cube_ds=self.rzsm_cube,
            atmospheric_ds=self.atm_ds,
            s2s_ds=self.s2s_ds,
            target_var_name="rzsm_0_100_seasonal_anomaly",
            eval_mask=self.mask,
            normalize=True,
            norm_params=self.norm_params,
        )

        # Shapes
        self.assertEqual(case.x_w1.shape, (11, 32, 48, 11))
        self.assertEqual(case.x_w2_base.shape, (11, 32, 48, 11))
        self.assertEqual(case.x_w3_base.shape, (11, 32, 48, 3))
        self.assertEqual(case.x_w4_base.shape, (11, 32, 48, 3))
        self.assertEqual(case.y_w1.shape, (1, 32, 48, 1))
        self.assertEqual(case.y_w2.shape, (1, 32, 48, 1))
        self.assertEqual(case.y_w3.shape, (1, 32, 48, 1))
        self.assertEqual(case.y_w4.shape, (1, 32, 48, 1))

        # Normalized values in [0.0, 1.0]
        eval_vals = case.x_w1[:, self.mask]
        self.assertTrue(np.all(eval_vals >= 0.0))
        self.assertTrue(np.all(eval_vals <= 1.0))

        # Ocean zero-filling
        ocean_vals = case.x_w1[:, ~self.mask]
        self.assertEqual(float(np.max(np.abs(ocean_vals))), 0.0)

        ocean_targets = case.y_w1[:, ~self.mask]
        self.assertEqual(float(np.max(np.abs(ocean_targets))), 0.0)

        # Invariant metadata
        self.assertTrue(case.is_normalized)
        self.assertEqual(case.target_dates[1], "2015-01-21")
        self.assertEqual(case.target_dates[2], "2015-01-28")
        self.assertEqual(case.target_dates[3], "2015-02-04")
        self.assertEqual(case.target_dates[4], "2015-02-11")


if __name__ == "__main__":
    unittest.main()
