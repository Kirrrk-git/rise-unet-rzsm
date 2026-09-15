"""
tests/test_14_validation_atmospheric_pipeline.py
------------------------------------------------
Automated unit test suite verifying the 2022-2023 validation atmospheric pipeline:
  1. Complete census of 210 validation forecast cycles (105 in 2022, 105 in 2023).
  2. End-to-end data flow through production atmospheric preprocessing path.
  3. Strict detection of missing predictor channels, missing dates, and active-domain NaNs.
  4. Safe normalization scaling adhering strictly to frozen training parameters.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

import importlib.util

spec = importlib.util.spec_from_file_location(
    "verify_atmos_pipeline",
    Path(__file__).resolve().parent.parent / "scripts" / "15_verify_validation_atmospheric_pipeline.py"
)
verify_atmos_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_atmos_module)

load_validation_manifest = verify_atmos_module.load_validation_manifest
load_eval_mask = verify_atmos_module.load_eval_mask
create_mock_validation_atmospheric_dataset = verify_atmos_module.create_mock_validation_atmospheric_dataset
verify_atmospheric_pipeline = verify_atmos_module.verify_atmospheric_pipeline
EXPECTED_ATM_VARS = verify_atmos_module.EXPECTED_ATM_VARS
EXPECTED_GRID_SHAPE = verify_atmos_module.EXPECTED_GRID_SHAPE
EXPECTED_ACTIVE_CELLS = verify_atmos_module.EXPECTED_ACTIVE_CELLS
from src.data.case_builder import load_frozen_normalization_contract


class TestValidationAtmosphericPipeline(unittest.TestCase):
    """Test suite for validation atmospheric pipeline preflight."""

    def setUp(self):
        self.val_df = load_validation_manifest()
        self.norm_contract = load_frozen_normalization_contract()
        self.eval_mask = load_eval_mask()
        self.val_dates = pd.to_datetime(self.val_df["issue_date"].values).tolist()
        self.mock_ds = create_mock_validation_atmospheric_dataset(self.val_dates)

    def test_val_manifest_census_and_yearly_accounting(self):
        """Verifies exact census: 210 cycles, 105 in 2022, 105 in 2023."""
        self.assertEqual(len(self.val_df), 210)
        dt = pd.to_datetime(self.val_df["issue_date"])
        self.assertEqual((dt.dt.year == 2022).sum(), 105)
        self.assertEqual((dt.dt.year == 2023).sum(), 105)

    def test_eval_mask_geometry_and_cells(self):
        """Verifies active evaluation mask is (32, 48) with exactly 126 land cells."""
        self.assertEqual(self.eval_mask.shape, EXPECTED_GRID_SHAPE)
        self.assertEqual(int(self.eval_mask.sum()), EXPECTED_ACTIVE_CELLS)

    def test_validation_pipeline_clean_flow(self):
        """Verifies clean pass across all 210 validation cases with frozen normalization."""
        res = verify_atmospheric_pipeline(
            self.mock_ds, self.val_df, self.norm_contract, self.eval_mask, verbose=False
        )
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["dates_checked"], 210)
        self.assertEqual(len(res["missing_dates"]), 0)
        self.assertEqual(len(res["missing_vars"]), 0)
        self.assertEqual(res["nan_inf_count"], 0)
        self.assertTrue(res["shape_conforming"])
        self.assertTrue(res["channel_ordering_correct"])

    def test_missing_variable_detection(self):
        """Verifies hard failure if any of the 5 atmospheric variables is missing."""
        tampered_ds = self.mock_ds.drop_vars("pwat")
        res = verify_atmospheric_pipeline(
            tampered_ds, self.val_df, self.norm_contract, self.eval_mask, verbose=False
        )
        self.assertEqual(res["status"], "FAIL")
        self.assertIn("pwat", res["missing_vars"])

    def test_nan_in_active_cell_detection(self):
        """Verifies hard failure if a NaN is injected into an active evaluation land cell."""
        tampered_ds = self.mock_ds.copy(deep=True)
        # Find first active cell coordinate
        r, c = np.where(self.eval_mask)
        # Inject NaN into first active cell on first day for variable tmax
        tampered_ds["tmax"].values[0, r[0], c[0]] = np.nan
        res = verify_atmospheric_pipeline(
            tampered_ds, self.val_df, self.norm_contract, self.eval_mask, verbose=False
        )
        self.assertEqual(res["status"], "FAIL")
        self.assertGreater(res["nan_inf_count"], 0)

    def test_missing_date_detection(self):
        """Verifies failure if any validation issue date is missing from atmospheric data."""
        # Drop year 2023 dates
        trimmed_ds = self.mock_ds.sel(time=slice("2022-01-01", "2022-12-31"))
        res = verify_atmospheric_pipeline(
            trimmed_ds, self.val_df, self.norm_contract, self.eval_mask, verbose=False
        )
        self.assertEqual(res["status"], "FAIL")
        self.assertEqual(len(res["missing_dates"]), 105)


if __name__ == "__main__":
    unittest.main()
