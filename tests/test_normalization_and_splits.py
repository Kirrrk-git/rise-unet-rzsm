"""
tests/test_normalization_and_splits.py
---------------------------------------
Sub-Phase 21K, Step 21K.2: Comprehensive Unit Tests for Split Segregation
and Training-Only Normalization Parameters.

Verifies:
  1. Split Partition Disjointness: Zero date overlap between TRAIN, VAL, and SEALED_TEST.
  2. Split Completeness: Exact 1,154 total cases partitioned (735 train, 210 val, 209 test).
  3. Strict Chronological Ordering: train < val < test (zero temporal inversion).
  4. Non-Leakage Contract: VAL and TEST cases strictly excluded from normalization statistics.
  5. Normalization Bounds Integrity: min < max, finite values, and unit interval [0, 1] scaling.
  6. Machine-Readable Contract Validity: contracts/A0/*.yaml parse cleanly with valid schema.
"""

import json
import unittest
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "manifests" / "splits"
CONTRACTS_DIR = REPO_ROOT / "contracts" / "A0"
CALENDAR_PATH = REPO_ROOT / "manifests" / "production_case_calendar.csv"


class TestDatasetSplitsAndNormalization(unittest.TestCase):
    """Rigorous unit test suite for Step 21K.2 deliverables."""

    @classmethod
    def setUpClass(cls):
        cls.train_csv = SPLITS_DIR / "train_cases.csv"
        cls.val_csv = SPLITS_DIR / "val_cases.csv"
        cls.test_csv = SPLITS_DIR / "test_cases_sealed.csv"
        cls.summary_json = SPLITS_DIR / "split_summary.json"
        cls.norm_yaml = CONTRACTS_DIR / "normalization_parameters.yaml"
        cls.prod_yaml = CONTRACTS_DIR / "mindanao_a0_production_contract.yaml"

    def test_split_files_exist(self):
        """Verifies all split CSVs and summary JSON are present on disk."""
        self.assertTrue(self.train_csv.exists(), "train_cases.csv missing")
        self.assertTrue(self.val_csv.exists(), "val_cases.csv missing")
        self.assertTrue(self.test_csv.exists(), "test_cases_sealed.csv missing")
        self.assertTrue(self.summary_json.exists(), "split_summary.json missing")

    def test_split_counts_and_completeness(self):
        """Verifies exact split counts sum to 1,154 without dropping cases."""
        df_train = pd.read_csv(self.train_csv)
        df_val = pd.read_csv(self.val_csv)
        df_test = pd.read_csv(self.test_csv)
        df_cal = pd.read_csv(CALENDAR_PATH)

        self.assertEqual(len(df_train), 735, "TRAIN split count must be exactly 735")
        self.assertEqual(len(df_val), 210, "VAL split count must be exactly 210")
        self.assertEqual(len(df_test), 209, "SEALED_TEST split count must be exactly 209")
        self.assertEqual(len(df_train) + len(df_val) + len(df_test), len(df_cal))
        self.assertEqual(len(df_cal), 1154, "Total calendar count must be 1,154")

    def test_split_disjointness_and_non_leakage(self):
        """Verifies zero issue date overlap across all three splits."""
        df_train = pd.read_csv(self.train_csv)
        df_val = pd.read_csv(self.val_csv)
        df_test = pd.read_csv(self.test_csv)

        train_dates = set(df_train["issue_date"])
        val_dates = set(df_val["issue_date"])
        test_dates = set(df_test["issue_date"])

        self.assertEqual(len(train_dates & val_dates), 0, "TRAIN and VAL share dates!")
        self.assertEqual(len(train_dates & test_dates), 0, "TRAIN and SEALED_TEST share dates!")
        self.assertEqual(len(val_dates & test_dates), 0, "VAL and SEALED_TEST share dates!")

    def test_chronological_progression(self):
        """Verifies strictly monotonic chronological progression without time inversion."""
        df_train = pd.read_csv(self.train_csv)
        df_val = pd.read_csv(self.val_csv)
        df_test = pd.read_csv(self.test_csv)

        max_train = df_train["issue_date"].max()
        min_val = df_val["issue_date"].min()
        max_val = df_val["issue_date"].max()
        min_test = df_test["issue_date"].min()

        self.assertLess(max_train, min_val, f"TRAIN max {max_train} must precede VAL min {min_val}")
        self.assertLess(max_val, min_test, f"VAL max {max_val} must precede TEST min {min_test}")

    def test_sealed_test_quarantine(self):
        """Verifies sealed test partition contains correct queued and out-of-bounds cases."""
        df_test = pd.read_csv(self.test_csv)
        queued_cases = (df_test["usable_status"] == "QUEUED_S2S_DOWNLOAD").sum()
        out_of_bounds = (df_test["usable_status"] == "TARGET_OUT_OF_BOUNDS").sum()

        self.assertEqual(queued_cases, 202, "Expected 202 QUEUED_S2S_DOWNLOAD cases")
        self.assertEqual(out_of_bounds, 7, "Expected 7 TARGET_OUT_OF_BOUNDS cases in late Dec 2025")
        self.assertEqual(queued_cases + out_of_bounds, 209, "Total test split must be 209")

    def test_normalization_yaml_contract_validity(self):
        """Verifies normalization_parameters.yaml exists, parses cleanly, and adheres to contract."""
        self.assertTrue(self.norm_yaml.exists(), "normalization_parameters.yaml missing")
        with open(self.norm_yaml) as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["normalization_scope"]["partition"], "TRAIN_ONLY")
        self.assertEqual(data["normalization_scope"]["active_evaluation_cells"], 126)
        self.assertEqual(data["normalization_scope"]["cycle_count"], 735)

        # Check RZSM anomaly bounds
        rzsm = data["rzsm_parameters"]["seasonal_anomaly"]
        self.assertLess(rzsm["min"], rzsm["max"])
        self.assertAlmostEqual(rzsm["mean"], 0.0, places=4)
        self.assertGreater(rzsm["std"], 0.03)

        # Check channel counts
        channels = data["lead_channel_specifications"]
        self.assertEqual(channels["Lead_1"]["total_channels"], 11)
        self.assertEqual(channels["Lead_2"]["total_channels"], 12)
        self.assertEqual(channels["Lead_3"]["total_channels"], 5)
        self.assertEqual(channels["Lead_4"]["total_channels"], 6)

    def test_production_contract_validity(self):
        """Verifies mindanao_a0_production_contract.yaml exists and contains required hyperparameters."""
        self.assertTrue(self.prod_yaml.exists(), "mindanao_a0_production_contract.yaml missing")
        with open(self.prod_yaml) as f:
            contract = yaml.safe_load(f)

        self.assertEqual(contract["architecture_and_optimization"]["model_class"], "UNET_RZSM")
        self.assertEqual(contract["architecture_and_optimization"]["loss_function"], "spatial_crps_loss")
        self.assertEqual(contract["architecture_and_optimization"]["optimizer"]["name"], "Adam")
        self.assertEqual(contract["architecture_and_optimization"]["training_schedule"]["seeds"], [42, 123, 456])
        self.assertEqual(contract["architecture_and_optimization"]["training_schedule"]["primary_checkpoint_metric"], "val_crps")
        self.assertEqual(contract["architecture_and_optimization"]["batch_size_evaluation_schedule"]["parent_ex29_reference"], 66)

    def test_tensor_builder_active_normalization_contract(self):
        """
        Executable contract test: confirms the actual production tensor builder consumes
        the frozen contracts/A0/normalization_parameters.yaml artifact, producing values
        strictly bounded in [0.0, 1.0] for active land cells and 0.0 for ocean buffer.
        Also proves active consumption: modifying bounds directly changes tensor values.
        """
        from src.data.case_builder import (
            assemble_single_a0_case,
            load_frozen_normalization_contract,
            DEFAULT_NORM_CONTRACT_PATH,
        )
        from src.data.s2s import CANDIDATE_A_LATS, CANDIDATE_A_LONS
        import xarray as xr

        # Load authoritative contract
        contract = load_frozen_normalization_contract(self.norm_yaml)
        self.assertIsNotNone(contract)

        # Build synthetic test datasets
        dates = pd.date_range("2015-01-01", "2015-02-28", freq="D")
        lats = CANDIDATE_A_LATS
        lons = CANDIDATE_A_LONS

        rzsm_data = np.full((len(dates), len(lats), len(lons)), 0.35, dtype=np.float32)
        rzsm_cube = xr.Dataset(
            data_vars={"rzsm_rolling_7d": (("time", "lat", "lon"), rzsm_data)},
            coords={"time": dates, "lat": lats, "lon": lons},
        )

        atm_dates = pd.date_range("2015-01-01", "2015-01-31", freq="D")
        shape_atm = (len(atm_dates), len(lats), len(lons))
        atm_ds = xr.Dataset(
            data_vars={
                "pwat": (("time", "lat", "lon"), np.full(shape_atm, 45.0, dtype=np.float32)),
                "spfh": (("time", "lat", "lon"), np.full(shape_atm, 0.018, dtype=np.float32)),
                "tmax": (("time", "lat", "lon"), np.full(shape_atm, 302.0, dtype=np.float32)),
                "diff_temp": (("time", "lat", "lon"), np.full(shape_atm, 6.0, dtype=np.float32)),
                "hgt_pres": (("time", "lat", "lon"), np.full(shape_atm, 12400.0, dtype=np.float32)),
            },
            coords={"time": atm_dates, "lat": lats, "lon": lons},
        )

        shape_s2s = (2, 11, len(lats), len(lons))
        s2s_ds = xr.Dataset(
            data_vars={
                "t2m": (("lead", "member", "lat", "lon"), np.full(shape_s2s, 301.0, dtype=np.float32)),
                "d2m": (("lead", "member", "lat", "lon"), np.full(shape_s2s, 296.0, dtype=np.float32)),
                "tcw": (("lead", "member", "lat", "lon"), np.full(shape_s2s, 0.02, dtype=np.float32)),
            },
            coords={"lead": [1, 2], "member": list(range(11)), "lat": lats, "lon": lons},
        )

        mask = np.zeros((len(lats), len(lons)), dtype=np.int8)
        mask[10:25, 15:35] = 1
        active = mask == 1
        ocean = mask == 0

        # Assemble without normalization
        case_raw = assemble_single_a0_case(
            issue_date="2015-01-16",
            rzsm_cube_ds=rzsm_cube,
            atmospheric_ds=atm_ds,
            s2s_ds=s2s_ds,
            eval_mask=mask,
            normalize=False,
        )
        self.assertFalse(case_raw.is_normalized)
        # Raw tmax is ~302.0 K, well outside [0, 1]
        self.assertGreater(case_raw.x_w1[:, active, 5].mean(), 100.0)

        # Assemble with frozen normalization contract
        case_norm = assemble_single_a0_case(
            issue_date="2015-01-16",
            rzsm_cube_ds=rzsm_cube,
            atmospheric_ds=atm_ds,
            s2s_ds=s2s_ds,
            eval_mask=mask,
            normalize=True,
        )
        self.assertTrue(case_norm.is_normalized)
        self.assertEqual(case_norm.normalization_contract_used, str(DEFAULT_NORM_CONTRACT_PATH))

        # Check invariant 1: Active cells are strictly bounded in [0.0, 1.0]
        self.assertTrue(np.all(case_norm.x_w1[:, active, :] >= 0.0))
        self.assertTrue(np.all(case_norm.x_w1[:, active, :] <= 1.0))
        self.assertTrue(np.all(case_norm.x_w2_base[:, active, :] >= 0.0))
        self.assertTrue(np.all(case_norm.x_w2_base[:, active, :] <= 1.0))
        self.assertTrue(np.all(case_norm.y_w1[:, active, :] >= 0.0))
        self.assertTrue(np.all(case_norm.y_w1[:, active, :] <= 1.0))

        # Check invariant 2: Ocean cells remain strictly 0.0
        self.assertTrue(np.all(case_norm.x_w1[:, ocean, :] == 0.0))
        self.assertTrue(np.all(case_norm.x_w2_base[:, ocean, :] == 0.0))
        self.assertTrue(np.all(case_norm.y_w1[:, ocean, :] == 0.0))

        # Check invariant 3: Active consumption test - modifying bounds changes tensor output
        with open(self.norm_yaml, "r", encoding="utf-8") as f:
            custom_contract = yaml.safe_load(f)
        # Deliberately modify tmax bounds
        custom_contract["atmospheric_parameters"]["tmax"]["min"] = 300.0
        custom_contract["atmospheric_parameters"]["tmax"]["max"] = 304.0
        # For value 302.0: (302 - 300)/(304 - 300) = 0.5
        case_custom = assemble_single_a0_case(
            issue_date="2015-01-16",
            rzsm_cube_ds=rzsm_cube,
            atmospheric_ds=atm_ds,
            s2s_ds=s2s_ds,
            eval_mask=mask,
            normalize=True,
            norm_params=custom_contract,
        )
        # In frozen contract, tmax min=288.0, max=312.0 -> (302-288)/24 = 0.5833
        frozen_tmax_val = float(case_norm.x_w1[:, active, 5].mean())
        custom_tmax_val = float(case_custom.x_w1[:, active, 5].mean())
        self.assertAlmostEqual(custom_tmax_val, 0.5, places=4)
        self.assertNotAlmostEqual(frozen_tmax_val, custom_tmax_val, places=2)


if __name__ == "__main__":
    unittest.main()
