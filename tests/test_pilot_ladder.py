"""
test_pilot_ladder.py
--------------------
Automated unit tests for Sub-Phase 21G (8-Case Pilot Ladder & Manifest Integrity).
Verifies:
1. Member-level manifest schema and 88-row completeness (8 cases x 11 members).
2. Summary manifest schema and 8-row completeness.
3. Zero-defect tensor shapes: X_w1 (11, 32, 48, 11), Y_w1..Y_w4 (1, 32, 48, 1).
4. Verified parent target indexing offsets L = [6, 13, 20, 27].
5. Active evaluation cell census (126 cells) with 0 NaNs and 0 Infs.
6. Ocean buffer cells (1,410 cells) strictly 0.0-filled.
7. SHA-256 checksum consistency between serialized files and manifest entries.
"""

import unittest
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

class TestPilotLadder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.members_csv = cls.repo_root / "manifests" / "cases_pilot_v001.csv"
        cls.summary_csv = cls.repo_root / "manifests" / "cases_pilot_summary_v001.csv"
        cls.cases_dir = cls.repo_root / "processed" / "cases" / "pilot"
        cls.mask_path = cls.repo_root / "processed" / "grid" / "mindanao_eval_mask_025.nc"

        cls.eval_mask = xr.open_dataset(cls.mask_path)["evaluation_mask"].values.astype(bool)

    def test_manifest_existence_and_row_counts(self):
        """Verify existence and row counts: 88 member rows, 8 summary rows."""
        self.assertTrue(self.members_csv.exists(), f"Missing {self.members_csv}")
        self.assertTrue(self.summary_csv.exists(), f"Missing {self.summary_csv}")

        df_members = pd.read_csv(self.members_csv)
        df_summary = pd.read_csv(self.summary_csv)

        self.assertEqual(len(df_members), 88, f"Expected 88 rows in member manifest, got {len(df_members)}")
        self.assertEqual(len(df_summary), 8, f"Expected 8 rows in summary manifest, got {len(df_summary)}")

        # Check unique case IDs
        unique_cases = df_members["case_id"].unique()
        self.assertEqual(len(unique_cases), 8)
        self.assertTrue((df_members["status"] == "VALID").all())
        self.assertTrue((df_summary["status"] == "VALID").all())

    def test_member_level_structure(self):
        """Verify each case has exactly 11 members (0 to 10) with 1 CF and 10 PF."""
        df_members = pd.read_csv(self.members_csv)
        for case_id, group in df_members.groupby("case_id"):
            self.assertEqual(len(group), 11)
            members = sorted(group["ensemble_member"].tolist())
            self.assertEqual(members, list(range(11)))
            cf_count = (group["member_type"] == "CF").sum()
            pf_count = (group["member_type"] == "PF").sum()
            self.assertEqual(cf_count, 1)
            self.assertEqual(pf_count, 10)

    def test_serialized_case_tensors_and_checksums(self):
        """Verify tensor shapes, targets, evaluation mask census, and checksums."""
        df_summary = pd.read_csv(self.summary_csv)

        for _, row in df_summary.iterrows():
            case_id = row["case_id"]
            npz_path = self.cases_dir / f"{case_id}.npz"
            self.assertTrue(npz_path.exists(), f"Missing {npz_path}")

            # Verify Checksum
            hasher = hashlib.sha256()
            with open(npz_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            calc_hash = hasher.hexdigest()
            self.assertEqual(calc_hash, row["sha256_checksum"])

            # Load NPZ
            data = np.load(npz_path)
            x_w1 = data["x_w1"]
            x_w2_base = data["x_w2_base"]
            x_w3_base = data["x_w3_base"]
            x_w4_base = data["x_w4_base"]
            y_w1 = data["y_w1"]
            y_w2 = data["y_w2"]
            y_w3 = data["y_w3"]
            y_w4 = data["y_w4"]

            # Shapes
            self.assertEqual(x_w1.shape, (11, 32, 48, 11))
            self.assertEqual(x_w2_base.shape, (11, 32, 48, 11))
            self.assertEqual(x_w3_base.shape, (11, 32, 48, 3))
            self.assertEqual(x_w4_base.shape, (11, 32, 48, 3))
            self.assertEqual(y_w1.shape, (1, 32, 48, 1))
            self.assertEqual(y_w2.shape, (1, 32, 48, 1))
            self.assertEqual(y_w3.shape, (1, 32, 48, 1))
            self.assertEqual(y_w4.shape, (1, 32, 48, 1))

            # Quality census over 126 cells
            for arr in [x_w1, x_w2_base, x_w3_base, x_w4_base, y_w1, y_w2, y_w3, y_w4]:
                eval_vals = arr[:, self.eval_mask]
                ocean_vals = arr[:, ~self.eval_mask]
                self.assertEqual(np.isnan(eval_vals).sum(), 0)
                self.assertEqual(np.isinf(eval_vals).sum(), 0)
                self.assertEqual(float(np.max(np.abs(ocean_vals))), 0.0)

            # Target date verification L = [6, 13, 20, 27]
            t0 = pd.Timestamp(row["issue_time"])
            self.assertEqual(row["target_w1_date"], (t0 + pd.Timedelta(days=6)).strftime("%Y-%m-%d"))
            self.assertEqual(row["target_w2_date"], (t0 + pd.Timedelta(days=13)).strftime("%Y-%m-%d"))
            self.assertEqual(row["target_w3_date"], (t0 + pd.Timedelta(days=20)).strftime("%Y-%m-%d"))
            self.assertEqual(row["target_w4_date"], (t0 + pd.Timedelta(days=27)).strftime("%Y-%m-%d"))

    def test_grand_total_evaluation_domain_census(self):
        """Verify grand total census across all 8 cases equals exactly 314,496 evaluation points."""
        df_summary = pd.read_csv(self.summary_csv)
        total_eval_points = 0
        for _, row in df_summary.iterrows():
            case_id = row["case_id"]
            npz_path = self.cases_dir / f"{case_id}.npz"
            data = np.load(npz_path)
            # x_w1 (11*11=121), x_w2_base (11*11=121), x_w3_base (11*3=33), x_w4_base (11*3=33), y_w1..w4 (4*1=4)
            # Total channels = 312 channels * 126 evaluation cells = 39,312 points per case.
            case_eval_points = 0
            for arr in [data["x_w1"], data["x_w2_base"], data["x_w3_base"], data["x_w4_base"],
                        data["y_w1"], data["y_w2"], data["y_w3"], data["y_w4"]]:
                eval_vals = arr[:, self.eval_mask]
                case_eval_points += eval_vals.size
            self.assertEqual(case_eval_points, 39312)
            total_eval_points += case_eval_points
        self.assertEqual(total_eval_points, 314496)

if __name__ == "__main__":
    unittest.main()
