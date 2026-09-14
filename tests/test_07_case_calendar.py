"""
Unit Test Suite: Production Case Calendar Verification

Authoritative Reference:
- Master Plan: mindanao_adaptation_master_plan.md
- Operational Cycle Calendar: ECMWF CY48R1 Schedule-Referenced Forecast Origin Calendar
- Parent Study: Lesinger & Tian (2025), Nature Communications, DOI: 10.1038/s41467-025-62761-3
"""

import csv
import datetime
import unittest
from pathlib import Path

from src.data.calendar import export_case_calendar, generate_operational_cycles

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCaseCalendarIntegrity(unittest.TestCase):
    """Rigorous verification of the production case calendar."""

    @classmethod
    def setUpClass(cls):
        cls.calendar_path = REPO_ROOT / "manifests" / "production_case_calendar.csv"
        if not cls.calendar_path.exists():
            export_case_calendar(cls.calendar_path)

        cls.rows = []
        with open(cls.calendar_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cls.rows = list(reader)

    def test_calendar_total_cycles_and_indexing(self):
        """Asserts exactly 1,154 operational cycles are indexed monotonically."""
        self.assertEqual(len(self.rows), 1154, "Production calendar must contain exactly 1,154 cycles.")
        for idx, r in enumerate(self.rows, start=1):
            self.assertEqual(int(r["cycle_index"]), idx, f"Row index mismatch at row {idx}")
            self.assertTrue(r["case_id"].startswith("CASE_"), f"Malformed case_id: {r['case_id']}")

    def test_split_temporal_boundaries(self):
        """Verifies strict adherence to temporal splits (Train 15-21, Val 22-23, Test 24-25)."""
        train_rows = [r for r in self.rows if r["split"] == "TRAIN"]
        val_rows = [r for r in self.rows if r["split"] == "VAL"]
        test_rows = [r for r in self.rows if r["split"] == "SEALED_TEST"]

        self.assertEqual(len(train_rows), 735, "Training split must have exactly 735 cycles (2015-2021).")
        self.assertEqual(len(val_rows), 210, "Validation split must have exactly 210 cycles (2022-2023).")
        self.assertEqual(len(test_rows), 209, "Sealed test split must have exactly 209 cycles (2024-2025).")

        for r in train_rows:
            year = int(r["issue_date"].split("-")[0])
            self.assertTrue(2015 <= year <= 2021, f"Train row {r['case_id']} outside 2015-2021: {year}")

        for r in val_rows:
            year = int(r["issue_date"].split("-")[0])
            self.assertTrue(2022 <= year <= 2023, f"Val row {r['case_id']} outside 2022-2023: {year}")

        for r in test_rows:
            year = int(r["issue_date"].split("-")[0])
            self.assertTrue(2024 <= year <= 2025, f"Test row {r['case_id']} outside 2024-2025: {year}")

    def test_deterministic_lag_and_target_offsets(self):
        """Asserts exact mathematical date arithmetic for all 1,154 cycles."""
        for r in self.rows:
            t0 = datetime.date.fromisoformat(r["issue_date"])
            
            # Antecedent lags
            lag_1 = datetime.date.fromisoformat(r["lag_1d_date"])
            lag_7 = datetime.date.fromisoformat(r["lag_7d_date"])
            lag_14 = datetime.date.fromisoformat(r["lag_14d_date"])
            self.assertEqual(lag_1, t0 - datetime.timedelta(days=1))
            self.assertEqual(lag_7, t0 - datetime.timedelta(days=7))
            self.assertEqual(lag_14, t0 - datetime.timedelta(days=14))

            # Verification targets
            tgt_1 = datetime.date.fromisoformat(r["target_w1_date"])
            tgt_2 = datetime.date.fromisoformat(r["target_w2_date"])
            tgt_3 = datetime.date.fromisoformat(r["target_w3_date"])
            tgt_4 = datetime.date.fromisoformat(r["target_w4_date"])
            self.assertEqual(tgt_1, t0 + datetime.timedelta(days=6))
            self.assertEqual(tgt_2, t0 + datetime.timedelta(days=13))
            self.assertEqual(tgt_3, t0 + datetime.timedelta(days=20))
            self.assertEqual(tgt_4, t0 + datetime.timedelta(days=27))

    def test_boundary_clipping_isolation(self):
        """Asserts that exactly 7 cycles exceeding the 2025-12-31 cube are flagged TARGET_OUT_OF_BOUNDS."""
        oob_rows = [r for r in self.rows if r["usable_status"] == "TARGET_OUT_OF_BOUNDS"]
        self.assertEqual(len(oob_rows), 7, "Exactly 7 trailing cycles in Dec 2025 must be TARGET_OUT_OF_BOUNDS.")
        for r in oob_rows:
            tgt_4 = datetime.date.fromisoformat(r["target_w4_date"])
            self.assertGreater(tgt_4, datetime.date(2025, 12, 31), f"Row {r['case_id']} flagged OOB but target within cube.")

    def test_ecmwf_operational_schedule_rules(self):
        """Asserts that cycles follow the ECMWF CY48R1 operational reference calendar schedule."""
        cycles = generate_operational_cycles(2015, 2025)
        self.assertEqual(len(cycles), 1154, "Total operational cycles must equal exactly 1,154.")

        # Annual breakdown: 105 per year for 2015-2024, 104 for 2025
        year_counts = {}
        for c in cycles:
            y = int(c["hyear"])
            year_counts[y] = year_counts.get(y, 0) + 1

        for y in range(2015, 2025):
            self.assertEqual(year_counts[y], 105, f"Year {y} must have exactly 105 cycles.")
        self.assertEqual(year_counts[2025], 104, "Year 2025 must have exactly 104 cycles.")

        # In operational years 2024 and 2025, every issue date is a Monday (0) or Thursday (3)
        for c in cycles:
            d = datetime.date.fromisoformat(c["cycle_date_str"])
            if d.year in (2024, 2025):
                self.assertIn(d.weekday(), (0, 3), f"Operational cycle {d} in {d.year} is not Mon/Thu")


if __name__ == "__main__":
    unittest.main()
