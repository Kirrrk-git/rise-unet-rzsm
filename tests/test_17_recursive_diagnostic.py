"""
tests/test_17_recursive_diagnostic.py
--------------------------------------
Hermetic unit tests for Phase 23 Recursive Degradation Diagnostic framework.
Tests:
  1. Oracle counterfactual prior assembly and tensor shape invariants.
  2. Integration of Oracle priors with prepare_case_lead_tensors.
  3. Boundary clipping rule (y' = clip(y + eps, 0, 1)) and cell-clipping census.
  4. Case-level metric calculation (MAE, RMSE, ACC, exact CRPS, proxy CRPS).
  5. Paired difference calculations and Cohen's d_z effect size.
  6. Moving-block bootstrap (MBB) point estimates and confidence intervals.
  7. Paired Wilcoxon signed-rank test behavior.
  8. Step-down Holm-Bonferroni multiple testing correction.
  9. Authoritative Gate 2 GO / NO-GO rule evaluator.
  10. Cross-contract reconciliation between Phase 23 Contract and Baseline Contract A0.
"""

import unittest
from pathlib import Path
import numpy as np
import yaml

from src.data.tf_dataset import (
    ENSEMBLE_MEMBERS,
    GRID_HEIGHT,
    GRID_WIDTH,
    LEAD_CHANNELS,
    prepare_case_lead_tensors,
)
from src.evaluation.recursive_diagnostic import (
    assemble_oracle_priors,
    apply_boundary_clipped_perturbation,
    compute_case_level_metrics,
    compute_paired_differences,
    moving_block_bootstrap,
    wilcoxon_paired_test,
    holm_bonferroni_correction,
    compute_cohens_d_z,
    check_perturbation_monotonicity,
    evaluate_gate2_rules,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestRecursiveDiagnosticFramework(unittest.TestCase):
    """Test suite for Phase 23 diagnostic mathematical and procedural components."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        # Mock evaluation mask with exactly 126 active cells
        self.eval_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        flat_indices = self.rng.choice(GRID_HEIGHT * GRID_WIDTH, size=126, replace=False)
        self.eval_mask.flat[flat_indices] = True
        self.assertEqual(int(np.sum(self.eval_mask)), 126)

        # Mock case data dictionary with valid shapes
        self.mock_case_data = {
            "x_w1": self.rng.uniform(0.1, 0.9, size=(ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 11)).astype(np.float32),
            "x_w2_base": self.rng.uniform(0.1, 0.9, size=(ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 11)).astype(np.float32),
            "x_w3_base": self.rng.uniform(0.1, 0.9, size=(ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 3)).astype(np.float32),
            "x_w4_base": self.rng.uniform(0.1, 0.9, size=(ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 3)).astype(np.float32),
            "y_w1": self.rng.uniform(0.2, 0.8, size=(1, GRID_HEIGHT, GRID_WIDTH, 1)).astype(np.float32),
            "y_w2": self.rng.uniform(0.2, 0.8, size=(1, GRID_HEIGHT, GRID_WIDTH, 1)).astype(np.float32),
            "y_w3": self.rng.uniform(0.2, 0.8, size=(1, GRID_HEIGHT, GRID_WIDTH, 1)).astype(np.float32),
            "y_w4": self.rng.uniform(0.2, 0.8, size=(1, GRID_HEIGHT, GRID_WIDTH, 1)).astype(np.float32),
        }

    def test_assemble_oracle_priors_shapes_and_invariants(self):
        """Validates Oracle counterfactual prior assembly across Leads 1 to 4."""
        # Lead 1 has no recursive inputs
        priors_w1 = assemble_oracle_priors(self.mock_case_data, lead=1)
        self.assertEqual(priors_w1, {})

        # Lead 2 requires y_true_w1
        priors_w2 = assemble_oracle_priors(self.mock_case_data, lead=2)
        self.assertIn(1, priors_w2)
        self.assertEqual(priors_w2[1].shape, (ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 1))
        # Ensure all 11 ensemble members received exact identical ground truth values
        for m in range(1, ENSEMBLE_MEMBERS):
            np.testing.assert_array_equal(priors_w2[1][m], priors_w2[1][0])
        np.testing.assert_array_equal(priors_w2[1][0], self.mock_case_data["y_w1"][0])

        # Lead 3 requires y_true_w1 and y_true_w2
        priors_w3 = assemble_oracle_priors(self.mock_case_data, lead=3)
        self.assertEqual(sorted(list(priors_w3.keys())), [1, 2])
        self.assertEqual(priors_w3[2].shape, (ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 1))

        # Lead 4 requires y_true_w1, y_true_w2, and y_true_w3
        priors_w4 = assemble_oracle_priors(self.mock_case_data, lead=4)
        self.assertEqual(sorted(list(priors_w4.keys())), [1, 2, 3])
        self.assertEqual(priors_w4[3].shape, (ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 1))

    def test_oracle_priors_integration_with_prepare_case_lead_tensors(self):
        """Verifies that Oracle priors construct exact full tensors matching LEAD_CHANNELS."""
        for lead in (1, 2, 3, 4):
            oracle_priors = assemble_oracle_priors(self.mock_case_data, lead=lead)
            x_tensor, y_tensor = prepare_case_lead_tensors(
                case_data=self.mock_case_data,
                lead=lead,
                y_hat_prev=oracle_priors,
            )
            expected_c = LEAD_CHANNELS[lead]
            self.assertEqual(x_tensor.shape, (ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, expected_c))
            self.assertEqual(y_tensor.shape, (ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 1))

    def test_boundary_clipping_and_census(self):
        """Verifies boundary clipping rule y' = clip(y + eps, 0, 1) and active cell census."""
        # Create an array with edge values
        base_pred = np.zeros((ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 1), dtype=np.float32)
        base_pred.fill(0.95)

        # Test positive perturbation of +0.10: 0.95 + 0.10 = 1.05 -> should clip to 1.0
        clipped, count, pct = apply_boundary_clipped_perturbation(base_pred, epsilon=0.10, eval_mask=self.eval_mask)
        self.assertEqual(clipped.shape, base_pred.shape)
        self.assertTrue(np.all(clipped <= 1.0))
        self.assertTrue(np.all(clipped >= 0.0))
        # Since all cells were 0.95, all 11 * 126 active evaluations should have clipped
        self.assertEqual(count, ENSEMBLE_MEMBERS * 126)
        self.assertEqual(pct, 100.0)

        # Test negative perturbation of -0.05 on 0.95: 0.95 - 0.05 = 0.90 -> 0 clipped
        clipped_no, count_no, pct_no = apply_boundary_clipped_perturbation(base_pred, epsilon=-0.05, eval_mask=self.eval_mask)
        self.assertEqual(count_no, 0)
        self.assertEqual(pct_no, 0.0)
        np.testing.assert_allclose(clipped_no, 0.90, atol=1e-6)

    def test_case_level_metrics_calculation(self):
        """Verifies case-level MAE, RMSE, ACC, exact CRPS, and proxy CRPS calculation."""
        y_true = np.full((1, GRID_HEIGHT, GRID_WIDTH, 1), 0.5, dtype=np.float32)
        # Prediction with small spread
        y_pred = np.empty((ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 1), dtype=np.float32)
        for m in range(ENSEMBLE_MEMBERS):
            y_pred[m] = 0.5 + 0.01 * (m - 5)

        metrics = compute_case_level_metrics(y_true, y_pred, self.eval_mask)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("acc", metrics)
        self.assertIn("exact_crps", metrics)
        self.assertIn("spatial_crps_proxy", metrics)

        self.assertGreaterEqual(metrics["mae"], 0.0)
        self.assertGreaterEqual(metrics["rmse"], 0.0)
        self.assertGreaterEqual(metrics["exact_crps"], 0.0)

    def test_paired_differences_and_cohens_d_z(self):
        """Verifies paired difference calculation and Cohen's d_z effect size."""
        rec_errors = [0.060, 0.065, 0.070, 0.075, 0.080]
        ora_errors = [0.050, 0.052, 0.055, 0.058, 0.060]
        diffs = compute_paired_differences(rec_errors, ora_errors)
        self.assertEqual(len(diffs), 5)
        self.assertTrue(np.all(diffs > 0.0))

        dz = compute_cohens_d_z(diffs)
        self.assertGreater(dz, 0.0)
        # Expected d_z = mean(diff) / std(diff)
        expected_dz = np.mean(diffs) / np.std(diffs, ddof=1)
        self.assertAlmostEqual(dz, expected_dz, places=6)

    def test_moving_block_bootstrap_ci(self):
        """Verifies Moving-Block Bootstrap point estimate and confidence intervals."""
        synthetic_diffs = self.rng.normal(loc=0.005, scale=0.002, size=194)
        mean_val, ci_lo, ci_hi, p_boot = moving_block_bootstrap(
            synthetic_diffs,
            block_length=4,
            n_boot=1000,
            alpha=0.05,
            seed=42,
        )
        self.assertAlmostEqual(mean_val, float(np.mean(synthetic_diffs)), places=6)
        self.assertLess(ci_lo, mean_val)
        self.assertGreater(ci_hi, mean_val)
        self.assertLess(p_boot, 0.05)

    def test_wilcoxon_paired_test(self):
        """Verifies paired Wilcoxon test detects positive degradation with statistical significance."""
        # Strongly positive differences
        pos_diffs = self.rng.normal(loc=0.01, scale=0.002, size=100)
        stat, pval = wilcoxon_paired_test(pos_diffs)
        self.assertLess(pval, 0.001)

        # Zero-centered symmetric differences
        zero_diffs = self.rng.normal(loc=0.0, scale=0.01, size=100)
        stat_z, pval_z = wilcoxon_paired_test(zero_diffs)
        self.assertGreater(pval_z, 0.05)

    def test_holm_bonferroni_correction(self):
        """Verifies step-down Holm-Bonferroni multiple testing adjustment."""
        p_vals = {
            "W2": 0.04,
            "W3": 0.01,
            "W4": 0.001,
        }
        corrected = holm_bonferroni_correction(p_vals, alpha=0.05)

        # Sorted raw order: W4 (0.001, m=3) -> 0.003
        #                   W3 (0.01, m=2)  -> 0.02
        #                   W2 (0.04, m=1)  -> 0.04
        self.assertEqual(corrected["W4"]["rank"], 1)
        self.assertAlmostEqual(corrected["W4"]["adj_p"], 0.003, places=5)
        self.assertTrue(corrected["W4"]["rejected"])

        self.assertEqual(corrected["W3"]["rank"], 2)
        self.assertAlmostEqual(corrected["W3"]["adj_p"], 0.02, places=5)
        self.assertTrue(corrected["W3"]["rejected"])

        self.assertEqual(corrected["W2"]["rank"], 3)
        self.assertAlmostEqual(corrected["W2"]["adj_p"], 0.04, places=5)
        self.assertTrue(corrected["W2"]["rejected"])

    def test_evaluate_gate2_rules_go_and_no_go(self):
        """Verifies Gate 2 rule evaluator correctly produces GO vs NO-GO verdicts."""
        # Case A: Clear GO scenario
        lead_summary_go = {
            2: {"rejected": True, "cohens_d_z": 0.25, "relative_gap_pct": 2.5},
            3: {"rejected": True, "cohens_d_z": 0.35, "relative_gap_pct": 3.8},
            4: {"rejected": True, "cohens_d_z": 0.40, "relative_gap_pct": 4.5},
        }
        res_go = evaluate_gate2_rules(lead_summary_go, seed_consistency=True, divergence_monotonic=True)
        self.assertEqual(res_go["verdict"], "GO")
        self.assertTrue(res_go["criteria"]["GO-1_statistical_significance"]["passed"])
        self.assertTrue(res_go["criteria"]["GO-2_practical_effect_size"]["passed"])

        # Case B: NO-GO due to lack of statistical significance
        lead_summary_no_sig = {
            2: {"rejected": False, "cohens_d_z": 0.10, "relative_gap_pct": 0.5},
            3: {"rejected": False, "cohens_d_z": 0.12, "relative_gap_pct": 0.8},
            4: {"rejected": False, "cohens_d_z": 0.14, "relative_gap_pct": 1.1},
        }
        res_no_sig = evaluate_gate2_rules(lead_summary_no_sig, seed_consistency=True, divergence_monotonic=True)
        self.assertEqual(res_no_sig["verdict"], "NO-GO")
        self.assertFalse(res_no_sig["criteria"]["GO-1_statistical_significance"]["passed"])

        # Case C: NO-GO due to lack of replicate consistency across seeds
        res_no_seed = evaluate_gate2_rules(lead_summary_go, seed_consistency=False, divergence_monotonic=True)
        self.assertEqual(res_no_seed["verdict"], "NO-GO")
        self.assertFalse(res_no_seed["criteria"]["GO-3_replicate_consistency"]["passed"])

    def test_check_perturbation_monotonicity_magnitude(self):
        """Verifies monotonicity evaluation on perturbation magnitude |eps| separately by sign."""
        # Case 1: Monotonic growth across both positive and negative realistic perturbations
        valid_div = {
            "3": {"0.05": 0.012, "0.10": 0.024, "-0.05": 0.011, "-0.10": 0.021, "0.25": 0.045, "0.50": 0.060},
            "4": {"0.05": 0.015, "0.10": 0.030, "-0.05": 0.014, "-0.10": 0.028, "0.25": 0.050, "0.50": 0.065},
        }
        is_mono, details = check_perturbation_monotonicity(valid_div, target_leads=(3, 4))
        self.assertTrue(is_mono)
        self.assertTrue(details["lead_3"]["passed"])
        self.assertTrue(details["lead_4"]["passed"])

        # Case 2: Positive is monotonic but negative is not (e.g. D(-0.10) <= D(-0.05))
        non_mono_div = {
            "3": {"0.05": 0.012, "0.10": 0.024, "-0.05": 0.015, "-0.10": 0.010},
            "4": {"0.05": 0.015, "0.10": 0.030, "-0.05": 0.018, "-0.10": 0.012},
        }
        is_mono_f, details_f = check_perturbation_monotonicity(non_mono_div, target_leads=(3, 4))
        self.assertFalse(is_mono_f)
        self.assertFalse(details_f["lead_3"]["passed"])
        self.assertFalse(details_f["lead_4"]["passed"])

    def test_contract_reconciliation_phase23_vs_a0_baseline(self):
        """Cross-checks PHASE_23_DIAGNOSTIC_CONTRACT against A0_Mindanao_Baseline_Contract."""
        p23_path = REPO_ROOT / "contracts" / "A0" / "PHASE_23_DIAGNOSTIC_CONTRACT.yaml"
        a0_path = REPO_ROOT / "contracts" / "A0" / "A0_Mindanao_Baseline_Contract.yaml"

        self.assertTrue(p23_path.is_file(), "PHASE_23_DIAGNOSTIC_CONTRACT.yaml must exist.")
        self.assertTrue(a0_path.is_file(), "A0_Mindanao_Baseline_Contract.yaml must exist.")

        with open(p23_path, "r", encoding="utf-8") as f:
            c23 = yaml.safe_load(f)
        with open(a0_path, "r", encoding="utf-8") as f:
            ca0 = yaml.safe_load(f)

        # 1. Validation denominator parity (194 cases)
        self.assertEqual(
            c23["evaluation_cohort"]["usable_validation_denominator"],
            ca0["cohort_accounting_ledger"]["validation_partition"]["usable_count"],
        )
        self.assertEqual(c23["evaluation_cohort"]["usable_validation_denominator"], 194)

        # 2. Active evaluation cells parity (126 cells)
        self.assertEqual(
            c23["evaluation_cohort"]["spatial_domain"]["active_evaluation_cells"],
            ca0["domain_and_geometry"]["active_evaluation_cells"],
        )
        self.assertEqual(c23["evaluation_cohort"]["spatial_domain"]["active_evaluation_cells"], 126)

        # 3. Sealed test quarantine parity (202 usable cases)
        self.assertEqual(
            c23["evaluation_cohort"]["sealed_test_quarantine"]["usable_denominator"],
            ca0["cohort_accounting_ledger"]["sealed_test_partition"]["usable_evaluation_denominator"],
        )
        self.assertEqual(c23["evaluation_cohort"]["sealed_test_quarantine"]["usable_denominator"], 202)

        # 4. Seeds parity
        self.assertEqual(
            sorted(c23["evaluated_models"]["replicate_seeds"]),
            sorted(ca0["production_checkpoints_manifest"]["seeds"]),
        )
        self.assertEqual(c23["evaluated_models"]["replicate_seeds"], [42, 123, 456])


if __name__ == "__main__":
    unittest.main()
