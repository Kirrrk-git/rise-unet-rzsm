"""
tests/test_tf_dataset.py
------------------------
Automated unit test suite verifying the A0 TensorFlow data pipeline, ensemble grouping semantics,
target broadcasting, CRPS loss mathematics, and checkpoint persistence (Sub-Phase 21H).
"""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.data.tf_dataset import (
    A0CaseBatchGenerator,
    validate_batch_size,
    prepare_case_lead_tensors,
    load_case_npz,
    crps2d_numpy,
    crps_exact_analytical,
    save_a0_checkpoint,
    restore_a0_checkpoint,
    ENSEMBLE_MEMBERS,
    GRID_HEIGHT,
    GRID_WIDTH,
    LEAD_CHANNELS,
    OUTPUT_HEADS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_CASES_DIR = REPO_ROOT / "processed" / "cases" / "pilot"


class TestBatchSizeValidation(unittest.TestCase):
    """Tests enforcement of the 11-member ensemble grouping constraint."""

    def test_valid_batch_sizes(self):
        # Explicitly verify the complete valid set implied by range(11, 121, 11):
        # {11, 22, 33, 44, 55, 66, 77, 88, 99, 110}
        complete_valid_set = list(range(11, 121, 11))
        for valid_b in complete_valid_set:
            try:
                validate_batch_size(valid_b)
            except ValueError as e:
                self.fail(f"Valid batch size {valid_b} raised unexpected error: {e}")

    def test_invalid_batch_sizes(self):
        # Representative invalid batch sizes: negative, zero, non-multiples of 11
        for invalid_b in [-22, -11, 0, 1, 5, 10, 12, 15, 20, 23, 30, 45, 65, 78, 100, 111]:
            with self.assertRaises(ValueError, msg=f"Should fail for batch_size={invalid_b}"):
                validate_batch_size(invalid_b)


class TestCaseTensorPreparation(unittest.TestCase):
    """Tests lead-specific channel concatenation and target broadcasting on real pilot cases."""

    def setUp(self):
        pilot_files = sorted(list(PILOT_CASES_DIR.glob("CASE_*.npz")))
        self.assertTrue(len(pilot_files) >= 1, "At least 1 pilot case required for testing.")
        self.sample_case_path = pilot_files[0]
        self.case_data = load_case_npz(self.sample_case_path)

    def test_lead_1_preparation(self):
        x, y = prepare_case_lead_tensors(self.case_data, lead=1)
        self.assertEqual(x.shape, (11, 32, 48, 11))
        self.assertEqual(y.shape, (11, 32, 48, 1))

        # Check that target is broadcast identically across all 11 ensemble members
        y_true_single = self.case_data["y_w1"][0]  # shape: (32, 48, 1)
        for m in range(11):
            np.testing.assert_array_equal(y[m], y_true_single)

    def test_lead_2_preparation_with_recursive(self):
        mock_y_hat_w1 = np.full((11, 32, 48, 1), 0.42, dtype=np.float32)
        x, y = prepare_case_lead_tensors(
            self.case_data, lead=2, y_hat_prev={1: mock_y_hat_w1}
        )
        self.assertEqual(x.shape, (11, 32, 48, 12))
        self.assertEqual(y.shape, (11, 32, 48, 1))
        np.testing.assert_allclose(x[:, :, :, 11:12], mock_y_hat_w1)

    def test_lead_3_preparation_with_recursive(self):
        mock_y_hat_w1 = np.full((11, 32, 48, 1), 0.40, dtype=np.float32)
        mock_y_hat_w2 = np.full((11, 32, 48, 1), 0.45, dtype=np.float32)
        x, y = prepare_case_lead_tensors(
            self.case_data, lead=3, y_hat_prev={1: mock_y_hat_w1, 2: mock_y_hat_w2}
        )
        self.assertEqual(x.shape, (11, 32, 48, 5))
        self.assertEqual(y.shape, (11, 32, 48, 1))
        np.testing.assert_allclose(x[:, :, :, 3:4], mock_y_hat_w1)
        np.testing.assert_allclose(x[:, :, :, 4:5], mock_y_hat_w2)

    def test_lead_4_preparation_with_recursive(self):
        mock_y_hat_w1 = np.full((11, 32, 48, 1), 0.38, dtype=np.float32)
        mock_y_hat_w2 = np.full((11, 32, 48, 1), 0.42, dtype=np.float32)
        mock_y_hat_w3 = np.full((11, 32, 48, 1), 0.46, dtype=np.float32)
        x, y = prepare_case_lead_tensors(
            self.case_data,
            lead=4,
            y_hat_prev={1: mock_y_hat_w1, 2: mock_y_hat_w2, 3: mock_y_hat_w3},
        )
        self.assertEqual(x.shape, (11, 32, 48, 6))
        self.assertEqual(y.shape, (11, 32, 48, 1))
        np.testing.assert_allclose(x[:, :, :, 3:4], mock_y_hat_w1)
        np.testing.assert_allclose(x[:, :, :, 4:5], mock_y_hat_w2)
        np.testing.assert_allclose(x[:, :, :, 5:6], mock_y_hat_w3)


class TestA0CaseBatchGenerator(unittest.TestCase):
    """Tests batch iteration, deep supervision dictionary structure, and case-level batching."""

    def setUp(self):
        self.case_paths = sorted(list(PILOT_CASES_DIR.glob("CASE_*.npz")))
        self.assertEqual(len(self.case_paths), 8, "Expected exactly 8 pilot cases in processed/cases/pilot/")

    def test_batch_size_11(self):
        gen = A0CaseBatchGenerator(
            case_paths=self.case_paths,
            lead=1,
            batch_size=11,
            shuffle=False,
        )
        self.assertEqual(len(gen), 8)

        batches = list(gen)
        self.assertEqual(len(batches), 8)

        for x_b, y_dict in batches:
            self.assertEqual(x_b.shape, (11, 32, 48, 11))
            self.assertEqual(x_b.dtype, np.float32)
            self.assertFalse(np.isnan(x_b).any())

            for head in OUTPUT_HEADS:
                self.assertIn(head, y_dict)
                self.assertEqual(y_dict[head].shape, (11, 32, 48, 1))
                self.assertEqual(y_dict[head].dtype, np.float32)
                self.assertFalse(np.isnan(y_dict[head]).any())

    def test_batch_size_22(self):
        gen = A0CaseBatchGenerator(
            case_paths=self.case_paths,
            lead=1,
            batch_size=22,
            shuffle=False,
        )
        self.assertEqual(len(gen), 4)

        batches = list(gen)
        self.assertEqual(len(batches), 4)

        for x_b, y_dict in batches:
            self.assertEqual(x_b.shape, (22, 32, 48, 11))
            for head in OUTPUT_HEADS:
                self.assertEqual(y_dict[head].shape, (22, 32, 48, 1))

    def test_case_level_shuffle_preserves_internal_member_order(self):
        # Even when shuffled, the 11 members belonging to each case must stay together
        gen_shuffled = A0CaseBatchGenerator(
            case_paths=self.case_paths,
            lead=1,
            batch_size=11,
            shuffle=True,
            seed=123,
        )
        for x_b, y_dict in gen_shuffled:
            y_b = y_dict["RZSM_output_1"]
            # All 11 members of a single batch should have identical target values
            for m in range(1, 11):
                np.testing.assert_array_equal(y_b[m], y_b[0])

    def test_end_to_end_ensemble_ordering_preservation(self):
        # Multi-case batch (B=22, exactly 2 cases)
        gen = A0CaseBatchGenerator(
            case_paths=self.case_paths[:2],
            lead=1,
            batch_size=22,
            shuffle=False,
        )
        x_b, y_dict = next(iter(gen))
        y_b = y_dict["RZSM_output_1"]

        # Case 1 occupies [0:11], Case 2 occupies [11:22]
        case1_target = y_b[0]
        case2_target = y_b[11]

        # Slices 0..10 must all match Case 1
        for m in range(11):
            np.testing.assert_array_equal(y_b[m], case1_target)
        # Slices 11..21 must all match Case 2
        for m in range(11, 22):
            np.testing.assert_array_equal(y_b[m], case2_target)

        # Case 1 and Case 2 targets must NOT be identical (distinct calendar weeks)
        self.assertFalse(np.array_equal(case1_target, case2_target))


class TestCRPSMathematicalEquivalence(unittest.TestCase):
    """Tests CRPS formulation against analytical reference calculations and target invariance."""

    def test_crps_exact_analytical_reconciliation(self):
        # Deterministic toy ensemble: M=11 members with known observation y=0.45
        y_obs = 0.45
        x_ens = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70], dtype=np.float32)
        M = 11

        # Hand-calculated analytical reference values:
        # MAE = mean(|x_m - y|) = 0.13636364
        # Pairwise diff = sum(|x_m - x_n|) / (2 * M^2) = 0.090909086
        # CRPS_exact = MAE - Pairwise = 0.045454554
        mae_ref = float(np.mean(np.abs(x_ens - y_obs)))
        diff_matrix = np.abs(x_ens[:, None] - x_ens[None, :])
        pairwise_ref = float(np.sum(diff_matrix) / (2.0 * M * M))
        crps_exact_ref = mae_ref - pairwise_ref

        # Compute via independent reference function
        crps_exact_calc = crps_exact_analytical(np.array([y_obs]), x_ens)
        self.assertAlmostEqual(crps_exact_calc, crps_exact_ref, places=6)
        self.assertAlmostEqual(crps_exact_calc, 0.045454554, places=6)

        # Author's spatial approximation: CRPS_exp = MAE - factor * std
        std_ref = float(np.std(x_ens))
        crps_author_ref = mae_ref - 0.08 * std_ref
        self.assertAlmostEqual(crps_author_ref, 0.12371453, places=6)

        # Grid evaluation via crps2d_numpy
        y_true_grid = np.full((11, 32, 48, 1), y_obs, dtype=np.float32)
        y_pred_grid = np.zeros((11, 32, 48, 1), dtype=np.float32)
        for m in range(11):
            y_pred_grid[m, :, :, 0] = x_ens[m]

        crps_approx_calc = crps2d_numpy(y_true_grid, y_pred_grid, factor=0.08)
        self.assertAlmostEqual(crps_approx_calc, crps_author_ref, places=6)

    def test_crps_target_broadcasting_invariance(self):
        # Verify that CRPS(y_single, y_pred) == CRPS(repeat(y_single, 11), y_pred) to machine precision
        rng = np.random.default_rng(42)
        y_single = rng.uniform(0.2, 0.6, size=(1, 32, 48, 1)).astype(np.float32)
        y_broadcast = np.repeat(y_single, 11, axis=0)
        y_pred = rng.uniform(0.2, 0.6, size=(11, 32, 48, 1)).astype(np.float32)

        crps_from_single = crps2d_numpy(y_single, y_pred, factor=0.08)
        crps_from_broadcast = crps2d_numpy(y_broadcast, y_pred, factor=0.08)
        self.assertEqual(crps_from_single, crps_from_broadcast)

        # Multi-case batch invariance (B=22, 2 cases)
        y_multi_single = rng.uniform(0.2, 0.6, size=(2, 32, 48, 1)).astype(np.float32)
        y_multi_broadcast = np.repeat(y_multi_single, 11, axis=0)
        y_multi_pred = rng.uniform(0.2, 0.6, size=(22, 32, 48, 1)).astype(np.float32)

        multi_single_score = crps2d_numpy(y_multi_single, y_multi_pred, factor=0.08)
        multi_broadcast_score = crps2d_numpy(y_multi_broadcast, y_multi_pred, factor=0.08)
        self.assertEqual(multi_single_score, multi_broadcast_score)

    def test_crps_zero_error_zero_spread(self):
        # Perfect prediction with zero ensemble spread: CRPS = 0 - 0 = 0
        y_true = np.ones((11, 32, 48, 1), dtype=np.float32)
        y_pred = np.ones((11, 32, 48, 1), dtype=np.float32)
        score = crps2d_numpy(y_true, y_pred, factor=0.08)
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_crps_known_error_and_spread(self):
        # 11 ensemble members with mean 0.5 and spread
        y_true = np.zeros((11, 32, 48, 1), dtype=np.float32)
        # Create predictions: 0.1, 0.2, ..., 1.1 -> mean = 0.6, MAE = 0.6
        vals = np.linspace(0.1, 1.1, 11, dtype=np.float32)
        y_pred = np.zeros((11, 32, 48, 1), dtype=np.float32)
        for m in range(11):
            y_pred[m, :, :, 0] = vals[m]

        expected_mae = float(np.mean(vals))
        expected_std = float(np.std(vals))
        expected_crps = expected_mae - 0.08 * expected_std

        computed_crps = crps2d_numpy(y_true, y_pred, factor=0.08)
        self.assertAlmostEqual(computed_crps, expected_crps, places=5)

    def test_multi_case_crps_averaging(self):
        # Two cases (B=22)
        y_true = np.zeros((22, 32, 48, 1), dtype=np.float32)
        y_pred = np.zeros((22, 32, 48, 1), dtype=np.float32)

        # Case 1: constant 0.5 (mae=0.5, std=0 -> crps=0.5)
        y_pred[:11] = 0.5
        # Case 2: constant 0.3 (mae=0.3, std=0 -> crps=0.3)
        y_pred[11:] = 0.3

        expected_score = (0.5 + 0.3) / 2.0
        score = crps2d_numpy(y_true, y_pred, factor=0.08)
        self.assertAlmostEqual(score, expected_score, places=6)


class MockModel:
    """Mock model providing get_weights and set_weights for testing checkpoint mechanics."""

    def __init__(self, weights):
        self._weights = [w.copy() for w in weights]

    def get_weights(self):
        return [w.copy() for w in self._weights]

    def set_weights(self, weights):
        self._weights = [w.copy() for w in weights]


class TestCheckpointPersistence(unittest.TestCase):
    """Tests model weights serialization and exact numerical restoration."""

    def test_save_and_restore_cycle(self):
        dummy_weights = [
            np.random.normal(size=(3, 3, 11, 16)).astype(np.float32),
            np.zeros((16,), dtype=np.float32),
            np.random.normal(size=(1, 1, 16, 1)).astype(np.float32),
        ]
        model = MockModel(dummy_weights)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = save_a0_checkpoint(
                model=model,
                epoch=5,
                loss=0.12345,
                checkpoint_dir=tmpdir,
                filename_prefix="test_model",
                metadata={"seed": 42, "lr": 0.001},
            )

            # Create clean model with different initial weights
            new_weights = [np.ones_like(w) for w in dummy_weights]
            clean_model = MockModel(new_weights)

            # Restore
            restore_a0_checkpoint(clean_model, save_path)

            restored_weights = clean_model.get_weights()
            for orig, rest in zip(dummy_weights, restored_weights):
                np.testing.assert_allclose(rest, orig, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
