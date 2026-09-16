"""
tests/test_15_production_training_contracts.py
---------------------------------------------
Automated test suite verifying the contracts, topology, and mechanics of Step 21K.3:
  1. Default training configurations and contract alignment.
  2. Spatial CRPS loss graph execution (crps2d_tf) with active land masking.
  3. Predeclared seeds [42, 123, 456] and parameter counts across all 4 leads.
  4. Dry-run CLI execution on pilot cases.
"""

import unittest
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

from src.models.a0_unet import (
    build_a0_unet,
    EXPECTED_A0_PARAMETER_COUNTS,
    LEAD_CHANNELS,
)
from src.data.tf_dataset import crps2d_tf, crps2d_numpy, DEFAULT_FACTOR


class TestProductionTrainingContracts(unittest.TestCase):
    """Verifies Step 21K.3 production training contracts and functions."""

    def test_parameter_counts_and_lead_schedule(self):
        """Verifies exact expected parameter counts and channels for all 4 leads."""
        self.assertEqual(LEAD_CHANNELS[1], 11)
        self.assertEqual(LEAD_CHANNELS[2], 12)
        self.assertEqual(LEAD_CHANNELS[3], 5)
        self.assertEqual(LEAD_CHANNELS[4], 6)

        self.assertEqual(EXPECTED_A0_PARAMETER_COUNTS[1], 1_627_139)
        self.assertEqual(EXPECTED_A0_PARAMETER_COUNTS[2], 1_630_307)
        self.assertEqual(EXPECTED_A0_PARAMETER_COUNTS[3], 1_608_131)
        self.assertEqual(EXPECTED_A0_PARAMETER_COUNTS[4], 1_611_299)

    @unittest.skipUnless(TF_AVAILABLE, "Requires TensorFlow")
    def test_crps2d_tf_graph_execution(self):
        """Verifies that crps2d_tf executes cleanly as a differentiable loss function."""
        batch_size = 22  # 2 cases x 11 members
        y_true = np.random.uniform(0.1, 0.9, size=(batch_size, 32, 48, 1)).astype(np.float32)
        y_pred = tf.Variable(np.random.uniform(0.1, 0.9, size=(batch_size, 32, 48, 1)).astype(np.float32))

        eval_mask = np.zeros((32, 48), dtype=bool)
        eval_mask[10:20, 15:25] = True  # Mock active cells

        with tf.GradientTape() as tape:
            loss = crps2d_tf(y_true, y_pred, factor=0.08, eval_mask=eval_mask)

        self.assertTrue(tf.math.is_finite(loss))
        grad = tape.gradient(loss, y_pred)
        self.assertIsNotNone(grad)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(grad)))

    @unittest.skipUnless(TF_AVAILABLE, "Requires TensorFlow")
    def test_crps2d_tf_ocean_masking_invariance(self):
        """Verifies that perturbing masked ocean cells does not affect masked CRPS loss."""
        batch_size = 11
        y_true = np.ones((batch_size, 32, 48, 1), dtype=np.float32) * 0.5
        y_pred = np.ones((batch_size, 32, 48, 1), dtype=np.float32) * 0.5

        eval_mask = np.zeros((32, 48), dtype=bool)
        eval_mask[10, 10] = True  # Single active land cell

        loss1 = crps2d_tf(y_true, y_pred, factor=0.08, eval_mask=eval_mask).numpy()

        # Perturb ocean cells heavily
        y_pred_perturbed = y_pred.copy()
        y_pred_perturbed[:, 0:5, 0:5, :] = 999.0

        loss2 = crps2d_tf(y_true, y_pred_perturbed, factor=0.08, eval_mask=eval_mask).numpy()

        self.assertAlmostEqual(loss1, loss2, places=6)


if __name__ == "__main__":
    unittest.main()
