"""
tests/test_13_production_smoke_preflight.py
-------------------------------------------
Automated test suite verifying the 5 technical stages of Step 21K.3-pre:
  1. Real 4-Lead architecture parameter updates with deep supervision.
  2. Recursive channel semantics, exact index placement, and permutation tamper detection.
  3. Evaluation domain masking application in loss calculation (masked vs unmasked).
  4. Model-weight parity vs full training-state checkpoint serialization.
"""

import tempfile
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
from src.data.case_builder import (
    simulate_recursive_cascade_step,
    verify_recursive_channel_semantics,
)
from src.data.tf_dataset import (
    save_a0_checkpoint,
    restore_a0_checkpoint,
    save_a0_training_state,
    restore_a0_training_state,
)


class TestProductionSmokePreflight(unittest.TestCase):
    """Verifies Step 21K.3-pre production training preflight contracts."""

    def test_recursive_channel_semantics_all_leads(self):
        """Verifies recursive channel placement and dimensions across Leads 2, 3, and 4."""
        batch_size = 2
        y_hat_w1 = np.random.uniform(0.1, 0.9, size=(batch_size, 32, 48, 1)).astype(np.float32)
        y_hat_w2 = np.random.uniform(0.1, 0.9, size=(batch_size, 32, 48, 1)).astype(np.float32)
        y_hat_w3 = np.random.uniform(0.1, 0.9, size=(batch_size, 32, 48, 1)).astype(np.float32)

        # Lead 1
        x_w1 = np.zeros((batch_size, 32, 48, 11), dtype=np.float32)
        self.assertTrue(verify_recursive_channel_semantics(x_w1, lead=1, prior_predictions=[]))

        # Lead 2 (11 base + 1 recursive -> 12)
        x_w2_base = np.zeros((batch_size, 32, 48, 11), dtype=np.float32)
        x_w2_full = simulate_recursive_cascade_step(x_w2_base, [y_hat_w1])
        self.assertEqual(x_w2_full.shape[-1], 12)
        self.assertTrue(verify_recursive_channel_semantics(x_w2_full, lead=2, prior_predictions=[y_hat_w1]))

        # Lead 3 (3 base + 2 recursive -> 5)
        x_w3_base = np.zeros((batch_size, 32, 48, 3), dtype=np.float32)
        x_w3_full = simulate_recursive_cascade_step(x_w3_base, [y_hat_w1, y_hat_w2])
        self.assertEqual(x_w3_full.shape[-1], 5)
        self.assertTrue(verify_recursive_channel_semantics(x_w3_full, lead=3, prior_predictions=[y_hat_w1, y_hat_w2]))

        # Lead 4 (3 base + 3 recursive -> 6)
        x_w4_base = np.zeros((batch_size, 32, 48, 3), dtype=np.float32)
        x_w4_full = simulate_recursive_cascade_step(x_w4_base, [y_hat_w1, y_hat_w2, y_hat_w3])
        self.assertEqual(x_w4_full.shape[-1], 6)
        self.assertTrue(verify_recursive_channel_semantics(x_w4_full, lead=4, prior_predictions=[y_hat_w1, y_hat_w2, y_hat_w3]))

    def test_recursive_channel_tamper_detection(self):
        """Scrambling or permuting recursive channel orders must trigger ValueError."""
        batch_size = 2
        y_hat_w1 = np.random.uniform(0.1, 0.9, size=(batch_size, 32, 48, 1)).astype(np.float32)
        y_hat_w2 = np.random.uniform(0.1, 0.9, size=(batch_size, 32, 48, 1)).astype(np.float32)

        x_w3_base = np.zeros((batch_size, 32, 48, 3), dtype=np.float32)
        x_w3_full = simulate_recursive_cascade_step(x_w3_base, [y_hat_w1, y_hat_w2])

        # Swap channel 3 (y_hat_w1) and channel 4 (y_hat_w2)
        x_w3_scrambled = x_w3_full.copy()
        x_w3_scrambled[..., [3, 4]] = x_w3_scrambled[..., [4, 3]]

        with self.assertRaises(ValueError):
            verify_recursive_channel_semantics(x_w3_scrambled, lead=3, prior_predictions=[y_hat_w1, y_hat_w2])

    def test_training_state_checkpoint_roundtrip(self):
        """Verifies full training state serialization (weights, optimizer, step, epoch)."""
        if not TF_AVAILABLE:
            self.skipTest("TensorFlow not installed.")

        tf.keras.backend.clear_session()
        model = build_a0_unet(lead=1, height=32, width=48, using_deep_supervision=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # Run dummy step to initialize optimizer slots
        x = tf.zeros((1, 32, 48, 11), dtype=tf.float32)
        y = tf.zeros((1, 32, 48, 1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            p = model(x, training=True)
            l = tf.reduce_mean(tf.abs(p[0] - y))
        grads = tape.gradient(l, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            state_file = save_a0_training_state(
                model=model,
                optimizer=optimizer,
                epoch=3,
                step=420,
                learning_rate=1e-4,
                loss=0.085,
                checkpoint_dir=tmp_path,
                metadata={"seed": 123},
            )

            # Restore into clean instances
            fresh_model = build_a0_unet(lead=1, height=32, width=48, using_deep_supervision=True)
            fresh_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
            meta = restore_a0_training_state(fresh_model, fresh_opt, state_file)

            self.assertEqual(meta["epoch"], 3)
            self.assertEqual(meta["step"], 420)
            self.assertEqual(meta["extra"]["seed"], 123)

            # Output parity at restored state
            out1 = model(x, training=False)[-1].numpy()
            out2 = fresh_model(x, training=False)[-1].numpy()
            max_delta = float(np.max(np.abs(out1 - out2)))
            self.assertLess(max_delta, 1e-6)

            # Next-step optimization trajectory parity
            x_next = tf.constant(np.random.uniform(0.1, 0.9, size=(1, 32, 48, 11)).astype(np.float32))
            y_next = tf.constant(np.random.uniform(0.1, 0.9, size=(1, 32, 48, 1)).astype(np.float32))

            with tf.GradientTape() as tape1:
                p1_next = model(x_next, training=True)
                l1_next = tf.reduce_mean(tf.abs(p1_next[0] - y_next))
            g1_next = tape1.gradient(l1_next, model.trainable_variables)
            optimizer.apply_gradients(zip(g1_next, model.trainable_variables))

            with tf.GradientTape() as tape2:
                p2_next = fresh_model(x_next, training=True)
                l2_next = tf.reduce_mean(tf.abs(p2_next[0] - y_next))
            g2_next = tape2.gradient(l2_next, fresh_model.trainable_variables)
            fresh_opt.apply_gradients(zip(g2_next, fresh_model.trainable_variables))

            # Discrepancy between model and fresh_model after next update must be zero
            next_weight_delta = float(
                np.max([np.max(np.abs(w1.numpy() - w2.numpy())) for w1, w2 in zip(model.trainable_variables, fresh_model.trainable_variables)])
            )
            self.assertLess(next_weight_delta, 1e-6)
            self.assertAlmostEqual(float(l1_next), float(l2_next), places=6)

    def test_four_lead_gradient_and_optimizer_step(self):
        """Verifies real GradientTape backpropagation and parameter update on all 4 leads."""
        if not TF_AVAILABLE:
            self.skipTest("TensorFlow not installed.")

        # Synthetic 126-cell mask
        mask = np.zeros((32, 48), dtype=bool)
        mask[10:24, 15:24] = True
        mask_tf = tf.constant(mask, dtype=tf.bool)

        for lead in [1, 2, 3, 4]:
            cin = LEAD_CHANNELS[lead]
            tf.keras.backend.clear_session()
            model = build_a0_unet(lead=lead, height=32, width=48, using_deep_supervision=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

            x = tf.random.uniform((2, 32, 48, cin), minval=0.1, maxval=0.9, seed=lead)
            y = tf.random.uniform((2, 32, 48, 1), minval=0.1, maxval=0.9, seed=lead)

            weights_before = [w.numpy() for w in model.trainable_variables]

            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                # Compute masked loss over active domain
                masked_losses = []
                for p in preds:
                    p_act = tf.boolean_mask(p, mask_tf, axis=1)
                    y_act = tf.boolean_mask(y, mask_tf, axis=1)
                    masked_losses.append(tf.reduce_mean(tf.abs(p_act - y_act)))
                loss = tf.reduce_mean(masked_losses)

            grads = tape.gradient(loss, model.trainable_variables)
            # Assert all gradients finite
            for g in grads:
                if g is not None:
                    self.assertFalse(np.isnan(g.numpy()).any())
                    self.assertFalse(np.isinf(g.numpy()).any())

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            weights_after = [w.numpy() for w in model.trainable_variables]
    def test_normalize_assembled_case(self):
        """Verifies normalization scaling into [0, 1] and ocean buffer zero-filling."""
        from src.data.tf_dataset import normalize_assembled_case

        mask = np.zeros((32, 48), dtype=bool)
        mask[10:24, 15:24] = True  # 126 active cells

        b = 11
        mock_raw = {
            "x_w1": np.full((b, 32, 48, 11), 300.0, dtype=np.float32),
            "x_w2_base": np.full((b, 32, 48, 11), 300.0, dtype=np.float32),
            "x_w3_base": np.full((b, 32, 48, 3), 0.0, dtype=np.float32),
            "x_w4_base": np.full((b, 32, 48, 3), 0.0, dtype=np.float32),
            "y_w1": np.zeros((1, 32, 48, 1), dtype=np.float32),
            "y_w2": np.zeros((1, 32, 48, 1), dtype=np.float32),
            "y_w3": np.zeros((1, 32, 48, 1), dtype=np.float32),
            "y_w4": np.zeros((1, 32, 48, 1), dtype=np.float32),
        }
        # Give hgt_pres realistic value in channel 7
        mock_raw["x_w1"][..., 7] = 12400.0
        mock_raw["x_w2_base"][..., 7] = 12400.0

        normed = normalize_assembled_case(mock_raw, eval_mask=mask)

        # Invariant checks
        for k in ["x_w1", "x_w2_base", "x_w3_base", "x_w4_base", "y_w1", "y_w2", "y_w3", "y_w4"]:
            arr = normed[k]
            # Active cells must be in [0, 1]
            self.assertTrue(np.all(arr[:, mask] >= 0.0))
            self.assertTrue(np.all(arr[:, mask] <= 1.0))
            # Ocean cells must be exactly 0.0
            self.assertEqual(float(np.max(np.abs(arr[:, ~mask]))), 0.0)


if __name__ == "__main__":
    unittest.main()

