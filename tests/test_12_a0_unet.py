"""
tests/test_a0_unet.py
---------------------
Unit tests for Model A0 UNET_RZSM architecture factory and specifications.
"""

import unittest
from src.models.a0_unet import (
    LEAD_CHANNELS,
    GRID_HEIGHT,
    GRID_WIDTH,
    TOTAL_A0_PARAMETERS,
    EXPECTED_A0_PARAMETER_COUNTS,
    build_a0_unet,
)


class TestA0UnetFactory(unittest.TestCase):
    """Tests Model A0 architectural constants, specifications, and instantiation logic."""

    def test_grid_dimensions(self):
        self.assertEqual(GRID_HEIGHT, 32)
        self.assertEqual(GRID_WIDTH, 48)
        self.assertEqual(GRID_HEIGHT % 16, 0)
        self.assertEqual(GRID_WIDTH % 16, 0)

    def test_geometry_guard_rejects_non_divisible_by_16(self):
        """Verifies architectural invariant guard rejecting non-divisible by 16 dimensions."""
        with self.assertRaises(ValueError):
            build_a0_unet(lead=1, height=30, width=48)
        with self.assertRaises(ValueError):
            build_a0_unet(lead=1, height=32, width=50)

    def test_lead_channel_schedule(self):
        self.assertEqual(LEAD_CHANNELS[1], 11)
        self.assertEqual(LEAD_CHANNELS[2], 12)
        self.assertEqual(LEAD_CHANNELS[3], 5)
        self.assertEqual(LEAD_CHANNELS[4], 6)

    def test_invalid_lead_raises_error(self):
        with self.assertRaises(ValueError):
            build_a0_unet(lead=0)
        with self.assertRaises(ValueError):
            build_a0_unet(lead=5)

    def test_parent_parameter_constants(self):
        self.assertEqual(TOTAL_A0_PARAMETERS, 1_630_307)
        self.assertEqual(EXPECTED_A0_PARAMETER_COUNTS[1], 1_627_139)
        self.assertEqual(EXPECTED_A0_PARAMETER_COUNTS[2], 1_630_307)
        self.assertEqual(EXPECTED_A0_PARAMETER_COUNTS[3], 1_608_131)
        self.assertEqual(EXPECTED_A0_PARAMETER_COUNTS[4], 1_611_299)

    def test_real_model_instantiation_and_architecture_contract(self):
        """
        When TensorFlow is present, independently instantiate the authentic UNET_RZSM
        model and verify layer topologies, per-lead parameter counts, and output heads.
        """
        try:
            import tensorflow as tf
            tf_available = True
        except ImportError:
            tf_available = False

        if not tf_available:
            self.skipTest("TensorFlow not installed in current environment; runtime instantiation deferred to Colab GPU.")

        # 1. Lead 1 Instantiation & Parity Assertions
        model_w1 = build_a0_unet(lead=1, height=GRID_HEIGHT, width=GRID_WIDTH)
        
        # Exact per-lead parameter count assertion (Lead 1 = 1,627,139)
        actual_params_w1 = model_w1.count_params()
        self.assertEqual(
            actual_params_w1,
            EXPECTED_A0_PARAMETER_COUNTS[1],
            f"Actual parameters ({actual_params_w1:,}) must match Lead 1 contract ({EXPECTED_A0_PARAMETER_COUNTS[1]:,}) exactly."
        )

        # Trainable weight tensors assertion
        trainable_weights_count = len(model_w1.trainable_weights)
        self.assertEqual(
            trainable_weights_count,
            298,
            f"Trainable weight count ({trainable_weights_count}) must equal 298 tensors."
        )

        # Multi-head deep supervision assertion (3 heads)
        self.assertEqual(len(model_w1.outputs), 3, "Model A0 must possess exactly 3 multi-scale output heads.")
        for i, out in enumerate(model_w1.outputs, 1):
            self.assertEqual(
                out.shape[1:],
                (GRID_HEIGHT, GRID_WIDTH, 1),
                f"Head {i} output shape must be ({GRID_HEIGHT}, {GRID_WIDTH}, 1)."
            )

        # Expected layer classes verification
        layer_classes = set(layer.__class__.__name__ for layer in model_w1.layers)
        required_classes = {"InputLayer", "Conv2D", "Activation", "Concatenate"}
        for req in required_classes:
            self.assertIn(req, layer_classes, f"Model A0 must contain {req} layers.")

        # 2. Verify all 4 leads have exact input channels and per-lead parameter counts
        for lead, expected_ch in LEAD_CHANNELS.items():
            m_lead = build_a0_unet(lead=lead, height=GRID_HEIGHT, width=GRID_WIDTH)
            self.assertEqual(
                m_lead.input_shape,
                (None, GRID_HEIGHT, GRID_WIDTH, expected_ch),
                f"Lead {lead} input shape must be (None, {GRID_HEIGHT}, {GRID_WIDTH}, {expected_ch})."
            )
            self.assertEqual(
                m_lead.count_params(),
                EXPECTED_A0_PARAMETER_COUNTS[lead],
                f"Lead {lead} parameters ({m_lead.count_params():,}) must equal {EXPECTED_A0_PARAMETER_COUNTS[lead]:,}."
            )

        # 3. Forward pass finiteness & shape assertion
        dummy_x = tf.random.normal((2, GRID_HEIGHT, GRID_WIDTH, LEAD_CHANNELS[1]), dtype=tf.float32)
        preds = model_w1(dummy_x, training=False)
        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertEqual(p.shape, (2, GRID_HEIGHT, GRID_WIDTH, 1))
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(p)).numpy()))


if __name__ == "__main__":
    unittest.main()

