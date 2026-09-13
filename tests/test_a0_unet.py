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
    build_a0_unet,
)


class TestA0UnetFactory(unittest.TestCase):
    """Tests Model A0 architectural constants, specifications, and instantiation logic."""

    def test_grid_dimensions(self):
        self.assertEqual(GRID_HEIGHT, 32)
        self.assertEqual(GRID_WIDTH, 48)
        self.assertEqual(GRID_HEIGHT % 16, 0)
        self.assertEqual(GRID_WIDTH % 16, 0)

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

    def test_parent_parameter_constant(self):
        self.assertEqual(TOTAL_A0_PARAMETERS, 1_630_307)

    def test_real_model_instantiation_and_architecture_contract(self):
        """
        When TensorFlow is present, independently instantiate the authentic UNET_RZSM
        model and verify layer topologies, parameter counts, and output heads.
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
        
        # Exact parameter count assertion
        actual_params = model_w1.count_params()
        self.assertEqual(
            actual_params,
            TOTAL_A0_PARAMETERS,
            f"Actual parameters ({actual_params:,}) must match parent EX29 contract ({TOTAL_A0_PARAMETERS:,}) exactly."
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

        # 2. Verify all 4 leads have exact input channel dimensions
        for lead, expected_ch in LEAD_CHANNELS.items():
            m_lead = build_a0_unet(lead=lead, height=GRID_HEIGHT, width=GRID_WIDTH)
            self.assertEqual(
                m_lead.input_shape,
                (None, GRID_HEIGHT, GRID_WIDTH, expected_ch),
                f"Lead {lead} input shape must be (None, {GRID_HEIGHT}, {GRID_WIDTH}, {expected_ch})."
            )
            self.assertEqual(
                m_lead.count_params(),
                TOTAL_A0_PARAMETERS,
                f"Lead {lead} parameters must equal {TOTAL_A0_PARAMETERS}."
            )


if __name__ == "__main__":
    unittest.main()

