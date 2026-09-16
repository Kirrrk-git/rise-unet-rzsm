"""
src/models/a0_unet.py
---------------------
Authoritative Model A0 (UNET_RZSM) architecture factory for the Mindanao regional adaptation.
Reconciles Kyle Lesinger & Di Tian (2025) Nature Communications architecture with Mindanao grid geometry.
"""

import logging
from typing import Tuple, Optional, Any, Dict

logger = logging.getLogger(__name__)

# Mindanao Candidate A spatial grid constants
GRID_HEIGHT: int = 32
GRID_WIDTH: int = 48

# Multi-lead channel schedule (EX29 contract)
LEAD_CHANNELS: Dict[int, int] = {
    1: 11,  # 3 RZSM lags + 5 ERA5 atmospheric + 3 S2S W1 forecast
    2: 12,  # 11 base + 1 recursive y_hat_W1
    3: 5,   # 3 RZSM lags + 2 recursive (y_hat_W1, y_hat_W2)
    4: 6,   # 3 RZSM lags + 3 recursive (y_hat_W1, y_hat_W2, y_hat_W3)
}

# Reconciled per-lead parameter counts for UNET_RZSM with Candidate A geometry (32x48)
# Each input channel variance accounts for exactly 3,168 parameters in the initial inception block:
# Lead 1 (11 channels) = 1,627,139 params
# Lead 2 (12 channels) = 1,630,307 params (Parent EX29 parity target)
# Lead 3 (5 channels)  = 1,608,131 params
# Lead 4 (6 channels)  = 1,611,299 params
EXPECTED_A0_PARAMETER_COUNTS: Dict[int, int] = {
    1: 1_627_139,
    2: 1_630_307,
    3: 1_608_131,
    4: 1_611_299,
}

# Retained alias for backwards compatibility: parent Lead 2 reference parameter count
TOTAL_A0_PARAMETERS: int = 1_630_307


def ensure_keras_compatibility() -> None:
    """
    Applies runtime compatibility shims for protobuf runtime validation
    and Keras 3 DepthwiseConv2D keyword translation (kernel_initializer -> depthwise_initializer).
    Ensures genuine UNET_RZSM architecture instantiation succeeds across modern Keras 3 / TF 2.16+ environments.
    """
    try:
        import google.protobuf.runtime_version as _rt
        _rt.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
    except (ImportError, AttributeError):
        pass

    try:
        import tensorflow as tf
        import keras.layers
        try:
            import keras.src.layers.convolutional.depthwise_conv2d as dw_mod
            base_dw = dw_mod.DepthwiseConv2D
        except Exception:
            base_dw = keras.layers.DepthwiseConv2D

        if getattr(base_dw, "_is_mindanao_compatible", False):
            return

        class CompatibleDepthwiseConv2D(base_dw):
            """Keras 3 compatibility shim translating legacy kwargs to depthwise kwargs."""
            _is_mindanao_compatible = True

            def __init__(self, *args, **kwargs):
                if "kernel_initializer" in kwargs:
                    kwargs["depthwise_initializer"] = kwargs.pop("kernel_initializer")
                if "kernel_constraint" in kwargs:
                    kwargs["depthwise_constraint"] = kwargs.pop("kernel_constraint")
                super().__init__(*args, **kwargs)

        keras.layers.DepthwiseConv2D = CompatibleDepthwiseConv2D
        if hasattr(tf, "keras") and hasattr(tf.keras, "layers"):
            tf.keras.layers.DepthwiseConv2D = CompatibleDepthwiseConv2D
    except (ImportError, AttributeError):
        pass


# Automatically ensure compatibility on module load
ensure_keras_compatibility()


def build_a0_unet(
    lead: int = 1,
    height: int = GRID_HEIGHT,
    width: int = GRID_WIDTH,
    output_channels: int = 1,
    using_deep_supervision: bool = True,
    name: Optional[str] = None,
) -> Any:
    """
    Constructs the genuine Model A0 UNET_RZSM architecture for a given forecast lead.

    Architecture & Masking Boundary:
    --------------------------------
    The UNET_RZSM neural network outputs unmasked raw continuous predictions across
    the full (height, width) grid. Output zero-filling for inactive ocean cells is
    NOT performed inside the neural network graph itself. The regional adaptation
    enforces spatial boundaries through three distinct, decoupled mechanisms:
      1. Input inactive-cell zero filling (pre-inference conditioning in case_builder.py).
      2. Loss and evaluation masking (eval_mask applied during loss computation and metrics).
      3. Postprocessing masking (optional zeroing applied to final predictions for visualization).

    Parameters
    ----------
    lead : int
        Forecast lead week (1, 2, 3, or 4).
    height : int
        Spatial grid height (default: 32; must be divisible by 16).
    width : int
        Spatial grid width (default: 48; must be divisible by 16).
    output_channels : int
        Target channels per head (default: 1).
    using_deep_supervision : bool
        Whether to enable the 3 multi-scale output heads (default: True, required for production A0).
    name : Optional[str]
        Keras model name.

    Returns
    -------
    keras.Model
        Instantiated Model A0 architecture with exact per-lead parameter count.
    """
    if lead not in LEAD_CHANNELS:
        raise ValueError(f"Invalid lead {lead}. Supported leads: {list(LEAD_CHANNELS.keys())}")

    # Architectural invariant guard: 4 levels of 2x2 max-pooling require divisibility by 16
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            f"Spatial grid dimensions ({height}, {width}) must be divisible by 16 "
            "for 4-level UNet pooling."
        )

    model_name = name or f"UNET_RZSM_Mindanao_A0_Lead_{lead}"
    num_channels = LEAD_CHANNELS[lead]

    ensure_keras_compatibility()

    try:
        import tensorflow as tf
        Input = tf.keras.layers.Input
        Model = tf.keras.models.Model
        from parent_study_ex29.function import modelRzsmRelu as UNETRzsm
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"TensorFlow / Keras / parent_study_ex29.function.modelRzsmRelu required to instantiate genuine UNET_RZSM: {e}"
        ) from e

    inputs = Input(shape=(height, width, num_channels), name=f"input_lead_{lead}")
    outputs = UNETRzsm.model_build_func(
        inputs=inputs,
        output_channels=output_channels,
        using_deep_supervision=using_deep_supervision,
        kernel_norm=None,
        var_name="RZSM",
        number_of_UNET_backbone_max_pool=4,
    )

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    logger.info(
        f"Instantiated genuine Model A0 ({model.name}): "
        f"input={model.input_shape}, outputs={len(model.outputs)}, "
        f"total_params={model.count_params():,}"
    )
    return model
