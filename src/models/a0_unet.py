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

TOTAL_A0_PARAMETERS: int = 1_630_307  # Reconciled parent EX29 parameter count for UNET_RZSM


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

    Parameters
    ----------
    lead : int
        Forecast lead week (1, 2, 3, or 4).
    height : int
        Spatial grid height (default: 32).
    width : int
        Spatial grid width (default: 48).
    output_channels : int
        Target channels per head (default: 1).
    using_deep_supervision : bool
        Whether to enable the 3 multi-scale output heads (default: True).
    name : Optional[str]
        Keras model name.

    Returns
    -------
    keras.Model
        Instantiated Model A0 architecture with 1.63M parameters.
    """
    if lead not in LEAD_CHANNELS:
        raise ValueError(f"Invalid lead {lead}. Supported leads: {list(LEAD_CHANNELS.keys())}")

    model_name = name or f"UNET_RZSM_Mindanao_A0_Lead_{lead}"
    num_channels = LEAD_CHANNELS[lead]

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
