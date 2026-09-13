"""
src/models/__init__.py
----------------------
Model factory and architectural specifications for the Mindanao RISE-UNet project.
"""

from .a0_unet import (
    build_a0_unet,
    LEAD_CHANNELS,
    GRID_HEIGHT,
    GRID_WIDTH,
    TOTAL_A0_PARAMETERS,
)

__all__ = [
    "build_a0_unet",
    "LEAD_CHANNELS",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "TOTAL_A0_PARAMETERS",
]
