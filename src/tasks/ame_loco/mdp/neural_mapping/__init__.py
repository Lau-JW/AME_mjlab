"""AME-2 neural mapping package (paper Sec V)."""

from .gated_unet import GatedElevationUNet, beta_nll_loss, total_variation_weights
from .pipeline import (
    NeuralMappingPipeline,
    create_depth_cloud_sensor_cfg,
    sample_neural_elevation_map,
)

__all__ = [
    "GatedElevationUNet",
    "beta_nll_loss",
    "total_variation_weights",
    "NeuralMappingPipeline",
    "create_depth_cloud_sensor_cfg",
    "sample_neural_elevation_map",
]
