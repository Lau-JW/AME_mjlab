"""AME-2 terrain configuration — simplified 3-type curriculum.

Only flat, pyramid_stairs, pyramid_stairs_inv.
Each type has equal weight, arranged in a curriculum grid.
"""

from dataclasses import dataclass, field, replace
from typing import Any
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG


# Build a simplified terrain generator with only 3 flat/stairs types
_full_sub = ROUGH_TERRAINS_CFG.sub_terrains
KEEP_KEYS = ["flat", "pyramid_stairs", "pyramid_stairs_inv"]

SIMPLE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=ROUGH_TERRAINS_CFG.seed,
    curriculum=True,
    size=ROUGH_TERRAINS_CFG.size,
    border_width=ROUGH_TERRAINS_CFG.border_width,
    border_height=ROUGH_TERRAINS_CFG.border_height,
    num_rows=10,
    num_cols=15,  # 3 types × 5 columns each
    color_scheme=ROUGH_TERRAINS_CFG.color_scheme,
    sub_terrains={
        "flat": replace(_full_sub["flat"], proportion=1/3),
        "pyramid_stairs": replace(
            _full_sub["pyramid_stairs"],
            proportion=1/3,
            step_height_range=(0.15, 0.21),
        ),
        "pyramid_stairs_inv": replace(
            _full_sub["pyramid_stairs_inv"],
            proportion=1/3,
            step_height_range=(0.15, 0.21),
        ),
    },
    difficulty_range=(0.0, 1.0),
    add_lights=ROUGH_TERRAINS_CFG.add_lights,
)
