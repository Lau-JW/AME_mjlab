"""AME-2 terrain curriculum configuration.

Paper: Section IV-D3, Appendix A
12 terrain types in 3 categories with progressive difficulty scaling.
"""

from dataclasses import dataclass, field
from typing import Any
import math


@dataclass
class TerrainCurriculumCfg:
    """Terrain curriculum configuration matching paper Appendix A.

    Each terrain type has a difficulty range that scales during training.
    """

    # ── Dense terrains (each ~5%) ──
    rough_noise_range: tuple[float, float] = (0.0, 0.2)
    """Heightfield noise range in meters. Scales from 0 to 0.2m."""

    stair_slope_range: tuple[float, float] = (5.0, 45.0)
    """Stair slope in degrees. Scales from 5 to 45 deg."""

    box_max_height_range: tuple[float, float] = (0.05, 0.4)
    """Max box height in meters for Boxes terrain."""

    obstacle_density_range: tuple[float, float] = (0.0, 0.5)
    """Obstacle density in m^-2."""

    # ── Climbing terrains ──
    climb_up_height: tuple[float, float] = (0.1, 1.0)
    """Climbing Up pit height in meters. Scales from 0.1 to 1.0m."""

    climb_down_height: tuple[float, float] = (0.2, 1.0)
    """Climbing Down platform height in meters. Scales from 0.2 to 1.0m."""

    climb_consecutive_first: tuple[float, float] = (0.05, 0.5)
    """Consecutive climbing first ring height."""

    climb_consecutive_second: tuple[float, float] = (0.05, 0.4)
    """Consecutive climbing second ring height."""

    # ── Sparse terrains ──
    gap_distance: tuple[float, float] = (0.1, 1.1)
    """Gap distance in meters. Scales from 0.1 to 1.1m."""

    beam_width: tuple[float, float] = (0.4, 0.16)
    """Beam width in meters. DECREASING from 0.4 to 0.16m."""

    pallet_gap: tuple[float, float] = (0.08, 0.35)
    """Pallet gap width. Scales from 0.08 to 0.35m."""

    pallet_height_diff: tuple[float, float] = (0.0, 0.3)
    """Inter-pallet height difference. Scales from 0 to 0.3m."""


# Terrain proportions matching paper Appendix A
TERRAIN_PROPORTIONS: dict[str, float] = {
    # Dense (20%)
    "rough": 0.05,
    "stair_down": 0.05,
    "stair_up": 0.05,
    "boxes": 0.05,
    # Climbing (30%)
    "climbing_up": 0.20,
    "climbing_down": 0.05,
    "climbing_consecutive": 0.05,
    # Sparse (45%)
    "gap": 0.05,
    "pallets": 0.05,
    "stones": 0.30,
    "beam": 0.05,
    # Remaining 5%
    "obstacles": 0.05,
}

# G1-specific max values (paper: TRON1 is smaller than ANYmal-D)
G1_TERRAIN_LIMITS = {
    "climb_up_height": 0.48,  # TRON1 max: 0.48m
    "climb_down_height": 0.88,  # TRON1 max: 0.88m
    "gap_distance": 0.6,  # TRON1 max: 0.6m
    "beam_width": 0.16,  # both robots
}
