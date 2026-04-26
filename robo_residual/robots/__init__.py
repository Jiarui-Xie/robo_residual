"""Pre-built joint configs for common robots."""

from robo_residual.robots.g1 import g1_conservative, g1_energy_efficient
from robo_residual.robots.h1 import h1_conservative, h1_energy_efficient
from robo_residual.robots.go2 import go2_conservative, go2_energy_efficient

__all__ = [
    "g1_conservative",
    "g1_energy_efficient",
    "h1_conservative",
    "h1_energy_efficient",
    "go2_conservative",
    "go2_energy_efficient",
]
