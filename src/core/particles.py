from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ParticleBatch:
    """Container for host-side particle attributes before uploading to Taichi."""

    positions: np.ndarray
    velocities: Optional[np.ndarray]
    material_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.positions.ndim != 2:
            raise ValueError("positions must be of shape (N, dim)")
        num = self.positions.shape[0]
        if (
            self.velocities is not None
            and self.velocities.shape != self.positions.shape
        ):
            raise ValueError("velocities must match positions shape")
        if self.material_ids.shape[0] != num:
            raise ValueError("material_ids length must match number of particles")
        if self.material_ids.ndim != 1:
            raise ValueError("material_ids must be a flat array")
