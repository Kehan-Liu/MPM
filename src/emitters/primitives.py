from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.particles import ParticleBatch


@dataclass
class BoxEmitterConfig:
    lower_corner: Sequence[float]
    upper_corner: Sequence[float]
    spacing: float
    velocity: Sequence[float]
    material_id: int
    jitter: float = 0.0


def build_box(config: BoxEmitterConfig) -> ParticleBatch:
    lower = np.array(config.lower_corner, dtype=np.float32)
    upper = np.array(config.upper_corner, dtype=np.float32)
    if lower.shape != upper.shape:
        raise ValueError("lower_corner and upper_corner must have the same dimension")
    dim = lower.shape[0]
    spacing = float(config.spacing)
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    axes = [
        np.arange(lower[d], upper[d], spacing, dtype=np.float32) for d in range(dim)
    ]
    if any(len(axis) == 0 for axis in axes):
        raise ValueError("Box emitter produced zero particles; check bounds/spacing")
    mesh = np.meshgrid(*axes, indexing="ij")
    positions = np.stack([m.reshape(-1) for m in mesh], axis=-1)
    if config.jitter > 0:
        rand = (
            (np.random.rand(*positions.shape).astype(np.float32) - 0.5)
            * config.jitter
            * spacing
        )
        positions += rand
    velocities = np.broadcast_to(
        np.array(config.velocity, dtype=np.float32), positions.shape
    ).copy()
    material_ids = np.full((positions.shape[0],), config.material_id, dtype=np.int32)
    return ParticleBatch(
        positions=positions, velocities=velocities, material_ids=material_ids
    )
