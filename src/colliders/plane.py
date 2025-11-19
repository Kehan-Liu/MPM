from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PlaneCollider:
    point: Sequence[float]
    normal: Sequence[float]
    friction: float = 0.2

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        point_arr = np.array(self.point, dtype=np.float32)
        normal_arr = np.array(self.normal, dtype=np.float32)
        norm = np.linalg.norm(normal_arr)
        if norm == 0:
            raise ValueError("Collider normal cannot be zero")
        normal_arr /= norm
        return point_arr, normal_arr
