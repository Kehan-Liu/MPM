from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import numpy as np


@dataclass
class ParticleIO:
    output_dir: Path
    grid_resolution: int
    dimension: int

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "grid_resolution": self.grid_resolution,
            "dimension": self.dimension,
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def save_frame(
        self,
        frame_id: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        material_ids: np.ndarray,
    ) -> Path:
        path = self.output_dir / f"frame_{frame_id:04d}.npz"
        np.savez_compressed(
            path,
            positions=positions.astype(np.float32),
            velocities=velocities.astype(np.float32),
            material_ids=material_ids.astype(np.int32),
        )
        return path

    @staticmethod
    def load_sequence(folder: Path) -> Iterable[np.lib.npyio.NpzFile]:
        for npz_path in sorted(folder.glob("frame_*.npz")):
            yield np.load(npz_path)
