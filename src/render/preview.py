from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import taichi as ti


class Preview2D:
    """Tiny Taichi GUI helper for quick inspection of particle dumps."""

    def __init__(self, resolution: int = 720, background: int = 0x112F41) -> None:
        self.resolution = resolution
        self.gui = ti.GUI("MPM Preview", res=(resolution, resolution))
        self.background = background

    def draw_frame(
        self,
        positions: np.ndarray,
        colors: Optional[Sequence[Sequence[float]]] = None,
        radius: float = 2.0,
        display: bool = True,
    ) -> np.ndarray:
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("Preview2D only supports 2D positions of shape (N, 2)")
        self.gui.clear(self.background)
        pos = np.clip(positions.astype(np.float32), 0.0, 0.99)
        if colors is None:
            palette = np.ones((positions.shape[0], 3), dtype=np.float32) * 0.8
        else:
            palette = np.asarray(colors, dtype=np.float32)
            if palette.ndim == 1:
                palette = np.tile(palette[None, :], (positions.shape[0], 1))
        gui_colors = self._as_gui_colors(palette)
        if pos.size:
            self.gui.circles(pos, radius=radius, color=gui_colors)
        image = np.clip(self.gui.get_image(), 0.0, 1.0)
        if display:
            self.gui.show()
        return (image * 255).astype(np.uint8)

    def show(
        self,
        positions: np.ndarray,
        colors: Optional[Sequence[Sequence[float]]] = None,
        radius: float = 2.0,
        pause_frames: int = 240,
    ) -> None:
        for _ in range(pause_frames):
            if not self.gui.running:
                break
            self.draw_frame(positions, colors=colors, radius=radius, display=True)

    def write_video(
        self,
        frames: Iterable[Tuple[np.ndarray, Optional[np.ndarray]]],
        radius: float = 2.0,
        output_dir: Path | str = "outputs/previews",
        framerate: int = 24,
        display: bool = False,
        build_gif: bool = False,
    ) -> Path:
        video_manager = ti.tools.VideoManager(
            output_dir=str(output_dir), framerate=framerate, automatic_build=False
        )
        for idx, (pos, cols) in enumerate(frames):
            frame = self.draw_frame(pos, colors=cols, radius=radius, display=display)
            video_manager.write_frame(frame)
            if display and not self.gui.running:
                break
        video_manager.make_video(mp4=True, gif=build_gif)
        return Path(video_manager.get_output_filename(".mp4"))

    @staticmethod
    def _as_gui_colors(palette: np.ndarray) -> np.ndarray:
        if palette.ndim == 1:
            return palette
        if palette.shape[1] != 3:
            raise ValueError("Colors must have shape (N, 3)")
        ints = np.clip(palette, 0.0, 1.0)
        ints = (ints * 255.0).astype(np.uint8)
        packed = (
            (ints[:, 0].astype(np.uint32) << 16)
            | (ints[:, 1].astype(np.uint32) << 8)
            | ints[:, 2].astype(np.uint32)
        )
        return packed
