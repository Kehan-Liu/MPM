from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import taichi as ti

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runtime import auto_init
from src.render.preview import Preview2D

COLOR_CYCLE = np.array(
    [
        (0.91, 0.3, 0.24),
        (0.18, 0.8, 0.44),
        (0.2, 0.6, 0.86),
        (0.61, 0.35, 0.71),
        (0.95, 0.77, 0.06),
        (0.99, 0.46, 0.38),
        (0.17, 0.63, 0.8),
        (0.56, 0.93, 0.56),
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview cached particle frames and export a video via Taichi GUI",
    )
    parser.add_argument(
        "cache_dir",
        type=Path,
        help="Directory containing frame_XXXX.npz files produced by scripts/run_scene.py",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "vulkan"],
        help="Taichi backend for GUI rendering",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=720,
        help="GUI canvas resolution (square)",
    )
    parser.add_argument(
        "--radius", type=float, default=2.0, help="Particle draw radius"
    )
    parser.add_argument(
        "--framerate", type=int, default=24, help="Output video frame rate"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to preview",
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=1,
        help="Number of GUI refresh cycles per simulation frame (>=1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to store the preview video (defaults to <cache_dir>/preview)",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also export a GIF alongside the MP4",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip showing the GUI window while still recording frames",
    )
    return parser.parse_args()


def resolve_arch(name: str):
    name = name.lower()
    if name == "cpu":
        return ti.cpu
    if name == "cuda":
        return ti.cuda
    if name == "vulkan":
        return ti.vulkan
    return None


def enumerate_frames(folder: Path, limit: int | None) -> Iterator[Path]:
    candidates = sorted(folder.glob("frame_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No frames found under {folder}")
    if limit is not None:
        return iter(candidates[:limit])
    return iter(candidates)


def colorize(
    material_ids: np.ndarray,
    palette: Dict[int, np.ndarray],
    wheel: Iterator[Tuple[float, float, float]],
) -> np.ndarray:
    colors = np.empty((material_ids.shape[0], 3), dtype=np.float32)
    for mat_id in np.unique(material_ids):
        if mat_id not in palette:
            palette[mat_id] = np.array(next(wheel))
    for idx, mat_id in enumerate(material_ids):
        colors[idx] = palette[mat_id]
    return colors


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir
    output_dir = args.output or (cache_dir / "preview")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_supported_cache(cache_dir)

    arch = resolve_arch(args.arch)
    used_arch = auto_init(arch=arch)
    print(f"[Taichi] using backend: {used_arch}")

    preview = Preview2D(resolution=args.resolution)
    palette: Dict[int, np.ndarray] = {}
    color_wheel = cycle(COLOR_CYCLE)
    frame_iter = enumerate_frames(cache_dir, args.max_frames)

    labeled_frames: Iterator[Tuple[np.ndarray, np.ndarray]] = (
        _load_frame(path, palette, color_wheel) for path in frame_iter
    )

    mp4_path = preview.write_video(
        frames=generate_draw_batches(
            labeled_frames, preview, args.pause, args.headless
        ),
        radius=args.radius,
        output_dir=output_dir,
        framerate=args.framerate,
        display=not args.headless,
        build_gif=args.gif,
    )
    print(f"Video saved to {mp4_path}")


def _load_frame(
    path: Path,
    palette: Dict[int, np.ndarray],
    wheel: Iterator[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    positions = data["positions"]
    material_ids = data["material_ids"].astype(np.int32)
    colors = colorize(material_ids, palette, wheel)
    return positions, colors


def generate_draw_batches(
    labeled_frames: Iterator[Tuple[np.ndarray, np.ndarray]],
    preview: Preview2D,
    pause_cycles: int,
    headless: bool,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    for positions, colors in labeled_frames:
        yield positions, colors
        if not headless:
            for _ in range(max(0, pause_cycles - 1)):
                preview.draw_frame(positions, colors=colors, display=True)


def ensure_supported_cache(cache_dir: Path) -> None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    dim = meta.get("dimension")
    if dim is None:
        return
    if dim != 2:
        raise ValueError(
            f"Previewer currently supports only 2D caches; got dimension={dim}"
        )


if __name__ == "__main__":
    main()
