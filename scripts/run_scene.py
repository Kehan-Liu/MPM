from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import taichi as ti

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runtime import auto_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an MLS-MPM scene")
    parser.add_argument(
        "--scene",
        type=str,
        default="fluid_cloth_demo",
        help="Scene module name under scenes/",
    )
    parser.add_argument(
        "--frames", type=int, default=5, help="Number of frames to simulate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/demo"),
        help="Folder to flush particle caches",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "vulkan"],
        help="Taichi backend",
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


def main() -> None:
    args = parse_args()
    arch = resolve_arch(args.arch)
    used_arch = auto_init(arch=arch)
    print(f"[Taichi] using backend: {used_arch}")
    scene_module = importlib.import_module(f"scenes.{args.scene}")
    solver, frames, output_dir = scene_module.build_scene(
        output_dir=args.output, frames=args.frames
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solver.run(total_frames=frames, output_dir=output_dir)
    print(f"Simulation completed. Frames stored in {output_dir}")


if __name__ == "__main__":
    main()
