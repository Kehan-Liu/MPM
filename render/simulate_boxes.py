from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Box:
    cx: float
    cy: float
    cz: float
    hx: float
    hy: float
    hz: float
    vx: float
    vy: float
    vz: float


def resolve_collision_x(b0: Box, b1: Box, restitution: float) -> None:
    dx = b1.cx - b0.cx
    overlap = (b0.hx + b1.hx) - abs(dx)
    if overlap > 0.0:
        n = 1.0 if dx >= 0.0 else -1.0
        b0.cx -= n * overlap * 0.5
        b1.cx += n * overlap * 0.5
        v_rel = b1.vx - b0.vx
        new_rel = -restitution * v_rel
        v_com = 0.5 * (b0.vx + b1.vx)
        b0.vx = v_com - 0.5 * new_rel
        b1.vx = v_com + 0.5 * new_rel


def simulate(frames: int, dt: float, restitution: float, speed0: float, speed1: float) -> Dict:
    b0 = Box(cx=-1.5, cy=0.5, cz=0.0, hx=0.5, hy=0.5, hz=0.5, vx=abs(speed0), vy=0.0, vz=0.0)
    b1 = Box(cx=1.5, cy=0.5, cz=0.0, hx=0.5, hy=0.5, hz=0.5, vx=-abs(speed1), vy=0.0, vz=0.0)

    data: Dict = {"fps": int(round(1.0 / dt)), "frames": []}

    for _ in range(frames):
        b0.cx += b0.vx * dt
        b1.cx += b1.vx * dt
        resolve_collision_x(b0, b1, restitution)
        frame_boxes = [
            {
                "pos": [b0.cx, b0.cy, b0.cz],
                "rot": [0.0, 0.0, 0.0],
                "scale": [b0.hx * 2.0, b0.hy * 2.0, b0.hz * 2.0],
                "color": [0.2, 0.6, 0.9],
            },
            {
                "pos": [b1.cx, b1.cy, b1.cz],
                "rot": [0.0, 0.0, 0.0],
                "scale": [b1.hx * 2.0, b1.hy * 2.0, b1.hz * 2.0],
                "color": [0.9, 0.4, 0.2],
            },
        ]
        data["frames"].append({"boxes": frame_boxes})

    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple two-box 1D collision simulation for Blender rendering")
    ap.add_argument("--frames", type=int, default=240, help="Total frames to simulate")
    ap.add_argument("--dt", type=float, default=1.0 / 60.0, help="Time step per frame")
    ap.add_argument("--restitution", type=float, default=0.6, help="Coefficient of restitution for collision")
    ap.add_argument("--speed0", type=float, default=1.5, help="Initial speed of box 0 along +x")
    ap.add_argument("--speed1", type=float, default=1.0, help="Initial speed of box 1 along -x")
    ap.add_argument("--out", type=str, default=os.path.join("render", "output", "boxes_sim.json"), help="Output JSON path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sim = simulate(args.frames, args.dt, args.restitution, args.speed0, args.speed1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sim, f)

    print(f"Wrote simulation to {args.out}")


if __name__ == "__main__":
    main()
