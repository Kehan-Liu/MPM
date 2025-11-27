from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2


def build_video(input_dir: str, out_path: str, fps: int) -> None:
    pattern = os.path.join(input_dir, "frame_*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No frames found under {input_dir}")

    first = cv2.imread(files[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {files[0]}")
    h, w, _ = first.shape

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer. Try a different fourcc or check OpenCV build.")

    for idx, f in enumerate(files, 1):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: skip unreadable frame: {f}")
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)
        if idx % 50 == 0:
            print(f"Written {idx}/{len(files)} frames...")

    writer.release()
    print(f"Video saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble PNG frames into MP4 using OpenCV")
    ap.add_argument("--input", required=True, help="Directory containing frame_XXXX.png")
    ap.add_argument("--out", required=True, help="Output MP4 path")
    ap.add_argument("--fps", type=int, default=60, help="Frames per second")
    args = ap.parse_args()

    build_video(args.input, args.out, args.fps)


if __name__ == "__main__":
    main()
