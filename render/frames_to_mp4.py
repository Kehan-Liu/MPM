from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2


def build_video(
    input_dir: str, out_path: str, source_fps: int, target_fps: int = 60
) -> None:
    pattern = os.path.join(input_dir, "frame_*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No frames found under {input_dir}")

    first = cv2.imread(files[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {files[0]}")
    h, w, _ = first.shape

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Try H.264 (avc1) first as it is compatible with VS Code / Web browsers
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, target_fps, (w, h))

    if not writer.isOpened():
        print("Warning: 'avc1' codec not found. Falling back to 'mp4v'.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, target_fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(
            "Failed to open video writer. Try a different fourcc or check OpenCV build."
        )
    # Resample frames in time so duration = (num_source_frames / source_fps)
    # and output is written at `target_fps`. We map each output frame index j
    # to a source frame index src_idx = floor(j * N / M).
    N = len(files)
    if source_fps <= 0:
        raise ValueError("source_fps must be > 0")
    duration = N / float(source_fps)
    M = max(1, int(round(duration * float(target_fps))))

    def src_index_for_out(j: int) -> int:
        # map j in [0..M-1] -> src in [0..N-1]
        return min(N - 1, int((j * N) / M))

    last_src = -1
    last_img = None
    for j in range(M):
        si = src_index_for_out(j)
        if si != last_src:
            f = files[si]
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: skip unreadable frame: {f}")
                # reuse last_img if available
                if last_img is None:
                    continue
                img = last_img
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            last_img = img
            last_src = si
        writer.write(last_img)
        if (j + 1) % 50 == 0 or j == M - 1:
            print(f"Written {j+1}/{M} output frames (source frames: {N})...")

    writer.release()
    print(
        f"Video saved to {out_path} (duration {duration:.3f}s, target_fps={target_fps}, output_frames={M})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assemble PNG frames into MP4 using OpenCV"
    )
    ap.add_argument(
        "--input", required=True, help="Directory containing frame_XXXX.png"
    )
    ap.add_argument("--out", required=True, help="Output MP4 path")
    ap.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Source frames per second (original capture fps)",
    )
    ap.add_argument(
        "--target-fps",
        type=int,
        default=60,
        help="Output video fps (playback rate), default 60",
    )
    args = ap.parse_args()

    build_video(args.input, args.out, args.fps, args.target_fps)


if __name__ == "__main__":
    main()
