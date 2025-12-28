import argparse
import concurrent.futures
import glob
import os
import subprocess
import sys


def _parse_frame_str_from_ply(ply_path: str) -> str:
    # supports: frame_0000.ply
    name = os.path.basename(ply_path)
    if not (name.startswith("frame_") and name.endswith(".ply")):
        raise ValueError(f"Unexpected particle filename: {name}")
    return name[len("frame_") : -len(".ply")]


def _render_one(task):
    (
        workspace_root,
        ply_path,
        rigid_dir,
        frame_str,
        output_png,
        skip_existing,
        suppress_blender_output,
    ) = task

    if skip_existing and os.path.exists(output_png):
        return True

    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    cmd = [
        "blender",
        os.path.join(workspace_root, "assets", "assets.blend"),
        "-b",
        "-P",
        os.path.join(workspace_root, "render", "render_granular.py"),
        "--",
        ply_path,
        rigid_dir,
        frame_str,
        output_png,
    ]

    try:
        if suppress_blender_output:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        else:
            subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error rendering frame {frame_str}")
        if getattr(e, "stderr", None):
            try:
                print(e.stderr.decode())
            except Exception:
                print(e.stderr)
        return False


def render_demo(
    demo_name: str,
    start: int,
    end: int | None,
    skip: bool,
    threads: int,
    out_dir: str | None,
    quiet: bool,
):
    workspace_root = os.getcwd()
    results_dir = os.path.join(workspace_root, "results", demo_name)
    particle_dir = os.path.join(results_dir, "particles")
    rigid_dir = os.path.join(results_dir, "rigid")

    if out_dir is None:
        output_dir = os.path.join(workspace_root, "render_output", demo_name)
    else:
        output_dir = out_dir

    ply_files = sorted(glob.glob(os.path.join(particle_dir, "frame_*.ply")))
    if not ply_files:
        print(f"No particle PLYs found in {particle_dir}")
        return

    tasks = []
    for ply_path in ply_files:
        try:
            frame_str = _parse_frame_str_from_ply(ply_path)
            frame_num = int(frame_str)
        except Exception:
            continue

        if frame_num < start:
            continue
        if end is not None and frame_num > end:
            continue

        output_png = os.path.join(output_dir, f"frame_{frame_str}.png")
        tasks.append(
            (
                workspace_root,
                ply_path,
                rigid_dir,
                frame_str,
                output_png,
                skip,
                quiet,
            )
        )

    if not tasks:
        print("No frames selected.")
        return

    threads = max(1, int(threads))
    print(f"Rendering {len(tasks)} frames with {threads} worker(s)")

    if threads == 1:
        ok = sum(_render_one(t) for t in tasks)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
            ok = sum(ex.map(_render_one, tasks))

    print(f"Done. Successful frames: {ok}/{len(tasks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render granular (particle) demos without reconstruction."
    )
    parser.add_argument("demo_name", help="Folder name under results/")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--skip", action="store_true", help="Skip frames whose PNG already exists"
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of parallel Blender processes"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: render_output/<demo_name>)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Blender stdout; print stderr on failure",
    )

    args = parser.parse_args()
    render_demo(
        args.demo_name,
        args.start,
        args.end,
        args.skip,
        args.threads,
        args.out,
        args.quiet,
    )
