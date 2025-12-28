import argparse
import os
import subprocess
import sys
import glob
import concurrent.futures

# Add the current directory to sys.path to allow importing frames_to_mp4
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from frames_to_mp4 import build_video
except ImportError:
    print("Warning: Could not import frames_to_mp4. Video generation might fail.")
    build_video = None


def process_frame(args):
    (
        frame_num,
        frame_str,
        ply_file,
        results_dir,
        output_dir,
        skip_existing,
        workspace_root,
    ) = args

    # Construct paths
    rigid_dir = os.path.join(results_dir, "rigid")
    output_png = os.path.join(output_dir, f"frame_{frame_str}.png")
    obj_file = os.path.join(results_dir, "particles", f"frame_{frame_str}_mpm.obj")

    if skip_existing and os.path.exists(output_png):
        print(f"Skipping existing frame {frame_num}")
        return True

    # Reconstruct
    if not os.path.exists(obj_file):
        print(f"Reconstructing frame {frame_num}...")
        reconstruct_cmd = [
            sys.executable,
            os.path.join(workspace_root, "render", "reconstruct_mesh.py"),
            ply_file,
            obj_file,
        ]
        try:
            subprocess.run(reconstruct_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error reconstructing frame {frame_num}: {e}")
            return False

    print(f"Rendering frame {frame_num}...")

    # Command
    cmd = [
        "blender",
        os.path.join(workspace_root, "assets", "assets.blend"),
        "-b",
        "-P",
        os.path.join(workspace_root, "render", "render_frame.py"),
        "--",
        obj_file,
        rigid_dir,
        frame_str,
        output_png,
    ]

    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error rendering frame {frame_num}:")
        if e.stderr:
            print(e.stderr.decode())
        return False


def render_demo(
    demo_name, start_frame=0, end_frame=None, skip_existing=False, num_threads=4
):
    # Paths
    workspace_root = os.getcwd()  # Assuming run from root as per user example
    results_dir = os.path.join(workspace_root, "results", demo_name)
    output_dir = os.path.join(workspace_root, "render_output", demo_name)

    os.makedirs(output_dir, exist_ok=True)

    # Find all mpm ply files
    ply_pattern = os.path.join(results_dir, "particles/frame_*.ply")
    ply_files = sorted(glob.glob(ply_pattern))

    if not ply_files:
        print(f"No ply files found in {results_dir}")
        return

    tasks = []
    for ply_file in ply_files:
        filename = os.path.basename(ply_file)
        # filename is frame_XXXX.ply
        # extract frame number
        try:
            parts = filename.split("_")
            # parts: ['frame', '0200.ply']
            frame_str = parts[1].split(".")[0]
            frame_num = int(frame_str)
        except (IndexError, ValueError):
            print(f"Skipping malformed filename: {filename}")
            continue

        if start_frame is not None and frame_num < start_frame:
            continue
        if end_frame is not None and frame_num > end_frame:
            continue

        tasks.append(
            (
                frame_num,
                frame_str,
                ply_file,
                results_dir,
                output_dir,
                skip_existing,
                workspace_root,
            )
        )

    print(
        f"Found {len(tasks)} frames to render for demo '{demo_name}' using {num_threads} threads"
    )

    rendered_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_frame, tasks))
        rendered_count = sum(results)

    print(f"Finished rendering {rendered_count} frames.")

    # Generate Video
    video_path = os.path.join(output_dir, f"{demo_name}.mp4")
    print(f"Generating video: {video_path}")

    if build_video:
        try:
            # build_video(input_dir, out_path, source_fps, target_fps=60)
            # The frames are in output_dir named frame_XXXX.png
            # frames_to_mp4 expects frame_*.png
            build_video(output_dir, video_path, source_fps=24, target_fps=30)
            print("Video generation complete.")
        except Exception as e:
            print(f"Failed to generate video: {e}")
    else:
        print("Video generation skipped (module not found).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render MPM demo frames and create video."
    )
    parser.add_argument("demo_name", help="Name of the demo (folder in results/)")
    parser.add_argument("--start", type=int, default=0, help="Start frame number")
    parser.add_argument("--end", type=int, default=None, help="End frame number")
    parser.add_argument("--skip", action="store_true", help="Skip existing frames")
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of threads to use"
    )

    args = parser.parse_args()

    render_demo(args.demo_name, args.start, args.end, args.skip, args.threads)
