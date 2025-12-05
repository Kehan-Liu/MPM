import argparse
import os
import subprocess
import sys
import glob

# Add the current directory to sys.path to allow importing frames_to_mp4
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from frames_to_mp4 import build_video
except ImportError:
    print("Warning: Could not import frames_to_mp4. Video generation might fail.")
    build_video = None


def render_demo(demo_name, start_frame=0, end_frame=None, skip_existing=False):
    # Paths
    workspace_root = os.getcwd()  # Assuming run from root as per user example
    results_dir = os.path.join(workspace_root, "results", demo_name)
    output_dir = os.path.join(workspace_root, "render_output", demo_name)

    os.makedirs(output_dir, exist_ok=True)

    # Find all mpm ply files
    ply_pattern = os.path.join(results_dir, "frame_*_mpm.ply")
    ply_files = sorted(glob.glob(ply_pattern))

    if not ply_files:
        print(f"No ply files found in {results_dir}")
        return

    print(f"Found {len(ply_files)} frames to render for demo '{demo_name}'")

    rendered_count = 0

    for ply_file in ply_files:
        filename = os.path.basename(ply_file)
        # filename is frame_XXXX_mpm.ply
        # extract frame number
        try:
            parts = filename.split("_")
            # parts: ['frame', '0200', 'mpm.ply']
            frame_str = parts[1]
            frame_num = int(frame_str)
        except (IndexError, ValueError):
            print(f"Skipping malformed filename: {filename}")
            continue

        if start_frame is not None and frame_num < start_frame:
            continue
        if end_frame is not None and frame_num > end_frame:
            continue

        # Construct paths
        json_file = os.path.join(results_dir, f"frame_{frame_str}_rigid.json")
        output_png = os.path.join(output_dir, f"frame_{frame_str}.png")

        if not os.path.exists(json_file):
            print(f"Warning: Missing rigid json for frame {frame_num}: {json_file}")

        if skip_existing and os.path.exists(output_png):
            print(f"Skipping existing frame {frame_num}")
            rendered_count += 1
            continue

        print(f"Rendering frame {frame_num}...")

        # Command
        # blender ./assets/assets.blend -b -P render/render_frame.py -- <ply> <json> <out>
        cmd = [
            "blender",
            os.path.join("assets", "assets.blend"),
            "-b",
            "-P",
            os.path.join("render", "render_frame.py"),
            "--",
            ply_file,
            json_file,
            output_png,
        ]

        try:
            # Suppress stdout/stderr to keep it clean, or let it show?
            # Blender output is verbose. Maybe redirect to DEVNULL unless error?
            # But user might want to see progress.
            # Let's keep it visible but maybe we can just print a progress bar?
            # For now, let's just run it.
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            rendered_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error rendering frame {frame_num}:")
            if e.stderr:
                print(e.stderr.decode())
            # Decide whether to stop or continue. Usually continue.

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

    args = parser.parse_args()

    render_demo(args.demo_name, args.start, args.end, args.skip)
