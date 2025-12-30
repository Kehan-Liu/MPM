import subprocess
import sys
import os


def reconstruct_mesh(ply_path, output_obj_path):
    """
    Reconstructs mesh from PLY particles using splashsurf.
    """
    if not os.path.exists(ply_path):
        print(f"Error: Input file {ply_path} does not exist.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_obj_path)), exist_ok=True)

    # Parameters based on user request
    # -r=0.02 (particle radius)
    # -l=2.0 (smoothing kernel radius)
    # -c=0.2 (cube radius / voxel size)
    # -t=0.6 (density threshold)

    cmd = [
        "splashsurf",
        "reconstruct",
        ply_path,
        "-r=0.02",
        "-l=1.0",
        "-c=0.2",
        "-t=0.6",
        "--mesh-smoothing-weights=on",
        "--mesh-smoothing-iters=2",
        "--normals=on",
        "--normals-smoothing-iters=2",
        "-o",
        output_obj_path,
    ]

    print(f"Running reconstruction: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Reconstruction successful: {output_obj_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during reconstruction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python reconstruct_mesh.py <input_ply> <output_obj>")
        sys.exit(1)

    reconstruct_mesh(sys.argv[1], sys.argv[2])
