import argparse
import os
import subprocess
import sys
import tempfile
import shutil

import numpy as np
from plyfile import PlyData, PlyElement


def reconstruct_mesh(ply_path, output_obj_path):
    """
    Reconstructs mesh from PLY particles using pysplashsurf.
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

    # Prefer the console entrypoint if available; otherwise fall back to
    # `python -m pysplashsurf` which is more robust across environments.
    exe = shutil.which("pysplashsurf")
    if exe is None:
        cmd = [sys.executable, "-m", "pysplashsurf", "reconstruct", ply_path]
    else:
        cmd = [exe, "reconstruct", ply_path]

    cmd += [
        "-r=0.02",
        "-l=2.0",
        "-c=0.2",
        "-t=0.6",
        "--mesh-smoothing-weights=on",
        "--mesh-smoothing-iters=5",
        "--normals=on",
        "--normals-smoothing-iters=5",
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


def _sanitize_ply_positions(
    ply_path: str, *, keep: bool = False
) -> tuple[str, int, int]:
    """Drop vertices with non-finite x/y/z.

    SplashSurf (0.14.0) can panic with `Option::unwrap()` when NaN/Inf values are
    present in the input point cloud. We defensively filter them.

    Returns (path_to_use, n_in, n_out).
    """

    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError(f"PLY has no 'vertex' element: {ply_path}")
    v = ply["vertex"].data
    for key in ("x", "y", "z"):
        if key not in v.dtype.names:
            raise ValueError(f"PLY vertex element missing '{key}': {ply_path}")

    x = np.asarray(v["x"], dtype=np.float64)
    y = np.asarray(v["y"], dtype=np.float64)
    z = np.asarray(v["z"], dtype=np.float64)
    xyz = np.stack([x, y, z], axis=1)
    finite = np.isfinite(xyz).all(axis=1)

    n_in = int(xyz.shape[0])
    n_out = int(finite.sum())
    if n_out == n_in:
        return ply_path, n_in, n_out

    if n_out == 0:
        raise ValueError(f"All {n_in} vertices are non-finite in {ply_path}")

    sanitized = xyz[finite].astype(np.float32, copy=False)
    out_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    out_data = np.empty(n_out, dtype=out_dtype)
    out_data["x"] = sanitized[:, 0]
    out_data["y"] = sanitized[:, 1]
    out_data["z"] = sanitized[:, 2]

    ply_el = PlyElement.describe(out_data, "vertex")

    if keep:
        base, _ = os.path.splitext(ply_path)
        out_path = base + "_sanitized.ply"
    else:
        tmp = tempfile.NamedTemporaryFile(
            prefix="mpm_sanitized_", suffix=".ply", delete=False
        )
        out_path = tmp.name
        tmp.close()

    PlyData([ply_el], text=False).write(out_path)
    return out_path, n_in, n_out


def reconstruct_mesh_cli(
    ply_path: str, output_obj_path: str, *, sanitize: bool, keep_sanitized: bool
):
    ply_to_use = ply_path
    tmp_to_delete: str | None = None
    if sanitize:
        ply_to_use, n_in, n_out = _sanitize_ply_positions(ply_path, keep=keep_sanitized)
        if ply_to_use != ply_path:
            print(f"Sanitized PLY: dropped {n_in - n_out} / {n_in} non-finite vertices")
            if not keep_sanitized:
                tmp_to_delete = ply_to_use

    try:
        reconstruct_mesh(ply_to_use, output_obj_path)
    finally:
        if tmp_to_delete is not None:
            try:
                os.remove(tmp_to_delete)
            except OSError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct mesh from particle PLY using pysplashsurf."
    )
    parser.add_argument("input_ply", help="Input PLY with vertex x/y/z")
    parser.add_argument("output_obj", help="Output OBJ path")
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable filtering NaN/Inf vertices before reconstruction",
    )
    parser.add_argument(
        "--keep-sanitized",
        action="store_true",
        help="Keep the sanitized PLY file next to the input (for debugging)",
    )
    args = parser.parse_args()

    reconstruct_mesh_cli(
        args.input_ply,
        args.output_obj,
        sanitize=not args.no_sanitize,
        keep_sanitized=args.keep_sanitized,
    )
