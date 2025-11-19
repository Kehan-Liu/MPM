# Simulation & Rendering Pipeline

1. **Scene authoring**
   - Define emitters, colliders, and materials inside `scenes/` using Python configs.
   - Each scene chooses which material models to register, their constitutive parameters, and frame ranges.

2. **Simulation**
   - Core MLS-MPM solver (Taichi) runs on CPU or GPU depending on configuration.
   - Outputs per-frame particle/grid snapshots into `outputs/<scene_name>/frame_xxxx.npz` via the IO module.

3. **Post-processing**
   - Optional filters (smoothing, attribute baking, meshing) convert NPZ caches into PLY, USD, or OpenVDB for rendering.
   - Preview rendering via `src/render/preview.py` for interactive inspection.

4. **Offline rendering**
   - Blender Cycles: import PLY or USD caches, assign shading, and render to EXR.
   - USD pipeline: feed USD stages to Omniverse or Houdini Solaris/Karma for large scenes.

5. **Automation**
   - Scripts under `scripts/` orchestrate batch simulations (`scripts/run_scene.py`) and conversions (`scripts/export_usd.py`).
