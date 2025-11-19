# Taichi MLS-MPM Laboratory

This repository hosts a modular Material Point Method sandbox built with [Taichi](https://www.taichi-lang.org/) for experimenting with coupled fluid, deformable solid, and cloth simulations, and exporting them for offline rendering.

## Repository layout

```
mpm/
├── assets/              # Static geometry, textures, reference meshes for emitters/colliders
├── docs/                # Design notes, equations, pipeline diagrams
├── scenes/              # High-level simulation setups (Python or YAML configs)
├── scripts/             # CLI utilities (batch runners, render submitters)
├── src/
│   ├── core/            # Grid-particle data structures, MLS-MPM solver loop
│   ├── materials/       # Constitutive models + registry (fluids, elastoplastics, cloth)
│   ├── emitters/        # Volume/mesh emitters and particle seeding utilities
│   ├── colliders/       # Analytic/mesh colliders and coupling hooks
│   ├── io/              # Checkpointing, particle caches, USD/PLY/NPZ exporters
│   └── render/          # Preview viewers, offline export shims
└── requirements.txt / pyproject.toml
```

## High-level roadmap

1. **Core solver** – Implement a modular MLS-MPM solver that supports multiple materials by keeping material IDs and constitutive parameters per particle.
2. **Scene authoring** – Provide emitter/collider factories and configuration loading so fluids, deformables, and cloth can be combined quickly.

## Getting started

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Run the reference scene

```bash
python scripts/run_scene.py --scene fluid_cloth_demo --frames 5 --output outputs/fluid_cloth
```

This launches the Taichi runtime, builds the coupled fluid/elastic/cloth scene defined in `scenes/fluid_cloth_demo.py`, and writes compressed particle caches (`frame_XXXX.npz`) plus a `manifest.json` to the chosen output folder.

### 3. Build a GUI video preview for 2D result

```bash
python scripts/view_particles.py outputs/fluid_cloth_cpu --arch cpu --framerate 24 --max-frames 120 --gif
```

This script replays cached particle frames with the Taichi GUI, records them via `ti.tools.VideoManager`, and emits MP4 (and optional GIF) previews in `outputs/fluid_cloth_cpu/preview`.