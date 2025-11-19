from __future__ import annotations

from pathlib import Path

from src.core.particles import ParticleBatch
from src.core.solver import MLSMPMSolver, SolverConfig
from src.emitters import BoxEmitterConfig, build_box
from src.materials import MaterialCategory, MaterialConfig, MaterialRegistry


def build_scene(
    output_dir: str | Path, frames: int = 5
) -> tuple[MLSMPMSolver, int, Path]:
    registry = MaterialRegistry()
    water_id = registry.register(
        MaterialConfig(
            name="water",
            density=1000.0,
            youngs_modulus=1e4,
            poisson_ratio=0.3,
            viscosity=0.02,
            category=MaterialCategory.FLUID,
            color=(0.2, 0.4, 0.9),
        )
    )
    jelly_id = registry.register(
        MaterialConfig(
            name="jelly",
            density=800.0,
            youngs_modulus=4e3,
            poisson_ratio=0.1,
            hardening=5.0,
            category=MaterialCategory.ELASTIC,
            color=(0.95, 0.6, 0.3),
        )
    )
    cloth_id = registry.register(
        MaterialConfig(
            name="cloth",
            density=500.0,
            youngs_modulus=2e5,
            poisson_ratio=0.3,
            hardening=2.0,
            category=MaterialCategory.CLOTH,
            color=(0.8, 0.8, 0.8),
        )
    )

    solver = MLSMPMSolver(
        material_registry=registry,
        config=SolverConfig(
            dim=2, grid_resolution=128, dt=2e-4, substeps=25, max_particles=120_000
        ),
    )
    solver.add_collider(point=(0.0, 0.02), normal=(0.0, 1.0), friction=0.3)

    batches: list[ParticleBatch] = []
    batches.append(
        build_box(
            BoxEmitterConfig(
                lower_corner=(0.25, 0.3),
                upper_corner=(0.45, 0.55),
                spacing=0.01,
                velocity=(0.0, -0.5),
                material_id=water_id,
                jitter=0.1,
            )
        )
    )
    batches.append(
        build_box(
            BoxEmitterConfig(
                lower_corner=(0.55, 0.35),
                upper_corner=(0.75, 0.55),
                spacing=0.0125,
                velocity=(0.0, 0.0),
                material_id=jelly_id,
            )
        )
    )
    # batches.append(
    #     build_box(
    #         BoxEmitterConfig(
    #             lower_corner=(0.2, 0.65),
    #             upper_corner=(0.8, 0.7),
    #             spacing=0.01,
    #             velocity=(0.0, -0.2),
    #             material_id=cloth_id,
    #         )
    #     )
    # )
    for batch in batches:
        solver.add_particles(batch)

    return solver, frames, Path(output_dir)
