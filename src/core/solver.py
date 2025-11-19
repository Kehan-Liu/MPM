from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import taichi as ti

from .particles import ParticleBatch
from ..io import ParticleIO
from ..materials import MaterialCategory, MaterialRegistry

CATEGORY_FLUID = int(MaterialCategory.FLUID)
CATEGORY_CLOTH = int(MaterialCategory.CLOTH)


@dataclass
class SolverConfig:
    dim: int = 2
    grid_resolution: int = 128
    dt: float = 2e-4
    substeps: int = 10
    max_particles: int = 200_000
    particle_spacing: float = 0.5


@ti.data_oriented
class MLSMPMSolver:
    def __init__(
        self,
        material_registry: MaterialRegistry,
        config: SolverConfig,
    ) -> None:
        if config.dim not in (2, 3):
            raise ValueError("Only 2D or 3D simulations are supported")
        self.dim = config.dim
        self.cfg = config
        self.registry = material_registry
        payload = material_registry.to_solver_payload()
        self.num_materials = len(payload["density"])
        self.dx = 1.0 / config.grid_resolution
        self.inv_dx = 1.0 / self.dx
        self.particles_ready = False
        self.pending_batches: List[ParticleBatch] = []

        grid_shape = (config.grid_resolution,) * self.dim
        self.grid_v = ti.Vector.field(self.dim, dtype=ti.f32, shape=grid_shape)
        self.grid_m = ti.field(dtype=ti.f32, shape=grid_shape)

        self.max_particles = config.max_particles
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_particles)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_particles)
        self.F = ti.Matrix.field(
            self.dim, self.dim, dtype=ti.f32, shape=self.max_particles
        )
        self.C = ti.Matrix.field(
            self.dim, self.dim, dtype=ti.f32, shape=self.max_particles
        )
        self.Jp = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.material_id = ti.field(dtype=ti.i32, shape=self.max_particles)
        self.particle_count = ti.field(dtype=ti.i32, shape=())

        self.gravity = ti.Vector.field(self.dim, dtype=ti.f32, shape=())
        default_g = [0.0] * self.dim
        if self.dim >= 2:
            default_g[1] = -9.81
        else:
            default_g[0] = -9.81
        self.gravity[None] = ti.Vector(default_g)

        self.collider_capacity = 8
        self.collider_count = ti.field(dtype=ti.i32, shape=())
        self.collider_points = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=self.collider_capacity
        )
        self.collider_normals = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=self.collider_capacity
        )
        self.collider_friction = ti.field(dtype=ti.f32, shape=self.collider_capacity)

        self.material_density = ti.field(dtype=ti.f32, shape=self.num_materials)
        self.material_mu = ti.field(dtype=ti.f32, shape=self.num_materials)
        self.material_lambda = ti.field(dtype=ti.f32, shape=self.num_materials)
        self.material_bulk = ti.field(dtype=ti.f32, shape=self.num_materials)
        self.material_hardening = ti.field(dtype=ti.f32, shape=self.num_materials)
        self.material_viscosity = ti.field(dtype=ti.f32, shape=self.num_materials)
        self.material_category = ti.field(dtype=ti.i32, shape=self.num_materials)
        self.material_color = ti.Vector.field(3, dtype=ti.f32, shape=self.num_materials)

        for idx in range(self.num_materials):
            self.material_density[idx] = payload["density"][idx]
            self.material_mu[idx] = payload["mu"][idx]
            self.material_lambda[idx] = payload["lambda"][idx]
            self.material_bulk[idx] = payload["bulk"][idx]
            self.material_hardening[idx] = payload["hardening"][idx]
            self.material_viscosity[idx] = payload["viscosity"][idx]
            self.material_category[idx] = payload["category"][idx]
            self.material_color[idx] = ti.Vector(payload["color"][idx])

        self.p_vol = (self.cfg.particle_spacing * self.dx) ** self.dim

    def set_gravity(self, vector: Sequence[float]) -> None:
        arr = list(vector)
        if len(arr) != self.dim:
            raise ValueError("Gravity vector dimension mismatch")
        self.gravity[None] = ti.Vector(list(vector))

    def add_collider(
        self, point: Sequence[float], normal: Sequence[float], friction: float = 0.2
    ) -> None:
        if self.collider_count[None] >= self.collider_capacity:
            raise RuntimeError("Exceeded collider capacity")
        import numpy as np

        point_arr = np.asarray(point, dtype=np.float32)
        normal_arr = np.asarray(normal, dtype=np.float32)
        if point_arr.shape[0] != self.dim or normal_arr.shape[0] != self.dim:
            raise ValueError("Collider dimension mismatch")
        norm = float(np.linalg.norm(normal_arr))
        if norm == 0.0:
            raise ValueError("Collider normal cannot be zero")
        normal_arr = normal_arr / norm
        point_vec = ti.Vector(point_arr.tolist())
        normal_vec = ti.Vector(normal_arr.tolist())
        idx = self.collider_count[None]
        self.collider_points[idx] = point_vec
        self.collider_normals[idx] = normal_vec.normalized()
        self.collider_friction[idx] = friction
        self.collider_count[None] += 1

    def add_particles(self, batch: ParticleBatch) -> None:
        if batch.positions.shape[1] != self.dim:
            raise ValueError("Particle dimension mismatch")
        self.pending_batches.append(batch)
        total = sum(b.positions.shape[0] for b in self.pending_batches)
        if total > self.max_particles:
            raise RuntimeError("Particle buffer exceeds solver capacity")
        self.particles_ready = False

    def _ensure_device_particles(self) -> None:
        if self.particles_ready:
            return
        if not self.pending_batches:
            raise RuntimeError("No particles have been registered")
        positions = np.concatenate([b.positions for b in self.pending_batches], axis=0)
        velocities = np.concatenate(
            [
                b.velocities if b.velocities is not None else np.zeros_like(b.positions)
                for b in self.pending_batches
            ],
            axis=0,
        )
        material_ids = np.concatenate(
            [b.material_ids for b in self.pending_batches], axis=0
        ).astype(np.int32)
        count = positions.shape[0]
        self.x.from_numpy(pad_array(positions, self.max_particles))
        self.v.from_numpy(pad_array(velocities, self.max_particles))
        self.material_id.from_numpy(pad_scalar_array(material_ids, self.max_particles))
        self.particle_count[None] = count
        self._init_particle_state()
        self.particles_ready = True

    def step(self, num_substeps: int | None = None) -> None:
        self._ensure_device_particles()
        steps = num_substeps or self.cfg.substeps
        for _ in range(steps):
            self._substep()

    def run(self, total_frames: int, output_dir: Path, save_every: int = 1) -> None:
        self._ensure_device_particles()
        writer = ParticleIO(
            output_dir=Path(output_dir),
            grid_resolution=self.cfg.grid_resolution,
            dimension=self.dim,
        )
        for frame in range(total_frames):
            self.step(self.cfg.substeps)
            if frame % save_every == 0:
                data = self.to_numpy()
                writer.save_frame(frame, data["x"], data["v"], data["material_id"])

    def to_numpy(self) -> dict[str, np.ndarray]:
        count = self.particle_count[None]
        return {
            "x": self.x.to_numpy()[:count],
            "v": self.v.to_numpy()[:count],
            "material_id": self.material_id.to_numpy()[:count],
        }

    @ti.kernel
    def _substep(self):
        # for I in ti.grouped(self.grid_m):
        #     self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
        #     self.grid_m[I] = 0.0
        self.grid_v.fill(0)
        self.grid_m.fill(0)

        # P2G
        for p in range(self.particle_count[None]):
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.cfg.dt * self.C[p]) @ self.F[p]
            Xp = self.x[p]
            Vp = self.v[p]
            Cp = self.C[p]
            Fp = self.F[p]
            mat = self.material_id[p]
            base = (Xp * self.inv_dx - 0.5).cast(int)
            fx = Xp * self.inv_dx - base.cast(float)
            w = [
                ti.Vector(
                    [
                        0.5 * (1.5 - fx[d]) ** 2,
                        0.75 - (fx[d] - 1.0) ** 2,
                        0.5 * (fx[d] - 0.5) ** 2,
                    ]
                )
                for d in ti.static(range(self.dim))
            ]
            Je = Fp.determinant()
            harden = ti.exp(self.material_hardening[mat] * (1.0 - Je))
            mu = self.material_mu[mat] * harden
            la = self.material_lambda[mat] * harden
            if self.material_category[mat] == CATEGORY_FLUID:
                mu = 0.0
                la = self.material_bulk[mat]
            r, _ = ti.polar_decompose(Fp)
            stress = (
                2 * mu * (Fp - r) @ Fp.transpose()
                + ti.Matrix.identity(ti.f32, self.dim) * la * (Je - 1.0) * Je
            )
            stress = -self.cfg.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx * stress
            mass_p = self.material_density[mat] * self.p_vol
            affine = stress + mass_p * Cp

            for offset in ti.grouped(ti.ndrange(*((3,) * self.dim))):
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[d][offset[d]]
                grid_idx = base + offset
                inside = True
                for d in ti.static(range(self.dim)):
                    inside = inside and 0 <= grid_idx[d] < self.cfg.grid_resolution
                if not inside:
                    continue
                dpos = (offset.cast(float) - fx) * self.dx
                ti.atomic_add(self.grid_v[grid_idx], weight * (mass_p * Vp + affine @ dpos))
                ti.atomic_add(self.grid_m[grid_idx], weight * mass_p)

        # Grid update
        for I in ti.grouped(self.grid_m):
            mass = self.grid_m[I]
            if mass > 0:
                v = (1.0 / mass) * self.grid_v[I]
                v += self.cfg.dt * self.gravity[None]
                coord = (I.cast(float) + 0.5) * self.dx
                for cid in range(self.collider_count[None]):
                    normal = self.collider_normals[cid]
                    point = self.collider_points[cid]
                    signed = (coord - point).dot(normal)
                    if signed < 0:
                        vn = v.dot(normal)
                        vt = v - vn * normal
                        if vn < 0:
                            friction = self.collider_friction[cid]
                            vt_norm = vt.norm() + 1e-6
                            max_t = -vn * friction
                            if vt_norm > max_t:
                                vt = vt / vt_norm * max_t
                            v = vt
                for d in ti.static(range(self.dim)):
                    if I[d] < 2 and v[d] < 0:
                        v[d] = 0
                    if I[d] > self.cfg.grid_resolution - 3 and v[d] > 0:
                        v[d] = 0
                self.grid_v[I] = v

        # G2P
        for p in range(self.particle_count[None]):
            Xp = self.x[p]
            base = (Xp * self.inv_dx - 0.5).cast(int)
            fx = Xp * self.inv_dx - base.cast(float)
            w = [
                ti.Vector(
                    [
                        0.5 * (1.5 - fx[d]) ** 2,
                        0.75 - (fx[d] - 1.0) ** 2,
                        0.5 * (fx[d] - 0.5) ** 2,
                    ]
                )
                for d in ti.static(range(self.dim))
            ]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            for offset in ti.grouped(ti.ndrange(*((3,) * self.dim))):
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[d][offset[d]]
                grid_idx = base + offset
                inside = True
                for d in ti.static(range(self.dim)):
                    inside = inside and 0 <= grid_idx[d] < self.cfg.grid_resolution
                if not inside:
                    continue
                g_v = self.grid_v[grid_idx]
                dpos = offset.cast(float) - fx
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p] = new_v
            self.x[p] = Xp + self.cfg.dt * new_v
            self.C[p] = new_C
            for d in ti.static(range(self.dim)):
                self.x[p][d] = ti.min(ti.max(self.x[p][d], 0.0), 0.999)

    @ti.kernel
    def _init_particle_state(self):
        for p in range(self.particle_count[None]):
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            self.Jp[p] = 1.0


def pad_array(data: np.ndarray, target: int) -> np.ndarray:
    if data.shape[0] == target:
        return data
    padded = np.zeros((target,) + data.shape[1:], dtype=data.dtype)
    padded[: data.shape[0]] = data
    return padded


def pad_scalar_array(data: np.ndarray, target: int) -> np.ndarray:
    if data.shape[0] == target:
        return data
    padded = np.zeros((target,), dtype=data.dtype)
    padded[: data.shape[0]] = data
    return padded
