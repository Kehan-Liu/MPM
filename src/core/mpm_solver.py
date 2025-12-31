import taichi as ti
from src.core.rigid import Rigid
from src.core.utils import *

# from src.core.cloth import Cloth
from src.objects import RigidObject, MPMObject, ClothObject, MPMModel
from src.scene import Scene
from typing import List
import numpy as np
from plyfile import PlyData, PlyElement
import os

CATEGORY_WATER = int(MPMModel.WATER.value)
CATEGORY_JELLY = int(MPMModel.JELLY.value)
CATEGORY_SNOW = int(MPMModel.SNOW.value)
CATEGORY_SAND = int(MPMModel.SAND.value)


@ti.data_oriented
class MPMSolver:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.dt = scene.dt
        self.n_grid = scene.n_grid
        self.dx = 1.0 / self.n_grid  # grid spacing
        self.inv_dx = float(self.n_grid)

        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity[None] = ti.Vector(scene.gravity)

        self.rigid = Rigid(scene.rigid_objects, self.dx, self.dt, scene.gravity, penalty_parameter=scene.penalty_parameter, clamp_factor=scene.clamp_factor)

        # self.cloth = Cloth(scene.cloth_objects, self.dx, self.dt)

        print("Initializing MPM Solver...")
        self.mpm_objects = scene.mpm_objects
        num_particles = sum([obj.num_particles for obj in scene.mpm_objects])
        self.p_x = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.p_v = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.p_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)
        self.p_Jp = ti.field(dtype=ti.f32, shape=num_particles)  # plastic deformation
        self.p_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)
        self.p_material = ti.field(dtype=ti.i32, shape=num_particles)
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.n_particles[None] = num_particles
        # self.p_vol = (self.dx * 0.5) ** 3

        self.num_mpm = len(scene.mpm_objects)
        self.material_density = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_mu = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_lambda = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_viscosity = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_hardening = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_stiffness = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_power = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_type = ti.field(dtype=ti.i32, shape=self.num_mpm)
        self.object_p_vol = ti.field(dtype=ti.f32, shape=self.num_mpm)

        grid_shape = (self.n_grid,) * 3
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=grid_shape)
        self.grid_m = ti.field(dtype=ti.f32, shape=grid_shape)

        # CPIC
        self.grid_d = ti.Vector.field(
            self.rigid.n_rigid, dtype=ti.f32, shape=grid_shape
        )  # shortest distance to rigid surface
        self.grid_A = ti.Vector.field(
            self.rigid.n_rigid, dtype=ti.i32, shape=grid_shape
        )
        self.grid_T = ti.Vector.field(
            self.rigid.n_rigid, dtype=ti.i32, shape=grid_shape
        )
        self.grid_r = ti.field(
            dtype=ti.i32, shape=grid_shape
        )  # which rigid body is the closest

        self.p_d = ti.Vector.field(
            self.rigid.n_rigid, dtype=ti.f32, shape=num_particles
        )  # shortest distance to rigid surface
        self.p_A = ti.Vector.field(
            self.rigid.n_rigid, dtype=ti.i32, shape=num_particles
        )
        self.p_T = ti.Vector.field(
            self.rigid.n_rigid, dtype=ti.i32, shape=num_particles
        )
        self.p_n = ti.Vector.field(
            3, dtype=ti.f32, shape=(num_particles, self.rigid.n_rigid)
        )  # normal of the closest rigid surface

        self.init_mpm_materials()
        self.init_mpm_particles()

        # Helpful hint: with higher resolution, stable/consistent dt often must shrink.
        self._print_dt_recommendation()

    def init_mpm_materials(self):
        print("Initializing MPM materials...")
        for i, obj in enumerate(self.mpm_objects):
            mat = obj.material
            self.material_density[i] = mat.density
            E = mat.E
            nu = mat.nu
            self.material_mu[i] = E / (2 * (1 + nu))
            self.material_lambda[i] = E * nu / ((1 + nu) * (1 - 2 * nu))
            self.material_viscosity[i] = float(getattr(mat, "viscosity", 0.0))
            self.material_hardening[i] = mat.hardening
            self.material_type[i] = int(mat.model.value)
            self.material_stiffness[i] = mat.stiffness
            self.material_power[i] = mat.power

    def _print_dt_recommendation(self, cfl: float = 0.4):
        """Print a rough CFL dt recommendation based on elastic wave speed.

        This is not a guarantee (contacts/collisions can require smaller dt),
        but it helps explain resolution-dependent instability/energy.
        """

        dx = self.dx
        c_max = 0.0
        for i, obj in enumerate(self.mpm_objects):
            mat = obj.material
            rho = float(getattr(mat, "density", 1000.0))
            if mat.model == MPMModel.WATER:
                # Weakly-compressible EOS: p = k * ((rho/rho0)^gamma - 1)
                # dp/drho|rho0 = k * gamma / rho0  =>  c = sqrt(k*gamma/rho0)
                k = float(getattr(mat, "stiffness", 200.0))
                gamma = float(getattr(mat, "power", 7))
                c = float(np.sqrt(max(k * gamma, 0.0) / max(rho, 1e-12)))
            else:
                E = float(getattr(mat, "E", 1e5))
                nu = float(getattr(mat, "nu", 0.2))
                # Lame parameters (same formulas used in Taichi fields)
                mu = E / (2 * (1 + nu))
                lam = E * nu / ((1 + nu) * (1 - 2 * nu + 1e-12))
                # Longitudinal wave speed for isotropic linear elastic material.
                c = float(np.sqrt(max(lam + 2.0 * mu, 0.0) / max(rho, 1e-12)))
            c_max = max(c_max, c)

        if c_max <= 0:
            return
        dt_rec = cfl * dx / c_max
        print(
            f"[MPM] dt={self.dt:.3e}, dx={dx:.3e}. Rough CFL dt≈{dt_rec:.3e} (CFL={cfl}, c_max≈{c_max:.3e})."
        )

    def init_mpm_particles(self):
        print("Initializing MPM particles...")
        all_p_x = []
        all_p_material = []

        for i, obj in enumerate(self.mpm_objects):
            # Load mesh
            try:
                if obj.meshdir in GEOMS:
                    mesh = GEOMS[obj.meshdir](obj.scale)
                else:
                    mesh = trimesh.load(obj.meshdir, force="mesh")
                if mesh.is_watertight:
                    vol = mesh.volume
                else:
                    vol = mesh.convex_hull.volume
            except Exception as e:
                print(f"Error loading mesh {obj.meshdir}: {e}")
                # Fallback: create a small box around position
                mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
                vol = 0.1**3

            self.object_p_vol[i] = vol / obj.num_particles

            # Apply translation
            mesh.apply_translation(obj.position)

            samples = []
            bounds = mesh.bounds

            iter_count = 0
            while len(samples) < obj.num_particles and iter_count < 100:
                print(
                    f"Sampling particles for object {i}, iteration {iter_count}, collected {len(samples)} particles"
                )
                needed = obj.num_particles - len(samples)
                batch_size = min(max(needed * 2, 5000), 50000)
                points = np.random.uniform(bounds[0], bounds[1], (batch_size, 3))
                print("start checking containment")
                inside = mesh.contains(points)
                print("containment check done")
                samples.extend(points[inside])
                iter_count += 1

            # Handle count mismatch
            if len(samples) > obj.num_particles:
                samples = samples[: obj.num_particles]
            while len(samples) < obj.num_particles:
                if len(samples) > 0:
                    samples.append(samples[0])
                else:
                    samples.append(obj.position)

            print(f"Final particle count for object {i}: {len(samples)}")
            all_p_x.append(np.array(samples))
            all_p_material.append(np.full(obj.num_particles, i, dtype=np.int32))

        if all_p_x:
            all_p_x_np = np.concatenate(all_p_x, axis=0)
            all_p_material_np = np.concatenate(all_p_material, axis=0)

            self.p_x.from_numpy(all_p_x_np.astype(np.float32))
            self.p_material.from_numpy(all_p_material_np)

            self.p_v.fill(0)

            F_np = np.repeat(
                np.eye(3, dtype=np.float32)[np.newaxis, :, :],
                self.n_particles[None],
                axis=0,
            )
            self.p_F.from_numpy(F_np)

            self.p_Jp.fill(1.0)
            self.p_C.fill(0)

    @ti.func
    def is_valid(self, I):
        return (
            0 <= I[0] < self.n_grid
            and 0 <= I[1] < self.n_grid
            and 0 <= I[2] < self.n_grid
        )

    @ti.kernel
    def reset_grid(self):
        self.grid_A.fill(0)
        self.grid_T.fill(0)
        self.grid_d.fill(1e10)
        self.grid_r.fill(-1)

    @ti.kernel
    def build_rigid_cdf(self):
        for p in range(self.rigid.n_boundary_particles[None]):
            rb_id = self.rigid.bp2rb[p]
            f_id = self.rigid.bp2f[p]
            bp_pos = self.rigid.get_bp_position(p)
            v0, v1, v2 = self.rigid.get_vertices_of_face(f_id)
            v0v1 = v1 - v0
            v0v2 = v2 - v0
            d00 = v0v1.dot(v0v1)
            d01 = v0v1.dot(v0v2)
            d11 = v0v2.dot(v0v2)
            # Guard against degenerate/near-degenerate triangles.
            denom = d00 * d11 - d01 * d01
            denom = ti.select(ti.abs(denom) < 1e-12, 1e-12, denom)
            face_normal = self.rigid.face_normals[f_id]  # normalized
            base = (bp_pos * self.inv_dx - 0.5).cast(int)
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                gi = base + offset
                if self.is_valid(gi):
                    gpos = (gi).cast(float) * self.dx
                    dist = (gpos - v0).dot(face_normal)
                    g_proj = gpos - dist * face_normal
                    v0gp = g_proj - v0
                    d20 = v0gp.dot(v0v1)
                    d21 = v0gp.dot(v0v2)
                    v = (d11 * d20 - d01 * d21) / denom
                    w = (d00 * d21 - d01 * d20) / denom
                    u = 1.0 - v - w
                    if u >= 0 and v >= 0 and w >= 0:
                        abs_dist = abs(dist)
                        if abs_dist < self.grid_d[gi][rb_id]:
                            self.grid_d[gi][rb_id] = abs_dist
                            self.grid_A[gi][rb_id] = 1
                            self.grid_T[gi][rb_id] = 1 if dist > 0 else -1

    @ti.kernel
    def update_grid_rigid_id(self):
        for I in ti.grouped(self.grid_r):
            d_min = 1e10
            for r in range(self.rigid.n_rigid):
                if self.grid_A[I][r] == 1 and self.grid_d[I][r] < d_min:
                    d_min = self.grid_d[I][r]
                    self.grid_r[I] = r

    @ti.kernel
    def update_particle_rigid_properties(self):
        for r in ti.static(range(self.rigid.n_rigid)):
            for p in range(self.n_particles[None]):
                self.p_A[p][r] = 0
                self.p_d[p][r] = 0.0

                base = (self.p_x[p] * self.inv_dx - 0.5).cast(int)
                fx = self.p_x[p] * self.inv_dx - base.cast(float)
                w = [
                    0.5 * (1.5 - fx) ** 2,
                    0.75 - (fx - 1.0) ** 2,
                    0.5 * (fx - 0.5) ** 2,
                ]
                Tpr = 0.0

                # Moving Least Squares
                # Optimized to avoid large matrix construction (27x27) which slows down compilation

                for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                    gi = base + offset
                    if self.is_valid(gi) and self.grid_A[gi][r] == 1:
                        self.p_A[p][r] = 1
                        weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                        d_signed = self.grid_T[gi][r] * self.grid_d[gi][r]
                        Tpr += weight * d_signed

                if self.p_A[p][r] == 1:
                    if self.p_T[p][r] == 0:
                        self.p_T[p][r] = 1 if Tpr > 0 else -1

                    M = ti.Matrix.zero(ti.f32, 4, 4)
                    rhs = ti.Vector.zero(ti.f32, 4)

                    for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                        gi = base + offset
                        if self.is_valid(gi):
                            weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                            d_signed = (
                                self.grid_T[gi][r] * self.grid_d[gi][r] * self.p_T[p][r]
                            )
                            gpos = (offset.cast(float) - fx) * self.dx

                            q = ti.Vector([1.0, gpos[0], gpos[1], gpos[2]])
                            M += weight * q.outer_product(q)
                            rhs += weight * d_signed * q

                    # Regularize MLS matrix to avoid singular inverse near contact boundaries.
                    for k in ti.static(range(4)):
                        M[k, k] += 1e-8

                    M_inv = M.inverse()
                    beta = M_inv @ rhs
                    self.p_d[p][r] = beta[0]
                    n_raw = ti.Vector([beta[1], beta[2], beta[3]])
                    # Avoid NaNs when n_raw is (near) zero.
                    self.p_n[p, r] = n_raw / ti.max(n_raw.norm(), 1e-12)
                else:
                    self.p_T[p][r] = 0

    @ti.kernel
    def p2g(self):
        # MPM begin
        self.grid_v.fill(0)
        self.grid_m.fill(0)

        # P2G_1, distribute mass and velocity
        for p in range(self.n_particles[None]):
            self.p_v[p] += self.dt * self.gravity[None]  # Apply gravity to particle
            base = (self.p_x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            op = self.p_material[p]  # object index
            p_vol = self.object_p_vol[op]
            p_mass = self.material_density[op] * p_vol
            p_C = self.p_C[p]

            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                gi = base + offset
                if self.is_valid(gi):
                    flag = 1
                    # check compatibility
                    for k in ti.static(range(self.rigid.n_rigid)):
                        if (
                            self.p_T[p][k] != self.grid_T[gi][k]
                            and self.p_A[p][k] * self.grid_A[gi][k] != 0
                        ):
                            flag = 0
                    if flag == 1:
                        dpos = (offset.cast(float) - fx) * self.dx
                        weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                        self.grid_v[gi] += weight * (
                            p_mass * (self.p_v[p] + p_C @ dpos)
                        )
                        self.grid_m[gi] += weight * p_mass

        # P2G_2, strain-stess
        for p in range(self.n_particles[None]):
            base = (self.p_x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            op = self.p_material[p]  # object index
            p_vol = self.object_p_vol[op]
            p_mass = self.material_density[op] * p_vol
            mat = self.material_type[op]
            self.p_F[p] = (
                ti.Matrix.identity(ti.f32, 3) + self.dt * self.p_C[p]
            ) @ self.p_F[p]
            U, sig, V = ti.svd(self.p_F[p])
            J = 1.0
            for d in ti.static(range(3)):
                J *= sig[d, d]

            stress = ti.Matrix.zero(ti.f32, 3, 3)
            if mat == CATEGORY_WATER:
                stiffness = self.material_stiffness[op]
                power = self.material_power[op]
                eta = self.material_viscosity[op]

                # Clamp J to prevent numerical instability with J^(-power)
                J_clamped = ti.max(0.05, ti.min(J, 20.0))
                new_F = ti.Matrix.identity(ti.f32, 3)
                new_F[0, 0] = J_clamped
                self.p_F[p] = new_F

                volume = p_vol * J_clamped
                # Tait-Murnaghan EOS: p = k * (J^(-gamma) - 1)
                # Clamp negative pressure (tension) relative to stiffness
                pressure = ti.max(
                    -stiffness * 0.5,
                    stiffness * (ti.pow(J_clamped, -power) - 1.0),
                )

                stress = -pressure * ti.Matrix.identity(ti.f32, 3)
                # Newtonian viscosity: sigma_visc = 2 * eta * D, D = sym(grad v)
                D = self.p_C[p] + self.p_C[p].transpose()
                stress += eta * D
                div_v = self.p_C[p].trace()
                if div_v < 0:
                    bulk_visc = 0.1 * stiffness * self.dx
                    stress += bulk_visc * div_v * ti.Matrix.identity(ti.f32, 3)
                stress *= -volume * 4.0 * self.dt * self.inv_dx * self.inv_dx

            else:
                mu = self.material_mu[op]
                lam = self.material_lambda[op]
                hardening = ti.exp(self.material_hardening[op] * (1.0 - self.p_Jp[p]))
                mu *= hardening
                lam *= hardening
                if mat == CATEGORY_SAND:
                    mu = 50.0

                if mat == CATEGORY_SNOW:
                    for d in ti.static(range(3)):
                        new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                        self.p_Jp[p] *= sig[d, d] / new_sig
                        sig[d, d] = new_sig
                    self.p_F[p] = U @ sig @ V.transpose()
                    J = sig[0, 0] * sig[1, 1] * sig[2, 2]
                elif mat == CATEGORY_SAND:
                    new_F = ti.Matrix.identity(ti.f32, 3)
                    new_F[0, 0] = J
                    self.p_F[p] = new_F
                    self.p_Jp[p] = J

                stress = 2 * mu * (self.p_F[p] - U @ V.transpose()) @ self.p_F[
                    p
                ].transpose() + ti.Matrix.identity(ti.f32, 3) * lam * J * (J - 1.0)

                eta = self.material_viscosity[op]
                if eta > 0:
                    D = self.p_C[p] + self.p_C[p].transpose()
                    stress += eta * D

                stress *= -p_vol * 4 * self.inv_dx * self.inv_dx * self.dt

            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                gi = base + offset
                if self.is_valid(gi):
                    flag = 1
                    # check compatibility
                    for k in ti.static(range(self.rigid.n_rigid)):
                        if (
                            self.p_T[p][k] != self.grid_T[gi][k]
                            and self.p_A[p][k] * self.grid_A[gi][k] != 0
                        ):
                            flag = 0
                    if flag == 1:
                        dpos = (offset.cast(float) - fx) * self.dx
                        weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                        self.grid_v[gi] += weight * stress @ dpos

    @ti.kernel
    def update_grid(self):
        # Grid operations
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] = (1 / self.grid_m[I]) * self.grid_v[I]
                # self.grid_v[I] += self.dt * self.gravity[None] # Gravity applied to particles
                # Simple boundary condition | Add friction now
                boundary_friction = 0.3
                for d in ti.static(range(3)):
                    if I[d] < 3 and self.grid_v[I][d] < 0:
                        v_n = self.grid_v[I][d]
                        v_t = self.grid_v[I]
                        v_t[d] = 0
                        if v_t.norm() > 1e-10:
                            self.grid_v[I] = v_t.normalized() * ti.max(
                                0, v_t.norm() + v_n * boundary_friction
                            )
                        else:
                            self.grid_v[I] = ti.Vector.zero(ti.f32, 3)
                        self.grid_v[I][d] = 0
                    if I[d] > self.n_grid - 3 and self.grid_v[I][d] > 0:
                        v_n = self.grid_v[I][d]
                        v_t = self.grid_v[I]
                        v_t[d] = 0
                        if v_t.norm() > 1e-10:
                            self.grid_v[I] = v_t.normalized() * ti.max(
                                0, v_t.norm() - v_n * boundary_friction
                            )
                        else:
                            self.grid_v[I] = ti.Vector.zero(ti.f32, 3)
                        self.grid_v[I][d] = 0

    @ti.kernel
    def g2p(self):
        # G2P
        for p in range(self.n_particles[None]):
            base = (self.p_x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            op = self.p_material[p]
            p_mass = self.material_density[op] * self.object_p_vol[op]

            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                gi = base + offset
                if self.is_valid(gi):
                    flag = 1
                    # check compatibility
                    for k in ti.static(range(self.rigid.n_rigid)):
                        if (
                            self.p_T[p][k] != self.grid_T[gi][k]
                            and self.p_A[p][k] * self.grid_A[gi][k] != 0
                        ):
                            flag = 0
                    weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                    g_v = self.grid_v[gi]
                    if flag == 0:
                        gpos = (gi.cast(float)) * self.dx
                        gr = self.grid_r[gi]
                        g_rigid_vel = self.rigid.get_rigid_velocity(gr, gpos)
                        delta_v = self.p_v[p] - g_rigid_vel
                        n = self.p_n[p, gr]
                        n = n / ti.max(n.norm(), 1e-12)
                        dv_dot_n = delta_v.dot(n)
                        if dv_dot_n < 0:
                            delta_vt = delta_v - dv_dot_n * n
                            vt_norm = delta_vt.norm()
                            new_g_vt = self.p_v[p]
                            if vt_norm > 1e-10:
                                new_g_vt = ti.max(
                                    0.0,
                                    vt_norm + self.rigid.friction[gr] * dv_dot_n,
                                ) * (delta_vt / vt_norm)
                            new_g_vn = self.rigid.restitution[gr] * (-dv_dot_n) * n
                            g_v = new_g_vt + new_g_vn + g_rigid_vel
                            self.rigid.apply_impulse(
                                gr,
                                gpos,
                                (self.p_v[p] - g_v) * p_mass * weight,
                            )
                        else:
                            g_v = self.p_v[p]
                        g_v += n * self.rigid.splitter[gr]  # splitting them apart
                    new_v += weight * g_v
                    dpos = offset.cast(float) - fx
                    new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.p_v[p] = new_v
            self.p_C[p] = new_C
            self.p_x[p] += self.dt * self.p_v[p]

            # penalty force
            for r in ti.static(range(self.rigid.n_rigid)):
                if self.p_d[p][r] < 0:
                    delta_v_penalty = self.rigid.kh[r] * (-self.p_d[p][r]) * self.p_n[p, r]
                    self.p_v[p] += delta_v_penalty
                    self.rigid.apply_impulse(r, self.p_x[p], -delta_v_penalty * p_mass)

    def export(self, frame, output_dir: str):
        self.rigid.export(frame, output_dir)
        particle_path = os.path.join(output_dir, "particles")
        num_p = self.n_particles[None]
        pos_np = self.p_x.to_numpy()[:num_p]
        vel_np = self.p_v.to_numpy()[:num_p]

        dtype_list = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("vx", "f4"),
            ("vy", "f4"),
            ("vz", "f4"),
        ]
        data = np.empty(num_p, dtype=dtype_list)

        data["x"] = pos_np[:, 0]
        data["y"] = pos_np[:, 1]
        data["z"] = pos_np[:, 2]
        data["vx"] = vel_np[:, 0]
        data["vy"] = vel_np[:, 1]
        data["vz"] = vel_np[:, 2]

        ply_el = PlyElement.describe(data, "vertex")
        filename = os.path.join(particle_path, f"frame_{frame:04d}.ply")
        PlyData([ply_el], text=False).write(filename)

    def mpm_step(self):
        self.reset_grid()
        self.build_rigid_cdf()
        self.update_grid_rigid_id()
        self.update_particle_rigid_properties()
        self.p2g()
        self.update_grid()
        self.g2p()

    def step(self, time):
        self.mpm_step()
        self.rigid.step(time)
        # self.cloth.step()
