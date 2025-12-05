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

        self.rigid = Rigid(scene.rigid_objects, self.dx, self.dt, scene.gravity)

        # self.cloth = Cloth(scene.cloth_objects, self.dx, self.dt)

        print("Initializing MPM Solver...")
        self.mpm_objects = scene.mpm_objects
        num_particles = sum([obj.num_particles for obj in scene.mpm_objects])
        self.p_x = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.p_v = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
        self.p_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)
        self.p_Jp = ti.field(dtype=ti.f32, shape=num_particles)
        self.p_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)
        self.p_material = ti.field(dtype=ti.i32, shape=num_particles)
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.n_particles[None] = num_particles
        # self.p_vol = (self.dx * 0.5) ** 3

        self.num_mpm = len(scene.mpm_objects)
        self.material_density = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_mu = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_lambda = ti.field(dtype=ti.f32, shape=self.num_mpm)
        self.material_hardening = ti.field(dtype=ti.f32, shape=self.num_mpm)
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

    def init_mpm_materials(self):
        print("Initializing MPM materials...")
        for i, obj in enumerate(self.mpm_objects):
            mat = obj.material
            self.material_density[i] = mat.density
            E = mat.E
            nu = mat.nu
            self.material_mu[i] = E / (2 * (1 + nu))
            self.material_lambda[i] = E * nu / ((1 + nu) * (1 - 2 * nu))
            self.material_hardening[i] = mat.hardening
            self.material_type[i] = int(mat.model.value)

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

            # Rejection sampling
            samples = []
            bounds = mesh.bounds

            iter_count = 0
            while len(samples) < obj.num_particles and iter_count < 100:
                print(f"Sampling particles for object {i}, iteration {iter_count}, collected {len(samples)} particles")
                needed = obj.num_particles - len(samples)
                batch_size = min(max(needed * 2, 1000), 10000)
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
            denom = d00 * d11 - d01 * d01
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
                zeta = ti.Matrix.identity(ti.f32, 27)
                d_grid = ti.Vector.zero(ti.f32, 27)
                Q = ti.Matrix.zero(ti.f32, 27, 4)

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

                    for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                        gi = base + offset
                        if self.is_valid(gi):
                            weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                            d_signed = (
                                self.grid_T[gi][r] * self.grid_d[gi][r] * self.p_T[p][r]
                            )
                            row_id = offset[0] * 9 + offset[1] * 3 + offset[2]
                            gpos = (offset.cast(float) - fx) * self.dx
                            d_grid[row_id] = d_signed
                            Q[row_id, 0] = 1.0
                            Q[row_id, 1] = gpos[0]
                            Q[row_id, 2] = gpos[1]
                            Q[row_id, 3] = gpos[2]
                            zeta[row_id, row_id] = weight
                            Tpr += weight * d_signed

                    M = Q.transpose() @ zeta @ Q
                    M_inv = M.inverse()
                    beta = M_inv @ Q.transpose() @ zeta @ d_grid
                    self.p_d[p][r] = beta[0]
                    self.p_n[p, r] = ti.Vector([beta[1], beta[2], beta[3]]).normalized()
                else:
                    self.p_T[p][r] = 0

    @ti.kernel
    def p2g(self):
        # MPM begin
        self.grid_v.fill(0)
        self.grid_m.fill(0)

        # P2G
        for p in range(self.n_particles[None]):
            base = (self.p_x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.p_x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            op = self.p_material[p]  # object index
            p_vol = self.object_p_vol[op]
            p_mass = self.material_density[op] * p_vol
            mat = self.material_type[op]
            mu = self.material_mu[op]
            lam = self.material_lambda[op]
            hardening = ti.exp(self.material_hardening[op] * (1.0 - self.p_Jp[p]))
            if mat == CATEGORY_SNOW:
                mu *= hardening
                lam *= hardening
            if mat == CATEGORY_WATER:
                mu = 0.0

            self.p_F[p] = (
                ti.Matrix.identity(ti.f32, 3) + self.dt * self.p_C[p]
            ) @ self.p_F[p]
            U, sig, V = ti.svd(self.p_F[p])
            J = 1.0
            for d in ti.static(range(3)):
                new_sig = sig[d, d]
                if mat == CATEGORY_SNOW:
                    new_sig = max(
                        min(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3
                    )  # Allow customization later !!
                self.p_Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if mat == CATEGORY_WATER:
                new_F = ti.Matrix.identity(ti.f32, 3)
                new_F[0, 0] = J
                self.p_F[p] = new_F
            elif mat == CATEGORY_SNOW:
                self.p_F[p] = U @ sig @ V.transpose()
            stress = 2 * mu * (self.p_F[p] - U @ V.transpose()) @ self.p_F[
                p
            ].transpose() + ti.Matrix.identity(ti.f32, 3) * lam * J * (J - 1.0)
            stress *= -p_vol * 4 * self.inv_dx * self.inv_dx * self.dt
            affine = stress + p_mass * self.p_C[p]

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
                            p_mass * self.p_v[p] + affine @ dpos
                        )
                        self.grid_m[gi] += weight * p_mass

    @ti.kernel
    def update_grid(self):
        # Grid operations
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] = (1 / self.grid_m[I]) * self.grid_v[I]
                self.grid_v[I] += self.dt * self.gravity[None]
                # Simple boundary condition !!
                for d in ti.static(range(3)):
                    if I[d] < 3 and self.grid_v[I][d] < 0:
                        self.grid_v[I][d] = 0
                    if I[d] > self.n_grid - 3 and self.grid_v[I][d] > 0:
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
                        g_rigid_vel = self.rigid.get_rigid_velocity(
                            gr, gpos
                        )
                        delta_v = self.p_v[p] - g_rigid_vel
                        dv_dot_n = delta_v.dot(self.p_n[p, gr])
                        if dv_dot_n < 0:
                            delta_vt = delta_v - dv_dot_n * self.p_n[p, gr]
                            new_g_vt = (
                                ti.max(
                                    0.0,
                                    delta_vt.norm()
                                    + self.rigid.friction[gr] * dv_dot_n,
                                )
                                * delta_vt.normalized()
                            )
                            new_g_vn = (
                                self.rigid.restitution[gr]
                                * (-dv_dot_n)
                                * self.p_n[p, gr]
                            )
                            g_v = new_g_vt + new_g_vn + g_rigid_vel
                            self.rigid.apply_impulse(
                                gr,
                                gpos,
                                (self.p_v[p] - g_v) * p_mass * weight,
                            )
                        g_v += self.p_n[p, gr] * self.rigid.splitter[gr] # splitting them apart
                    new_v += weight * g_v
                    dpos = offset.cast(float) - fx
                    new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.p_v[p] = new_v
            self.p_C[p] = new_C
            self.p_x[p] += self.dt * self.p_v[p]

            # penalty force
            for r in ti.static(range(self.rigid.n_rigid)):
                if self.p_d[p][r] < 0:
                    penalty_force = (
                        self.rigid.kh[r] * (-self.p_d[p][r]) * self.p_n[p, r]
                    )
                    self.p_v[p] += (self.dt / p_mass) * penalty_force
                    self.rigid.apply_impulse(r, self.p_x[p], -penalty_force * self.dt)

    def export(self, frame, output_dir: str):
        self.rigid.export(frame, output_dir)
        num_p = self.n_particles[None]
        pos_np = self.p_x.to_numpy()[:num_p]

        vertex = np.array(
            [(p[0], p[1], p[2]) for p in pos_np],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        )

        ply_el = PlyElement.describe(vertex, "vertex")
        filename = os.path.join(output_dir, f"frame_{frame:04d}_mpm.ply")
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
        print("MPM step at time:", time)
        self.mpm_step()
        self.rigid.step(time)
        # self.cloth.step()
