from src.core.utils import *
from src.objects import RigidObject

import taichi as ti
import numpy as np
import os


@ti.func
def tri_normal(a, b, c):
    n = (b - a).cross(c - a)
    ln = n.norm()
    res = ti.Vector([0.0, 1.0, 0.0])
    if ln > 1e-8:
        res = n / ln
    return res


@ti.func
def closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    result = a
    if not ((d1 <= 0) and (d2 <= 0)):
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        if not ((d3 >= 0) and (d4 <= d3)):
            vc = d1 * d4 - d3 * d2
            if (vc <= 0) and (d1 >= 0) and (d3 <= 0):
                v = d1 / (d1 - d3 + 1e-8)
                result = a + v * ab
            else:
                cp = p - c
                d5 = ab.dot(cp)
                d6 = ac.dot(cp)
                if not ((d6 >= 0) and (d5 <= d6)):
                    vb = d5 * d2 - d1 * d6
                    if (vb <= 0) and (d2 >= 0) and (d6 <= 0):
                        w = d2 / (d2 - d6 + 1e-8)
                        result = a + w * ac
                    else:
                        va = d3 * d6 - d5 * d4
                        if (va <= 0) and ((d4 - d3) >= 0) and ((d5 - d6) >= 0):
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-8)
                            result = b + w * (c - b)
                        else:
                            denom = va + vb + vc + 1e-8
                            v = vb / denom
                            w = vc / denom
                            result = a + ab * v + ac * w
                else:
                    result = c
        else:
            result = b
    return result


@ti.data_oriented
class RigidBody:
    def __init__(self, rigid_object: RigidObject):
        # Optional helper: convert quaternion (w,x,y,z) to rotation matrix

        try:
            # mesh: load from path in RigidObject.meshdir
            if rigid_object.meshdir in GEOMS:
                mesh = GEOMS[rigid_object.meshdir](rigid_object.scale)
            else:
                mesh = trimesh.load(rigid_object.meshdir, force="mesh")
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh from RigidObject.meshdir: {e}")

        self.mesh = mesh
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.vertices))
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=len(mesh.faces))
        self.mass_center_offset = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.position = ti.Vector.field(
            3, dtype=ti.f32, shape=()
        )  # position of the center of mass
        self.orientation = ti.Vector.field(
            4, dtype=ti.f32, shape=()
        )  # orientation matrix of the body
        self.rotation_matrix = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=()
        )  # rotation matrix of the body
        self.velocity = ti.Vector.field(
            3, dtype=ti.f32, shape=()
        )  # velocity of the center of mass
        self.angular_velocity = ti.Vector.field(
            3, dtype=ti.f32, shape=()
        )  # angular velocity of the body
        self.mass, self.volume = ti.field(dtype=ti.f32, shape=()), ti.field(
            dtype=ti.f32, shape=()
        )
        self.inertia = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=()
        )  # inertia tensor relative to the center of mass
        self.inv_inertia = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=()
        )  # inverse inertia tensor relative to the center of mass
        self.impulse = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.angular_impulse = ti.Vector.field(
            3, dtype=ti.f32, shape=()
        )  # torque relative to the center of mass

        mesh.vertices = mesh.vertices.astype(np.float32)
        mesh.faces = mesh.faces.astype(np.int32)
        self.vertices.from_numpy(mesh.vertices)
        self.faces.from_numpy(mesh.faces)

        self.render_vertices = ti.Vector.field(
            3, dtype=ti.f32, shape=len(mesh.vertices)
        )
        self.render_indices = ti.field(dtype=ti.i32, shape=len(mesh.faces) * 3)
        self.render_indices.from_numpy(mesh.faces.flatten())

        self.mass[None] = rigid_object.mass
        self.centralize()  # centralize the mesh
        self.mesh = trimesh.Trimesh(
            vertices=self.vertices.to_numpy(), faces=self.faces.to_numpy()
        )
        if rigid_object.is_dynamic and rigid_object.scripted_trajectory is None:
            self.get_volume()
            self.get_inertia()
        self.voxel = None
        self.num_particles = 0
        self.get_voxel()
        self.position[None] = ti.Vector(rigid_object.position)
        self.velocity[None] = ti.Vector(rigid_object.velocity)
        self.orientation[None] = ti.Vector(rigid_object.orientation)
        self.rotation_matrix[None] = self.quat_wxyz_to_matrix(rigid_object.orientation)
        self.collision_threshold = rigid_object.collision_threshold
        self.fixed = not rigid_object.is_dynamic
        self.restitution = ti.field(dtype=ti.f32, shape=())
        self.restitution[None] = rigid_object.material.restitution
        self.friction = ti.field(dtype=ti.f32, shape=())
        self.friction[None] = rigid_object.material.friction
        self.num_contacts = ti.field(dtype=ti.i32, shape=())

        self.scripted_trajectory = rigid_object.scripted_trajectory

    @ti.kernel
    def update_render_vertices(self):
        for i in self.vertices:
            self.render_vertices[i] = (
                self.position[None] + self.rotation_matrix[None] @ self.vertices[i]
            )

    def quat_wxyz_to_matrix(self, qwxyz):
        qw, qx, qy, qz = (
            float(qwxyz[0]),
            float(qwxyz[1]),
            float(qwxyz[2]),
            float(qwxyz[3]),
        )
        # normalized not strictly required, but safer
        norm = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
        if norm > 1e-8:
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
        # standard quaternion to rotation matrix (w,x,y,z)
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    @ti.func
    def mass_center(self):
        mesh_area = ti.float32(0.0)
        temp = ti.Vector([0.0, 0.0, 0.0])

        for i in range(self.faces.shape[0]):
            v0 = self.vertices[self.faces[i][0]]
            v1 = self.vertices[self.faces[i][1]]
            v2 = self.vertices[self.faces[i][2]]

            center = (v0 + v1 + v2) / 3.0
            area = 0.5 * (v1 - v0).cross(v2 - v0).norm()

            mesh_area += area
            temp += center * area

        return temp / mesh_area

    @ti.kernel
    def get_volume(self):
        vol = ti.float32(0.0)
        for i in range(self.faces.shape[0]):
            v0 = self.vertices[self.faces[i][0]]
            v1 = self.vertices[self.faces[i][1]]
            v2 = self.vertices[self.faces[i][2]]
            vol += v0.dot(v1.cross(v2)) / 6.0
        self.volume[None] = ti.abs(vol)

    @ti.kernel
    def centralize(self):
        center = self.mass_center()
        self.mass_center_offset[None] = center
        for i in range(self.vertices.shape[0]):
            self.vertices[i] -= center

    def get_voxel(self):
        mesh = self.mesh.copy()
        # mesh.apply_transform(np.vstack((np.hstack((self.orientation.to_numpy(), self.position.to_numpy().reshape(-1, 1))), [0, 0, 0, 1])))
        self.voxel = mesh.voxelized(pitch=0.01).fill().points.astype(np.float32)
        self.num_particles = self.voxel.shape[0]

    @ti.kernel
    def get_inertia(self):
        covariance_tensor = ti.Matrix(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        canoical_inertia_tensor = (
            ti.Matrix([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]) / 60
        )

        for i in range(self.faces.shape[0]):
            # ti.static_print(self.faces[i, 0])
            transform = ti.Matrix.cols(
                [
                    self.vertices[self.faces[i][0]],
                    self.vertices[self.faces[i][1]],
                    self.vertices[self.faces[i][2]],
                ]
            )
            covariance_tensor += (
                transform.determinant()
                * transform
                @ canoical_inertia_tensor
                @ transform.transpose()
            )
        covariance_tensor *= self.mass[None] / self.volume[None]
        inertia_tensor = (
            ti.Matrix.identity(ti.f32, 3) * ti.Matrix.trace(covariance_tensor)
            - covariance_tensor
        )
        self.inertia[None] = inertia_tensor
        self.inv_inertia[None] = inertia_tensor.inverse()

    @ti.func
    def apply_impulse_at_point(
        self, impulse: ti.types.vector(3, ti.f32), point: ti.types.vector(3, ti.f32)
    ):
        # Use atomic adds to avoid race conditions when multiple contact samples
        # accumulate impulses concurrently from parallel kernels.
        for k in ti.static(range(3)):
            ti.atomic_add(self.impulse[None][k], impulse[k])
        torque = ti.math.cross(point - self.position[None], impulse)
        for k in ti.static(range(3)):
            ti.atomic_add(self.angular_impulse[None][k], torque[k])

    @ti.func
    def signed_distance(self, p):
        min_abs = 1e30
        best_phi = 0.0
        best_n = ti.Vector([0.0, 1.0, 0.0])
        best_cp = ti.Vector([0.0, 0.0, 0.0])
        best_f = -1
        center = self.position[None]
        for f in range(self.faces.shape[0]):
            idx = self.faces[f]
            a = center + self.rotation_matrix[None] @ self.vertices[idx[0]]
            b = center + self.rotation_matrix[None] @ self.vertices[idx[1]]
            c = center + self.rotation_matrix[None] @ self.vertices[idx[2]]
            cp = closest_point_on_triangle(p, a, b, c)
            # Use actual distance from p to closest point as magnitude,
            # and choose normal from surface toward exterior.
            dir_vec = p - cp
            dist = dir_vec.norm()
            phi = ti.cast(0.0, ti.f32)
            n = ti.Vector([0.0, 0.0, 0.0])
            if dist > 1e-8:
                dir_n = dir_vec / dist
                # if dir points toward center, p is inside -> negative phi
                if (center - cp).dot(dir_vec) > 0:
                    phi = -dist
                    n = -dir_n
                else:
                    phi = dist
                    n = dir_n
            else:
                # p is effectively on the surface; fall back to triangle normal
                n = tri_normal(a, b, c)
                # determine sign by checking if n points outward relative to center
                if (center - cp).dot(n) > 0:
                    n = -n
                phi = (p - cp).dot(n)
            ap = ti.abs(phi)
            if ap < min_abs:
                min_abs = ap
                best_phi = phi
                best_n = n
                best_cp = cp
                best_f = f
        return best_phi, best_n, best_cp, best_f

    @ti.func
    def get_velocity_at_point(
        self, point: ti.types.vector(3, ti.f32)
    ) -> ti.types.vector(3, ti.f32):
        # Velocity coupling relation
        r = point - self.position[None]

        linear_velocity = self.velocity[None]

        angular_velocity_at_point = self.angular_velocity[None].cross(r)

        return linear_velocity + angular_velocity_at_point

    @ti.kernel
    def update(self, dt_old: float, max_speed: float, max_omega: float):
        # Linear motion
        if not self.fixed:
            dt = ti.cast(dt_old, ti.f32)

            v_new = self.velocity[None] + self.impulse[None] / self.mass[None]
            # clamp linear speed
            v_len = v_new.norm()
            if v_len > max_speed:
                v_new = v_new * (ti.cast(max_speed, ti.f32) / (v_len + 1e-8))
            self.velocity[None] = v_new
            self.position[None] += self.velocity[None] * dt

            # Angular motion
            # angular_acceleration = self.torque[None] / self.mass  # Simplified, should use inertia tensor
            I_inv_world = (
                self.rotation_matrix[None]
                @ self.inv_inertia[None]
                @ self.rotation_matrix[None].transpose()
            )
            w_new = (
                self.angular_velocity[None] + I_inv_world @ self.angular_impulse[None]
            )
            # clamp angular speed
            w_len = w_new.norm()
            if w_len > max_omega:
                w_new = w_new * (ti.cast(max_omega, ti.f32) / (w_len + 1e-8))
            self.angular_velocity[None] = w_new

            # Orientation
            q = self.orientation[None]
            wx, wy, wz = w_new[0], w_new[1], w_new[2]

            qw, qx, qy, qz = q[0], q[1], q[2], q[3]

            dqw = -0.5 * (wx * qx + wy * qy + wz * qz)
            dqx = 0.5 * (wx * qw + wy * qz - wz * qy)
            dqy = 0.5 * (wy * qw + wz * qx - wx * qz)
            dqz = 0.5 * (wz * qw + wx * qy - wy * qx)

            self.orientation[None][0] += dqw * dt
            self.orientation[None][1] += dqx * dt
            self.orientation[None][2] += dqy * dt
            self.orientation[None][3] += dqz * dt

            self.orientation[None] = self.orientation[None].normalized()

        # Reset forces and torques
        self.impulse[None] = ti.Vector([0.0, 0.0, 0.0])
        self.angular_impulse[None] = ti.Vector([0.0, 0.0, 0.0])

    def follow_trajectory(self, t: float, dt: float):
        """Follow a scripted trajectory at time t, ignoring all forces.

        scripted_trajectory must be a Python callable returning (pos, quat_wxyz)
        where pos is iterable of 3 floats and quat_wxyz is (w,x,y,z).
        This method sets position/orientation/velocity/angular_velocity directly
        and clears accumulated forces/torques.
        """
        if self.scripted_trajectory is None:
            return
        # Sample pose at t and t+dt
        pos_t, quat_t = self.scripted_trajectory(t)
        pos_next, quat_next = self.scripted_trajectory(t + dt)
        p_t = np.array(pos_t, dtype=np.float32)
        p_next = np.array(pos_next, dtype=np.float32)
        # Linear velocity from finite difference
        v = (p_next - p_t) / float(dt if dt != 0 else 1e-8)

        # Normalize quaternions
        q1 = np.array(quat_t, dtype=np.float32)
        q2 = np.array(quat_next, dtype=np.float32)

        def _norm_quat(q):
            n = float(np.linalg.norm(q))
            return q / n if n > 1e-8 else q

        q1 = _norm_quat(q1)
        q2 = _norm_quat(q2)
        # q_diff = q2 * conj(q1) in (w,x,y,z)
        w1, x1, y1, z1 = float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])
        w2, x2, y2, z2 = float(q1[0]), -float(q1[1]), -float(q1[2]), -float(q1[3])
        w_diff = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x_diff = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y_diff = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z_diff = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        if w_diff < 0:
            w_diff, x_diff, y_diff, z_diff = -w_diff, -x_diff, -y_diff, -z_diff
        omega = (
            2.0
            * np.array([x_diff, y_diff, z_diff], dtype=np.float32)
            / float(dt if dt != 0 else 1e-8)
        )

        # Write to Taichi fields
        self.position[None] = p_t
        self.velocity[None] = v
        self.orientation[None] = ti.Vector(q1)
        self.angular_velocity[None] = omega
        # Clear forces/torques to avoid accumulation
        self.impulse[None] = ti.Vector([0.0, 0.0, 0.0])
        self.angular_impulse[None] = ti.Vector([0.0, 0.0, 0.0])

    def step(self, t: float, dt: float):
        """Advance body by dt. If scripted, follow trajectory; otherwise integrate physics."""
        if self.scripted_trajectory is not None:
            self.follow_trajectory(t, dt)
        else:
            self.update(dt, max_speed=100.0, max_omega=50.0)
        self.rotation_matrix[None] = self.quat_wxyz_to_matrix(self.orientation[None])

    def export(self, frame: int, output_dir: str, rb_id: int):
        filename = os.path.join(
            output_dir, f"rigid_body_{rb_id:04d}_frame_{frame:04d}.obj"
        )
        self.update_render_vertices()
        vertices_np = self.render_vertices.to_numpy()
        # Prefer authoritative face indices from the original per-body mesh stored
        # in `self.mesh`. Fall back to Taichi fields if necessary.
        try:
            faces_np = np.array(self.mesh.faces, dtype=np.int32)
        except Exception:
            faces_np = self.faces.to_numpy()

        # Validate face indices to avoid trimesh IndexError; emit diagnostics if invalid.
        if faces_np.size == 0:
            print(f"Warning: rigid {rb_id} has no faces to export (frame {frame}). Skipping.")
            return

        max_idx = int(np.max(faces_np))
        n_verts = int(vertices_np.shape[0])
        if max_idx >= n_verts:
            print(
                f"Export skipped for rigid {rb_id} frame {frame}: max face index {max_idx} >= vertex count {n_verts}."
            )
            # Provide sample diagnostics
            print(f"Vertex count: {n_verts}, faces shape: {faces_np.shape}")
            print(f"Faces sample: {faces_np.flatten()[:12]}")
            return

        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
        mesh.export(filename)
