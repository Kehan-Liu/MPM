import taichi as ti
import numpy as np
from src.objects import RigidObject
from typing import List, Tuple
import trimesh

MAX_VERTICES = 100000
MAX_FACES = 200000
MAX_BP = 400000


@ti.data_oriented
class Rigid:
    def __init__(
        self,
        rigid_objects: List[RigidObject],
        dx: float,
        dt: float,
        gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
    ):
        self.rigid_objects = rigid_objects
        self.n_rigid = len(rigid_objects)
        self.dt = dt
        self.dx = dx
        self.gravity = ti.Vector(gravity)

        self.scripted_trajectories = [obj.scripted_trajectory for obj in rigid_objects]

        # Initialize Taichi fields for rigid body properties
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.orientations = ti.Vector.field(4, dtype=ti.f32, shape=self.n_rigid)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.angular_velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.masses = ti.field(dtype=ti.f32, shape=self.n_rigid)
        self.inertia = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_rigid)
        self.inv_inertia = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_rigid)
        self.rotation_matrices = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_rigid)

        self.impulses = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.angular_impulses = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.kh = ti.field(dtype=ti.f32, shape=self.n_rigid)
        self.friction = ti.field(dtype=ti.f32, shape=self.n_rigid)
        self.restitution = ti.field(dtype=ti.f32, shape=self.n_rigid)
        self.splitter = ti.field(dtype=ti.f32, shape=self.n_rigid)

        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
        self.v2rb = ti.field(dtype=ti.i32, shape=MAX_VERTICES)

        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=MAX_FACES)
        self.face_normals = ti.Vector.field(3, dtype=ti.f32, shape=MAX_FACES)
        self.f2rb = ti.field(dtype=ti.i32, shape=MAX_FACES)

        self.boundary_particles = ti.Vector.field(
            3, dtype=ti.f32, shape=MAX_BP
        )  # offset to CoM
        self.bp2rb = ti.field(dtype=ti.i32, shape=MAX_BP)
        self.bp2f = ti.field(dtype=ti.i32, shape=MAX_BP)

        self.n_vertices = ti.field(dtype=ti.i32, shape=())
        self.n_faces = ti.field(dtype=ti.i32, shape=())
        self.n_boundary_particles = ti.field(dtype=ti.i32, shape=())

        self.vert_offset = ti.field(dtype=ti.i32, shape=self.n_rigid + 1)
        self.face_offset = ti.field(dtype=ti.i32, shape=self.n_rigid + 1)
        self.bp_offset = ti.field(dtype=ti.i32, shape=self.n_rigid + 1)

        self.init_rigid_states()
        self.load_mesh_data()

    def init_rigid_states(self):
        print("Initializing rigid body states...")
        for i, obj in enumerate(self.rigid_objects):
            self.positions[i] = ti.Vector(obj.position)
            self.orientations[i] = ti.Vector(obj.orientation)
            self.velocities[i] = ti.Vector(obj.velocity)
            self.angular_velocities[i] = ti.Vector(obj.angular_velocity)
            self.masses[i] = obj.mass
            self.kh[i] = obj.material.kh
            self.friction[i] = obj.material.friction
            self.restitution[i] = obj.material.restitution
            self.splitter[i] = obj.material.splitter

    def load_mesh_data(self):
        vertex_count = 0
        face_count = 0
        bp_count = 0

        print("Loading rigid body mesh data...")
        for rb_id, obj in enumerate(self.rigid_objects):
            try:
                mesh = trimesh.load(obj.meshdir, force="mesh")
            except Exception as e:
                print(f"Error loading mesh {obj.meshdir}: {e}")
                continue

            # Compute CoM and Inertia manually (Shell)
            v = mesh.vertices
            faces = mesh.faces

            # Get vertices for each face: (N, 3, 3)
            face_verts = v[faces]

            # Compute cross product for area
            e1 = face_verts[:, 1] - face_verts[:, 0]
            e2 = face_verts[:, 2] - face_verts[:, 0]
            cross = np.cross(e1, e2)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            total_area = np.sum(areas)

            # Face centroids
            centroids = np.mean(face_verts, axis=1)

            # Body CoM
            if total_area > 1e-8:
                com = np.average(centroids, axis=0, weights=areas)
            else:
                com = np.mean(v, axis=0)

            # Shift vertices to CoM
            verts = v - com

            # Re-get face vertices with shifted verts
            face_verts = verts[faces]
            centroids = np.mean(
                face_verts, axis=1
            )  # Recompute centroids relative to CoM

            # Compute Inertia Tensor
            mass = obj.mass
            if total_area > 1e-8:
                face_masses = mass * (areas / total_area)
            else:
                face_masses = np.zeros(len(faces))

            # Vectorized covariance computation
            v0 = face_verts[:, 0]
            v1 = face_verts[:, 1]
            v2 = face_verts[:, 2]
            c = centroids

            def outer_sum(vecs, weights):
                weighted_vecs = vecs * np.sqrt(weights)[:, np.newaxis]
                return weighted_vecs.T @ weighted_vecs

            C = (1.0 / 12.0) * (
                outer_sum(v0, face_masses)
                + outer_sum(v1, face_masses)
                + outer_sum(v2, face_masses)
                + 9.0 * outer_sum(c, face_masses)
            )

            I_matrix = np.trace(C) * np.eye(3) - C

            self.inertia[rb_id] = ti.Matrix(I_matrix)
            self.inv_inertia[rb_id] = ti.Matrix(np.linalg.inv(I_matrix))

            n_verts = len(verts)
            n_faces = len(faces)

            self.vert_offset[rb_id] = vertex_count
            self.face_offset[rb_id] = face_count
            self.bp_offset[rb_id] = bp_count

            for i in range(n_verts):
                self.vertices[vertex_count + i] = ti.Vector(verts[i])
                self.v2rb[vertex_count + i] = rb_id
            for i in range(n_faces):
                self.faces[face_count + i] = ti.Vector(faces[i] + vertex_count)
                self.f2rb[face_count + i] = rb_id

                vx = verts[faces[i][1]] - verts[faces[i][0]]
                vy = verts[faces[i][2]] - verts[faces[i][0]]
                face_normal = np.cross(vx, vy)
                norm_val = np.linalg.norm(face_normal)
                if norm_val > 1e-10:
                    self.face_normals[face_count + i] = ti.Vector(
                        face_normal / norm_val
                    )
                else:
                    self.face_normals[face_count + i] = ti.Vector([0, 1, 0])

                rx = np.linalg.norm(vx)
                ry = np.linalg.norm(vy)
                nx = vx / rx
                ny = vy / ry
                x_length = min(rx / 3, self.dx * 0.5)
                while x_length < rx + self.dx:
                    y_length = min(ry / 3, self.dx * 0.5)
                    while y_length < ry + self.dx:
                        if (x_length / rx + y_length / ry) < 1.0:
                            point = verts[faces[i][0]] + nx * x_length + ny * y_length
                            self.boundary_particles[bp_count] = ti.Vector(point)
                            self.bp2rb[bp_count] = rb_id
                            self.bp2f[bp_count] = face_count + i
                            bp_count += 1
                        y_length += self.dx
                    x_length += self.dx

            vertex_count += n_verts
            face_count += n_faces

        self.vert_offset[self.n_rigid] = vertex_count
        self.face_offset[self.n_rigid] = face_count
        self.bp_offset[self.n_rigid] = bp_count

        self.n_vertices[None] = vertex_count
        self.n_faces[None] = face_count
        self.n_boundary_particles[None] = bp_count

    @ti.kernel
    def update_rotation_matrices(self):
        for i in range(self.n_rigid):
            q = self.orientations[i]
            w, x, y, z = q[0], q[1], q[2], q[3]
            R = ti.Matrix(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            self.rotation_matrices[i] = R

    @ti.func
    def get_bp_position(self, p):
        rb_id = self.bp2rb[p]
        com_pos = self.positions[rb_id]
        bp_offset = self.boundary_particles[p]
        rotation = self.rotation_matrices[rb_id]
        world_pos = com_pos + rotation @ bp_offset
        return world_pos

    @ti.func
    def get_vertices_of_face(self, f_id):
        v0_id = self.faces[f_id][0]
        v1_id = self.faces[f_id][1]
        v2_id = self.faces[f_id][2]
        rb_id = self.v2rb[v0_id]
        com_pos = self.positions[rb_id]
        rotation = self.rotation_matrices[rb_id]
        v0 = com_pos + rotation @ self.vertices[v0_id]
        v1 = com_pos + rotation @ self.vertices[v1_id]
        v2 = com_pos + rotation @ self.vertices[v2_id]
        return v0, v1, v2

    @ti.func
    def get_rigid_velocity(self, rb_id, point):
        com_pos = self.positions[rb_id]
        vel = self.velocities[rb_id]
        ang_vel = self.angular_velocities[rb_id]
        r = point - com_pos
        rigid_vel = vel + ang_vel.cross(r)
        return rigid_vel

    @ti.func
    def apply_impulse(self, rb_id, contact_point, impulse):
        self.impulses[rb_id] += impulse
        r = contact_point - self.positions[rb_id]
        self.angular_impulses[rb_id] += r.cross(impulse)

    def step(self, time: float):
        self.resolve_collisions()
        self.integrate(time)
        self.update_rotation_matrices()

    def integrate(self, time: float):
        for i in range(self.n_rigid):
            if self.scripted_trajectories[i] is not None:
                pos_t, ori_t = self.scripted_trajectories[i](time)
                pos_next, ori_next = self.scripted_trajectories[i](time + self.dt)

                vel = (np.array(pos_next) - np.array(pos_t)) / self.dt

                q_t = np.array(ori_t)
                q_next = np.array(ori_next)
                q_t_inv = np.array([q_t[0], -q_t[1], -q_t[2], -q_t[3]])

                w1, x1, y1, z1 = q_next
                w2, x2, y2, z2 = q_t_inv

                w_diff = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x_diff = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y_diff = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                z_diff = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

                if w_diff < 0:
                    w_diff = -w_diff
                    x_diff = -x_diff
                    y_diff = -y_diff
                    z_diff = -z_diff

                ang_vel = 2.0 * np.array([x_diff, y_diff, z_diff]) / self.dt

                self.positions[i] = ti.Vector(pos_t)
                self.orientations[i] = ti.Vector(ori_t).normalized()
                self.velocities[i] = ti.Vector(vel)
                self.angular_velocities[i] = ti.Vector(ang_vel)
                self.impulses[i] = ti.Vector([0.0, 0.0, 0.0])
                self.angular_impulses[i] = ti.Vector([0.0, 0.0, 0.0])

            else:
                self.velocities[i] += self.impulses[i] / self.masses[i]
                self.velocities[i] += self.gravity * self.dt
                self.positions[i] += self.velocities[i] * self.dt

                R = self.rotation_matrices[i]
                I_inv_world = R @ self.inv_inertia[i] @ R.transpose()
                self.angular_velocities[i] += I_inv_world @ self.angular_impulses[i]

                self.impulses[i] = ti.Vector([0.0, 0.0, 0.0])
                self.angular_impulses[i] = ti.Vector([0.0, 0.0, 0.0])

                w = self.angular_velocities[i]
                q = self.orientations[i]

                wx, wy, wz = w[0], w[1], w[2]
                qw, qx, qy, qz = q[0], q[1], q[2], q[3]

                dqw = -0.5 * (wx * qx + wy * qy + wz * qz)
                dqx = 0.5 * (wx * qw + wy * qz - wz * qy)
                dqy = 0.5 * (wy * qw + wz * qx - wx * qz)
                dqz = 0.5 * (wz * qw + wx * qy - wy * qx)

                self.orientations[i][0] += dqw * self.dt
                self.orientations[i][1] += dqx * self.dt
                self.orientations[i][2] += dqy * self.dt
                self.orientations[i][3] += dqz * self.dt

                self.orientations[i] = self.orientations[i].normalized()

    def resolve_collisions(self):
        pass
