import taichi as ti
import numpy as np
from src.objects import RigidObject
from typing import List, Tuple
from src.core.rigid_body import RigidBody

MAX_VERTICES = 100000
MAX_FACES = 200000
MAX_BP = 400000


@ti.kernel
def rigid_rigid_penalty(
    A: ti.template(),
    B: ti.template(),
    stiffness: ti.f32,
    damping: ti.f32,
    margin: ti.f32,
    dt: ti.f32,
):
    # For each vertex of A, compute signed distance to mesh B; if penetrating, apply penalty+damping
    for i in range(A.vertices.shape[0]):
        xA = A.position[None] + A.rotation_matrix[None] @ A.vertices[i]
        phi, n = B.signed_distance(xA)
        if phi < 0 and ti.abs(phi) > margin:
            cp = xA - phi * n
            vA = A.get_velocity_at_point(cp)
            vB = B.get_velocity_at_point(cp)
            vn = (vA - vB).dot(n)
            F = (-stiffness * phi - damping * vn) * n
            A.apply_impulse_at_point(F * dt, cp)
            B.apply_impulse_at_point(-F * dt, cp)


@ti.kernel
def mesh_plane_penalty(
    rb: ti.template(),
    nx: ti.f32,
    ny: ti.f32,
    nz: ti.f32,
    h: ti.f32,
    stiffness: ti.f32,
    damping: ti.f32,
    margin: ti.f32,
    mu_t: ti.f32,
    c_t: ti.f32,
    max_fn: ti.f32,
    dt: ti.f32,
):
    # plane: n·x = h, with unit normal n
    n = ti.Vector([nx, ny, nz])
    for i in range(rb.faces.shape[0]):
        # iterate triangle vertices for contact sampling
        for k in ti.static(range(3)):
            vidx = rb.faces[i][k]
            x = rb.position[None] + rb.rotation_matrix[None] @ rb.vertices[vidx]
            phi = x.dot(n) - h
            if phi < 0 and ti.abs(phi) > margin:
                cp = x - phi * n
                v = rb.get_velocity_at_point(cp)
                vn = v.dot(n)
                vt = v - vn * n
                # normal spring-damper with soft cap
                f_n = -stiffness * phi - damping * vn
                if f_n > max_fn:
                    f_n = max_fn
                if f_n < -max_fn:
                    f_n = -max_fn
                # tangential viscous + Coulomb limit
                f_t = ti.Vector([0.0, 0.0, 0.0])
                vt_norm = vt.norm()
                if vt_norm > 1e-8:
                    f_visc = -c_t * vt
                    limit = mu_t * ti.abs(f_n)
                    f_visc_norm = f_visc.norm()
                    if f_visc_norm > limit:
                        f_t = -limit * (vt / vt_norm)
                    else:
                        f_t = f_visc
                F = f_n * n + f_t
                rb.apply_impulse_at_point(F * dt, cp)


@ti.data_oriented
class Rigid:
    def __init__(
        self,
        rigid_objects: List[RigidObject],
        dx: float,
        dt: float,
        gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
    ):
        self.n_rigid = len(rigid_objects)
        self.rigid_conf = rigid_objects
        self.rigid_objects: List[RigidBody] = []
        self.dt = dt
        self.dx = dx
        self.gravity = ti.Vector(gravity)

        self.scripted_trajectories = [obj.scripted_trajectory for obj in rigid_objects]

        # Initialize Taichi fields for rigid body properties
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.angular_velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.rotation_matrices = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_rigid)

        self.impulses = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.angular_impulses = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.kh = ti.field(dtype=ti.f32, shape=self.n_rigid)
        self.friction = ti.field(dtype=ti.f32, shape=self.n_rigid)
        self.restitution = ti.field(dtype=ti.f32, shape=self.n_rigid)
        self.splitter = ti.field(dtype=ti.f32, shape=self.n_rigid)

        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=MAX_FACES)
        self.face_normals = ti.Vector.field(3, dtype=ti.f32, shape=MAX_FACES)
        self.f2rb = ti.field(dtype=ti.i32, shape=MAX_FACES)

        self.boundary_particles = ti.Vector.field(
            3, dtype=ti.f32, shape=MAX_BP
        )  # offset to CoM
        self.bp2rb = ti.field(dtype=ti.i32, shape=MAX_BP)
        self.bp2f = ti.field(dtype=ti.i32, shape=MAX_BP)
        self.n_boundary_particles = ti.field(dtype=ti.i32, shape=())

        self.init_rigid_states()
        self.load_mesh_data()

    def init_rigid_states(self):
        print("Initializing rigid body states...")
        for i, obj in enumerate(self.rigid_conf):
            self.positions[i] = ti.Vector(obj.position)
            self.velocities[i] = ti.Vector(obj.velocity)
            self.angular_velocities[i] = ti.Vector(obj.angular_velocity)
            self.kh[i] = obj.material.kh
            self.friction[i] = obj.material.friction
            self.restitution[i] = obj.material.restitution
            self.splitter[i] = obj.material.splitter
            self.rigid_objects.append(RigidBody(obj))
            self.rotation_matrices[i] = self.rigid_objects[i].rotation_matrix[None]

    def load_mesh_data(self):
        vertex_count = 0
        face_count = 0
        bp_count = 0

        print("Loading rigid body mesh data...")
        for rb_id, obj in enumerate(self.rigid_objects):
            mesh = obj.mesh

            # Compute CoM and Inertia manually (Shell)
            verts = mesh.vertices - obj.mass_center_offset[None].to_numpy()
            faces = mesh.faces

            n_verts = len(verts)
            n_faces = len(faces)

            for i in range(n_verts):
                self.vertices[vertex_count + i] = ti.Vector(verts[i])
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

        self.n_boundary_particles[None] = bp_count

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
        rb_id = self.f2rb[f_id]
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

    def resolve_collisions(self):
        # Pairwise rigid-rigid collision
        stiffness = 5e4
        damping = 1e4
        margin = 1e-3
        
        print(self.n_rigid)
        print(self.rigid_objects)
        for i in range(self.n_rigid):
            for j in range(i + 1, self.n_rigid):
                rigid_rigid_penalty(
                    self.rigid_objects[i],
                    self.rigid_objects[j],
                    stiffness,
                    damping,
                    margin,
                    self.dt,
                )

        # Plane collision (box boundaries [0, 1]^3)
        mu_t = 0.2
        c_t = 5000.0
        max_force = 1e4

        print(self.n_rigid)
        print(self.rigid_objects)
        for i in range(self.n_rigid):
            # y > 0
            mesh_plane_penalty(
                self.rigid_objects[i],
                0.0,
                1.0,
                0.0,
                0.0,
                stiffness,
                damping,
                margin,
                mu_t,
                c_t,
                max_force,
                self.dt,
            )
            # y < 1
            mesh_plane_penalty(
                self.rigid_objects[i],
                0.0,
                -1.0,
                0.0,
                -1.0,
                stiffness,
                damping,
                margin,
                mu_t,
                c_t,
                max_force,
                self.dt,
            )
            # x > 0
            mesh_plane_penalty(
                self.rigid_objects[i],
                1.0,
                0.0,
                0.0,
                0.0,
                stiffness,
                damping,
                margin,
                mu_t,
                c_t,
                max_force,
                self.dt,
            )
            # x < 1
            mesh_plane_penalty(
                self.rigid_objects[i],
                -1.0,
                0.0,
                0.0,
                -1.0,
                stiffness,
                damping,
                margin,
                mu_t,
                c_t,
                max_force,
                self.dt,
            )
            # z > 0
            mesh_plane_penalty(
                self.rigid_objects[i],
                0.0,
                0.0,
                1.0,
                0.0,
                stiffness,
                damping,
                margin,
                mu_t,
                c_t,
                max_force,
                self.dt,
            )
            # z < 1
            mesh_plane_penalty(
                self.rigid_objects[i],
                0.0,
                0.0,
                -1.0,
                -1.0,
                stiffness,
                damping,
                margin,
                mu_t,
                c_t,
                max_force,
                self.dt,
            )

    def step(self, time: float):
        for i, obj in enumerate(self.rigid_objects):
            obj.impulse[None] = (
                self.impulses[i] + self.gravity * obj.mass[None] * self.dt
            )
            obj.angular_impulse[None] = self.angular_impulses[i]

        self.resolve_collisions()

        for i, obj in enumerate(self.rigid_objects):
            obj.step(time, self.dt)
            self.positions[i] = obj.position[None]
            self.velocities[i] = obj.velocity[None]
            self.angular_velocities[i] = obj.angular_velocity[None]
            self.rotation_matrices[i] = obj.rotation_matrix[None]
            self.impulses[i] = ti.Vector([0.0, 0.0, 0.0])
            self.angular_impulses[i] = ti.Vector([0.0, 0.0, 0.0])
