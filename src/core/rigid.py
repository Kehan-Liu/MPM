import taichi as ti
import numpy as np
from src.objects import RigidObject, ClothObject
from typing import List, Tuple, Optional
from src.core.rigid_body import RigidBody
from src.core.cloth import Cloth
import json
from tqdm import tqdm
import os

MAX_VERTICES = 100000
MAX_FACES = 200000
MAX_BP = 400000


@ti.kernel
def rigid_rigid_penalty(
    M: ti.template(),
    A: ti.template(),
    B: ti.template(),
    stiffness: ti.f32,
    damping: ti.f32,
    margin: ti.f32,
    dt: ti.f32,
    contact_flag: ti.template(),
    a_idx: ti.i32,
    diag_counts: ti.template(),
    diag_max_fn: ti.template(),
    diag_sum_imp: ti.template(),
):
    max_fn = ti.cast(1e3, ti.f32)
    # iterate boundary particles that belong to A
    n_bp = M.n_boundary_particles[None]
    total_area = M.rb_total_area[a_idx]
    for p in range(n_bp):
        if M.bp2rb[p] != a_idx:
            continue
        xA = M.get_bp_position(p)
        phi, n, cp, fidx = B.signed_distance(xA)

        if phi < 0.0:
            # dead-zone to reduce jitter
            phi_eff = ti.min(phi + margin, 0.0)

            vA = A.get_velocity_at_point(xA)
            vB = B.get_velocity_at_point(cp)
            vn = (vA - vB).dot(n)

            vn_closing = ti.min(vn, 0.0)

            # area-weighted stiffness/damping so total stiffness is independent of sampling
            area_i = M.bp_area[p]
            frac = area_i / total_area if total_area > 1e-12 else 1.0
            k_i = stiffness * frac * M.penalty_parameter[None]
            c_i = damping * frac * M.penalty_parameter[None]

            f_n = -k_i * phi_eff - c_i * vn_closing

            # prohibit attraction + cap
            f_n = ti.max(0.0, f_n)
            if f_n > max_fn:
                f_n = max_fn

            F = f_n * n
            A.apply_impulse_at_point(F * dt, cp)
            B.apply_impulse_at_point(-F * dt, cp)
            ti.atomic_add(contact_flag[None], 1)
            # diagnostics
            ti.atomic_add(diag_counts[a_idx], 1)
            if f_n > diag_max_fn[a_idx]:
                diag_max_fn[a_idx] = f_n
            ti.atomic_add(diag_sum_imp[a_idx][0], (F * dt)[0])
            ti.atomic_add(diag_sum_imp[a_idx][1], (F * dt)[1])
            ti.atomic_add(diag_sum_imp[a_idx][2], (F * dt)[2])


@ti.kernel
def mesh_box_penalty(
    M: ti.template(),
    rb: ti.template(),
    xmin: ti.f32,
    ymin: ti.f32,
    zmin: ti.f32,
    xmax: ti.f32,
    ymax: ti.f32,
    zmax: ti.f32,
    stiffness: ti.f32,
    damping: ti.f32,
    margin: ti.f32,
    mu_t: ti.f32,
    c_t: ti.f32,
    max_fn: ti.f32,
    dt: ti.f32,
    rb_idx: ti.i32,
    diag_counts: ti.template(),
    diag_max_fn: ti.template(),
    diag_sum_imp: ti.template(),
):
    # iterate boundary particles for this rigid body
    total_area = M.rb_total_area[rb_idx]
    n_bp = M.n_boundary_particles[None]
    for p in range(n_bp):
        if M.bp2rb[p] != rb_idx:
            continue
        x = M.get_bp_position(p)
        best_m = ti.float32(0.0)
        best_phi = ti.float32(0.0)
        best_n = ti.Vector([0.0, 0.0, 0.0])
        # x >= xmin plane (n=+X, h=xmin)
        phi = x[0] - xmin
        if phi < -margin:
            m = -phi
            if m > best_m:
                best_m = m
                best_phi = phi
                best_n = ti.Vector([1.0, 0.0, 0.0])
        # x <= xmax plane (n=-X, h=-xmax)
        phi = (-x[0]) - (-xmax)  # = xmax - x0
        if phi < -margin:
            m = -phi
            if m > best_m:
                best_m = m
                best_phi = phi
                best_n = ti.Vector([-1.0, 0.0, 0.0])
        # y >= ymin plane (n=+Y, h=ymin)
        phi = x[1] - ymin
        if phi < -margin:
            m = -phi
            if m > best_m:
                best_m = m
                best_phi = phi
                best_n = ti.Vector([0.0, 1.0, 0.0])
        # y <= ymax plane (n=-Y, h=-ymax)
        phi = (-x[1]) - (-ymax)  # = ymax - y
        if phi < -margin:
            m = -phi
            if m > best_m:
                best_m = m
                best_phi = phi
                best_n = ti.Vector([0.0, -1.0, 0.0])
        # z >= zmin plane (n=+Z, h=zmin)
        phi = x[2] - zmin
        if phi < -margin:
            m = -phi
            if m > best_m:
                best_m = m
                best_phi = phi
                best_n = ti.Vector([0.0, 0.0, 1.0])
        # z <= zmax plane (n=-Z, h=-zmax)
        phi = (-x[2]) - (-zmax)  # = zmax - z
        if phi < -margin:
            m = -phi
            if m > best_m:
                best_m = m
                best_phi = phi
                best_n = ti.Vector([0.0, 0.0, -1.0])
        if best_m > 0.0:
            cp = x - best_phi * best_n
            nn = best_n
            v = rb.get_velocity_at_point(cp)
            vn = v.dot(nn)
            vt = v - vn * nn
            # area-weighted stiffness/damping (scaled by global penalty parameter)
            area_i = M.bp_area[p]
            frac = area_i / total_area if total_area > 1e-12 else 1.0
            k_i = stiffness * frac * M.penalty_parameter[None]
            c_i = damping * frac * M.penalty_parameter[None]
            f_n = -k_i * best_phi - c_i * ti.min(vn, 0.0)
            # cap
            if f_n > max_fn:
                f_n = max_fn
            if f_n < -max_fn:
                f_n = -max_fn
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
            F = f_n * nn + f_t
            rb.apply_impulse_at_point(F * dt, cp)
            # diagnostics
            ti.atomic_add(diag_counts[rb_idx], 1)
            if ti.abs(f_n) > diag_max_fn[rb_idx]:
                diag_max_fn[rb_idx] = ti.abs(f_n)
            ti.atomic_add(diag_sum_imp[rb_idx][0], (F * dt)[0])
            ti.atomic_add(diag_sum_imp[rb_idx][1], (F * dt)[1])
            ti.atomic_add(diag_sum_imp[rb_idx][2], (F * dt)[2])


@ti.kernel
def mesh_plane_penalty(
    M: ti.template(),
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
    total_area = M.rb_total_area[0]  # not used per-rb here; caller should ensure appropriate rb usage
    n_bp = M.n_boundary_particles[None]
    for p in range(n_bp):
        rb_id = M.bp2rb[p]
        # use plane for all bodies; filter inside caller if needed
        x = M.get_bp_position(p)
        phi = x.dot(n) - h
        if phi < 0 and ti.abs(phi) > margin:
            cp = x - phi * n
            nn = n
            # find corresponding rb template? assume caller passes per-rb rb when needed
            # Here we perform per-sample penalty using the per-sample area weighting
            v = rb.get_velocity_at_point(cp)
            vn = v.dot(nn)
            vt = v - vn * nn
            area_i = M.bp_area[p]
            total_area_rb = M.rb_total_area[rb_id]
            frac = area_i / total_area_rb if total_area_rb > 1e-12 else 1.0
            k_i = stiffness * frac * M.penalty_parameter[None]
            c_i = damping * frac * M.penalty_parameter[None]
            f_n = -k_i * phi - c_i * vn
            if f_n > max_fn:
                f_n = max_fn
            if f_n < -max_fn:
                f_n = -max_fn
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
            F = f_n * nn + f_t
            rb.apply_impulse_at_point(F * dt, cp)


@ti.data_oriented
class Rigid:
    def __init__(
        self,
        rigid_objects: List[RigidObject],
        dx: float,
        dt: float,
        gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
        damping: int = 1e4,
        margin: float = 1e-5,
        stiffness: float = 5e4,
        penalty_parameter: float = 1.0,
        clamp_factor: float = 20.0,
        cloth_objects: Optional[List[ClothObject]] = None,
    ):
        """
        现在这个 Rigid 类负责：
        - 多刚体动力学（原有功能）
        - 多块布料（Cloth）的推进和刚体碰撞
        """
        self.n_rigid = len(rigid_objects)
        self.rigid_conf = rigid_objects
        self.rigid_objects: List[RigidBody] = []
        self.dt = dt
        self.dx = dx
        self.gravity = ti.Vector(gravity)

        self.scripted_trajectories = [obj.scripted_trajectory for obj in rigid_objects]

        # 布料
        self.cloth_conf: List[ClothObject] = cloth_objects or []
        self.cloths: List[Cloth] = []
        for c_obj in self.cloth_conf:
            self.cloths.append(Cloth(c_obj, dt, gravity))

        # Initialize Taichi fields for rigid body properties
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.angular_velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.rotation_matrices = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=self.n_rigid
        )

        self.impulses = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        self.angular_impulses = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
        # Flag to indicate a contact was detected during penalty kernels
        self.contact_flag = ti.field(dtype=ti.i32, shape=())
        # Diagnostics: per-rigid contact counts, max normal force, summed contact impulse
        self.diag_contact_counts = ti.field(dtype=ti.i32, shape=self.n_rigid)
        self.diag_max_fn = ti.field(dtype=ti.f32, shape=self.n_rigid)
        self.diag_sum_imp = ti.Vector.field(3, dtype=ti.f32, shape=self.n_rigid)
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
        self.bp_area = ti.field(dtype=ti.f32, shape=MAX_BP)
        self.bp2rb = ti.field(dtype=ti.i32, shape=MAX_BP)
        self.bp2f = ti.field(dtype=ti.i32, shape=MAX_BP)
        self.n_boundary_particles = ti.field(dtype=ti.i32, shape=())
        self.rb_total_area = ti.field(dtype=ti.f32, shape=self.n_rigid)
        # global penalty scaling parameter (can be tuned at runtime)
        self.penalty_parameter = ti.field(dtype=ti.f32, shape=())
        self.damping = damping
        self.margin = margin
        self.stiffness = stiffness
        # set default penalty scaling
        self.penalty_parameter[None] = penalty_parameter
        # per-step clamp factor (multiplier of gravity impulse). Can be tuned at runtime.
        self.clamp_factor = clamp_factor
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
        # helper containers to compute per-face sample counts and face areas
        face_sample_count = {}
        face_area = {}
        for rb_id, obj in enumerate(self.rigid_objects):
            mesh = obj.mesh

            # Compute CoM and Inertia manually (Shell)
            verts = mesh.vertices
            faces = mesh.faces

            n_verts = len(verts)
            n_faces = len(faces)

            for i in range(n_verts):
                self.vertices[vertex_count + i] = ti.Vector(verts[i])
            for i in tqdm(range(n_faces), desc="Loading faces", unit="face"):
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
                # increase sampling density: use a fner step (self.dx / 10)
                x_length = min(rx / 3, self.dx * 0.5)
                while x_length < rx + self.dx:
                    y_length = min(ry / 3, self.dx * 0.5)
                    while y_length < ry + self.dx:
                        if (x_length / rx + y_length / ry) < 1.0:
                            point = verts[faces[i][0]] + nx * x_length + ny * y_length
                            self.boundary_particles[bp_count] = ti.Vector(point)
                            self.bp2rb[bp_count] = rb_id
                            self.bp2f[bp_count] = face_count + i
                            # count sample for this face
                            face_sample_count.setdefault(face_count + i, 0)
                            face_sample_count[face_count + i] += 1
                            bp_count += 1
                        y_length += self.dx
                    x_length += self.dx

            vertex_count += n_verts
            face_count += n_faces

        # compute face areas and total area per rigid body
        for f_id in range(face_count - n_faces, face_count):
            # compute area using vertices
            v0 = np.array(self.vertices[self.faces[f_id][0]])
            v1 = np.array(self.vertices[self.faces[f_id][1]])
            v2 = np.array(self.vertices[self.faces[f_id][2]])
            a = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5
            face_area[f_id] = a

        # distribute per-sample area
        # first compute total area per rigid
        total_area_per_rb = [0.0] * self.n_rigid
        for f_id, a in face_area.items():
            rb = int(self.f2rb[f_id])
            total_area_per_rb[rb] += a

        # now assign area to each boundary particle proportional to its face area / samples
        for p in range(bp_count):
            f_id = int(self.bp2f[p])
            a = face_area.get(f_id, 0.0)
            samples = face_sample_count.get(f_id, 1)
            per = a / float(samples) if samples > 0 else 0.0
            self.bp_area[p] = float(per)

        # set total area per rigid in field
        for rb_id in range(self.n_rigid):
            self.rb_total_area[rb_id] = float(total_area_per_rb[rb_id])

        self.n_boundary_particles[None] = bp_count
        print(
            f"Loaded {vertex_count} vertices, {face_count} faces, {bp_count} boundary particles"
        )

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

    def resolve_collisions(self, time: float):
        # Pairwise rigid-rigid collision
        stiffness = self.stiffness
        damping = self.damping
        margin = self.margin

        for i in range(self.n_rigid):
            for j in range(i + 1, self.n_rigid):
                A = self.rigid_objects[i]
                B = self.rigid_objects[j]
                # If both are Balls, prefer analytic detection + impulse, log collision
                is_ball_i = (
                    isinstance(self.rigid_conf[i].meshdir, str)
                    and self.rigid_conf[i].meshdir.lower() == "ball"
                )
                is_ball_j = (
                    isinstance(self.rigid_conf[j].meshdir, str)
                    and self.rigid_conf[j].meshdir.lower() == "ball"
                )
                if is_ball_i and is_ball_j:
                    pi = A.position[None].to_numpy()
                    pj = B.position[None].to_numpy()
                    # estimate radius from centered mesh verts
                    try:
                        ri = float(
                            np.max(
                                np.linalg.norm(
                                    A.mesh.vertices
                                    - A.mass_center_offset[None].to_numpy(),
                                    axis=1,
                                )
                            )
                        )
                        rj = float(
                            np.max(
                                np.linalg.norm(
                                    B.mesh.vertices
                                    - B.mass_center_offset[None].to_numpy(),
                                    axis=1,
                                )
                            )
                        )
                    except Exception:
                        ri, rj = 0.1, 0.1
                    dist = float(np.linalg.norm(pi - pj))
                    collided = dist <= (ri + rj)
                    if collided:
                        n = pi - pj
                        n_norm = float(np.linalg.norm(n))
                        if n_norm > 1e-8:
                            n = n / n_norm
                            vA = A.velocity[None].to_numpy()
                            vB = B.velocity[None].to_numpy()
                            vn = float(np.dot(vA - vB, n))
                            if vn < 0.0:
                                try:
                                    mA = float(A.mass[None])
                                except Exception:
                                    mA = (
                                        float(A.mass)
                                        if not hasattr(A.mass, "__getitem__")
                                        else float(A.mass[None])
                                    )
                                try:
                                    mB = float(B.mass[None])
                                except Exception:
                                    mB = (
                                        float(B.mass)
                                        if not hasattr(B.mass, "__getitem__")
                                        else float(B.mass[None])
                                    )
                                e_pair = (
                                    float(max(self.restitution[i], self.restitution[j]))
                                    if hasattr(self, "restitution")
                                    else 0.0
                                )
                                J = -(1.0 + e_pair) * vn / (1.0 / mA + 1.0 / mB)
                                vA_new = vA + (J * n) / mA
                                vB_new = vB - (J * n) / mB
                                A.velocity[None] = vA_new.astype(np.float32)
                                B.velocity[None] = vB_new.astype(np.float32)
                    # Skip mesh-based penalty for ball-ball to avoid double correction
                    continue
                # Generic meshes
                # reset contact flag and diagnostics for this pair
                self.contact_flag[None] = 0
                self.diag_contact_counts[i] = 0
                self.diag_contact_counts[j] = 0
                self.diag_max_fn[i] = 0.0
                self.diag_max_fn[j] = 0.0
                self.diag_sum_imp[i] = ti.Vector([0.0, 0.0, 0.0])
                self.diag_sum_imp[j] = ti.Vector([0.0, 0.0, 0.0])
                rigid_rigid_penalty(
                    self,
                    A,
                    B,
                    stiffness,
                    damping,
                    margin,
                    self.dt,
                    self.contact_flag,
                    i,
                    self.diag_contact_counts,
                    self.diag_max_fn,
                    self.diag_sum_imp,
                )
                rigid_rigid_penalty(
                    self,
                    B,
                    A,
                    stiffness,
                    damping,
                    margin,
                    self.dt,
                    self.contact_flag,
                    j,
                    self.diag_contact_counts,
                    self.diag_max_fn,
                    self.diag_sum_imp,
                )
                if int(self.contact_flag[None]) > 0:
                    # print accumulated impulses on each body for debugging
                    try:
                        # total impulses stored on the RigidBody (includes external + gravity + contact-added)
                        impA = self.rigid_objects[i].impulse[None].to_numpy()
                        impB = self.rigid_objects[j].impulse[None].to_numpy()
                        angA = self.rigid_objects[i].angular_impulse[None].to_numpy()
                        angB = self.rigid_objects[j].angular_impulse[None].to_numpy()

                        # external impulses (those written into self.impulses[] before resolve_collisions)
                        extA = self.impulses[i].to_numpy()
                        extB = self.impulses[j].to_numpy()

                        # gravity impulse = gravity * mass * dt (compute in numpy space)
                        mA = float(self.rigid_objects[i].mass[None])
                        mB = float(self.rigid_objects[j].mass[None])
                        g_vec = np.array([float(self.gravity[0]), float(self.gravity[1]), float(self.gravity[2])], dtype=np.float32)
                        g_impA = (g_vec * mA * float(self.dt)).astype(np.float32)
                        g_impB = (g_vec * mB * float(self.dt)).astype(np.float32)

                        # contact-added impulse = total - external - gravity
                        contactA = impA - extA - g_impA
                        contactB = impB - extB - g_impB

                        print(
                            f"collision detected between rigid {i} and {j} at time {time};"
                            f" totalA={impA}, extA={extA}, gravityA={g_impA}, contactA={contactA};"
                            f" totalB={impB}, extB={extB}, gravityB={g_impB}, contactB={contactB};"
                            f" angA={angA}, angB={angB}"
                        )
                        # print diagnostic summary per body
                        try:
                            dc_i = int(self.diag_contact_counts[i])
                            dc_j = int(self.diag_contact_counts[j])
                            max_i = float(self.diag_max_fn[i])
                            max_j = float(self.diag_max_fn[j])
                            sum_i = self.diag_sum_imp[i].to_numpy()
                            sum_j = self.diag_sum_imp[j].to_numpy()
                            print(f"diag i(count,max_fn,sum_imp) = ({dc_i},{max_i},{sum_i}) | diag j = ({dc_j},{max_j},{sum_j})")
                        except Exception:
                            pass
                    except Exception:
                        print(f"collision detected between rigid {i} and {j} at time {time} (failed to read impulses)")

        # Plane collision (box boundaries [0, 1]^3)
        mu_t = 0.2
        c_t = 5000.0
        max_force = 1e4

        for i in range(self.n_rigid):
            # reset per-body diagnostics before boundary penalty
            self.diag_contact_counts[i] = 0
            self.diag_max_fn[i] = 0.0
            self.diag_sum_imp[i] = ti.Vector([0.0, 0.0, 0.0])
            mesh_box_penalty(
                self,
                self.rigid_objects[i],
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                stiffness,
                damping,
                margin,
                mu_t,
                c_t,
                max_force,
                self.dt,
                i,
                self.diag_contact_counts,
                self.diag_max_fn,
                self.diag_sum_imp,
            )
            # if boundary penalty produced contacts, print diagnostics for rigid vs box
            try:
                if int(self.diag_contact_counts[i]) > 0:
                    imp = self.rigid_objects[i].impulse[None].to_numpy()
                    ext = self.impulses[i].to_numpy()
                    ang = self.rigid_objects[i].angular_impulse[None].to_numpy()
                    m = float(self.rigid_objects[i].mass[None])
                    g_vec = np.array([float(self.gravity[0]), float(self.gravity[1]), float(self.gravity[2])], dtype=np.float32)
                    g_imp = (g_vec * m * float(self.dt)).astype(np.float32)
                    contact = imp - ext - g_imp
                    print(
                        f"collision detected between rigid {i} and box at time {time};"
                        f" total={imp}, ext={ext}, gravity={g_imp}, contact={contact}; ang={ang}"
                    )
                    try:
                        dc = int(self.diag_contact_counts[i])
                        mx = float(self.diag_max_fn[i])
                        sm = self.diag_sum_imp[i].to_numpy()
                        print(f"diag (count,max_fn,sum_imp) = ({dc},{mx},{sm})")
                    except Exception:
                        pass
            except Exception:
                pass

        # Per-step clamp: limit total contact impulse per rigid to avoid explosive corrections.
        # Limit chosen as `clamp_factor * mass * |gravity| * dt` (i.e. several times the gravity impulse).
        clamp_factor = self.clamp_factor
        for rb_id in range(self.n_rigid):
            try:
                rb_obj = self.rigid_objects[rb_id]
                total_imp = rb_obj.impulse[None].to_numpy()
                ext_imp = self.impulses[rb_id].to_numpy()
                mass = float(rb_obj.mass[None])
                g_vec = np.array([float(self.gravity[0]), float(self.gravity[1]), float(self.gravity[2])], dtype=np.float32)
                g_imp = (g_vec * mass * float(self.dt)).astype(np.float32)
                contact_imp = total_imp - ext_imp - g_imp
                contact_norm = float(np.linalg.norm(contact_imp))
                limit = float(clamp_factor * np.linalg.norm(g_vec) * mass * float(self.dt))
                if contact_norm > limit and contact_norm > 1e-12:
                    scaled = contact_imp * (limit / contact_norm)
                    new_total = ext_imp + g_imp + scaled
                    rb_obj.impulse[None] = new_total.astype(np.float32)
            except Exception:
                pass

    def step(self, time: float):
        """
        一个全局 step：
        1. 把 MPM 等外部给刚体的 impulse + 重力写进 RigidBody
        2. 处理刚体-刚体、刚体-边界碰撞（添加刚体冲量）
        3. 推进所有布料（内部弹簧 + 布料-刚体 + 布料-边界）
        4. 推进刚体（使用累积冲量）
        """
        # 1) 写入刚体 impulse / angular_impulse
        for i, obj in enumerate(self.rigid_objects):
            obj.impulse[None] = (
                self.impulses[i] + self.gravity * obj.mass[None] * self.dt
            )
            obj.angular_impulse[None] = self.angular_impulses[i]

        # 2) 刚体-刚体 & 刚体-边界碰撞
        self.resolve_collisions(time)

        # 3) 推进所有布料（内部会调用 cloth_rigid_penalty 给刚体施加冲量）
        if self.cloths:
            for cloth in self.cloths:
                cloth.step(time, self.rigid_objects)

        # 4) 推进刚体并同步外层字段
        for i, obj in enumerate(self.rigid_objects):
            obj.step(time, self.dt)
            self.positions[i] = obj.position[None]
            self.velocities[i] = obj.velocity[None]
            self.angular_velocities[i] = obj.angular_velocity[None]
            self.rotation_matrices[i] = obj.rotation_matrix[None]
            self.impulses[i] = ti.Vector([0.0, 0.0, 0.0])
            self.angular_impulses[i] = ti.Vector([0.0, 0.0, 0.0])

    def export(self, frame, output_dir: str):
        rigid_path = os.path.join(output_dir, "rigid")
        for i, obj in enumerate(self.rigid_objects):
            obj.export(frame, rigid_path, rb_id=i)
