"""Impulse-based rigid body simulation (minimal prototype).

Features:
- RigidBody class holding mass properties, pose, velocities
- Simple collision detection: sphere-sphere, AABB vs AABB, ground plane (z=0)
- Impulse resolution with restitution and Coulomb friction
- Quaternion based orientation integration
- Per-frame mesh export using trimesh (geometry transformed)

Usage example (PowerShell):
    python simulators\solid\rigid_body_sim.py \
        --dt 0.016 --frames 240 --gravity 0,0,-9.8 \
        --body box:mass=2.0:size=1,1,1:pos=-2,0,1:vel=2,0,0:color=0.2,0.6,0.9 \
        --body box:mass=2.0:size=1,1,1:pos= 2,0,1:vel=-2,0,0:color=0.9,0.4,0.2 \
        --restitution 0.6 --friction 0.4 --out render\output\rigid_frames

Mesh import variant:
    python simulators\solid\rigid_body_sim.py --dt 0.01 --frames 300 \
        --body mesh:mesh=render/assets/demo_cube.obj:mass=1.0:pos=0,0,1:vel=0,0,0:color=1,1,1 \
        --body mesh:mesh=render/assets/demo_ball.obj:mass=1.5:pos=2,0,1:vel=-3,0,0:color=1,0.8,0.3 \
        --out render/output/rigid_frames

Exports each frame multiple obj files: frame_XXXX_bodyN.obj
A separate JSON summary frames.json also saved (pose per body per frame).

This is a deliberately simplified implementation intended for extension.
"""
from __future__ import annotations
import numpy as np
import argparse
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
from tqdm import tqdm
import math

try:
    import trimesh  # type: ignore
except Exception:  # broad catch: if not installed or other import issue
    print('[WARNING] trimesh not installed, simulation might fail')
    trimesh = None  # type: ignore

# ------------------ Math helpers ------------------

def quat_identity():
    return np.array([0.0, 0.0, 0.0, 1.0])  # (x,y,z,w)

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ])

def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return quat_identity()
    axis = axis / n
    s = math.sin(angle/2)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, math.cos(angle/2)], dtype=float)

def quat_rotate(q, v):
    # Rotate vector v by quaternion q
    qv = np.array([v[0], v[1], v[2], 0.0])
    qi = np.array([-q[0], -q[1], -q[2], q[3]])
    return quat_mul(quat_mul(q, qv), qi)[:3]

def omega_to_quat_delta(omega, dt):
    # omega is angular velocity vector
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-12:
        return quat_identity()
    axis = omega / np.linalg.norm(omega)
    return quat_from_axis_angle(axis, angle)

def quat_to_matrix(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ])

# ------------------ Data structures ------------------

@dataclass
class RigidBody:
    shape: str  # 'box' | 'sphere' | 'mesh'
    mass: float
    inv_mass: float
    size: np.ndarray  # for box (lx,ly,lz), for sphere (radius, radius, radius)
    position: np.ndarray
    rotation: np.ndarray  # quaternion (x,y,z,w)
    lin_vel: np.ndarray
    ang_vel: np.ndarray
    inertia_local: np.ndarray  # 3x3
    inv_inertia_local: np.ndarray  # 3x3
    mesh: Optional[object] = None  # hold a trimesh.Trimesh; kept generic to avoid type checker issues
    color: Tuple[float,float,float] = (1.0,1.0,1.0)
    restitution: float = 0.6
    friction: float = 0.4

    def world_inv_inertia(self):
        R = quat_to_matrix(self.rotation)
        return R @ self.inv_inertia_local @ R.T

    def apply_impulse(self, impulse, contact_r):
        # impulse: vector applied at contact point; contact_r: vector from COM to contact point
        if self.inv_mass == 0.0:
            return
        self.lin_vel += impulse * self.inv_mass
        ang_imp = np.cross(contact_r, impulse)
        self.ang_vel += self.world_inv_inertia() @ ang_imp

    def integrate(self, dt, gravity):
        if self.inv_mass != 0.0:
            self.lin_vel += gravity * dt
        self.position += self.lin_vel * dt
        # orientation integration
        dq = omega_to_quat_delta(self.ang_vel, dt)
        self.rotation = quat_normalize(quat_mul(dq, self.rotation))

@dataclass
class Contact:
    a: int
    b: int  # -1 for ground
    point: np.ndarray
    normal: np.ndarray
    penetration: float
    r_a: np.ndarray
    r_b: np.ndarray

    accum_jn: float = 0.0   # normal impulse
    accum_jt: float = 0.0   # tangent impulse
    tangent: Optional[np.ndarray] = None


def warm_start(contacts, bodies):
    """Apply previous accumulated impulses to stabilize resting contacts."""
    for c in contacts:
        a = bodies[c.a]
        n = c.normal
        # normal impulse
        Jn = c.accum_jn * n
        Jt = np.zeros(3)
        if c.tangent is not None:
            Jt = c.accum_jt * c.tangent

        # apply to body A
        a.apply_impulse(-(Jn + Jt), c.r_a)

        # body B
        if c.b >= 0:
            b = bodies[c.b]
            b.apply_impulse((Jn + Jt), c.r_b)


# ------------------ Inertia computation ------------------

def box_inertia(mass, size):
    lx, ly, lz = size
    return np.diag([
        (1/12)*mass*(ly*ly + lz*lz),
        (1/12)*mass*(lx*lx + lz*lz),
        (1/12)*mass*(lx*lx + ly*ly),
    ])

def sphere_inertia(mass, radius):
    i = (2/5)*mass*radius*radius
    return np.diag([i, i, i])

# ------------------ Collision detection ------------------

def compute_aabb(body: RigidBody):
    # Prefer exact AABB from mesh if available
    if body.mesh is not None and hasattr(body.mesh, 'vertices'):
        verts_local = np.asarray(body.mesh.vertices)
        if verts_local.size == 0:
            c = body.position
            return c.copy(), c.copy()
        R = quat_to_matrix(body.rotation)
        verts_world = (R @ verts_local.T).T + body.position
        return verts_world.min(axis=0), verts_world.max(axis=0)
    if body.shape == 'sphere':
        r = body.size[0]
        c = body.position
        return c - r, c + r
    # approximate oriented box by axis-aligned using rotation matrix
    half = body.size * 0.5
    R = quat_to_matrix(body.rotation)
    # extents = sum of absolute columns * half
    absR = np.abs(R)
    ext = absR @ half
    c = body.position
    return c - ext, c + ext

GROUND_Z = 0.0
# 注意：如果你发现物体开始往下掉，把这个改成 np.array([0.0, 0.0, 1.0])
GROUND_NORMAL = np.array([0.0, 0.0, -1.0])


def add_ground_contacts_for_body(
    body_index: int,
    body: RigidBody,
    contacts: List[Contact],
    ground_z: float = GROUND_Z,
    n_ground: np.ndarray = GROUND_NORMAL,
    eps: float = 1e-4,
):
    """为单个刚体与地面 (z=ground_z) 生成多个接触点.

    思路：
    - 把刚体的“碰撞顶点”变到世界坐标；
    - 找出 z < ground_z 的点（穿透地面）；
    - 每个点投影到地面，作为一个 contact point。
    """
    if body.inv_mass == 0.0:
        # 静态物体（比如你以后把地面 mesh 真加到 bodies 里）就跳过
        return

    verts_w = get_collision_vertices_world(body)

    # 找到所有在地面下方的顶点
    below_mask = verts_w[:, 2] < ground_z - eps
    if not np.any(below_mask):
        return

    for v in verts_w[below_mask]:
        # 穿透量：地面高度 - 顶点高度
        penetration = ground_z - v[2]
        if penetration <= 0.0:
            continue

        # 接触点 = 顶点在地面上的投影
        contact_point = v.copy()
        contact_point[2] = ground_z

        r_a = contact_point - body.position

        contacts.append(
            Contact(
                a=body_index,
                b=-1,                       # -1 表示地面
                point=contact_point,
                normal=n_ground.copy(),     # 地面法线
                penetration=float(penetration),
                r_a=r_a,
                r_b=np.zeros(3),
            )
        )

def get_collision_vertices_world(body: RigidBody) -> np.ndarray:
    """返回刚体用于碰撞检测的顶点（世界坐标）.

    - box: 8 个角点
    - sphere: 用 6 个轴向采样点近似
    - mesh: 使用 mesh.vertices
    """
    R = quat_to_matrix(body.rotation)

    # Prefer mesh vertices if present (unify all shapes to mesh-mode)
    if body.mesh is not None and hasattr(body.mesh, 'vertices'):
        verts_local = np.asarray(body.mesh.vertices)
        if verts_local.size == 0:
            return np.zeros((0, 3), dtype=float)
        return (R @ verts_local.T).T + body.position

    if body.shape == 'box':
        half = body.size * 0.5
        # 8 个角点（局部坐标）
        corners_local = np.array([
            [+half[0], +half[1], +half[2]],
            [+half[0], +half[1], -half[2]],
            [+half[0], -half[1], +half[2]],
            [+half[0], -half[1], -half[2]],
            [-half[0], +half[1], +half[2]],
            [-half[0], +half[1], -half[2]],
            [-half[0], -half[1], +half[2]],
            [-half[0], -half[1], -half[2]],
        ])
        return (R @ corners_local.T).T + body.position

    elif body.shape == 'sphere':
        r = body.size[0]
        # 6 个轴向点, 让球也有多个接触点
        samples_local = np.array([
            [ r, 0, 0], [-r, 0, 0],
            [ 0, r, 0], [ 0,-r, 0],
            [ 0, 0, r], [ 0, 0,-r],
        ])
        return (R @ samples_local.T).T + body.position

    elif body.shape == 'mesh' and body.mesh is not None:
        # handled by early return above; keep for safety
        verts_local = np.asarray(body.mesh.vertices)
        return (R @ verts_local.T).T + body.position

    else:
        # fallback: 按 box 处理
        half = body.size * 0.5
        corners_local = np.array([
            [+half[0], +half[1], +half[2]],
            [+half[0], +half[1], -half[2]],
            [+half[0], -half[1], +half[2]],
            [+half[0], -half[1], -half[2]],
            [-half[0], +half[1], +half[2]],
            [-half[0], +half[1], -half[2]],
            [-half[0], -half[1], +half[2]],
            [-half[0], -half[1], -half[2]],
        ])
        return (R @ corners_local.T).T + body.position

def aabb_overlap(a_min, a_max, b_min, b_max):
    return np.all(a_min <= b_max) and np.all(b_min <= a_max)

def compute_aabb_overlap_volume(a_min, a_max, b_min, b_max):
    """
    Compute overlap volume of two AABBs.
    Returns:
        volume (float)
        overlap_min (np.ndarray)
        overlap_max (np.ndarray)
    """
    overlap_min = np.maximum(a_min, b_min)
    overlap_max = np.minimum(a_max, b_max)

    # if no overlap
    if np.any(overlap_min >= overlap_max):
        return 0.0, overlap_min, overlap_max

    size = overlap_max - overlap_min
    volume = size[0] * size[1] * size[2]
    return float(volume), overlap_min, overlap_max
def debug_overlap_analysis(bodyA, bodyB, label=""):
    """
    Print debug information about overlap between bodyA and bodyB.
    Uses AABB overlap volume for simplicity.
    """
    a_min, a_max = compute_aabb(bodyA)
    b_min, b_max = compute_aabb(bodyB)

    volume, o_min, o_max = compute_aabb_overlap_volume(a_min, a_max, b_min, b_max)

    if volume > 0:
        print(f"\n====== OVERLAP DEBUG {label} ======")
        print("A position:", bodyA.position)
        print("B position:", bodyB.position)
        print("Overlap volume:", volume)
        print("Overlap AABB min:", o_min)
        print("Overlap AABB max:", o_max)

        volA = np.prod(bodyA.size)
        volB = np.prod(bodyB.size)

        print(f"Overlap ratio A: {volume/volA:.4f}")
        print(f"Overlap ratio B: {volume/volB:.4f}")
        print("====================================\n")

# ------------------ Mesh distance helpers ------------------

def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # Algorithm from Real-Time Collision Detection (Christer Ericson)
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3 + 1e-12)
        return a + v * ab

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6 + 1e-12)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-12)
        return b + w * (c - b)

    # inside face region
    denom = (va + vb + vc + 1e-12)
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w


def _triangle_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        return np.array([0.0, 1.0, 0.0])
    return n / ln


def signed_distance_to_mesh_point(p_world: np.ndarray, body: RigidBody):
    """Compute signed distance from point p_world to body.mesh surface.

    Returns: (phi, n_out, closest_point)
    - phi < 0 means inside the mesh
    - n_out is the outward normal of body at the closest point
    - closest_point is the world-space closest point on the triangle
    """
    assert body.mesh is not None, "signed_distance_to_mesh_point requires body.mesh"
    verts_local = np.asarray(body.mesh.vertices)
    faces = np.asarray(body.mesh.faces, dtype=np.int32)
    R = quat_to_matrix(body.rotation)
    pos = body.position

    best_abs = 1e30
    best_phi = 0.0
    best_n = np.array([0.0, 1.0, 0.0])
    best_cp = p_world

    for tri in faces:
        a = pos + R @ verts_local[tri[0]]
        b = pos + R @ verts_local[tri[1]]
        c = pos + R @ verts_local[tri[2]]
        cp = _closest_point_on_triangle(p_world, a, b, c)
        n = _triangle_normal(a, b, c)
        phi = np.dot(p_world - cp, n)
        # orient sign by body center
        if np.dot(body.position - cp, n) > 0.0:
            phi = -phi
            n = -n
        ap = abs(phi)
        if ap < best_abs:
            best_abs = ap
            best_phi = phi
            best_n = n
            best_cp = cp

    return best_phi, best_n, best_cp

# Removed convex support, GJK, and EPA; mesh-only contact generation is used below.



def detect_contacts(bodies: List[RigidBody], args):
    contacts: List[Contact] = []
    n_bodies = len(bodies)

    # helper: subsample indices evenly
    def subsample(n: int, limit: int):
        if n <= 0:
            return np.zeros((0,), dtype=int)
        if limit <= 0 or n <= limit:
            return np.arange(n, dtype=int)
        return np.linspace(0, n - 1, num=limit, dtype=int)

    # ---------- 1. 刚体 vs 地面 (z = 0) ----------
    for i, body in enumerate(bodies):
        add_ground_contacts_for_body(i, body, contacts)

    # ---------- 2. 刚体 vs 刚体：broad-phase(AABB) + mesh vertex-to-mesh contacts ----------
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            A = bodies[i]
            B = bodies[j]

            # 两个都是静态就跳过
            if A.inv_mass == 0.0 and B.inv_mass == 0.0:
                continue

            # AABB 粗检测（基于 mesh 的精确/近似 AABB）
            a_min, a_max = compute_aabb(A)
            b_min, b_max = compute_aabb(B)
            if not aabb_overlap(a_min, a_max, b_min, b_max):
                continue

            margin = getattr(args, 'contact_margin', 1e-4)
            vlimit = int(getattr(args, 'mesh_vertex_limit', 800))

            # 顶点 from A 测 B
            vertsA = get_collision_vertices_world(A)
            idxA = subsample(len(vertsA), vlimit)
            for idx in idxA:
                p = vertsA[idx]
                if B.mesh is None:
                    continue
                phi, n_out, cp = signed_distance_to_mesh_point(p, B)
                if phi < -margin:  # p inside B
                    normal = -n_out  # A->B方向：反向于B的外法线
                    penetration = -phi
                    contact_point = cp
                    r_a = contact_point - A.position
                    r_b = contact_point - B.position
                    contacts.append(Contact(a=i, b=j, point=contact_point, normal=normal,
                                            penetration=float(penetration), r_a=r_a, r_b=r_b))

            # 顶点 from B 测 A（对称）
            vertsB = get_collision_vertices_world(B)
            idxB = subsample(len(vertsB), vlimit)
            for idx in idxB:
                p = vertsB[idx]
                if A.mesh is None:
                    continue
                phi, n_out, cp = signed_distance_to_mesh_point(p, A)
                if phi < -margin:
                    normal = n_out  # A->B方向：此时点来自B，法线取A外法线即从A指向外 -> 即 A->B
                    penetration = -phi
                    contact_point = cp
                    r_a = contact_point - A.position
                    r_b = contact_point - B.position
                    contacts.append(Contact(a=i, b=j, point=contact_point, normal=normal,
                                            penetration=float(penetration), r_a=r_a, r_b=r_b))

    return contacts


# ------------------ Impulse solver ------------------

def resolve_contacts(bodies, contacts, dt):
    """Sequential Impulse Solver (Box2D-style)."""
    for c in contacts:
        a = bodies[c.a]
        n = c.normal / (np.linalg.norm(c.normal) + 1e-9)
        b = bodies[c.b] if c.b >= 0 else None

        # relative velocity
        v_a = a.lin_vel + np.cross(a.ang_vel, c.r_a)
        if b:
            v_b = b.lin_vel + np.cross(b.ang_vel, c.r_b)
            v_rel = v_b - v_a
        else:
            v_rel = -v_a
        vn = np.dot(v_rel, n)

        # ---------- Normal impulse ----------
        # Baumgarte bias (penetration correction)
        beta = 0.05
        bias = beta * max(c.penetration - 1e-3, 0.0) / dt

        # Effective mass
        denom = a.inv_mass
        denom += n @ (np.cross(a.world_inv_inertia() @ np.cross(c.r_a, n), c.r_a))
        if b:
            denom += b.inv_mass
            denom += n @ (np.cross(b.world_inv_inertia() @ np.cross(c.r_b, n), c.r_b))
        if denom < 1e-9: continue

        e = a.restitution if not b else min(a.restitution, b.restitution)

        jn_raw = -(1 + e) * vn + bias
        jn_raw /= denom

        # ⭐ clamp + accumulate (warm starting)
        jn_old = c.accum_jn
        c.accum_jn = max(0.0, jn_old + jn_raw)
        jn = c.accum_jn - jn_old

        # apply impulse
        impulse_n = jn * n
        a.apply_impulse(-impulse_n, c.r_a)
        if b:
            b.apply_impulse(impulse_n, c.r_b)

        # ---------- Tangential friction ----------
        v_a2 = a.lin_vel + np.cross(a.ang_vel, c.r_a)
        if b:
            v_b2 = b.lin_vel + np.cross(b.ang_vel, c.r_b)
            v_rel2 = v_b2 - v_a2
        else:
            v_rel2 = -v_a2

        vt = v_rel2 - np.dot(v_rel2, n) * n
        vt_len = np.linalg.norm(vt)
        if vt_len > 1e-9:
            t = vt / vt_len
            c.tangent = t

            denom_t = a.inv_mass
            denom_t += t @ (np.cross(a.world_inv_inertia() @ np.cross(c.r_a, t), c.r_a))
            if b:
                denom_t += b.inv_mass
                denom_t += t @ (np.cross(b.world_inv_inertia() @ np.cross(c.r_b, t), c.r_b))

            jt_raw = -vt_len / denom_t

            # Coulomb friction cone
            mu = a.friction if not b else math.sqrt(a.friction * b.friction)
            jt_limit = mu * c.accum_jn

            jt_old = c.accum_jt
            c.accum_jt = np.clip(jt_old + jt_raw, -jt_limit, jt_limit)
            jt = c.accum_jt - jt_old

            impulse_t = jt * t
            a.apply_impulse(-impulse_t, c.r_a)
            if b:
                b.apply_impulse(impulse_t, c.r_b)


# ------------------ Position Correction ------------------

def position_correction(bodies, contacts):
    k = 0.5     # position correction strength
    slop = 0.01 # penetration allowance

    for c in contacts:
        if c.penetration <= slop:
            continue
        n = c.normal / (np.linalg.norm(c.normal) + 1e-9)
        correction = k * (c.penetration - slop) * n

        a = bodies[c.a]
        if c.b >= 0:
            b = bodies[c.b]
            total_inv = a.inv_mass + b.inv_mass
            if total_inv > 0:
                a.position -= correction * (a.inv_mass / total_inv)
                b.position += correction * (b.inv_mass / total_inv)
        else:
            # ground
            a.position -= correction


# ------------------ Export ------------------

def export_frame(bodies: List[RigidBody], frame_idx: int, out_dir: str):
    # print(f'[Export Frame] exporting frame {frame_idx} to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    for i, b in enumerate(bodies):
        if b.mesh is None:
            # mesh-only: skip if mesh missing
            continue
        geom = b.mesh.copy()
        # print(geom)
        R = quat_to_matrix(b.rotation)
        geom.apply_transform(np.vstack([np.hstack([R, b.position.reshape(3,1)]), [0,0,0,1]]))

        path = os.path.join(out_dir, f"frame_{frame_idx:04d}_body{i}.obj")
        geom.export(path)


# ------------------ Body creation ------------------

def make_body(desc: str) -> RigidBody:
    # desc format examples:
    # box:mass=1.0:size=1,1,1:pos=0,1,0:vel=1,0,0:color=0.2,0.6,0.9
    # sphere:mass=2.0:radius=0.5:pos=0,3,0:vel=0,-1,0
    # mesh:mesh=render/assets/demo_cube.obj:mass=3.0:pos=0,1,0:vel=0,0,0
    parts = desc.split(':')
    shape = parts[0]
    kv = {}
    for p in parts[1:]:
        if '=' in p:
            k, v = p.split('=',1)
            kv[k] = v
    mass = float(kv.get('mass', '1.0'))
    inv_mass = 0.0 if mass <= 0 else 1.0/mass
    mesh = None
    if shape == 'box':
        size = np.array([float(x) for x in kv.get('size','1,1,1').split(',')])
        inertia = box_inertia(mass, size)
        if trimesh is None:
            raise RuntimeError('trimesh is required for mesh mode')
        mesh = trimesh.creation.box(extents=size)
    elif shape == 'sphere':
        r = float(kv.get('radius','0.5'))
        size = np.array([r,r,r])
        inertia = sphere_inertia(mass, r)
        if trimesh is None:
            raise RuntimeError('trimesh is required for mesh mode')
        subdivs = int(kv.get('subdiv','2'))
        mesh = trimesh.creation.icosphere(subdivisions=max(1, subdivs), radius=r)
    elif shape == 'mesh':
        if trimesh is None:
            raise RuntimeError('trimesh not installed for mesh import')
        mesh_path = kv.get('mesh')
        if not mesh_path or not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh path invalid: {mesh_path}")
        mesh = trimesh.load(mesh_path, force='mesh')
        # approximate size using bounding box extents
        size = mesh.extents
        # approximate inertia using box inertia
        inertia = box_inertia(mass, size)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    inv_inertia = np.linalg.inv(inertia) if mass > 0 else np.zeros((3,3))
    pos = np.array([float(x) for x in kv.get('pos','0,0,0').split(',')])
    vel = np.array([float(x) for x in kv.get('vel','0,0,0').split(',')])
    color = tuple(float(x) for x in kv.get('color','1,1,1').split(','))[:3]
    body = RigidBody(
        shape=shape,
        mass=mass,
        inv_mass=inv_mass,
        size=size,
        position=pos,
        rotation=quat_identity(),
        lin_vel=vel,
        ang_vel=np.zeros(3),
        inertia_local=inertia,
        inv_inertia_local=inv_inertia,
        mesh= mesh,
        color=color,
    )
    return body

# ------------------ Main simulation loop ------------------

def run_sim(args):
    print("[DEBUG] args.body =", args.body)
    bodies = [make_body(b) for b in args.body]
    print("[DEBUG] len(bodies) =", len(bodies))
    for b in bodies:
        b.restitution = args.restitution
        b.friction    = args.friction
    print('[Resitution] set to', args.restitution)
    print('[Friction] set to', args.friction)
    frames_output = []
    # ---------- Add a ground mesh for rendering ----------
    if trimesh is not None:
        ground_mesh = trimesh.creation.box(extents=[50, 50, 0.1])
        ground_body = RigidBody(
            shape='mesh',
            mass=0.0,
            inv_mass=0.0,
            size=np.array([50, 50, 0.1]),
            position=np.array([0, 0, -0.05]),   # so top face is exactly at z=0
            rotation=quat_identity(),
            lin_vel=np.zeros(3),
            ang_vel=np.zeros(3),
            inertia_local=np.zeros((3,3)),
            inv_inertia_local=np.zeros((3,3)),
            mesh=ground_mesh,
            color=(0.6, 0.6, 0.6)
        )
        # bodies.append(ground_body)

    start_time = time.time()
    for frame in tqdm(range(args.frames), desc="Simulating", unit="frame"):

        # ---- 1 integrate: move bodies ----
        for b in bodies:
            if b.inv_mass != 0:   # ground mass = 0
                b.integrate(args.dt, args.gravity)

        # ---- 2 detect (mesh-based) ----
        contacts = detect_contacts(bodies, args)

        # ---- 3 warm start ----
        warm_start(contacts, bodies)

        # ---- 4 resolve impulses ----
        for _ in range(10):
            resolve_contacts(bodies, contacts, args.dt)

        # ---- 5 positional correction ----
        position_correction(bodies, contacts)

        # print(len(bodies))
        if len(bodies) >= 2:
            debug_overlap_analysis(bodies[0], bodies[1], label=f"frame {frame}")


        # --------- THIS IS WHERE OBJ EXPORT HAPPENS ---------
        export_frame(bodies, frame, args.out)

        # -------- record json --------
        frames_output.append({
            "frame": frame,
            "bodies": [
                {
                    "pos": b.position.tolist(),
                    "rot": b.rotation.tolist(),
                    "size": b.size.tolist(),
                    "color": list(b.color),
                } for b in bodies
            ]
        })

        # periodic timing info every 50 frames
        if frame % 50 == 0:
            elapsed = time.time() - start_time
            avg = elapsed / (frame + 1) if frame >= 0 else 0.0
            eta = avg * (args.frames - frame - 1)
            print(f"[frame {frame:04d}] elapsed={elapsed:.2f}s avg/frame={avg:.3f}s ETA={eta:.2f}s")

    # -------- write summary --------
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "frames.json"), "w", encoding="utf-8") as f:
        json.dump({"dt": args.dt, "frames": frames_output}, f, indent=2)
    total = time.time() - start_time
    print('[RUN SIM] End')
    print(f"[Timing] frames={args.frames} total={total:.2f}s avg/frame={total/args.frames:.3f}s")


def parse_cli():
    ap = argparse.ArgumentParser(description='Impulse-based rigid body simulation prototype (mesh contacts)')
    ap.add_argument('--dt', type=float, default=1/60)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--gravity', type=str, default='0,0,-9.8')
    ap.add_argument('--restitution', type=float, default=0.9)
    ap.add_argument('--friction', type=float, default=0.2)
    ap.add_argument('--contact_margin', type=float, default=1e-4, help='penetration margin for mesh contacts')
    ap.add_argument('--mesh_vertex_limit', type=int, default=800, help='max sampled vertices per body when generating contacts')
    ap.add_argument('--body', action='append', default=[], help='Body descriptor string (see header examples). Can repeat.')
    ap.add_argument('--out', type=str, default='render/output/rigid_frames')
    args = ap.parse_args()
    g = np.array([float(x) for x in args.gravity.split(',')])
    args.gravity = g
    if not args.body:
        # default two boxes colliding
        args.body = [
            'box:mass=2.0:size=1,1,1:pos=-2,0,1:vel=2,0,0:color=0.2,0.6,0.9',
            'box:mass=2.0:size=1,1,1:pos=2,0,1:vel=-2,0,0:color=0.9,0.4,0.2'
        ]
    return args

if __name__ == '__main__':
    run_sim(parse_cli())
