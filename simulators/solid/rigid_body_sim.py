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
import math

try:
    import trimesh  # type: ignore
except Exception:  # broad catch: if not installed or other import issue
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
    normal: np.ndarray  # pointing from a to b
    penetration: float
    r_a: np.ndarray  # contact point relative to COM of a
    r_b: np.ndarray  # relative to COM of b (if b>=0)

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

def aabb_overlap(a_min, a_max, b_min, b_max):
    return np.all(a_min <= b_max) and np.all(b_min <= a_max)

def detect_contacts(bodies: List[RigidBody]):
    contacts: List[Contact] = []
    n = len(bodies)
    # ground plane z=0 (Z-up)
    for i, body in enumerate(bodies):
        # approximate bottom point
        a_min, a_max = compute_aabb(body)
        if a_min[2] < 0.0:
            penetration = -a_min[2]
            point = body.position + np.array([0.0, 0.0, -a_min[2]])
            normal = np.array([0.0, 0.0, 1.0])
            r_a = point - body.position
            contacts.append(Contact(i, -1, point, normal, penetration, r_a, np.zeros(3)))
    # pairwise
    for i in range(n):
        for j in range(i+1, n):
            a_min, a_max = compute_aabb(bodies[i])
            b_min, b_max = compute_aabb(bodies[j])
            if not aabb_overlap(a_min, a_max, b_min, b_max):
                continue
            # compute penetration depth & normal (simple): choose axis of minimum overlap
            overlap_axes = []
            for axis in range(3):
                o1 = a_max[axis] - b_min[axis]
                o2 = b_max[axis] - a_min[axis]
                if o1 <= 0 or o2 <= 0:
                    break
                ov = min(o1, o2)
                # normal direction
                if o1 < o2:
                    dir_sign = -1  # a towards -axis
                else:
                    dir_sign = 1
                axis_vec = np.zeros(3)
                axis_vec[axis] = dir_sign
                overlap_axes.append((ov, axis_vec))
            if not overlap_axes:
                continue
            # pick smallest
            overlap_axes.sort(key=lambda x: x[0])
            penetration, nvec = overlap_axes[0]
            # contact point approximate midpoint of centers projected
            pa = bodies[i].position
            pb = bodies[j].position
            point = (pa + pb)/2.0
            r_a = point - pa
            r_b = point - pb
            contacts.append(Contact(i, j, point, nvec, penetration, r_a, r_b))
    return contacts

# ------------------ Impulse solver ------------------

def resolve_contacts(bodies: List[RigidBody], contacts: List[Contact], dt: float, baumgarte=0.1):
    for c in contacts:
        A = bodies[c.a]
        B = bodies[c.b] if c.b >= 0 else None
        n = c.normal / (np.linalg.norm(c.normal) + 1e-9)
        # relative velocity at contact
        v_a = A.lin_vel + np.cross(A.ang_vel, c.r_a)
        if B:
            v_b = B.lin_vel + np.cross(B.ang_vel, c.r_b)
            v_rel = v_a - v_b
        else:
            v_rel = v_a  # ground treated as static
        vn = np.dot(v_rel, n)
        # positional correction (Baumgarte)
        bias = -(baumgarte/dt)*max(c.penetration - 1e-3, 0.0)
        # effective mass denominator
        inv_mass_term = A.inv_mass
        angular_term = n @ (np.cross(A.world_inv_inertia() @ np.cross(c.r_a, n), c.r_a))
        if B:
            inv_mass_term += B.inv_mass
            angular_term += n @ (np.cross(B.world_inv_inertia() @ np.cross(c.r_b, n), c.r_b))
        denom = inv_mass_term + angular_term
        if denom < 1e-9:
            continue
        e = min(A.restitution, B.restitution) if B else A.restitution
        jn = -(1 + e)*vn + bias
        jn /= denom
        if jn < 0:
            jn = 0  # do not pull
        impulse_n = jn * n
        A.apply_impulse( impulse_n, c.r_a)
        if B:
            B.apply_impulse(-impulse_n, c.r_b)
        # friction
        v_rel_post = v_a - (B.lin_vel + np.cross(B.ang_vel, c.r_b)) if B else v_a
        vt = v_rel_post - np.dot(v_rel_post, n)*n
        vt_len = np.linalg.norm(vt)
        if vt_len > 1e-9:
            t_dir = vt / vt_len
            # denom for tangential impulse
            angular_t = t_dir @ (np.cross(A.world_inv_inertia() @ np.cross(c.r_a, t_dir), c.r_a))
            if B:
                angular_t += t_dir @ (np.cross(B.world_inv_inertia() @ np.cross(c.r_b, t_dir), c.r_b))
            denom_t = inv_mass_term + angular_t
            jt = -vt_len / (denom_t + 1e-9)
            mu = math.sqrt(A.friction * (B.friction if B else A.friction))
            jt = np.clip(jt, -mu*jn, mu*jn)
            impulse_t = jt * t_dir
            A.apply_impulse( impulse_t, c.r_a)
            if B:
                B.apply_impulse(-impulse_t, c.r_b)

# ------------------ Export ------------------

def export_frame(bodies: List[RigidBody], frame_idx: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, b in enumerate(bodies):
        if b.mesh is None:
            # create a primitive on the fly (box or sphere) using trimesh
            if trimesh is None:
                continue
            if b.shape == 'box':
                geom = trimesh.creation.box(extents=b.size)
            elif b.shape == 'sphere':
                geom = trimesh.creation.icosphere(subdivisions=2, radius=b.size[0])
            else:
                continue
        else:
            geom = b.mesh.copy()
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
    if shape == 'box':
        size = np.array([float(x) for x in kv.get('size','1,1,1').split(',')])
        inertia = box_inertia(mass, size)
    elif shape == 'sphere':
        r = float(kv.get('radius','0.5'))
        size = np.array([r,r,r])
        inertia = sphere_inertia(mass, r)
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
        mesh= mesh if shape=='mesh' else None,
        color=color,
    )
    return body

# ------------------ Main simulation loop ------------------

def run_sim(args):
    bodies = [make_body(b) for b in args.body]
    frames_output = []
    for frame in range(args.frames):
        # broadphase & narrowphase
        contacts = detect_contacts(bodies)
        # resolve impulses
        resolve_contacts(bodies, contacts, args.dt)
        # integrate
        for b in bodies:
            b.integrate(args.dt, args.gravity)
        # export
        export_frame(bodies, frame, args.out)
        # record summary (position, rotation quaternion)
        frames_output.append({
            'frame': frame,
            'bodies': [
                {
                    'pos': b.position.tolist(),
                    'rot': b.rotation.tolist(),
                    'size': b.size.tolist(),
                    'color': list(b.color),
                } for b in bodies
            ]
        })
    # write JSON summary
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'frames.json'), 'w', encoding='utf-8') as f:
        json.dump({'dt': args.dt, 'frames': frames_output}, f, indent=2)
    print(f"[SIM] Finished {args.frames} frames -> {args.out}")


def parse_cli():
    ap = argparse.ArgumentParser(description='Impulse-based rigid body simulation prototype')
    ap.add_argument('--dt', type=float, default=1/60)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--gravity', type=str, default='0,0,-9.8')
    ap.add_argument('--restitution', type=float, default=0.6)
    ap.add_argument('--friction', type=float, default=0.4)
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
