import taichi as ti
import numpy as np
import trimesh
import time
ti.init(arch=ti.cuda)
def Ball(radius = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.icosphere(radius = radius, center = center, subdivisions = 2)

def Box(extents = [1,1,1], center = [0,0,0]):
    return trimesh.creation.box(extents = extents, transform=trimesh.transformations.translation_matrix(center))

def Cylinder(radius = 1, height = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.cylinder(radius = radius, height = height, center = center, resolution = resolution)

def Plane(size = [1,1], center = [0,0,0]):
    return trimesh.creation.box(extents = [size[0], 0.001, size[1]], center = center)

def Sphere(radius = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.uv_sphere(radius = radius, center = center, resolution = resolution)

def Cone(radius = 1, height = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.cone(radius = radius, height = height, center = center, resolution = resolution)

def Torus(inner_radius = 0.5, outer_radius = 1, center = [0,0,0], resolution = 100):
    return trimesh.creation.torus(inner_radius = inner_radius, outer_radius = outer_radius, center = center, resolution = resolution)
@ti.data_oriented
class RigidBody:
    def __init__(self, type=None, mesh=None, mass=1.0, 
                radius=1.0, height=1.0, center=[0.0, 0.0, 0.0],size=[1.0, 1.0],
                inner_radius=0.5, outer_radius=1.0, resolution=100,
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
                orientation=np.eye(3), 
                velocity=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
                angular_velocity=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                collision_threshold=np.finfo(np.float32).tiny,
                fixed=False):

        if type == 'Ball':
            mesh = Ball(radius, center, resolution)
        elif type == 'Box':
            mesh = Box(size, center)
        elif type == 'Cylinder':
            mesh = Cylinder(radius, height, center, resolution)
        elif type == 'Plane':
            mesh = Plane(size, center)
        elif type == 'Sphere':
            mesh = Sphere(radius, center, resolution)
        elif type == 'Cone':
            mesh = Cone(radius, height, center, resolution)
        elif type == 'Torus':
            mesh = Torus(inner_radius, outer_radius, center, resolution)
        elif mesh is None:
            raise ValueError("Please provide a mesh or a type of geometry")
        
        self.mesh = mesh
        mesh.vertices = mesh.vertices.astype(np.float32)
        mesh.faces = mesh.faces.astype(np.int32)
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(mesh.vertices))
        self.vertices.from_numpy(mesh.vertices)
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=len(mesh.faces))
        self.faces.from_numpy(mesh.faces)

        self.mass, self.volume = mass, ti.field(ti.f32, shape=())
        self.mass_center_offset = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.centralize() # centralize the mesh
        self.inertia_tensor = self.inertia() # inertia tensor relative to the center of mass with respect to the canonical frame
        self.mesh = trimesh.Trimesh(vertices=self.vertices.to_numpy(), faces=self.faces.to_numpy())
        self.voxel = None
        self.num_particles = 0
        self.get_voxel()
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=()) # position of the center of mass
        self.position[None] = position
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=()) # velocity of the center of mass
        self.velocity[None] = velocity
        self.orientation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=()) # orientation matrix of the body
        # cast to f32 to avoid f64->f32 warning
        self.orientation[None] = orientation.astype(np.float32)
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=()) # angular velocity of the body
        self.angular_velocity[None] = angular_velocity
        self.collision_threshold = collision_threshold
        
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.torque = ti.Vector.field(3, dtype=ti.f32, shape=()) # torque relative to the center of mass
        self.angular_momentum = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fixed = fixed
        # self.eular_angles = ti.Vector.field(3, dtype=ti.f32, shape=())
        
    @ti.func
    def mass_center(self) -> ti.types.vector(3, ti.f32):
        mesh_volume = ti.float32(0.0)
        temp = ti.Vector([0.0, 0.0, 0.0])
        
        for i in range(self.faces.shape[0]):
            # print(self.faces[i][0])
            center = 0.25 * (self.vertices[self.faces[i][0]] + self.vertices[self.faces[i][1]] + self.vertices[self.faces[i][2]])
            volume = ti.math.dot(self.vertices[self.faces[i][0]], ti.math.cross(self.vertices[self.faces[i][1]], self.vertices[self.faces[i][2]])) / 6
            mesh_volume += volume
            temp += center * volume
        
        self.volume[None] = ti.abs(mesh_volume)
        return temp / mesh_volume
    
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
    def inertia(self) -> ti.types.matrix(3, 3, ti.f32):
        covarience_tensor = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        canoical_inertia_tensor = ti.Matrix([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]) / 60
        
        for i in range(self.faces.shape[0]):
            # ti.static_print(self.faces[i, 0])
            transform = ti.Matrix.cols([self.vertices[self.faces[i][0]], self.vertices[self.faces[i][1]], self.vertices[self.faces[i][2]]])
            covarience_tensor += transform.determinant() * transform @ canoical_inertia_tensor @ transform.transpose()
        covarience_tensor *= self.mass / self.volume[None]
        inertia_tensor = ti.Matrix.identity(ti.f32, 3) * ti.Matrix.trace(covarience_tensor) - covarience_tensor
        return inertia_tensor

    @ti.func
    def apply_external_force(self, force: ti.types.vector(3, ti.f32), point: ti.types.vector(3, ti.f32)):
        self.force[None] += force
        self.torque[None] += ti.math.cross(point - self.position[None], force)

    @ti.func
    def apply_internal_force(self, force: ti.types.vector(3, ti.f32), point: ti.types.vector(3, ti.f32)):
        self.force[None] += force
        self.torque[None] += ti.math.cross(point - self.position[None], force)

    @ti.func
    def check_collision(self, point: ti.types.vector(3, ti.f32)) -> ti.types.vector(2, ti.f32):
        min_distance = float('inf')  # Used to record the minimum collision distance
        closest_normal = ti.Vector([0.0, 0.0, 0.0])  # Used to record the closest normal

        for i in range(self.faces.shape[0]):
            # Get the three vertices of a triangle
            v0 = self.position[None] + self.orientation[None] @ self.vertices[self.faces[i][0]]
            v1 = self.position[None] + self.orientation[None] @ self.vertices[self.faces[i][1]]
            v2 = self.position[None] + self.orientation[None] @ self.vertices[self.faces[i][2]]

            # Compute the normal vector of the triangle
            normal = (v1 - v0).cross(v2 - v0).normalized()

            # Compute the distance of the point to the face
            distance = abs((point - v0).dot(normal))

            # If the distance is less than the threshold, a collision is considered to have occurred
            if distance < self.collision_threshold:
                if distance < min_distance:
                    min_distance = distance
                    closest_normal = normal

        collision = 0
        normal = ti.Vector([0.0, 0.0, 0.0])
        
        if min_distance < float('inf'):
            collision = 1
            normal = closest_normal

        return collision, normal
        
    @ti.func
    def get_velocity_at_point(self, point: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
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

            acceleration = self.force[None] / self.mass
            v_new = self.velocity[None] + acceleration * dt
            # clamp linear speed
            v_len = v_new.norm()
            if v_len > max_speed:
                v_new = v_new * (ti.cast(max_speed, ti.f32) / (v_len + 1e-8))
            self.velocity[None] = v_new
            self.position[None] += self.velocity[None] * dt

            # Angular motion
            # angular_acceleration = self.torque[None] / self.mass  # Simplified, should use inertia tensor
            angular_acceleration = self.compute_angular_acceleration()
            w_new = self.angular_velocity[None] + angular_acceleration * dt
            # clamp angular speed
            w_len = w_new.norm()
            if w_len > max_omega:
                w_new = w_new * (ti.cast(max_omega, ti.f32) / (w_len + 1e-8))
            self.angular_velocity[None] = w_new
            angular_velocity_norm = self.angular_velocity[None].norm()
            exp_A = ti.Matrix.identity(ti.f32, 3)
            # print(angular_velocity_norm)
            if angular_velocity_norm > 1e-8:
                angular_velocity_matrix = ti.Matrix([
                    [0, -self.angular_velocity[None][2], self.angular_velocity[None][1]],
                    [self.angular_velocity[None][2], 0, -self.angular_velocity[None][0]],
                    [-self.angular_velocity[None][1], self.angular_velocity[None][0], 0]
                ]) / angular_velocity_norm

                exp_A = ti.Matrix.identity(ti.f32, 3) + angular_velocity_matrix * ti.sin(angular_velocity_norm * dt) + angular_velocity_matrix @ angular_velocity_matrix * (1 - ti.cos(angular_velocity_norm * dt))
            # print(exp_A)
            self.orientation[None] = exp_A @ self.orientation[None]

            # Reset forces and torques
            self.force[None] = ti.Vector([0.0, 0.0, 0.0])
            self.torque[None] = ti.Vector([0.0, 0.0, 0.0])
        
    @ti.func
    def compute_angular_acceleration(self) -> ti.types.vector(3, ti.f32):
        inertia_tensor_now = self.orientation[None] @ self.inertia_tensor @ self.orientation[None].transpose() # inertia tensor relative to the center of mass with respect to the current frame
        self.angular_momentum[None] = inertia_tensor_now @ self.angular_velocity[None]
        torque = self.torque[None] - ti.math.cross(self.angular_velocity[None], self.angular_momentum[None])
        return inertia_tensor_now.inverse() @ torque


# =============================
# Simulation helpers / kernels
# =============================

@ti.kernel
def apply_gravity(rb: ti.template(), mass: ti.f32, gx: ti.f32, gy: ti.f32, gz: ti.f32):
    # apply gravity at center of mass
    rb.apply_external_force(ti.Vector([gx, gy, gz]) * mass, rb.position[None])


@ti.kernel
def mesh_plane_penalty(rb: ti.template(), nx: ti.f32, ny: ti.f32, nz: ti.f32, h: ti.f32,
                       stiffness: ti.f32, damping: ti.f32, margin: ti.f32,
                       mu_t: ti.f32, c_t: ti.f32, max_fn: ti.f32):
    # plane: n·x = h, with unit normal n
    n = ti.Vector([nx, ny, nz])
    for i in range(rb.faces.shape[0]):
        # iterate triangle vertices for contact sampling
        for k in ti.static(range(3)):
            vidx = rb.faces[i][k]
            x = rb.position[None] + rb.orientation[None] @ rb.vertices[vidx]
            phi = x.dot(n) - h
            if phi < 0 and ti.abs(phi) > margin:
                cp = x - phi * n
                v = rb.get_velocity_at_point(cp)
                vn = v.dot(n)
                vt = v - vn * n
                # normal spring-damper with soft cap
                f_n = (-stiffness * phi - damping * vn)
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
                rb.apply_internal_force(F, cp)


def export_obj(rb: RigidBody, path: str):
    verts = rb.vertices.to_numpy()
    faces = rb.faces.to_numpy()
    pos = rb.position.to_numpy()
    R = rb.orientation.to_numpy()
    verts_w = (R @ verts.T).T + pos
    with open(path, 'w') as f:
        for v in verts_w:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")


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
                            denom = (va + vb + vc + 1e-8)
                            v = vb / denom
                            w = vc / denom
                            result = a + ab * v + ac * w
                else:
                    result = c
        else:
            result = b
    return result

@ti.func
def signed_distance_to_body(rb: ti.template(), p):
    min_abs = 1e30
    best_phi = 0.0
    best_n = ti.Vector([0.0, 1.0, 0.0])
    center = rb.position[None]
    for f in range(rb.faces.shape[0]):
        idx = rb.faces[f]
        a = center + rb.orientation[None] @ rb.vertices[idx[0]]
        b = center + rb.orientation[None] @ rb.vertices[idx[1]]
        c = center + rb.orientation[None] @ rb.vertices[idx[2]]
        cp = closest_point_on_triangle(p, a, b, c)
        n = tri_normal(a, b, c)
        phi = (p - cp).dot(n)
        # flip to make n point outward from rb
        if (center - cp).dot(n) > 0:
            phi = -phi
            n = -n
        ap = ti.abs(phi)
        if ap < min_abs:
            min_abs = ap
            best_phi = phi
            best_n = n
    return best_phi, best_n

@ti.kernel
def rigid_rigid_penalty(A: ti.template(), B: ti.template(), stiffness: ti.f32, damping: ti.f32, margin: ti.f32):
    # For each vertex of A, compute signed distance to mesh B; if penetrating, apply penalty+damping
    for i in range(A.vertices.shape[0]):
        xA = A.position[None] + A.orientation[None] @ A.vertices[i]
        phi, n = signed_distance_to_body(B, xA)
        if phi < 0 and ti.abs(phi) > margin:
            cp = xA - phi * n
            vA = A.get_velocity_at_point(cp)
            vB = B.get_velocity_at_point(cp)
            vn = (vA - vB).dot(n)
            F = (-stiffness * phi - damping * vn) * n
            A.apply_internal_force(F, cp)
            B.apply_internal_force(-F, cp)

@ti.kernel
def sphere_sphere_penalty(A: ti.template(), B: ti.template(), radiusA: ti.f32, radiusB: ti.f32,
                          stiffness: ti.f32, damping: ti.f32, margin: ti.f32):
    # Analytic contact for two spheres using centers and radii
    cA = A.position[None]
    cB = B.position[None]
    d = cA - cB
    dist = d.norm()
    # penetration depth: (rA + rB + margin) - dist
    total_r = radiusA + radiusB
    if dist < total_r + margin:
        # normal from B to A (avoid zero)
        n = ti.Vector([0.0, 1.0, 0.0])
        if dist > 1e-8:
            n = d / dist
        # contact point approx at mid along line
        cp = (cA + cB) * 0.5
        # relative normal velocity
        vA = A.get_velocity_at_point(cp)
        vB = B.get_velocity_at_point(cp)
        vn = (vA - vB).dot(n)
        # penetration amount
        phi = (total_r - dist)
        # spring + damping along normal
        F = (stiffness * phi - damping * vn) * n
        A.apply_internal_force(F, cp)
        B.apply_internal_force(-F, cp)


def apply_sphere_sphere_impulse(A, B, radiusA, radiusB, restitution=0.0, max_impulse=None):
    """Apply a simple linear collision impulse to control post-collision normal velocity.
    This ignores rotational impulse coupling (approximate), but guarantees that the
    post-impact normal relative velocity matches -restitution * pre-impact vn.
    """
    # positions
    cA = A.position.to_numpy()
    cB = B.position.to_numpy()
    d = cA - cB
    dist = np.linalg.norm(d)
    if dist <= 1e-8:
        return
    total_r = float(radiusA + radiusB)
    # only apply when in contact (allow small margin)
    if dist > total_r + 1e-6:
        return
    n = d / dist
    vA = A.velocity.to_numpy()
    vB = B.velocity.to_numpy()
    vn = float(np.dot(vA - vB, n))
    # if separating or resting, nothing to do
    if vn >= 0.0:
        return
    mA = float(A.mass)
    mB = float(B.mass)
    e = float(restitution)
    # scalar impulse magnitude (positive value)
    J = -(1.0 + e) * vn / (1.0 / mA + 1.0 / mB)
    if max_impulse is not None:
        J = np.sign(J) * min(abs(J), float(max_impulse))
    # apply linear impulse (neglect rotational coupling)
    vA_new = vA + (J * n) / mA
    vB_new = vB - (J * n) / mB
    A.velocity[None] = vA_new
    B.velocity[None] = vB_new

def run_sphere_drop(frames=240, dt=1/240.0, out_dir='render/output/rigid_ti_frames',
                    gravity=(0.0, -9.8, 0.0), stiffness=5e4, damping=1e4, margin=1e-3,
                    radius=0.5, mass=1.0, start_pos=(0.0, 2.0, 0.0), start_vel=(0.0, 0.0, 0.0), ground_y=0.0,
                    max_speed=25.0, max_omega=35.0, restitution=0.0, tangential_damping=0.0,
                    mu_t=0.2, c_t=5000.0):
    rb = RigidBody(type='Ball', mass=mass, radius=radius,
                   position=np.array(start_pos, dtype=np.float32),
                   velocity=np.array(start_vel, dtype=np.float32))

    gx, gy, gz = gravity
    import os
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for frame in range(frames):
        apply_gravity(rb, float(mass), float(gx), float(gy), float(gz))
        mesh_plane_penalty(rb, 0.0, 1.0, 0.0, float(ground_y), float(stiffness), float(damping), float(margin),
               float(mu_t), float(c_t), 1e4)
        rb.update(float(dt), float(max_speed), float(max_omega))
        # Position projection + restitution (simple corrective impulse for sphere-plane)
        pos = rb.position.to_numpy()
        vel = rb.velocity.to_numpy()
        # ground normal
        n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if pos[1] < ground_y + radius:
            # project out of plane
            pos[1] = ground_y + radius
            # correct normal velocity using restitution
            vn = float(np.dot(vel, n))
            if vn < 0.0:
                e = float(restitution)
                vel = vel - (1.0 + e) * vn * n
            # optional tangential damping to remove sliding energy at impact
            if tangential_damping > 0.0:
                vt = vel - np.dot(vel, n) * n
                vel = np.dot(vel, n) * n + (1.0 - float(tangential_damping)) * vt
            rb.position[None] = pos
            rb.velocity[None] = vel
        export_obj(rb, os.path.join(out_dir, f'frame_{frame:04d}_body0.obj'))
        if frame % 60 == 0:
            elapsed = time.time() - t0
            avg = elapsed / (frame + 1)
            eta = avg * (frames - frame - 1)
            print(f"[frame {frame:04d}] elapsed={elapsed:.2f}s avg={avg:.4f}s ETA={eta:.2f}s")


def run_two_spheres(frames=90, dt=1/90.0, out_dir='render/output/rigid_ti_frames',
                    gravity=(0.0, -9.8, 0.0), stiffness=5e4, damping=1e4, margin=1e-3,
                    radius=0.5, mass=1.0,
                    posA=(-1.0, 0.6, 0.0), velA=(2.0, 0.0, 0.0),
                    posB=(1.0, 0.6, 0.0), velB=(-2.0, 0.0, 0.0),
                    max_speed=20.0, max_omega=30.0, substeps=2,
                    ground_y=0.0, restitution=0.0, tangential_damping=0.0,
                    mu_t=0.2, c_t=5000.0):
    A = RigidBody(type='Ball', mass=mass, radius=radius, position=np.array(posA, dtype=np.float32), velocity=np.array(velA, dtype=np.float32))
    B = RigidBody(type='Ball', mass=mass, radius=radius, position=np.array(posB, dtype=np.float32), velocity=np.array(velB, dtype=np.float32))
    gx, gy, gz = gravity
    import os
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for frame in range(frames):
        # substeps for stability
        dt_step = float(dt) / float(substeps)
        for _ in range(int(substeps)):
            apply_gravity(A, float(mass), float(gx), float(gy), float(gz))
            apply_gravity(B, float(mass), float(gx), float(gy), float(gz))
            # Prefer analytic sphere-sphere contact for balls
            sphere_sphere_penalty(A, B, float(radius), float(radius), float(stiffness), float(damping), float(margin))
            # ground plane penalty for both spheres so they land on floor
            mesh_plane_penalty(A, 0.0, 1.0, 0.0, float(ground_y), float(stiffness), float(damping), float(margin), float(mu_t), float(c_t), 1e4)
            mesh_plane_penalty(B, 0.0, 1.0, 0.0, float(ground_y), float(stiffness), float(damping), float(margin), float(mu_t), float(c_t), 1e4)
            A.update(dt_step, float(max_speed), float(max_omega))
            B.update(dt_step, float(max_speed), float(max_omega))
            # Apply analytic linear impulse to control sphere-sphere collision velocities
            try:
                apply_sphere_sphere_impulse(A, B, float(radius), float(radius), restitution=restitution, max_impulse=None)
            except Exception:
                pass
            # Simple projection + restitution per substep for each sphere
            for rb in (A, B):
                pos = rb.position.to_numpy()
                vel = rb.velocity.to_numpy()
                n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                if pos[1] < ground_y + radius:
                    pos[1] = ground_y + radius
                    vn = float(np.dot(vel, n))
                    if vn < 0.0:
                        e = float(restitution)
                        vel = vel - (1.0 + e) * vn * n
                    if tangential_damping > 0.0:
                        vt = vel - np.dot(vel, n) * n
                        vel = np.dot(vel, n) * n + (1.0 - float(tangential_damping)) * vt
                    rb.position[None] = pos
                    rb.velocity[None] = vel
        export_obj(A, os.path.join(out_dir, f'frame_{frame:04d}_body0.obj'))
        export_obj(B, os.path.join(out_dir, f'frame_{frame:04d}_body1.obj'))
        if frame % 60 == 0:
            elapsed = time.time() - t0
            avg = elapsed / (frame + 1)
            eta = avg * (frames - frame - 1)
            print(f"[frame {frame:04d}] elapsed={elapsed:.2f}s avg={avg:.4f}s ETA={eta:.2f}s")


def measure_sphere_drop(frames=240, dt=1/240.0,
                        gravity=(0.0, -9.8, 0.0), radius=0.5, mass=1.0,
                        start_pos=(0.0, 2.0, 0.0), start_vel=(0.0, 0.0, 0.0), ground_y=0.0,
                        stiffness=5e4, damping=2e3, margin=1e-3,
                        max_speed=25.0, max_omega=35.0,
                        stop_on_contact=True, contact_margin=1e-3,
                        mu_t=0.2, c_t=5000.0):
    """采样单球下落的 y 位置轨迹，用于重力验证。

    返回: times(list[float]), ys(list[float])
    使用方法: 在测试中拟合 y = y0 + v0 t + 0.5 a t^2 比较 a 与输入重力的 gy。
    """
    rb = RigidBody(type='Ball', mass=mass, radius=radius,
                   position=np.array(start_pos, dtype=np.float32),
                   velocity=np.array(start_vel, dtype=np.float32))
    gx, gy, gz = gravity
    times = []
    ys = []
    for frame in range(frames):
        t = frame * dt
        apply_gravity(rb, float(mass), float(gx), float(gy), float(gz))
        # 地面碰撞（采样重力时可选择停止在接触前）
        if not stop_on_contact:
            mesh_plane_penalty(rb, 0.0, 1.0, 0.0, float(ground_y), float(stiffness), float(damping), float(margin),
                               float(mu_t), float(c_t), 1e4)
        rb.update(float(dt), float(max_speed), float(max_omega))
        y_val = float(rb.position.to_numpy()[1])
        times.append(t)
        ys.append(y_val)
        # 接触判定：球心到达 ground_y + radius 前稍提前停止
        if stop_on_contact and y_val <= ground_y + radius + contact_margin:
            break
    return times, ys


if __name__ == '__main__':
    # 默认：单球落地，可切换为双球碰撞
    # 支持 CLI：容器内多球碰撞
    import argparse, os, random
    ap = argparse.ArgumentParser(description='Taichi rigid bodies demo')
    ap.add_argument('--mode', type=str, default='sphere_drop', choices=['sphere_drop','two_spheres','spheres_box'])
    ap.add_argument('--frames', type=int, default=240)
    ap.add_argument('--dt', type=float, default=1/240.0)
    ap.add_argument('--out', type=str, default='render/output/rigid_ti_frames')
    ap.add_argument('--gravity', type=str, default='0,-9.8,0')
    ap.add_argument('--stiffness', type=float, default=5e4)
    ap.add_argument('--damping', type=float, default=1e4)
    ap.add_argument('--margin', type=float, default=1e-3)
    ap.add_argument('--max_speed', type=float, default=25.0)
    ap.add_argument('--max_omega', type=float, default=35.0)
    ap.add_argument('--substeps', type=int, default=2)
    ap.add_argument('--ground_y', type=float, default=0.0)
    ap.add_argument('--restitution', type=float, default=0.0, help='restitution coefficient for ground collisions (0..1)')
    ap.add_argument('--tangential_damping', type=float, default=0.0, help='fractional damping applied to tangential velocity on impact (0..1)')
    ap.add_argument('--mu_t', type=float, default=0.2, help='tangential Coulomb friction coefficient')
    ap.add_argument('--c_t', type=float, default=5000.0, help='tangential viscous damping for contacts')
    # spheres_box specific
    ap.add_argument('--num', type=int, default=8)
    ap.add_argument('--radius', type=float, default=0.2)
    ap.add_argument('--mass', type=float, default=1.0)
    ap.add_argument('--box', type=str, default='3,3,3')
    # two_spheres specific: accept three separate floats to avoid quoting/comma issues
    ap.add_argument('--posA', type=float, nargs=3, default=[-1.0, 0.6, 0.0],
                    help='position A as three floats: --posA x y z')
    ap.add_argument('--velA', type=float, nargs=3, default=[2.0, 0.0, 0.0],
                    help='velocity A as three floats: --velA vx vy vz')
    ap.add_argument('--posB', type=float, nargs=3, default=[1.0, 0.6, 0.0],
                    help='position B as three floats: --posB x y z')
    ap.add_argument('--velB', type=float, nargs=3, default=[-2.0, 0.0, 0.0],
                    help='velocity B as three floats: --velB vx vy vz')
    args = ap.parse_args()
    gx, gy, gz = [float(x) for x in args.gravity.split(',')]
    if args.mode == 'sphere_drop':
        run_sphere_drop(frames=args.frames, dt=args.dt, out_dir=args.out,
                        gravity=(gx,gy,gz), stiffness=args.stiffness, damping=args.damping, margin=args.margin,
                        restitution=args.restitution, tangential_damping=args.tangential_damping,
                        mu_t=args.mu_t, c_t=args.c_t)
    elif args.mode == 'two_spheres':
        # args.posA/velA/posB/velB are lists of three floats now
        posA = tuple(args.posA)
        velA = tuple(args.velA)
        posB = tuple(args.posB)
        velB = tuple(args.velB)
        run_two_spheres(frames=args.frames, dt=args.dt, out_dir=args.out,
                        gravity=(gx,gy,gz), stiffness=args.stiffness, damping=args.damping, margin=args.margin,
                        radius=args.radius, mass=args.mass,
                        posA=posA, velA=velA, posB=posB, velB=velB,
                        max_speed=args.max_speed, max_omega=args.max_omega, substeps=args.substeps,
                        ground_y=args.ground_y, restitution=args.restitution, tangential_damping=args.tangential_damping,
                        mu_t=args.mu_t, c_t=args.c_t)
    else:
        # spheres in box: 6 planes as container
        bx, by, bz = [float(x) for x in args.box.split(',')]
        os.makedirs(args.out, exist_ok=True)
        # create bodies
        bodies = []
        for i in range(args.num):
            # random positions within box, avoid overlap roughly
            px = (random.random()*0.8-0.4) * (bx-2*args.radius)
            py = (random.random()*0.8+0.1) * (by-2*args.radius)
            pz = (random.random()*0.8-0.4) * (bz-2*args.radius)
            vx = (random.random()*2-1)
            vy = (random.random()*2-1)
            vz = (random.random()*2-1)
            bodies.append(RigidBody(type='Ball', mass=args.mass, radius=args.radius,
                                    position=np.array([px, py, pz], dtype=np.float32),
                                    velocity=np.array([vx, vy, vz], dtype=np.float32)))
        t0 = time.time()
        for frame in range(args.frames):
            # gravity
            dt_step = float(args.dt) / float(args.substeps)
            for _ in range(int(args.substeps)):
                for rb in bodies:
                    apply_gravity(rb, float(args.mass), float(gx), float(gy), float(gz))
                # pairwise contacts
                n = len(bodies)
                for i in range(n):
                    for j in range(i+1, n):
                        rigid_rigid_penalty(bodies[i], bodies[j], float(args.stiffness), float(args.damping), float(args.margin))
                        rigid_rigid_penalty(bodies[j], bodies[i], float(args.stiffness), float(args.damping), float(args.margin))
                # box planes: +-x, +-y, +-z
                for rb in bodies:
                    mesh_plane_penalty(rb, 0.0, 1.0, 0.0,  by/2.0, float(args.stiffness), float(args.damping), float(args.margin), float(args.mu_t), float(args.c_t), 1e4)
                    mesh_plane_penalty(rb, 0.0,-1.0, 0.0,  by/2.0, float(args.stiffness), float(args.damping), float(args.margin), float(args.mu_t), float(args.c_t), 1e4)
                    mesh_plane_penalty(rb, 1.0, 0.0, 0.0,  bx/2.0, float(args.stiffness), float(args.damping), float(args.margin), float(args.mu_t), float(args.c_t), 1e4)
                    mesh_plane_penalty(rb,-1.0, 0.0, 0.0,  bx/2.0, float(args.stiffness), float(args.damping), float(args.margin), float(args.mu_t), float(args.c_t), 1e4)
                    mesh_plane_penalty(rb, 0.0, 0.0, 1.0,  bz/2.0, float(args.stiffness), float(args.damping), float(args.margin), float(args.mu_t), float(args.c_t), 1e4)
                    mesh_plane_penalty(rb, 0.0, 0.0,-1.0,  bz/2.0, float(args.stiffness), float(args.damping), float(args.margin), float(args.mu_t), float(args.c_t), 1e4)
                # integrate
                for k, rb in enumerate(bodies):
                    rb.update(dt_step, float(args.max_speed), float(args.max_omega))
            for k, rb in enumerate(bodies):
                export_obj(rb, os.path.join(args.out, f'frame_{frame:04d}_body{k}.obj'))
            if frame % 60 == 0:
                elapsed = time.time() - t0
                avg = elapsed / (frame + 1)
                eta = avg * (args.frames - frame - 1)
                print(f"[frame {frame:04d}] elapsed={elapsed:.2f}s avg={avg:.4f}s ETA={eta:.2f}s")
    # run_two_spheres(out_dir='render/output/rigid_ti_frames')
    
    def get_states(self):
        velocity = self.velocity.to_numpy()
        position = self.position.to_numpy()
        orientation = self.orientation.to_numpy()
        angular_velocity = self.angular_velocity.to_numpy()
        return {
            'velocity': velocity,
            'position': position,
            'orientation': orientation,
            'angular_velocity': angular_velocity
        }