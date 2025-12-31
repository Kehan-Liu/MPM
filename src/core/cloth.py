import taichi as ti
import numpy as np
from typing import List, Tuple
from src.objects import ClothObject
from src.core.rigid_body import RigidBody
import os


@ti.kernel
def cloth_rigid_penalty(
    positions: ti.template(),
    velocities: ti.template(),
    forces: ti.template(),
    n_vertices: ti.int32,
    rb: ti.template(),
    stiffness: ti.f32,
    damping: ti.f32,
    margin: ti.f32,
    dt: ti.f32,
):
    """
    对每个布料顶点，用刚体 mesh 的 signed_distance 做 penalty 碰撞。
    对布料：加力到 forces；
    对刚体：调用 rb.apply_impulse_at_point 施加反作用冲量。
    """
    for i in range(n_vertices):
        x = positions[i]
        phi, n, cp, fidx = rb.signed_distance(x)  # RigidBody.signed_distance now returns cp and face index
        if phi < 0 and ti.abs(phi) > margin:
            v_cloth = velocities[i]
            v_rigid = rb.get_velocity_at_point(cp)
            vn = (v_cloth - v_rigid).dot(n)
            # normal spring-damper
            F = (-stiffness * phi - damping * vn) * n
            # 作用在布料点
            forces[i] += F
            # 反作用在刚体（如果刚体只是静态障碍，可以注释掉这一行）
            rb.apply_impulse_at_point(-F * dt, cp)


@ti.kernel
def cloth_box_penalty(
    positions: ti.template(),
    velocities: ti.template(),
    forces: ti.template(),
    n_vertices: ti.int32,
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
):
    """
    布料顶点与 box 边界 [xmin,xmax]×[ymin,ymax]×[zmin,zmax] 的 penalty 碰撞。
    和刚体的 mesh_box_penalty 同风格。
    """
    for i in range(n_vertices):
        x = positions[i]
        v = velocities[i]

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
        phi = (-x[0]) - (-xmax)  # = xmax - x
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
            n = best_n
            phi = best_phi
            vn = v.dot(n)
            vt = v - vn * n

            f_n = -stiffness * phi - damping * vn
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

            F = f_n * n + f_t
            forces[i] += F


@ti.data_oriented
class Cloth:
    """
    单块布料（质点-弹簧）自求解器：
    - 每个 Cloth 对应一个 ClothObject（一个 mesh）
    - 内部维护：顶点、速度、力、质量、弹簧
    - step() 里完成：重力 + 弹簧力 + 和刚体/边界的碰撞 + 半隐式欧拉积分
    """

    def __init__(
        self,
        cloth_object: ClothObject,
        dt: float,
        gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
        k_struct: float = 2e4,
        k_bend: float = 5e3,
        spring_damping: float = 50.0,
        dx: float = 0.01,
    ):
        if isinstance(cloth_object, list):
            cloth_object = cloth_object[0]

        self.conf = cloth_object
        self.dt = dt
        self.dx = dx

        # 重力
        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity[None] = ti.Vector(gravity)

        # penalty / 碰撞参数
        self.margin = 1e-3
        self.penalty_stiffness = 5e4
        self.penalty_damping = 1e4

        # 解析 mesh
        mesh = cloth_object.mesh
        verts_np = mesh.vertices.astype(np.float32)
        faces_np = mesh.faces.astype(np.int32)

        n_verts = verts_np.shape[0]
        n_faces = faces_np.shape[0]

        self.n_vertices = ti.field(dtype=ti.i32, shape=())
        self.n_faces = ti.field(dtype=ti.i32, shape=())
        self.n_vertices[None] = n_verts
        self.n_faces[None] = n_faces

        # 顶点场
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
        self.mass = ti.field(dtype=ti.f32, shape=n_verts)
        self.fixed_mask = ti.field(dtype=ti.i32, shape=n_verts)  # 1 表示固定

        # 面拓扑
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=n_faces)

        # 弹簧场（长度由 Python 侧构造）
        self.k_struct = k_struct
        self.k_bend = k_bend
        self.spring_damping_default = spring_damping

        self.n_springs = ti.field(dtype=ti.i32, shape=())
        # 先用 Python 列表构造，再分配 Taichi field
        springs_py = self._build_spring_list(verts_np, faces_np)
        n_springs = len(springs_py)
        self.n_springs[None] = n_springs

        self.spring_i = ti.field(dtype=ti.i32, shape=n_springs)
        self.spring_j = ti.field(dtype=ti.i32, shape=n_springs)
        self.spring_rest_len = ti.field(dtype=ti.f32, shape=n_springs)
        self.spring_k = ti.field(dtype=ti.f32, shape=n_springs)
        self.spring_damping = ti.field(dtype=ti.f32, shape=n_springs)

        # 初始化顶点位置 / 速度 / 质量 / faces / springs
        self._init_vertices(verts_np)
        self._init_faces(faces_np)
        self._init_springs(springs_py)

    def _build_spring_list(self, verts_np: np.ndarray, faces_np: np.ndarray):
        """
        从三角 mesh 拓扑构造：
        - 结构弹簧：所有边
        - 弯曲弹簧：共享一条边的两三角的“对顶点”之间
        L0 用原始 mesh 的局部坐标距离。
        """
        struct_edges = set()
        edge2faces = {}

        # 收集结构边 & 构造 edge -> faces 映射
        for f_id, face in enumerate(faces_np):
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            tri = (a, b, c)
            for e in ((a, b), (b, c), (c, a)):
                e_sorted = tuple(sorted(e))
                struct_edges.add(e_sorted)
                edge2faces.setdefault(e_sorted, []).append((f_id, tri))

        # 弯曲弹簧：两三角共享边 -> 两个对顶点连接
        bend_edges = set()
        for e, flist in edge2faces.items():
            if len(flist) == 2:
                (_, tri1), (_, tri2) = flist
                edge_set = set(e)
                opp1 = list(set(tri1) - edge_set)[0]
                opp2 = list(set(tri2) - edge_set)[0]
                bend_edges.add(tuple(sorted((opp1, opp2))))

        springs = []

        # 结构弹簧
        for i, j in struct_edges:
            p_i = verts_np[i]
            p_j = verts_np[j]
            L0 = float(np.linalg.norm(p_i - p_j))
            springs.append((i, j, L0, self.k_struct))

        # 弯曲弹簧
        for i, j in bend_edges:
            p_i = verts_np[i]
            p_j = verts_np[j]
            L0 = float(np.linalg.norm(p_i - p_j))
            springs.append((i, j, L0, self.k_bend))

        return springs

    def _init_vertices(self, verts_np: np.ndarray):
        """
        根据 ClothObject 的配置，初始化顶点位置、速度和质量。
        """
        n_verts = verts_np.shape[0]
        pos_offset = np.array(
            getattr(self.conf, "position", (0.0, 0.0, 0.0)), dtype=np.float32
        )
        vel0 = np.array(
            getattr(self.conf, "velocity", (0.0, 0.0, 0.0)), dtype=np.float32
        )
        total_mass = float(getattr(self.conf, "mass", 1.0))
        mass_per_vert = total_mass / max(1, n_verts)

        for i in range(n_verts):
            world_pos = verts_np[i] + pos_offset
            self.positions[i] = ti.Vector(world_pos)
            self.velocities[i] = ti.Vector(vel0)
            self.mass[i] = mass_per_vert
            self.fixed_mask[i] = 0  # 默认全部不固定

    def _init_faces(self, faces_np: np.ndarray):
        n_faces = faces_np.shape[0]
        for i in range(n_faces):
            a, b, c = int(faces_np[i][0]), int(faces_np[i][1]), int(faces_np[i][2])
            self.faces[i] = ti.Vector([a, b, c])

    def _init_springs(self, springs_py):
        for s_id, (i, j, L0, k) in enumerate(springs_py):
            self.spring_i[s_id] = i
            self.spring_j[s_id] = j
            self.spring_rest_len[s_id] = L0
            self.spring_k[s_id] = k
            self.spring_damping[s_id] = self.spring_damping_default

    def pin_vertices(self, indices: List[int]):
        """
        固定一部分顶点（比如挂在天花板）：
            cloth.pin_vertices([0, 10, 20, ...])
        """
        for idx in indices:
            if 0 <= idx < self.n_vertices[None]:
                self.fixed_mask[idx] = 1

    @ti.kernel
    def clear_forces(self):
        for i in range(self.n_vertices[None]):
            if self.fixed_mask[i] == 0:
                self.forces[i] = self.mass[i] * self.gravity[None]
            else:
                self.forces[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_springs(self):
        """
        弹簧力 + 阻尼力
        """
        for s in range(self.n_springs[None]):
            i = self.spring_i[s]
            j = self.spring_j[s]
            xi = self.positions[i]
            xj = self.positions[j]
            vi = self.velocities[i]
            vj = self.velocities[j]
            k = self.spring_k[s]
            c = self.spring_damping[s]
            L0 = self.spring_rest_len[s]

            d = xi - xj
            len_d = d.norm() + 1e-8
            n = d / len_d

            # Hooke
            F_spring = -k * (len_d - L0) * n
            # damping 沿弹簧方向
            v_rel = (vi - vj).dot(n)
            F_damp = -c * v_rel * n

            F = F_spring + F_damp

            if self.fixed_mask[i] == 0:
                for k in ti.static(range(3)):
                    ti.atomic_add(self.forces[i][k], F[k])
            if self.fixed_mask[j] == 0:
                for k in ti.static(range(3)):
                    ti.atomic_add(self.forces[j][k], -F[k])

    @ti.kernel
    def integrate(self, dt: ti.f32):
        """
        半隐式欧拉积分
        """
        for i in range(self.n_vertices[None]):
            if self.fixed_mask[i] == 0:
                a = self.forces[i] / self.mass[i]
                self.velocities[i] += dt * a
                self.positions[i] += dt * self.velocities[i]

    def resolve_collisions_with_rigids(self, rigid_bodies: List[RigidBody]):
        """
        布料 ↔ 刚体（mesh SDF penalty） + 布料 ↔ box [0,1]^3 碰撞
        """
        stiffness = self.penalty_stiffness
        damping = self.penalty_damping
        margin = self.margin

        # 与所有刚体碰撞
        for rb in rigid_bodies:
            cloth_rigid_penalty(
                self.positions,
                self.velocities,
                self.forces,
                self.n_vertices[None],
                rb,
                stiffness,
                damping,
                margin,
                self.dt,
            )

        # box boundary，和刚体一样默认是 [0,1]^3，可按需要修改
        mu_t = 0.2
        c_t = 5000.0
        max_force = 1e4

        cloth_box_penalty(
            self.positions,
            self.velocities,
            self.forces,
            self.n_vertices[None],
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
        )

    def step(self, time: float, rigid_bodies: List[RigidBody]):
        """
        单块布料的一个 time step：
        1. 清空并加重力
        2. 弹簧力
        3. 布料-刚体 + 布料-边界 碰撞（通过 forces/impulse 修正）
        4. 积分更新 x, v
        """
        self.clear_forces()
        self.apply_springs()
        if rigid_bodies:
            self.resolve_collisions_with_rigids(rigid_bodies)
        self.integrate(self.dt)

    def export_obj(self, frame: int, output_dir: str, basename: str = "cloth"):
        """
        导出当前布料形状为 .obj，方便离线渲染。
        """
        os.makedirs(output_dir, exist_ok=True)
        n_v = self.n_vertices[None]
        n_f = self.n_faces[None]
        filename = os.path.join(output_dir, f"{basename}_{frame:04d}.obj")
        with open(filename, "w") as f:
            # vertices
            for i in range(n_v):
                p = self.positions[i]
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            # faces (1-based index)
            for i in range(n_f):
                v0 = int(self.faces[i][0]) + 1
                v1 = int(self.faces[i][1]) + 1
                v2 = int(self.faces[i][2]) + 1
                f.write(f"f {v0} {v1} {v2}\n")
        print(f"[Cloth] Exported {filename}")
