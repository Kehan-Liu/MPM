import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

quality = 1
n_particles, n_grid = 10000 * quality**2, 128 * quality
n_rseg = 100 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
substeps = int(5e-3 // dt)
p_vol, p_rho = (dx * 0.5) ** 2, 30
p_mass = p_vol * p_rho
E, nu = 1e4, 0.2
hardening = 1.0
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
gravity = ti.Vector([0, -9.81])
knife_length = 0.2
friction_coeff = 0.2
penalty_coeff = 10


p_x = ti.Vector.field(2, dtype=float, shape=n_particles)
p_v = ti.Vector.field(2, dtype=float, shape=n_particles)
p_C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
p_F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
p_Jp = ti.field(dtype=float, shape=n_particles)

grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))


grid_d = ti.field(dtype=float, shape=(n_grid, n_grid))
grid_A = ti.field(dtype=int, shape=(n_grid, n_grid))
grid_T = ti.field(dtype=int, shape=(n_grid, n_grid))

p_d = ti.field(dtype=float, shape=n_particles)
p_A = ti.field(dtype=int, shape=n_particles)
p_T = ti.field(dtype=int, shape=n_particles)

colors = ti.field(dtype=ti.uint32, shape=n_particles)

x_r = ti.Vector.field(2, dtype=float, shape=n_rseg + 1)
x_rp = ti.Vector.field(2, dtype=float, shape=n_rseg)

knife_pos = ti.Vector.field(2, dtype=float, shape=())
knife_v = ti.Vector.field(2, dtype=float, shape=())
knife_ang = ti.field(dtype=float, shape=())
knife_ang_vel = ti.field(dtype=float, shape=())


@ti.kernel
def update_knife_geom():
    ls = knife_pos[None] - 0.5 * knife_length * ti.Vector(
        [ti.cos(knife_ang[None]), ti.sin(knife_ang[None])]
    )
    le = knife_pos[None] + 0.5 * knife_length * ti.Vector(
        [ti.cos(knife_ang[None]), ti.sin(knife_ang[None])]
    )
    for i in range(n_rseg + 1):
        x_r[i] = ls + (le - ls) * (i / n_rseg)
    for i in range(n_rseg):
        x_rp[i] = (x_r[i] + x_r[i + 1]) / 2


@ti.func
def is_valid(I):
    return 0 <= I[0] < n_grid and 0 <= I[1] < n_grid


@ti.kernel
def update_colors():
    for i in range(n_particles):
        t = p_T[i]
        if t == -1:
            colors[i] = 0xFF0000
        elif t == 0:
            colors[i] = 0x00FF00
        elif t == 1:
            colors[i] = 0x0000FF
        if p_d[i] < 0:
            colors[i] = 0x6699FF


@ti.kernel
def substep():
    grid_v.fill(0)
    grid_m.fill(0)
    grid_d.fill(0)
    grid_A.fill(0)
    grid_T.fill(0)

    for p in x_rp:
        ba = x_r[p + 1] - x_r[p]
        base = (x_rp[p] * inv_dx - 0.5).cast(int)
        for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
            gi = base + offset
            if is_valid(gi):
                pa = gi.cast(float) * dx - x_r[p]
                h = pa.dot(ba) / (ba.dot(ba))

                if h <= 1 and h >= 0:
                    grid_d[gi] = (pa - h * ba).norm()
                    grid_A[gi] = 1
                    outer = -pa[0] * ba[1] + pa[1] * ba[0]

                    if outer > 0:
                        grid_T[gi] = 1
                    else:
                        grid_T[gi] = -1

    for p in p_x:
        p_A[p] = 0
        p_d[p] = 0.0

        base = (p_x[p] * inv_dx - 0.5).cast(int)
        fx = p_x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        Tpr = 0.0

        for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
            gi = base + offset
            if is_valid(gi) and grid_A[gi] == 1:
                p_A[p] = 1
                weight = w[offset[0]][0] * w[offset[1]][1]
                Tpr += weight * grid_d[gi] * grid_T[gi]

        zeta = ti.Matrix.identity(float, 9)
        d_grid = ti.Vector.zero(float, 9)
        Q = ti.Matrix.zero(float, 9, 3)

        if p_A[p] == 1:
            if p_T[p] == 0:
                p_T[p] = 1 if Tpr > 0 else -1

            for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
                gi = base + offset
                if is_valid(gi):
                    weight = w[offset[0]][0] * w[offset[1]][1]
                    d_signed = grid_T[gi] * grid_d[gi] * p_T[p]
                    row_id = offset[0] * 3 + offset[1]
                    gpos = (offset.cast(float) - fx) * dx
                    d_grid[row_id] = d_signed
                    Q[row_id, 0] = 1.0
                    Q[row_id, 1] = gpos[0]
                    Q[row_id, 2] = gpos[1]
                    zeta[row_id, row_id] = weight

            M = Q.transpose() @ zeta @ Q
            M_inv = M.inverse()
            beta = M_inv @ Q.transpose() @ zeta @ d_grid
            p_d[p] = beta[0]
        else:
            p_T[p] = 0

    # P2G
    for p in p_x:  # Particle state update and scatter to grid (P2G)
        p_v[p] += dt * gravity
        base = (p_x[p] * inv_dx - 0.5).cast(int)
        fx = p_x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        h = ti.exp(hardening * (1.0 - p_Jp[p]))
        mu, la = mu_0 * h, lambda_0 * h
        p_F[p] = (ti.Matrix.identity(float, 2) + dt * p_C[p]) @ p_F[p]
        U, sig, V = ti.svd(p_F[p])
        J = 1.0
        for d in ti.static(range(2)):
            # new_sig = max(min(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
            new_sig = sig[d, d]
            p_Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        stress = 2 * mu * (p_F[p] - U @ V.transpose()) @ p_F[
            p
        ].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * p_C[p]
        for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
            gi = base + offset
            if is_valid(gi) and p_T[p] * grid_T[gi] != -1:
                dpos = (offset.cast(float) - fx) * dx
                weight = w[offset[0]][0] * w[offset[1]][1]
                grid_v[gi] += weight * (p_mass * p_v[p] + affine @ dpos)
                grid_m[gi] += weight * p_mass

    # grid operation
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] = (1 / grid_m[I]) * grid_v[I]

            for d in ti.static(range(2)):
                if I[d] < 3 and grid_v[I][d] < 0:
                    grid_v[I][d] = 0
                if I[d] > n_grid - 3 and grid_v[I][d] > 0:
                    grid_v[I][d] = 0

    # G2P
    for p in p_x:  # grid to particle (G2P)
        base = (p_x[p] * inv_dx - 0.5).cast(int)
        fx = p_x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        p_n = ti.Vector([-ti.sin(knife_ang[None]), ti.cos(knife_ang[None])]) * p_T[p]
        for offset in ti.static(ti.grouped(ti.ndrange(3, 3))):
            gi = base + offset
            if is_valid(gi):
                weight = w[offset[0]][0] * w[offset[1]][1]
                g_v = grid_v[gi]
                if p_T[p] * grid_T[gi] == -1:
                    gpos = gi.cast(float) * dx
                    rel_pos = gpos - knife_pos[None]
                    rot_rel_pos = ti.Vector([-rel_pos[1], rel_pos[0]])
                    g_rigid_vel = knife_v[None] + knife_ang_vel[None] * rot_rel_pos
                    delta_v = p_v[p] - g_rigid_vel
                    dv_dot_n = delta_v.dot(p_n)
                    if dv_dot_n < 0:
                        delta_vt = delta_v - dv_dot_n * p_n
                        vt_norm = delta_vt.norm()
                        new_g_vt = p_v[p]
                        if vt_norm > 1e-10:
                            new_g_vt = ti.max(
                                0.0, vt_norm + friction_coeff * dv_dot_n
                            ) * (delta_vt / vt_norm)
                        g_v = new_g_vt + g_rigid_vel
                    else:
                        g_v = p_v[p]

                new_v += weight * g_v
                dpos = offset.cast(float) - fx
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        p_v[p], p_C[p] = new_v, new_C
        p_x[p] += dt * p_v[p]

        if p_d[p] < 0:
            penalty = penalty_coeff * (-p_d[p]) * p_n
            p_v[p] += penalty

    # Knife update
    knife_ang[None] += dt * knife_ang_vel[None]
    # knife_pos[None] += dt * knife_v[None]


@ti.kernel
def initialize():
    for i in range(n_particles):
        p_x[i] = [ti.random() * 0.9 + 0.05, ti.random() * 0.4 + 0.3]
        p_v[i] = ti.Matrix([0, 0])
        p_C[i] = ti.Matrix.zero(float, 2, 2)
        p_F[i] = ti.Matrix([[1, 0], [0, 1]])
        p_Jp[i] = 1

    knife_ang[None] = ti.math.pi / 2
    knife_pos[None] = ti.Vector([0.5, 0.8])
    knife_v[None] = ti.Vector([0, 0])
    knife_ang_vel[None] = 0.0


initialize()
gui = ti.GUI("Realtime Interaction", res=512, background_color=0xF0F0F0)

update_knife_geom()

prev_mouse_pos = None

while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):

    # knife_v[None] = ti.Vector([0.0, 0.0])
    # if gui.is_pressed(ti.GUI.LEFT) or gui.is_pressed("a"):
    #     knife_v[None][0] = -0.2
    # if gui.is_pressed(ti.GUI.RIGHT) or gui.is_pressed("d"):
    #     knife_v[None][0] = 0.2
    # if gui.is_pressed(ti.GUI.UP) or gui.is_pressed("w"):
    #     knife_v[None][1] = 0.2
    # if gui.is_pressed(ti.GUI.DOWN) or gui.is_pressed("s"):
    #     knife_v[None][1] = -0.2

    current_mouse_pos = ti.Vector(gui.get_cursor_pos())
    if prev_mouse_pos is not None:
        knife_v[None] = (current_mouse_pos - prev_mouse_pos) / (dt * substeps)
    else:
        knife_v[None] = ti.Vector([0.0, 0.0])
    prev_mouse_pos = current_mouse_pos
    knife_pos[None] = current_mouse_pos

    knife_ang_vel[None] = 0.0
    if gui.is_pressed("q"):
        knife_ang_vel[None] = 1.5
    if gui.is_pressed("e"):
        knife_ang_vel[None] = -1.5

    if gui.is_pressed(ti.GUI.SPACE):
        initialize()

    for s in range(substeps):
        substep()
        update_knife_geom()

    update_colors()
    gui.circles(p_x.to_numpy(), radius=2.0, color=0x3388FF)

    knife_start = knife_pos[None] - 0.5 * knife_length * ti.Vector(
        [ti.cos(knife_ang[None]), ti.sin(knife_ang[None])]
    )
    knife_end = knife_pos[None] + 0.5 * knife_length * ti.Vector(
        [ti.cos(knife_ang[None]), ti.sin(knife_ang[None])]
    )
    ks_np = knife_start.to_numpy()
    ke_np = knife_end.to_numpy()
    gui.line(ks_np, ke_np, radius=2, color=0xFF0000)

    gui.text("Control: Mouse to move, A/D to rotate", pos=(0.05, 0.95), color=0x000000)
    gui.show()
