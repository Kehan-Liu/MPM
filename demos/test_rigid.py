import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.rigid import Rigid
from src.scene import Scene
from src.objects import *
import taichi as ti
import numpy as np

ti.init(arch=ti.cuda)
# Choose a meeting point and time so balls starting at different heights collide in mid-air
g = (0.0, 0.0, -9.8)
meet_t = 0.6  # seconds until meeting
meet_p = (0.5, 0.5, 0.5)  # meeting position in the [0,1]^3 box

# initial positions (different heights and different x to approach each other)
pos1 = (0.5, 0.5, 0.85)
pos2 = (0.5, 0.5, 0.5)
pos3 = (0.5, 0.5, 0.15)


def compute_initial_velocity(pos, meet_p, meet_t, g):
    # v0 = (p - r0 - 0.5*g*t^2) / t
    gx, gy, gz = g
    rx, ry, rz = pos
    px, py, pz = meet_p
    vx = (px - rx - 0.5 * gx * meet_t * meet_t) / meet_t
    vy = (py - ry - 0.5 * gy * meet_t * meet_t) / meet_t
    vz = (pz - rz - 0.5 * gz * meet_t * meet_t) / meet_t
    return (vx, vy, vz)


# vel1 = compute_initial_velocity(pos1, meet_p, meet_t, g)
# vel2 = compute_initial_velocity(pos2, meet_p, meet_t, g)

Bunny1= RigidObject(
    meshdir="meshes/bunny.obj",
    position=pos1,
    mass=10.0,
    velocity= (0, 0, 0),
    material=RigidMaterial(kh=10, friction=0.5, restitution=0.2, splitter=0.1),
)

Bunny2 = RigidObject(
    meshdir="meshes/bunny.obj",
    position=pos3,
    mass=10.0,
    velocity= (0, 0, 0),
    material=RigidMaterial(kh=10, friction=0.5, restitution=0.2, splitter=0.1),
)
Box1 = RigidObject(
    meshdir="Box",
    position=pos1,
    mass=10.0,
    velocity= (0, 0, 0),
    material=RigidMaterial(kh=10, friction=0.5, restitution=0.2, splitter=0.1),
)

Box2 = RigidObject(
    meshdir="Box",
    position=pos3,
    mass=10.0,
    velocity= (0, 0, 0),
    material=RigidMaterial(kh=10, friction=0.5, restitution=0.2, splitter=0.1),
)

Ball = RigidObject(
    meshdir="Ball",
    position=pos2,
    mass=10.0,
    velocity= (0, 0, 0),
    material=RigidMaterial(kh=10, friction=0.5, restitution=0.2, splitter=0.1),
)
Cphere = RigidObject(
    meshdir="Sphere",
    position=pos3,
    mass=10.0,
    velocity= (0, 0, 0),
    material=RigidMaterial(kh=10, friction=0.5, restitution=0.2, splitter=0.1),
)
import time

current_time = 0.0
substeps = 40  # 降低每帧子步数以提升实时FPS；需要更精细物理可增大
last_t = time.time()
solver = Rigid(
    [Box1, Bunny2, Ball],
    dx=1 / 128,
    dt=5e-3 / substeps,
    gravity=(0, 0, -9.8), 
    damping=100,
    margin=1e-3,
    stiffness=5e5,
    penalty_parameter=1,
    clamp_factor=50.0
)

window = ti.ui.Window("Rigid", (1024, 1024), vsync=False)
canvas = window.get_canvas()
ui_scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 2.0, 0.5)
camera.lookat(0.5, 0.5, 0.5)
camera.up(0, 0, 1)

video_manager = ti.tools.VideoManager(
    output_dir="results", framerate=24, automatic_build=False
)


for frame in range(300):
    for _ in range(substeps):
        solver.step(current_time)
        current_time += solver.dt

    # Debug: print rigid positions and velocities to check force directions
    try:
        poses = solver.positions.to_numpy()
        vels = solver.velocities.to_numpy()
        print(f"poses=\n{poses}\nvels=\n{vels}")
    except Exception:
        pass

    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    ui_scene.set_camera(camera)
    ui_scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    ui_scene.ambient_light((0.5, 0.5, 0.5))

    for rb in solver.rigid_objects:
        rb.update_render_vertices()
        ui_scene.mesh(
            rb.render_vertices, indices=rb.render_indices, color=(0.6, 0.6, 0.6)
        )
    # Render each rigid body's mesh transformed by current pose
    # try:
    #     # read all poses in numpy (shape: n,3) and rotations (n,3,3)
    #     poses = solver.positions.to_numpy()
    #     rots = solver.rotation_matrices.to_numpy()
    #     for i, rb in enumerate(solver.rigid_objects):
    #         verts_local = rb.mesh.vertices.astype(np.float32)
    #         faces = rb.mesh.faces.astype(np.int32)
    #         R = rots[i]
    #         p = poses[i]
    #         verts_world = (verts_local @ R.T) + p.reshape(1, 3)
    #         # Taichi UI accepts numpy arrays for mesh
    #         ui_scene.mesh(verts_world, faces, color=(0.8, 0.6, 0.5))
    # except Exception:
    #     # fallback: draw centroids as particles
    #     ui_scene.particles(solver.positions, radius=0.15, color=(1.0, 0.5, 0.5))

    canvas.scene(ui_scene)
    video_manager.write_frame(window.get_image_buffer_as_numpy())
    window.show()
    os.makedirs("results/test_rigid", exist_ok=True)
    os.makedirs("results/test_rigid/rigid", exist_ok=True)
    os.makedirs("results/test_rigid/particles", exist_ok=True)
    solver.export(frame, "results/test_rigid")
    # 简易FPS统计
    now = time.time()
    dt = now - last_t
    last_t = now
    fps = 1.0 / dt if dt > 0 else 0.0
    print(f"Frame {frame} / 300 | FPS: {fps:.1f}")

video_manager.make_video(gif=True, mp4=True)
