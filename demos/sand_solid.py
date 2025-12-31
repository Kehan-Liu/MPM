import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.mpm_solver import MPMSolver
from src.scene import Scene
from src.objects import *
import taichi as ti

ti.init(arch=ti.gpu)

scene = Scene(gravity=(0, 0, -9.81), n_grid=64, dt=5e-5, penalty_parameter=1, clamp_factor=50.0)

sand = MPMObject(
    meshdir="Box",
    position=(0.3, 0.5, 0.3),
    scale=(1.0, 3.0, 1.5),
    num_particles=2000000,
    material=MPMMaterial(
        model=MPMModel.WATER,
        E=2e5,
        nu=0.2,
        density=1600,
        friction_angle=35.0,
    ),
)

Ball1 = RigidObject(
    meshdir="Ball",
    position=(0.8, 0.5, 0.1),
    mass=1.0,
    scale=(0.5, 0.5, 0.5),
    material=RigidMaterial(kh=100, friction=0.5),
)

Ball2 = RigidObject(
    meshdir="Ball",
    position=(0.8, 0.5, 0.25),
    mass=1.0,
    scale=(0.5, 0.5, 0.5),
    material=RigidMaterial(kh=100, friction=0.5),
)

Ball3 = RigidObject(
    meshdir="Ball",
    position=(0.8, 0.5, 0.4),
    mass=1.0,
    scale=(0.5, 0.5, 0.5),
    material=RigidMaterial(kh=100, friction=0.5),
)

Box1 = RigidObject(
    meshdir="Box",
    position=(0.8, 0.5, 0.1),
    mass=1.0,
    scale=(0.5, 0.5, 0.5),
    material=RigidMaterial(kh=100, friction=0.5),
)
Box2 = RigidObject(
    meshdir="Box",
    position=(0.8, 0.5, 0.25),
    mass=1.0,
    scale=(0.5, 0.5, 0.5),
    material=RigidMaterial(kh=100, friction=0.5),
)

Box3 = RigidObject(
    meshdir="Box",
    position=(0.8, 0.5, 0.4),
    mass=1.0,
    scale=(0.5, 0.5, 0.5),
    material=RigidMaterial(kh=100, friction=0.5),
)

Bunny = RigidObject(
    meshdir="meshes/bunny.obj",
    position=(0.3, 0.5, 0.8),
    mass=1.0,
    material=RigidMaterial(kh=100, friction=0.5),
)

Knife = RigidObject(
    meshdir="meshes/knife.obj",
    position=(0.5, 0.5, 0.8),
    mass=5.0,
    material=RigidMaterial(kh=100, friction=0.5),
)
scene.add_mpm_object(sand)
scene.add_rigid_object(Ball1)
scene.add_rigid_object(Box2)
scene.add_rigid_object(Ball3)
scene.add_rigid_object(Bunny)
solver = MPMSolver(scene)

colors = ti.Vector.field(3, dtype=ti.f32, shape=solver.n_particles[None])


@ti.kernel
def update_colors():
    for i in range(solver.n_particles[None]):
        t = solver.p_T[i][0]
        if t == -1:
            colors[i] = ti.Vector([1.0, 0.0, 0.0])
        elif t == 0:
            colors[i] = ti.Vector([0.0, 1.0, 0.0])
        elif t == 1:
            colors[i] = ti.Vector([0.0, 0.0, 1.0])
        if solver.p_d[i][0] < 0:
            colors[i] = ti.Vector([0.4, 0.7, 1.0])


window = ti.ui.Window("MPM", (1024, 1024), vsync=True)
canvas = window.get_canvas()
ui_scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 2.5, 0.8)
camera.lookat(0.5, 0.5, 0.5)
camera.up(0, 0, 1)

video_manager = ti.tools.VideoManager(
    output_dir="results", framerate=24, automatic_build=False
)

current_time = 0.0
for frame in range(300):
    for _ in range(100):
        solver.step(current_time)
        current_time += scene.dt

    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    ui_scene.set_camera(camera)
    ui_scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    ui_scene.ambient_light((0.5, 0.5, 0.5))

    update_colors()
    ui_scene.particles(solver.p_x, radius=0.005, per_vertex_color=colors)
    for rb in solver.rigid.rigid_objects:
        rb.update_render_vertices()
        ui_scene.mesh(
            rb.render_vertices, indices=rb.render_indices, color=(0.6, 0.6, 0.6)
        )

    canvas.scene(ui_scene)
    video_manager.write_frame(window.get_image_buffer_as_numpy())
    window.show()
    os.makedirs("results/sand_solid", exist_ok=True)
    os.makedirs("results/sand_solid/rigid", exist_ok=True)
    os.makedirs("results/sand_solid/particles", exist_ok=True)
    solver.export(frame, "results/sand_solid")
    print(f"Frame {frame} / 300")

video_manager.make_video(gif=True, mp4=True)
