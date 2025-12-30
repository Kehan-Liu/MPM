import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.mpm_solver import MPMSolver
from src.scene import Scene
from src.objects import *
import taichi as ti

ti.init(arch=ti.gpu)

scene = Scene(gravity=(0, 0, -9.81), n_grid=128, dt=2e-5)

water = MPMObject(
    meshdir="Box",
    position=(0.3, 0.5, 0.6),
    scale=(1.5, 3.0, 2.0),
    num_particles=200000,
    material=MPMMaterial(
        model=MPMModel.WATER, density=1000, viscosity=0.5, stiffness=200, power=7.0
    ),
)

Ball = RigidObject(
    meshdir="meshes/bunny.obj",
    position=(0.7, 0.5, 0.15),
    mass=3.0,
    material=RigidMaterial(kh=10, friction=0),
)

scene.add_mpm_object(water)
scene.add_rigid_object(Ball)
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
camera.position(3.0, 3.0, 1.5)
camera.lookat(0.5, 0.5, 0.5)
camera.up(0, 0, 1)

video_manager = ti.tools.VideoManager(
    output_dir="results", framerate=24, automatic_build=False
)

current_time = 0.0
for frame in range(300):
    for _ in range(250):
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
    os.makedirs("results/water_ball", exist_ok=True)
    os.makedirs("results/water_ball/rigid", exist_ok=True)
    os.makedirs("results/water_ball/particles", exist_ok=True)
    solver.export(frame, "results/water_ball")
    print(f"Frame {frame} / 300")

video_manager.make_video(gif=True, mp4=True)
