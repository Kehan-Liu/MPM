import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.mpm_solver import MPMSolver
from src.scene import Scene
from src.objects import *
import taichi as ti

ti.init(arch=ti.gpu)

dt = 5e-5
substeps = 100

scene = Scene(gravity=(0, 0, -9.81), n_grid=240, dt=dt)

bum = MPMObject(
    meshdir="Box",
    position=(0.65, 0.5, 0.151),
    scale=(2.0, 1.0, 1.0),
    num_particles=4000000,
    material=MPMMaterial(model=MPMModel.JELLY, E=5e4, nu=0.2, density=600),
)

def controller(t: float):
    move_down_time = 30 * substeps
    move_forward = 10 * substeps
    move_backward = 30 * substeps
    move_up_time = 30 * substeps
    interval = 0.15
    forward = 0.1
    step = int(t / dt + 0.5)
    base_x = 0.5 + float(step // (100 * substeps)) * interval
    step = step % (100 * substeps)
    z_high = 0.5
    z_low = 0.151
    y = 0.5
    x = base_x
    orientation = (1.0, 0.0, 0.0, 0.0)
    if step < move_down_time:
        z = z_high - (z_high - z_low) * (step / move_down_time)
    elif step < move_down_time + move_forward:
        z = z_low
        x = base_x - forward * ((step - move_down_time) / move_forward)
    elif step < move_down_time + move_forward + move_up_time:
        z = z_low + (z_high - z_low) * (
            (step - move_down_time - move_forward) / move_up_time
        )
        x = base_x - forward
    else:
        z = z_high
        x = base_x - forward + (forward + interval) * (
            (step - move_down_time - move_forward - move_up_time) / move_backward
        )
    return (x, y, z), orientation

Knife = RigidObject(
    meshdir="meshes/knife.obj",
    position=(0.5, 0.5, 0.3),
    mass=10.0,
    velocity=(0.0, 0.0, 0.0),
    material=RigidMaterial(kh=300, friction=0.0, splitter=1e-2),
    scripted_trajectory=controller,
)

scene.add_mpm_object(bum)
scene.add_rigid_object(Knife)
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
camera.position(-2.0, -2.0, 1.5)
camera.lookat(0.5, 0.5, 0.5)
camera.up(0, 0, 1)

video_manager = ti.tools.VideoManager(
    output_dir="results", framerate=24, automatic_build=False
)

current_time = 0.0
for frame in range(300):
    for _ in range(substeps):
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
    os.makedirs("results/cut_bum", exist_ok=True)
    os.makedirs("results/cut_bum/rigid", exist_ok=True)
    os.makedirs("results/cut_bum/particles", exist_ok=True)
    solver.export(frame, "results/cut_bum")
    print(f"Frame {frame} / 300")

video_manager.make_video(gif=True, mp4=True)
