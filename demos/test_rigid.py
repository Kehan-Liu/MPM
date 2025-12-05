from src.core.rigid import Rigid
from src.scene import Scene
from src.objects import *
import taichi as ti
import os

ti.init(arch=ti.gpu)

Water = MPMObject(
    meshdir="meshes/cube.obj",
    position=(0.3, 0.3, 0.03),
    num_particles=60000,
    material=MPMMaterial(model=MPMModel.JELLY, E=5e4, density=800),
)

Ball1 = RigidObject(
    meshdir="Ball",
    position=(0.5, 0.5, 0.8),
    mass=10.0,
    material=RigidMaterial(kh=10, friction=0.5, splitter=0.1),
)

Ball2 = RigidObject(
    meshdir="Ball",
    position=(0.5, 0.5, 0.2),
    mass=10.0,
    material=RigidMaterial(kh=10, friction=0.5, splitter=0.1),
)

solver = Rigid([Ball1, Ball2], dx=0.01, dt=1e-4, gravity=(0, 0, -9.8))

window = ti.ui.Window("Rigid", (1024, 1024), vsync=True)
canvas = window.get_canvas()
ui_scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 2.0, 0.5)
camera.lookat(0.5, 0.5, 0.5)
camera.up(0, 0, 1)

video_manager = ti.tools.VideoManager(
    output_dir="results", framerate=24, automatic_build=False
)

current_time = 0.0
for frame in range(300):
    for _ in range(50):
        solver.step(current_time)
        current_time += solver.dt

    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    ui_scene.set_camera(camera)
    ui_scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    ui_scene.ambient_light((0.5, 0.5, 0.5))

    ui_scene.particles(solver.positions, radius=0.15, color=(1.0, 0.5, 0.5))

    canvas.scene(ui_scene)
    video_manager.write_frame(window.get_image_buffer_as_numpy())
    window.show()
    solver.export(frame, "results/test_scene")
    print(f"Frame {frame} / 300")

video_manager.make_video(gif=True, mp4=True)
