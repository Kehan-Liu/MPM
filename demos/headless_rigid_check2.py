import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.rigid import Rigid
from src.objects import RigidObject, RigidMaterial
import taichi as ti
import numpy as np

print('starting headless test 2')
ti.init(arch=ti.cuda)
print('taichi initialized')

pos1 = (0.5, 0.5, 0.8)
pos3 = (0.5, 0.5, 0.2)

Box1 = RigidObject(
    meshdir='Box', position=pos1, mass=10.0, velocity=(0,0,0),
    material=RigidMaterial(kh=10, friction=0.5, restitution=0.2, splitter=0.1)
)
Box2 = RigidObject(
    meshdir='Box', position=pos3, mass=10.0, velocity=(0,0,0),
    material=RigidMaterial(kh=10, friction=0.5, restitution=0.2, splitter=0.1)
)

solver = Rigid([Box1, Box2], dx=0.01, dt=1e-3, gravity=(0,0,-9.8), damping=100, margin=1e-3, stiffness=5e4)
print('solver created')

steps = 2000
for s in range(steps):
    solver.step(s * solver.dt)
    poses = solver.positions.to_numpy()
    vels = solver.velocities.to_numpy()
    print(f"step {s:4d}: p0={poses[0]} v0={vels[0]} | p1={poses[1]} v1={vels[1]}")
    if abs(poses[0][2] - poses[1][2]) < 0.05:
        print('near contact at step', s)
        break
