# scripts/run_scripted_traj.py
import os
import numpy as np
from simulators.solid.rigid_body_sim_ti import RigidBody, export_obj

# 输出目录
out_dir = 'render/output/rigid_ti_frames_scripted'
os.makedirs(out_dir, exist_ok=True)

# 参数
frames = 240
dt = 1.0 / 60.0
radius = 0.5
mass = 1.0
max_speed = 50.0
max_omega = 100.0

# 示例轨迹函数：水平圆周，保持高度 y=1，顺时针转一圈/秒；不自转（四元数恒等）
def scripted_traj(t):
    # 位置 (x, y, z)
    x = 1.0 * np.cos(2.0 * np.pi * 0.5 * t)   # 半周期 -> 0.5 Hz 圆周
    z = 1.0 * np.sin(2.0 * np.pi * 0.5 * t)
    y = 1.0
    pos = np.array([x, y, z], dtype=np.float32)

    # 如果希望物体在轨迹上旋转，则返回不同的四元数
    # 这里返回恒等四元数 (w,x,y,z) = (1,0,0,0) -> 无旋转
    quat = (1.0, 0.0, 0.0, 0.0)
    return pos, quat

# 也可以返回旋转四元数，例如按 yaw 旋转:
def scripted_traj_with_yaw(t):
    pos = np.array([np.cos(t), 1.0, np.sin(t)], dtype=np.float32)
    # yaw angle (radians)
    yaw = 1.0 * t  # 1 rad/s
    half = 0.5 * yaw
    qw = float(np.cos(half))
    qx = 0.0
    qy = float(np.sin(half))  # rotation about Y axis -> quaternion (w,0,1*...,0)
    qz = 0.0
    return pos, (qw, qx, qy, qz)

# 创建带轨迹的刚体（直接传入 scripted_trajectory）
rb = RigidBody(type='Ball', radius=radius, mass=mass, scripted_trajectory=scripted_traj)

# 逐帧推进并导出 OBJ（每帧一个 OBJ，用 Blender 渲染）
for frame in range(frames):
    t = frame * dt
    # 统一使用 rb.step(t, dt, max_speed, max_omega)
    rb.step(t, dt, max_speed, max_omega)
    export_obj(rb, os.path.join(out_dir, f'frame_{frame:04d}_body0.obj'))