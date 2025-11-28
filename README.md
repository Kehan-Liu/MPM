# MPM / Simulation & Rendering Sandbox

本仓库包含多种物理/渲染实验组件：

## 目录结构概览
- `render/`：Blender 批量渲染脚本与示例（读取 JSON、关键帧动画、PNG/MP4 输出）。
- `simulators/solid/rigid_body_sim.py`：基于冲量的简易刚体模拟（盒 / 球 / 网格），逐帧导出变换后的 OBJ。
- 其它目录：`core/`, `collision/`, `configs/`, `scenes/`（后续可扩展）。

## 环境依赖
`requirements.txt` 包含：`taichi`, `numpy`, `trimesh`, 以及可选的渲染/分析库。安装：
```powershell
pip install -r requirements.txt
```

## Blender 渲染（见 `render/README.md`）
示例：模拟生成 JSON 后，用 `blender_render.py` 在无界面模式批量渲染为 PNG / MP4。支持 GPU 选择、HDR 环境、Filmic、地面阴影、导入外部 mesh。详见子目录 README。

## 刚体冲量模拟 (RigidBody)
文件：`simulators/solid/rigid_body_sim.py`

### 最小示例（两个盒体对撞）
```powershell
python simulators\solid\rigid_body_sim.py --dt 0.016 --frames 120 --out render\output\rigid_frames
```

### 自定义刚体（重复使用 `--body` 描述串）
```powershell
python simulators\solid\rigid_body_sim.py --dt 0.01 --frames 240 --gravity 0,-9.8,0 `
	--body box:mass=2.0:size=1,1,1:pos=-2,1,0:vel=3,0,0:color=0.2,0.6,0.9 `
	--body sphere:mass=1.5:radius=0.6:pos=2,1,0:vel=-3,0,0:color=0.9,0.4,0.2 `
	--out render\output\rigid_frames
```

### 导入网格
```powershell
python simulators\solid\rigid_body_sim.py --frames 180 --dt 0.016 `
	--body mesh:mesh=render/assets/demo_cube.obj:mass=3.0:pos=0,1,0:vel=0,0,0:color=1,1,1 `
	--body mesh:mesh=render/assets/demo_ball.obj:mass=2.0:pos=2,1,0:vel=-2,0,0:color=1,0.8,0.3 `
	--out render\output\rigid_frames
```

### 输出内容
- 每帧生成：`frame_XXXX_bodyN.obj`（单独网格，已应用位移 + 旋转）。
- 汇总：`frames.json`（包含 dt、每帧各刚体位置、四元数、尺寸、颜色）。

### 冲量解算特性
- 近似碰撞：AABB 重叠检测 + 选择最小重合轴作为法向。
- 支持地面平面 y=0 的接触与摩擦。
- 法向冲量：使用恢复系数（`restitution`），Baumgarte 位置校正简化穿透补偿。
- 摩擦冲量：库仑模型，限幅 `μ * j_n`。
- 姿态更新：四元数增量 `ω` -> `Δq`，并归一化。

### 后续可扩展方向
- 更精确的窄相检测：GJK + EPA 或 SAT for OBB。
- 连续碰撞检测（CCD）以防高速穿透。
- 约束稳定器（ERP / warm starting / sequential impulses）。
- 网格体积/惯性更精确估计（利用体素或凸分解）。
- Taichi 加速：把碰撞与积分循环迁移到 `@ti.kernel`。

# Bug to Fix
- 刚体模拟不太对, 刚体会弹一下以后穿过地面



