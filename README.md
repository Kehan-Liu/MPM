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
python simulators\solid\rigid_body_sim_ti.py --mode two_spheres --frames 240 --dt 0.004 --substeps 32 --gravity 0,-9.8,0 --restitution 0.5 --tangential_damping 0.2 --mu_t 0.3 --c_t 6000
```
这是模拟的流程,参数分别代表:
- `--mode` : 选择模拟场景（单球/双球/其他自定义场景）
- `--frames` : 总帧数
- `--dt` : 每帧时间步长
- `--substeps` : 每帧内的子步数（提高精度）
- `--gravity` : 重力向量
- `--restitution` : 碰撞恢复系数 (0~1) 过大会导致球反弹过高
- `--tangential_damping` : 切向阻尼系数 (0~1)
- `--mu_t` : 摩擦系数
- `--c_t` : 摩擦刚度系数（影响摩擦力计算）
### 输出内容
- 每帧生成：`frame_XXXX_bodyN.obj`（单独网格，已应用位移 + 旋转）(非常重要)
- 汇总：`frames.json`（包含 dt、每帧各刚体位置、四元数、尺寸、颜色）(并不重要)

### 冲量解算特性
- 近似碰撞：AABB 重叠检测 + 选择最小重合轴作为法向。
- 支持地面平面 y=0 的接触与摩擦。
- 法向冲量：使用恢复系数（`restitution`），Baumgarte 位置校正简化穿透补偿。
- 摩擦冲量：库仑模型，限幅 `μ * j_n`。
- 姿态更新：四元数增量 `ω` -> `Δq`，并归一化。

### 已经扩展方向
- 更精确的窄相检测：GJK + EPA 或 SAT for OBB。
- 连续碰撞检测（CCD）以防高速穿透。
- 约束稳定器（ERP / warm starting / sequential impulses）。
- 重复resolve 20次获得更精准法向量

## 渲染(Render)
假设你的blender路径在
```
C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe
```
使用以下命令来运行:
```powershell
 & "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P "f:\MPM\render\blender_render.py" -- `         
--mesh_sequence "f:\MPM\render\output\rigid_ti_frames" `
--output "f:\MPM\render\output\render_ti_frames" `
--fps 60 --resolution 1280x720 --engine CYCLES --samples 128 --use_gpu --gpu_only --ground `
--gravity "0,0,0" --show_gravity --gravity_marker_length 2.5 --body_colors "0:0.2,0.6,0.9;1:0.9,0.4,0.2"
```
这是渲染的整个流程.参数分别代表:
- `--mesh_sequence` : 读取的刚体模拟输出目录
- `--output` : 输出渲染图片的目录
- `--fps` : 帧率
- `--resolution` : 分辨率
- `--use_gpu`, `--gpu_only` : 使用GPU渲染
- `--ground` : 添加地面阴影
- `--body_colors` : 给不同刚体上色
## PNG 转化为 MP4
```powershell
python .\render\frames_to_mp4.py --input render\output\rigid_ti_frames --out render\output\ti_demo.mp4 --fps 60
```
以60帧每秒的速度将render\output\rigid_ti_frames目录下的png图片转化为mp4视频文件ti_demo.mp4
# Bug to Fix
- None





