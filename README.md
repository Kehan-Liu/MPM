# MPM / Simulation & Rendering Sandbox


## Environment Setup
`requirements.txt` 包含：`taichi`, `numpy`, `trimesh`, 以及可选的渲染/分析库。安装：
```powershell
pip install -r requirements.txt
```

### How to Get Started
```powershell
python test_rigid.py
```
Use this for testing collision of two balls.

```powershell
python test_bunny.py
```
Use this for testing collision of a ball and a bunny.

## 渲染(Render)
Ensure your blender path is at (if not, replace it with correct path)
```
C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe
```
use the following command to render the simulation output of previous tests:
```powershell
 & "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P render/blender_render.py -- --pose_dir f:\ACG\MPM\results\test_bunny --import_meshes f:\ACG\MPM\rigid_meshes --output f:\ACG\MPM\results\render_ti_frames --engine CYCLES --use_gpu --gpu_type OPTIX --fps 60 --resolution 1280x720 --ground --body_colors "0:0.2,0.6,0.9;1:0.9,0.4,0.2"
```
Parameters explanation:
- `--mesh_sequence` : 读取的刚体模拟输出目录
- `--output` : 输出渲染图片的目录
- `--fps` : 帧率
- `--resolution` : 分辨率
- `--use_gpu`, `--gpu_only` : 使用GPU渲染
- `--ground` : 添加地面阴影
- `--body_colors` : 给不同刚体上色
## PNG to MP4
```powershell
python .\render\frames_to_mp4.py --input results\render_ti_frames --out results\ti_demo.mp4 --fps 60
```

# Bug to Fix
- None





