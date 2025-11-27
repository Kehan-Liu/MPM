# Blender 渲染管线（render/）

此目录提供一个最简固体-固体（两立方体）碰撞的模拟与 Blender 批量渲染脚本。模拟在纯 Python 中生成每帧位姿，渲染脚本在 Blender 中读取并生成 PNG 帧。

## 目录结构
- `simulate_boxes.py`：两盒体沿 X 轴相向运动并发生弹性碰撞，输出 JSON。
- `blender_render.py`：Blender 内运行，读取模拟 JSON，创建场景并关键帧动画后渲染 PNG。
- `output/`：默认输出目录，包含 `boxes_sim.json` 与渲染帧。

## 运行步骤（Windows PowerShell）

1) 生成模拟数据（JSON）：
```powershell
python d:\ACG\MPM\render\simulate_boxes.py --frames 240 --dt 0.016 --restitution 0.6 --speed0 1.5 --speed1 1.0 --out d:\ACG\MPM\render\output\boxes_sim.json
```

2) 使用 Blender 无界面渲染：请替换为你的 Blender 安装路径。以下示例启用更真实的渲染设置（Cycles、GPU、地面、Filmic）。
```powershell
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P d:\ACG\MPM\render\blender_render.py -- \
  --sim d:\ACG\MPM\render\output\boxes_sim.json \
  --output d:\ACG\MPM\render\output\frames --fps 60 --resolution 1280x720 \
  --engine CYCLES --samples 128 --use_gpu --ground --film_filmic --contrast "Medium High Contrast" --exposure 0.5
```

渲染结束后，PNG 帧将位于 `d:\ACG\MPM\render\output\frames`。

## 参数说明
- `simulate_boxes.py`
  - `--frames`：总帧数（默认 240）。
  - `--dt`：每帧时间步长（默认 1/60）。
  - `--restitution`：碰撞恢复系数，0~1（默认 0.6）。
  - `--speed0`、`--speed1`：两盒体初速度大小（分别沿 +X 与 -X）。
  - `--out`：输出 JSON 路径。
- `blender_render.py`
  - `--sim`：模拟 JSON 路径（必填）。
  - `--output`：渲染输出目录（默认 `render/output/frames`）。
  - `--fps`：Blender 场景 FPS（若未指定则使用 JSON 中 fps）。
  - `--resolution`：输出分辨率，如 `1280x720`。
  - `--engine`：`CYCLES` 或 `EEVEE`（默认 `CYCLES`）。
  - `--samples`：采样数（Cycles：samples；EEVEE：taa_render_samples）。
  - `--use_gpu`：若启用，在 Cycles 中尝试切换 GPU（OPTIX/CUDA/HIP/METAL）并启用所有设备。
  - `--env_hdr`：HDRI 路径（.hdr/.exr），存在则作为环境贴图。
  - `--ground`：添加地面平面，Cycles 下作为 shadow catcher。
  - `--film_filmic`：启用 Filmic 视图变换；`--contrast` 设置对比度风格；`--exposure`/`--gamma` 控制曝光/伽马。
  - `--frame_start`/`--frame_end`：可选覆盖渲染帧范围。
  - `--video`/`--video_path`：直接输出 MP4（若构建不含 FFMPEG，此项会报错）。

## 备注
- `blender_render.py` 需在 Blender 内运行，普通 Python 环境无法导入 `bpy`。
- 本示例为教学目的的简化 1D 盒体碰撞，不含旋转与摩擦。若需更复杂的刚体/约束，请告知我以扩展。
 - 若 Blender 不支持 FFMPEG 导出，可先渲染 PNG，再使用 OpenCV 合成 MP4：
```powershell
python d:\ACG\MPM\render\frames_to_mp4.py --input d:\ACG\MPM\render\output\frames --out d:\ACG\MPM\render\output\demo.mp4 --fps 60
```