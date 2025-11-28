# Blender 渲染管线（render/）

此目录提供一个最简固体-固体（两立方体）碰撞的模拟与 Blender 批量渲染脚本。模拟在纯 Python 中生成每帧位姿，渲染脚本在 Blender 中读取并生成 PNG 帧。

## 目录结构
- `simulate_boxes.py`：两盒体沿 X 轴相向运动并发生弹性碰撞，输出 JSON。
- `blender_render.py`：Blender 内运行，读取模拟 JSON，创建场景并关键帧动画后渲染 PNG。
- `output/`：默认输出目录，包含 `boxes_sim.json` 与渲染帧。

## 运行步骤（Windows PowerShell）

1) 生成模拟数据（JSON）：
```powershell
python render\simulate_boxes.py --frames 240 --dt 0.016 --restitution 0.6 --speed0 1.5 --speed1 1.0 --out render\output\boxes_sim.json
```

2) 使用 Blender 无界面渲染：请替换为你的 Blender 安装路径。以下示例启用更真实的渲染设置（Cycles、GPU、地面、Filmic）。

注意：PowerShell 中不能用反斜杠 `\` 做续行（那是 Bash 的写法），应使用反引号 <code>`</code> 作为行尾续行符，或者写成单行。

多行（建议可读性）：（一定要在前面加调用运算符 `&`）
```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P render\blender_render.py -- `
  --sim .\render\output\boxes_sim.json `
  --output .\render\output\frames --fps 60 --resolution 1280x720 `
  --engine CYCLES --samples 128 --use_gpu --gpu_type OPTIX `
  --ground --film_filmic --contrast "Medium High Contrast" --exposure 0.5
```

单行（不易出错）：
```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P F:\ACG\MPM\render\blender_render.py -- --sim F:\ACG\MPM\render\output\boxes_sim.json --output F:\ACG\MPM\render\output\frames --fps 60 --resolution 1280x720 --engine CYCLES --samples 128 --use_gpu --ground --film_filmic --contrast "Medium High Contrast" --exposure 0.5
```

如果你不想每次写完整路径，可先定义变量：
```powershell
$blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
& $blender -b -P render\blender_render.py -- --sim render\output\boxes_sim.json --output render\output\frames --fps 60 --resolution 1280x720 --engine CYCLES --samples 128 --use_gpu --ground --film_filmic --contrast "Medium High Contrast" --exposure 0.5
```

或用环境变量（例如你设置了 `$env:BLENDER_EXE`）：
```powershell
& $env:BLENDER_EXE -b -P render\blender_render.py -- --sim render\output\boxes_sim.json --output render\output\frames --fps 60 --resolution 1280x720 --engine CYCLES --samples 128 --use_gpu --ground --film_filmic --contrast "Medium High Contrast" --exposure 0.5
```

常见错误来源：
- 把 Bash 的续行符 `\` 直接复制到 PowerShell；PowerShell 会把独立的 `\` 当成路径开头的字符，后一行参数被当作新的语句，导致 `-b` / `-P` 等被解析为“意外的标记”。
- 在复制时混入了不可见字符（从网页复制时可能发生）。若仍报错，先粘贴到纯文本编辑器再复制。
- 末行不需要反引号；仅在需要续行的行尾放反引号并紧跟换行（反引号后面不能有空格）。

### GPU 使用说明与性能
- `--use_gpu`：启用 Cycles GPU 渲染。
- `--gpu_type`：指定首选计算后端：`OPTIX` / `CUDA` / `HIP` / `METAL`。不指定则按此顺序自动尝试。
- `--gpu_only`：启用后仅使用 GPU 设备，不启用 CPU；可避免 CPU 分配任务导致 GPU 低占用。
- 控制台会打印 `[GPU] Requested/Selected/DeviceMode` 及每个设备，用于诊断。

#### 为什么 GPU 占用可能很低
1. 场景非常简单（两个立方体 + 低复杂材质），每帧耗时短，GPU 峰值不明显。
2. 同时启用 CPU 与多种后端设备（CUDA+OPTIX）会产生调度与同步，降低单 GPU 峰值。
3. 分辨率与采样数较低（例如 1280x720, 128 samples），GPU 很快完成工作，剩余时间等待 I/O 写 PNG。
4. 自适应采样 + 去噪减少实际有效路径数。

#### 提升 GPU 利用率的做法
- 加大分辨率或采样：如 `--resolution 1920x1080 --samples 512`。
- 使用 `--gpu_only` 避免 CPU 参与：减少调度开销。
- 固定单一后端：如 `--gpu_type OPTIX`（NVIDIA）或 `--gpu_type CUDA` 做对比测试。
- 关闭自适应采样与去噪（可在脚本里定制开关，如果后续需要可以添加参数）。
- 增加场景复杂度（更多物体 / 细分 / 纹理 / HDRI）。

#### 诊断最小示例
```powershell
& $blender -b -P render\blender_render.py -- --sim render\output\boxes_sim.json --use_gpu --gpu_type OPTIX --gpu_only --frame_start 1 --frame_end 5 --samples 64
```
出现 `DeviceMode: GPU` 且只列出 GPU 说明已成功，仅列 CPU 需检查驱动或 Blender Preferences > System。

### 导入自定义 Mesh
- 你可以通过 `--import_meshes` 指定一个或多个网格文件，脚本会按提供顺序导入并用来驱动每个刚体（若数量不足，将用默认立方体补齐）。
- 支持格式：`.obj` / `.fbx` / `.stl` / `.ply` / `.gltf` / `.glb`。
- 参数格式支持：
  - 用分号或逗号分隔多个文件：`--import_meshes "render\assets\a.obj;render\assets\b.obj"`
  - 传入目录：`--import_meshes render\assets`（将自动导入目录中所有支持的文件，按文件名排序）

示例：
```powershell
$blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
& $blender -b -P render\blender_render.py -- `
  --sim render\output\boxes_sim.json --import_meshes "render\assets\boxA.obj;render\assets\boxB.obj" `
  --use_gpu --gpu_type OPTIX --gpu_only --resolution 1280x720 --samples 256 --ground --film_filmic
```

注意：导入的文件可能包含多个对象，脚本会自动创建一个空对象作为父节点，并将组整体按模拟数据进行位姿/缩放动画。

### 生成简单 Mesh 文件
你也可以用内置脚本快速生成一个 OBJ 立方体：
```powershell
python render\make_mesh.py --shape cube --size 2.0 --out render\assets\demo_cube.obj
```
生成后即可用 `--import_meshes render\assets\demo_cube.obj` 参与渲染。

### 渲染逐帧 Mesh 目录（刚体模拟导出的 OBJ）
如果你使用 `simulators/solid/rigid_body_sim.py` 逐帧导出了 `frame_XXXX_bodyN.obj`，可以直接按目录渲染：
```powershell
$blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
& $blender -b -P render\blender_render.py -- `
  --mesh_sequence render\output\rigid_frames `
  --output render\output\frames --fps 60 --resolution 1280x720 `
  --engine CYCLES --samples 128 --use_gpu --gpu_type OPTIX --gpu_only --ground --film_filmic
```
说明：
- 目录内应包含若干 `frame_0000_body0.obj`, `frame_0000_body1.obj`, `frame_0001_body0.obj` ...
- 若目录内有 `frames.json`，脚本会读取每帧 `color` 并给对应 body 赋材质；否则可使用：
  - `--body_colors "0:0.2,0.6,0.9;1:0.9,0.4,0.2"` 为 body 索引统一上色。
- 你也可以用别名 `--rigid_frames_dir` 传同一目录。

渲染结束后，PNG 帧将位于 `render\output\frames`。
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
    - `--mesh_sequence`：指定一个目录，其中包含 `frame_XXXX_bodyN.obj` 形式的刚体模拟导出网格；脚本将按帧导入并逐帧渲染（不使用关键帧）。

  ### 渲染每帧网格序列（RigidBody 输出）
  假设使用 `rigid_body_sim.py` 输出在 `render/output/rigid_frames`：
  目录包含：`frame_0000_body0.obj`, `frame_0000_body1.obj`, ... 以及可选 `frames.json`（含颜色）。

  运行：
  ```powershell
  $blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
  & $blender -b -P render\blender_render.py -- `
    --mesh_sequence render\output\rigid_frames `
    --output render\output\rigid_frames_render `
    --resolution 1280x720 --engine CYCLES --samples 128 --use_gpu --gpu_type OPTIX --gpu_only --ground --film_filmic
  ```
  说明：
  - 不需要 `--sim`；检测到 `--mesh_sequence` 将进入“逐帧导入渲染”模式。
  - 每帧之前会清理上一帧导入的对象，避免内存膨胀。
  - 若存在 `frames.json`，将使用其中 `bodies[i].color` 赋材质；否则保持原 mesh 材质或默认灰色。
  - 输出 PNG 命名：`frame_XXXX.png`。

## 备注
- `blender_render.py` 需在 Blender 内运行，普通 Python 环境无法导入 `bpy`。
- 本示例为教学目的的简化 1D 盒体碰撞，不含旋转与摩擦。若需更复杂的刚体/约束，请告知我以扩展。
 - 若 Blender 不支持 FFMPEG 导出，可先渲染 PNG，再使用 OpenCV 合成 MP4：
```powershell
python render\frames_to_mp4.py --input .\render\output\frames --out .\render\output\demo.mp4 --fps 60
```