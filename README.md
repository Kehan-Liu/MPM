# Fluid, Deformable Body and Rigid Body Simulator


## Environment Setup
Please run the following command to create a virtual environment and install the required packages:
```powershell
conda create -n mpm python=3.10
conda activate mpm
pip install -r requirements.txt
```
For windows users, you may need to replace `blender` in `render_frame.py` with the full path to your blender executable, e.g., `C:\Program Files\Blender Foundation\Blender 3.5\blender.exe`.

## Getting Started
Inside `demos/` folder, you can find several demo scripts showcasing different features of the simulator. You can run them directly after setting up the environment, or create your own demo. For example:
```powershell
python .\demos\test_bunny.py
```

## Rendering Simulation Output
You can change the background HDRI image by replacing the file at `assets\background.exr` and add material textures in `assets\assets.blend` with blender. Specify the settings in the `render/render_frame.py` script, then run the following command to render simulation outputs:
```bash
python render/render_demo.py demo_name --start start_frame --end end_frame
```
You can find the result in `render_output/demo_name/`.

We also provide two other rendering scripts for specific use cases:
- `render/render_demo_blender.py`: Render with blender, less costly but noisy reconstruction.
- `render/render_granular_demo.py`: Render granular materials.

## Try the Realtime Interaction Demo
You can run the realtime interaction demo with the following command:
```powershell
python .\demos\realtime_demo.py
```
