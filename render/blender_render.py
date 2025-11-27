# Blender rendering script: run inside Blender
# Usage example (PowerShell):
# "C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P d:\\ACG\\MPM\\render\\blender_render.py -- \
#   --sim d:\\ACG\\MPM\\render\\output\\boxes_sim.json \
#   --output d:\\ACG\\MPM\\render\\output\\frames --fps 60 --resolution 1280x720 \
#   --engine CYCLES --samples 128 --use_gpu --env_hdr d:\\HDRI\\studio.hdr --ground --film_filmic --contrast "Medium High Contrast" --exposure 0.5 \
#   --frame_start 1 --frame_end 60

import bpy
import sys
import json
import os
import math


def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    # minimal manual parser
    args = {
        "sim": None,
        "output": os.path.join("render", "output", "frames"),
        "fps": None,
        "resolution": "1280x720",
        "video": False,
        "video_path": os.path.join("render", "output", "demo.mp4"),
        # realism & control
        "engine": "CYCLES",            # CYCLES or EEVEE
        "samples": 128,
        "use_gpu": False,
        "env_hdr": None,                # path to .hdr/.exr
        "ground": False,
        "ground_color": (0.2, 0.2, 0.2),
        "ground_size": 40.0,
        "film_filmic": True,
        "contrast": "Medium High Contrast",
        "exposure": 0.0,
        "gamma": 1.0,
        # lights
        "three_point": True,
        "sun": False,
        # frame range override
        "frame_start": None,
        "frame_end": None,
    }
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--sim" and i + 1 < len(argv):
            args["sim"] = argv[i + 1]
            i += 2
        elif k == "--output" and i + 1 < len(argv):
            args["output"] = argv[i + 1]
            i += 2
        elif k == "--fps" and i + 1 < len(argv):
            args["fps"] = int(argv[i + 1])
            i += 2
        elif k == "--resolution" and i + 1 < len(argv):
            args["resolution"] = argv[i + 1]
            i += 2
        elif k == "--video":
            args["video"] = True
            i += 1
        elif k == "--video_path" and i + 1 < len(argv):
            args["video_path"] = argv[i + 1]
            i += 2
        elif k == "--engine" and i + 1 < len(argv):
            args["engine"] = argv[i + 1].upper()
            i += 2
        elif k == "--samples" and i + 1 < len(argv):
            args["samples"] = int(argv[i + 1])
            i += 2
        elif k == "--use_gpu":
            args["use_gpu"] = True
            i += 1
        elif k == "--env_hdr" and i + 1 < len(argv):
            args["env_hdr"] = argv[i + 1]
            i += 2
        elif k == "--ground":
            args["ground"] = True
            i += 1
        elif k == "--ground_size" and i + 1 < len(argv):
            args["ground_size"] = float(argv[i + 1])
            i += 2
        elif k == "--film_filmic":
            args["film_filmic"] = True
            i += 1
        elif k == "--contrast" and i + 1 < len(argv):
            args["contrast"] = argv[i + 1]
            i += 2
        elif k == "--exposure" and i + 1 < len(argv):
            args["exposure"] = float(argv[i + 1])
            i += 2
        elif k == "--gamma" and i + 1 < len(argv):
            args["gamma"] = float(argv[i + 1])
            i += 2
        elif k == "--three_point":
            args["three_point"] = True
            i += 1
        elif k == "--sun":
            args["sun"] = True
            i += 1
        elif k == "--frame_start" and i + 1 < len(argv):
            args["frame_start"] = int(argv[i + 1])
            i += 2
        elif k == "--frame_end" and i + 1 < len(argv):
            args["frame_end"] = int(argv[i + 1])
            i += 2
        else:
            i += 1
    return args


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


def configure_engine(args):
    scene = bpy.context.scene
    engine = args["engine"].upper()
    if engine not in ("CYCLES", "BLENDER_EEVEE", "EEVEE"):
        engine = "CYCLES"
    if engine == "EEVEE":
        engine = "BLENDER_EEVEE"

    scene.render.engine = engine
    w_h = args["resolution"].lower().split("x")
    if len(w_h) == 2:
        scene.render.resolution_x = int(w_h[0])
        scene.render.resolution_y = int(w_h[1])
    scene.render.fps = args["fps"] or scene.render.fps

    if engine == "CYCLES":
        cycles = scene.cycles
        cycles.samples = int(args["samples"]) if args["samples"] else 128
        cycles.use_adaptive_sampling = True
        cycles.use_denoising = True
        try:
            if args["use_gpu"]:
                prefs = bpy.context.preferences.addons['cycles'].preferences
                # Prefer OPTIX if available, else CUDA
                for dev_type in ("OPTIX", "CUDA", "HIP", "METAL"):
                    try:
                        prefs.compute_device_type = dev_type
                        break
                    except Exception:
                        continue
                prefs.get_devices()
                for d in prefs.devices:
                    d.use = True
                cycles.device = 'GPU'
        except Exception:
            pass
    else:
        ee = scene.eevee
        ee.taa_render_samples = int(args["samples"]) if args["samples"] else 64
        ee.use_gtao = True
        ee.use_bloom = True
        ee.use_ssr = True
        ee.use_ssr_refraction = True


def setup_world(args):
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    # clear
    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new('ShaderNodeOutputWorld')
    out.location = (400, 0)
    bg = nodes.new('ShaderNodeBackground')
    bg.location = (150, 0)
    links.new(bg.outputs['Background'], out.inputs['Surface'])

    if args.get("env_hdr") and os.path.exists(args["env_hdr"]):
        env = nodes.new('ShaderNodeTexEnvironment')
        env.location = (-200, 0)
        try:
            env.image = bpy.data.images.load(args["env_hdr"])  # may raise if invalid
            links.new(env.outputs['Color'], bg.inputs['Color'])
        except Exception:
            pass

    # Color management
    try:
        view = scene.view_settings
        if args.get("film_filmic", True):
            view.view_transform = 'Filmic'
        if args.get("contrast"):
            view.look = args["contrast"]
        view.exposure = float(args.get("exposure", 0.0))
        view.gamma = float(args.get("gamma", 1.0))
    except Exception:
        pass


def setup_lights(args):
    # Optional sun
    if args.get("sun"):
        bpy.ops.object.light_add(type="SUN", location=(6.0, 8.0, 6.0))
        sun = bpy.context.active_object
        sun.data.energy = 3.0

    # Three-point lighting with area lights
    if args.get("three_point", True):
        def add_area(loc, rot, size, energy):
            bpy.ops.object.light_add(type='AREA', location=loc, rotation=rot)
            area = bpy.context.active_object
            area.data.size = size
            area.data.energy = energy
            return area
        add_area((4.0, 4.0, 4.0), (math.radians(-35), 0, math.radians(45)), 3.0, 1000)
        add_area((-4.0, 2.5, 2.5), (math.radians(-20), 0, math.radians(-35)), 2.5, 500)
        add_area((0.0, 6.0, -2.5), (math.radians(-80), 0, 0), 2.0, 300)


def add_ground(args):
    if not args.get("ground"):
        return None
    size = float(args.get("ground_size", 40.0))
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        c = args.get("ground_color", (0.2, 0.2, 0.2))
        bsdf.inputs['Base Color'].default_value = (c[0], c[1], c[2], 1.0)
        bsdf.inputs['Roughness'].default_value = 0.8
        bsdf.inputs['Metallic'].default_value = 0.0
    plane.data.materials.append(mat)
    # Shadow catcher (Cycles only)
    try:
        if bpy.context.scene.render.engine == 'CYCLES':
            plane.cycles.is_shadow_catcher = True
    except Exception:
        pass
    return plane


def make_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs[0].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs[7].default_value = 0.5
    return mat


def look_at(obj, target):
    direction = (target[0] - obj.location.x, target[1] - obj.location.y, target[2] - obj.location.z)
    dx, dy, dz = direction
    # simple yaw-pitch from direction
    yaw = math.atan2(dx, dz)
    dist = math.sqrt(dx * dx + dz * dz)
    pitch = math.atan2(-dy, dist)
    obj.rotation_euler = (pitch, 0.0, yaw)


def setup_camera_and_light():
    bpy.ops.object.camera_add(location=(0.0, 3.0, 6.0))
    cam = bpy.context.active_object
    look_at(cam, (0.0, 0.5, 0.0))
    # Ensure the scene uses this camera for rendering
    bpy.context.scene.camera = cam
    return cam


def create_cubes(colors):
    cubes = []
    for i in range(2):
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        mat = make_material(f"Box{i}_Mat", colors[i])
        cube.data.materials.append(mat)
        cubes.append(cube)
    return cubes


def main():
    args = parse_args(sys.argv)
    if not args["sim"] or not os.path.exists(args["sim"]):
        raise FileNotFoundError(f"Simulation JSON not found: {args['sim']}")

    with open(args["sim"], "r", encoding="utf-8") as f:
        sim = json.load(f)

    fps = args["fps"] or sim.get("fps", 60)
    frames = sim.get("frames", [])
    colors = [frames[0]["boxes"][0].get("color", [0.2, 0.6, 0.9]), frames[0]["boxes"][1].get("color", [0.9, 0.4, 0.2])]

    clear_scene()
    setup_camera_and_light()
    configure_engine(args)
    setup_world(args)
    setup_lights(args)
    add_ground(args)
    cubes = create_cubes(colors)

    scene = bpy.context.scene
    scene.frame_start = int(args.get("frame_start") or 1)
    scene.frame_end = int(args.get("frame_end") or len(frames))
    scene.render.fps = fps

    if args["video"]:
        # Configure FFmpeg video output (H.264 in MP4)
        out_dir = os.path.dirname(args["video_path"]) or "."
        os.makedirs(out_dir, exist_ok=True)
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "VERYFAST"
        scene.render.ffmpeg.audio_codec = "NONE"
        scene.render.filepath = args["video_path"]
    else:
        os.makedirs(args["output"], exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = os.path.join(args["output"], "frame_")

    for fi, frame in enumerate(frames, start=1):
        scene.frame_set(fi)
        for bi, box in enumerate(frame["boxes"]):
            cubes[bi].location = tuple(box["pos"])
            cubes[bi].rotation_euler = tuple(box["rot"])
            # Incoming JSON "scale" stores BOX DIMENSIONS (2*half_extents).
            # Blender default cube is 2x2x2 at scale=1.0, so to achieve given dimensions D,
            # we should set object.scale = D / 2.
            dims = box["scale"]
            cubes[bi].scale = (dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5)
            cubes[bi].keyframe_insert(data_path="location", frame=fi)
            cubes[bi].keyframe_insert(data_path="rotation_euler", frame=fi)
            cubes[bi].keyframe_insert(data_path="scale", frame=fi)

    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
