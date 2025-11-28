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
import datetime
LOG_PATH = None
DEBUG = False

def log(msg):
    if not DEBUG:
        return
    try:
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        s = f"[BR] {ts} {msg}"
        print(s)
        if LOG_PATH:
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(s + "\n")
    except Exception:
        pass


def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    # minimal manual parser
    args = {
        "sim": None,
        "mesh_sequence": None,        # directory containing frame_XXXX_bodyN.obj files
        "output": os.path.join("render", "output", "frames"),
        # alternative source: per-frame mesh files (exported by rigid sim)
        "rigid_frames_dir": None,      # directory containing frame_XXXX_bodyN.obj
        "body_colors": None,           # e.g. "0:0.2,0.6,0.9;1:0.9,0.4,0.2"
        "fps": None,
        "resolution": "1280x720",
        "video": False,
        "video_path": os.path.join("render", "output", "demo.mp4"),
        # realism & control
        "engine": "CYCLES",            # CYCLES or EEVEE
        "samples": 128,
        "use_gpu": False,
        "gpu_type": None,               # OPTIX/CUDA/HIP/METAL; auto if None
        "gpu_only": False,              # if True do not enable CPU device
        "import_meshes": None,          # path list string; supports ; or , separated or a directory
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
        # debug
        "debug": False,
    }
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--sim" and i + 1 < len(argv):
            args["sim"] = argv[i + 1]
            i += 2
        elif k == "--mesh_sequence" and i + 1 < len(argv):
            args["mesh_sequence"] = argv[i + 1]
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
        elif k == "--rigid_frames_dir" and i + 1 < len(argv):
            args["rigid_frames_dir"] = argv[i + 1]
            i += 2
        elif k == "--body_colors" and i + 1 < len(argv):
            args["body_colors"] = argv[i + 1]
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
        elif k == "--gpu_type" and i + 1 < len(argv):
            args["gpu_type"] = argv[i + 1].upper()
            i += 2
        elif k == "--gpu_only":
            args["gpu_only"] = True
            i += 1
        elif k == "--import_meshes" and i + 1 < len(argv):
            args["import_meshes"] = argv[i + 1]
            i += 2
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
        elif k == "--debug":
            args["debug"] = True
            i += 1
        else:
            i += 1
    # allow alias
    if args.get("rigid_frames_dir") and not args.get("mesh_sequence"):
        args["mesh_sequence"] = args["rigid_frames_dir"]
    return args


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


def clear_objects_except(keep_types={"CAMERA", "LIGHT"}):
    # Remove all objects except specified types
    for obj in list(bpy.data.objects):
        try:
            if obj.type not in keep_types:
                bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass


def _split_paths_list(path_list_str):
    if not path_list_str:
        return []
    s = path_list_str.strip()
    # accept ; or , as separators
    parts = []
    for sep in (';', ','):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            break
    if not parts:
        parts = [s]
    # If it's a directory, expand supported files inside it
    out = []
    for p in parts:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if os.path.splitext(fn)[1].lower() in ('.obj', '.fbx', '.stl', '.ply', '.gltf', '.glb'):
                    out.append(os.path.join(p, fn))
        else:
            out.append(p)
    return out


def _import_one_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.context.scene.objects)
    imported_ok = False
    # Try a few common importers depending on extension
    try:
        if ext == '.obj':
            # Blender 3.x: import_scene.obj; 4.x/5.x may also have wm.obj_import
            try:
                bpy.ops.import_scene.obj(filepath=path)
            except Exception:
                bpy.ops.wm.obj_import(filepath=path)
            imported_ok = True
        elif ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=path)
            imported_ok = True
        elif ext == '.stl':
            try:
                bpy.ops.import_mesh.stl(filepath=path)
            except Exception:
                bpy.ops.wm.stl_import(filepath=path)
            imported_ok = True
        elif ext in ('.gltf', '.glb'):
            try:
                bpy.ops.import_scene.gltf(filepath=path)
            except Exception:
                bpy.ops.wm.gltf_import(filepath=path)
            imported_ok = True
        elif ext == '.ply':
            try:
                bpy.ops.import_mesh.ply(filepath=path)
            except Exception:
                bpy.ops.wm.ply_import(filepath=path)
            imported_ok = True
    except Exception as e:
        print(f"[IMPORT] Failed to import {path}: {e}")

    after = set(bpy.context.scene.objects)
    new_objs = [o for o in after - before]
    if not imported_ok or not new_objs:
        print(f"[IMPORT] No objects imported from {path}")
        return None
    if len(new_objs) == 1:
        parent = new_objs[0]
    else:
        # Create an empty as parent to move/animate the group as one
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        parent = bpy.context.active_object
        parent.name = f"Imported_{os.path.basename(path)}"
        for o in new_objs:
            o.parent = parent
    print(f"[IMPORT] Imported {path} as object '{parent.name}' with {len(new_objs)} children")
    return parent


def import_mesh_files(path_list_str):
    paths = _split_paths_list(path_list_str)
    parents = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[IMPORT] Path not found: {p}")
            continue
        par = _import_one_mesh(p)
        if par:
            parents.append(par)
    return parents


def _list_rigid_frame_files(dir_path, frame_idx):
    files = []
    if not os.path.isdir(dir_path):
        return files
    target = f"frame_{frame_idx:04d}_"
    for fn in sorted(os.listdir(dir_path)):
        if not fn.lower().endswith('.obj'):
            continue
        if fn.startswith(target) and '_body' in fn:
            files.append(os.path.join(dir_path, fn))
    # sort by body index
    def body_index(p):
        name = os.path.basename(p)
        try:
            idx = name.split('_body')[-1].split('.')[0]
            return int(idx)
        except Exception:
            return 1e9
    files.sort(key=body_index)
    return files


def _all_frame_indices(dir_path):
    if not os.path.isdir(dir_path):
        return []
    idxs = set()
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith('.obj'):
            continue
        if fn.startswith('frame_') and '_body' in fn:
            try:
                s = fn.split('_')[1]  # XXXX
                idxs.add(int(s))
            except Exception:
                pass
    return sorted(list(idxs))


def _parse_body_colors(spec):
    # spec: "0:r,g,b;1:r,g,b"
    mapping = {}
    if not spec:
        return mapping
    for seg in spec.split(';'):
        seg = seg.strip()
        if not seg:
            continue
        if ':' not in seg:
            continue
        k, v = seg.split(':', 1)
        try:
            bi = int(k)
            c = tuple(float(x) for x in v.split(',')[:3])
            mapping[bi] = c
        except Exception:
            continue
    return mapping


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
                desired = args.get("gpu_type")
                dev_order = ["OPTIX", "CUDA", "HIP", "METAL"]
                if desired:
                    dev_order = [desired] + [d for d in dev_order if d != desired]
                selected = None
                for dev_type in dev_order:
                    try:
                        prefs.compute_device_type = dev_type
                        selected = dev_type
                        break
                    except Exception:
                        continue
                prefs.get_devices()
                any_enabled = False
                for d in prefs.devices:
                    # Enable only selected type devices; optionally skip CPU
                    if d.type == 'CPU' and args.get("gpu_only"):
                        d.use = False
                        continue
                    # Some builds list the same GPU twice under different backends; keep only those matching selected backend
                    if selected and d.type != selected and d.type != 'CPU':
                        d.use = False
                        continue
                    d.use = True
                    any_enabled = True or any_enabled
                if any_enabled:
                    cycles.device = 'GPU'
                # Diagnostics printout
                print("[GPU] Requested:", desired, "Selected:", selected, "DeviceMode:", cycles.device)
                for d in prefs.devices:
                    print(f"[GPU] Device: {d.name} Type={d.type} Use={d.use}")
        except Exception as e:
            print("[GPU] Failed to configure GPU, falling back to CPU:", e)
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


def assign_material_recursive(root_obj, mat):
    # Assign material to root if it's a mesh
    try:
        if hasattr(root_obj, 'type') and root_obj.type == 'MESH':
            if hasattr(root_obj, 'data') and hasattr(root_obj.data, 'materials'):
                # Replace existing materials with the provided one
                root_obj.data.materials.clear()
                root_obj.data.materials.append(mat)
    except Exception as e:
        print(f"[MATERIAL] Failed to assign material to {root_obj.name}: {e}")
    # Recursively assign to children
    for ch in getattr(root_obj, 'children', []) or []:
        assign_material_recursive(ch, mat)


def main():
    print("[Blender Render] Starting")
    args = parse_args(sys.argv)
    try:
        # set debug flag
        global DEBUG
        DEBUG = bool(args.get("debug"))
        if DEBUG:
            print("[BR] Debug logging enabled")
        # init logfile
        out_dir = args.get("output") or os.path.join("render", "output", "frames")
        os.makedirs(out_dir, exist_ok=True)
        global LOG_PATH
        LOG_PATH = os.path.join(out_dir, 'blender_render.log')
        log(f"Args keys: {list(args.keys())}")
        if args.get("mesh_sequence"):
            log(f"Mode=mesh_sequence Dir={args.get('mesh_sequence')}")
        else:
            log(f"Mode=json Sim={args.get('sim')}")
    except Exception:
        pass
    # Mesh sequence mode ---------------------------------
    if args.get("mesh_sequence"):
        render_mesh_sequence(args)
        return

    # JSON keyframe mode ---------------------------------
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
    imported = []
    if args.get("import_meshes"):
        try:
            imported = import_mesh_files(args["import_meshes"])
        except Exception as e:
            print("[IMPORT] Failed to import meshes:", e)
            imported = []
    if imported:
        objects = imported
        while len(objects) < 2:
            extra = create_cubes([(0.8,0.8,0.8), (0.6,0.6,0.6)])
            objects.extend(extra)
            break
    else:
        objects = create_cubes(colors)
    # Apply color materials
    try:
        for bi, col in enumerate(colors):
            if bi >= len(objects):
                break
            mat = make_material(f"Box{bi}_Mat", col)
            assign_material_recursive(objects[bi], mat)
    except Exception as e:
        print("[MATERIAL] Color assignment failed:", e)
    scene = bpy.context.scene
    scene.frame_start = int(args.get("frame_start") or 1)
    scene.frame_end = int(args.get("frame_end") or len(frames))
    scene.render.fps = fps
    if args["video"]:
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
            obj = objects[bi] if bi < len(objects) else objects[-1]
            obj.location = tuple(box["pos"])
            obj.rotation_euler = tuple(box["rot"])
            dims = box["scale"]
            obj.scale = (dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5)
            obj.keyframe_insert(data_path="location", frame=fi)
            obj.keyframe_insert(data_path="rotation_euler", frame=fi)
            obj.keyframe_insert(data_path="scale", frame=fi)
    bpy.ops.render.render(animation=True)


def render_mesh_sequence(args):
    seq_dir = args["mesh_sequence"]
    if not os.path.isdir(seq_dir):
        log(f"SEQ: directory not found: {seq_dir}")
        return
    log(f"SEQ: scanning directory: {seq_dir}")
    files = [f for f in os.listdir(seq_dir) if f.lower().endswith('.obj') and f.startswith('frame_')]
    frame_map = {}
    log(f"SEQ: found {len(files)} mesh files in {seq_dir}")
    for fn in files:
        base = os.path.splitext(fn)[0]
        parts = base.split('_')
        if len(parts) < 3:
            continue
        try:
            frame_idx = int(parts[1])
        except Exception:
            continue
        body_tag = parts[-1]
        if not body_tag.startswith('body'):
            continue
        try:
            body_idx = int(body_tag[4:])
        except Exception:
            body_idx = 0
        frame_map.setdefault(frame_idx, {})[body_idx] = os.path.join(seq_dir, fn)
    if not frame_map:
        log(f"SEQ: no frame_XXXX_bodyN.obj files found in {seq_dir}; exit")
        return
    colors_map = {}
    # Optional per-body static colors override
    static_colors = _parse_body_colors(args.get("body_colors")) if args.get("body_colors") else {}
    json_path = os.path.join(seq_dir, 'frames.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                jdata = json.load(jf)
            for fr in jdata.get('frames', []):
                idx = fr.get('frame')
                cols = [b.get('color', [1,1,1]) for b in fr.get('bodies', [])]
                colors_map[idx] = cols
        except Exception as e:
            print('[SEQ] Failed reading frames.json:', e)
    clear_scene()
    setup_camera_and_light()
    configure_engine(args)
    setup_world(args)
    setup_lights(args)
    add_ground(args)
    scene = bpy.context.scene
    if args.get('fps'):
        try:
            scene.render.fps = int(args['fps'])
        except Exception:
            pass
    scene.render.image_settings.file_format = 'PNG'
    os.makedirs(args['output'], exist_ok=True)
    base_objs = set(bpy.context.scene.objects)
    frame_indices = sorted(frame_map.keys())
    start = args.get('frame_start') or frame_indices[0]
    end = args.get('frame_end') or frame_indices[-1]
    log(f"SEQ: frame range: {start}..{end} (available {frame_indices[0]}..{frame_indices[-1]})")
    for frame_idx in frame_indices:
        if frame_idx < start or frame_idx > end:
            continue
        log(f"SEQ: preparing frame {frame_idx}")
        to_delete = [o for o in bpy.context.scene.objects if o not in base_objs]
        bpy.ops.object.select_all(action='DESELECT')
        for o in to_delete:
            o.select_set(True)
        if to_delete:
            bpy.ops.object.delete(use_global=False)
        body_files = frame_map[frame_idx]
        ordered = [body_files[k] for k in sorted(body_files.keys())]
        for bi, path in enumerate(ordered):
            parent = _import_one_mesh(path)
            if not parent:
                log(f"SEQ: import failed for {path}")
                continue
            # choose color: frames.json > static mapping > no material change
            col = None
            cols = colors_map.get(frame_idx)
            if cols and bi < len(cols):
                col = cols[bi]
            elif bi in static_colors:
                col = static_colors[bi]
            if col is not None:
                mat = make_material(f"SeqBody{bi}_Mat", col)
                assign_material_recursive(parent, mat)
        scene.frame_set(frame_idx)
        scene.render.filepath = os.path.join(args['output'], f'frame_{frame_idx:04d}.png')
        log(f"SEQ: rendering -> {scene.render.filepath}")
        bpy.ops.render.render(write_still=True)
        log(f"SEQ: rendered frame {frame_idx} -> {scene.render.filepath}")


if __name__ == "__main__":
    try:
        print("[BR] script entry")
        print("[BR] sys.argv:", sys.argv)
        main()
        print("[BR] script completed")
    except Exception as e:
        import traceback
        msg = f"[BR] UNCAUGHT ERROR: {e}\n{traceback.format_exc()}"
        try:
            print(msg)
        except Exception:
            pass
        try:
            if 'LOG_PATH' in globals() and LOG_PATH:
                with open(LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
        except Exception:
            pass
