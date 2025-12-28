import bpy
import sys
import os
import glob
import mathutils

# --- Configuration ---
ASSETS_PATH = os.path.join("assets", "assets.blend")

# Sand (granular) rendering settings
SAND_PARTICLE_RADIUS = 0.0032  # visual radius in Blender units
SAND_SUBDIVISIONS = 1  # ico sphere subdivisions (0-2 recommended for speed)

# Scene Settings (tuned for [0,1]^3)
CAMERA_POS = (2.2, 2.2, 1.2)
CAMERA_LOOKAT = (0.5, 0.5, 0.3)
LIGHT_POS = (5.0, 5.0, 10.0)
LIGHT_ENERGY = 5.0
HDRI_PATH = os.path.join("assets", "background.exr")


def ensure_material(
    name: str, base_color=(0.8, 0.8, 0.8, 1.0), roughness=0.6, metallic=0.0
):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = base_color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def cleanup_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def setup_camera():
    bpy.ops.object.camera_add(location=CAMERA_POS)
    cam = bpy.context.active_object
    cam.name = "RenderCamera"

    direction = mathutils.Vector(CAMERA_LOOKAT) - mathutils.Vector(CAMERA_POS)
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = cam


def setup_lighting():
    light_data = bpy.data.lights.new(name="SunLight", type="SUN")
    light_data.energy = LIGHT_ENERGY
    light_obj = bpy.data.objects.new(name="SunLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = LIGHT_POS

    direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector(LIGHT_POS)
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_obj.rotation_euler = rot_quat.to_euler()


def setup_environment():
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    node_bg = nodes.new(type="ShaderNodeBackground")
    node_out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(node_bg.outputs["Background"], node_out.inputs["Surface"])

    if HDRI_PATH and os.path.exists(HDRI_PATH):
        node_env = nodes.new(type="ShaderNodeTexEnvironment")
        try:
            img = bpy.data.images.load(HDRI_PATH)
            node_env.image = img
            links.new(node_env.outputs["Color"], node_bg.inputs["Color"])
        except Exception:
            node_bg.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1.0)
    else:
        node_bg.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1.0)


def setup_ground():
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground"

    mat = ensure_material(
        "GroundMaterial", base_color=(0.5, 0.5, 0.5, 1.0), roughness=1.0, metallic=0.0
    )
    if plane.data.materials:
        plane.data.materials[0] = mat
    else:
        plane.data.materials.append(mat)


def import_obj(filepath: str):
    bpy.ops.object.select_all(action="DESELECT")

    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=filepath, forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward="Y", axis_up="Z")

    sel = bpy.context.selected_objects
    if not sel:
        return None
    bpy.context.view_layer.objects.active = sel[0]
    if len(sel) > 1:
        bpy.ops.object.join()
    return bpy.context.active_object


def import_ply(filepath: str):
    bpy.ops.object.select_all(action="DESELECT")

    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=filepath)
    else:
        bpy.ops.import_mesh.ply(filepath=filepath)

    sel = bpy.context.selected_objects
    if not sel:
        return None
    bpy.context.view_layer.objects.active = sel[0]
    if len(sel) > 1:
        bpy.ops.object.join()
    return bpy.context.active_object


def load_rigid_meshes(rigid_dir: str, frame_str: str):
    pattern = os.path.join(rigid_dir, f"rigid_body_*_frame_{frame_str}.obj")
    mesh_files = sorted(glob.glob(pattern))
    if not mesh_files:
        print(f"Warning: no rigid meshes found: {pattern}")
        return

    # Different materials per rigid ball
    rb_mats = [
        ensure_material("RigidBall_0", base_color=(0.8, 0.2, 0.2, 1.0), roughness=0.4),
        ensure_material("RigidBall_1", base_color=(0.2, 0.8, 0.2, 1.0), roughness=0.4),
        ensure_material("RigidBall_2", base_color=(0.2, 0.2, 0.8, 1.0), roughness=0.4),
        ensure_material("RigidBall_3", base_color=(0.8, 0.8, 0.2, 1.0), roughness=0.4),
    ]

    for mesh_path in mesh_files:
        obj = import_obj(mesh_path)
        if obj is None:
            print(f"Warning: failed to import {mesh_path}")
            continue

        obj.name = os.path.splitext(os.path.basename(mesh_path))[0]

        # Parse rb id from filename: rigid_body_0000_frame_0000.obj
        rb_id = 0
        base = os.path.basename(mesh_path)
        try:
            rb_id = int(base.split("rigid_body_")[1].split("_frame_")[0])
        except Exception:
            rb_id = 0

        mat = rb_mats[rb_id % len(rb_mats)]
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        for poly in obj.data.polygons:
            poly.use_smooth = True


def add_sand_particle_instances(ply_path: str):
    sand_obj = import_ply(ply_path)
    if sand_obj is None:
        print(f"Warning: failed to import sand PLY: {ply_path}")
        return

    sand_obj.name = "SandParticles"

    sand_mat = ensure_material(
        "SandMaterial",
        base_color=(0.20, 0.55, 0.95, 1.0),
        roughness=0.7,
        metallic=0.0,
    )

    # Geometry Nodes: Mesh (vertices) -> Mesh to Points -> Instance on Points (IcoSphere)
    mod = sand_obj.modifiers.new(name="SandInstances", type="NODES")
    group = bpy.data.node_groups.new("SandInstanceGroup", "GeometryNodeTree")
    mod.node_group = group

    # Interface (Blender 4.x)
    if hasattr(group, "interface"):
        group.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        group.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
    else:
        group.inputs.new("NodeSocketGeometry", "Geometry")
        group.outputs.new("NodeSocketGeometry", "Geometry")

    nodes = group.nodes
    links = group.links

    in_node = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")

    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.mode = "VERTICES"

    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = SAND_PARTICLE_RADIUS
    ico.inputs["Subdivisions"].default_value = SAND_SUBDIVISIONS

    set_mat = nodes.new("GeometryNodeSetMaterial")
    if sand_mat:
        set_mat.inputs["Material"].default_value = sand_mat

    inst_on_points = nodes.new("GeometryNodeInstanceOnPoints")

    # Wire
    links.new(in_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(ico.outputs["Mesh"], set_mat.inputs["Geometry"])
    links.new(mesh_to_points.outputs["Points"], inst_on_points.inputs["Points"])
    links.new(set_mat.outputs["Geometry"], inst_on_points.inputs["Instance"])
    # IMPORTANT: do NOT Realize Instances here.
    # With millions of particles, realizing would explode geometry and become CPU-bound.
    links.new(inst_on_points.outputs["Instances"], out_node.inputs["Geometry"])


def setup_renderer():
    scene = bpy.context.scene

    # In background mode (-b), Eevee may fall back to software OpenGL (CPU/llvmpipe).
    # Cycles GPU is usually the fastest/most reliable headless path.
    if bpy.app.background:
        scene.render.engine = "CYCLES"
    else:
        for engine in ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"]:
            try:
                scene.render.engine = engine
                break
            except Exception:
                continue

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = 32
        scene.cycles.use_adaptive_sampling = True

        try:
            cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
            cycles_prefs.refresh_devices()
            gpu_found = False
            for device_type in ["OPTIX", "CUDA", "HIP", "METAL"]:
                try:
                    cycles_prefs.compute_device_type = device_type
                except Exception:
                    continue

                devices_of_type = [
                    d for d in cycles_prefs.devices if d.type == device_type
                ]
                if devices_of_type:
                    for device in cycles_prefs.devices:
                        device.use = device.type == device_type
                    gpu_found = True
                    break

            scene.cycles.device = "GPU" if gpu_found else "CPU"
            print(f"Render engine: CYCLES, device: {scene.cycles.device}")
        except Exception as e:
            print(f"GPU setup failed or not supported: {e}")
    else:
        print(f"Render engine: {scene.render.engine}")


def render_frame(ply_file: str, rigid_dir: str, frame_str: str, output_file: str):
    # Load base scene
    if os.path.exists(ASSETS_PATH):
        bpy.ops.wm.open_mainfile(filepath=ASSETS_PATH)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    cleanup_scene()
    setup_camera()
    setup_lighting()
    setup_environment()
    setup_ground()

    # Content
    load_rigid_meshes(rigid_dir, frame_str)
    add_sand_particle_instances(ply_file)

    # Render
    setup_renderer()
    bpy.context.scene.render.filepath = output_file
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    # CLI Args: ... -- <particle_ply> <rigid_dir> <frame_str> <output_png>
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1 :]
        if len(args) >= 4:
            render_frame(args[0], args[1], args[2], args[3])
        else:
            print(
                "Usage: blender -b -P render/render_granular.py -- <particle_ply> <rigid_dir> <frame_str> <output_png>"
            )
