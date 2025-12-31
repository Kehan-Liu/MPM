import bpy
import json
import sys
import os
import glob
import mathutils

# --- Configuration ---
ASSETS_PATH = "assets.blend"  # Path to your pre-made file
MPM_MATERIAL_NAME = "QieGao"  # Must match a material name in assets.blend
RB_MATERIAL_NAME = "RigidMat"  # Must match a material name in assets.blend

# Scene Settings
CAMERA_POS = (-2.0, -2.0, 1.2)
CAMERA_LOOKAT = (0.5, 0.5, 0.3)
LIGHT_POS = (5.0, 5.0, 10.0)
LIGHT_ENERGY = 5.0
HDRI_PATH = "assets/background.exr"  # Set to None or invalid path to skip


def load_rigid_bodies(rigid_dir, frame_str):
    """Loads rigid body meshes for the given frame."""
    if not os.path.exists(rigid_dir):
        print(f"Warning: {rigid_dir} not found")
        return

    pattern = os.path.join(rigid_dir, f"rigid_body_*_frame_{frame_str}.obj")
    mesh_files = sorted(glob.glob(pattern))

    mat = bpy.data.materials.get(RB_MATERIAL_NAME)

    for mesh_path in mesh_files:
        bpy.ops.object.select_all(action="DESELECT")

        # Use the new OBJ importer for Blender 4.0+
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis="Y", up_axis="Z")
        else:
            # Fallback for older Blender versions
            bpy.ops.import_scene.obj(filepath=mesh_path, axis_forward="Y", axis_up="Z")

        selected_objs = bpy.context.selected_objects
        if not selected_objs:
            print(f"Warning: No objects imported from {mesh_path}")
            continue

        # If multiple objects are imported, join them or pick the first one
        bpy.context.view_layer.objects.active = selected_objs[0]
        if len(selected_objs) > 1:
            bpy.ops.object.join()

        obj = bpy.context.active_object
        obj.name = os.path.splitext(os.path.basename(mesh_path))[0]

        if mat:
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)

        # Smooth shading
        for poly in obj.data.polygons:
            poly.use_smooth = True


def load_mpm_mesh(obj_path):
    """Imports OBJ mesh for MPM fluid."""
    if not os.path.exists(obj_path):
        print(f"Warning: {obj_path} not found")
        return

    bpy.ops.object.select_all(action="DESELECT")

    # Use the new OBJ importer for Blender 4.0+
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="Y", up_axis="Z")
    else:
        # Fallback for older Blender versions
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward="Y", axis_up="Z")

    selected_objs = bpy.context.selected_objects
    if not selected_objs:
        print(f"Warning: No objects imported from {obj_path}")
        return

    # If multiple objects are imported, join them or pick the first one
    bpy.context.view_layer.objects.active = selected_objs[0]
    if len(selected_objs) > 1:
        bpy.ops.object.join()

    obj = bpy.context.active_object
    obj.name = "MPM_Fluid"

    # Apply Material
    mat = bpy.data.materials.get(MPM_MATERIAL_NAME)
    if mat:
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    # Smooth shading
    for poly in obj.data.polygons:
        poly.use_smooth = True


def cleanup_scene():
    """Removes existing cameras and lights."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type in {"CAMERA", "LIGHT"}:
            obj.select_set(True)
    bpy.ops.object.delete()


def setup_camera():
    """Sets up the camera looking at a target."""
    bpy.ops.object.camera_add(location=CAMERA_POS)
    cam = bpy.context.active_object
    cam.name = "RenderCamera"

    # Look at target
    direction = mathutils.Vector(CAMERA_LOOKAT) - mathutils.Vector(CAMERA_POS)
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = cam


def setup_lighting():
    """Sets up a sun light."""
    # Create light datablock
    light_data = bpy.data.lights.new(name="SunLight", type="SUN")
    light_data.energy = LIGHT_ENERGY

    # Create object
    light_obj = bpy.data.objects.new(name="SunLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    light_obj.location = LIGHT_POS
    # Point light at center (optional for sun, but good for others)
    direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector(LIGHT_POS)
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_obj.rotation_euler = rot_quat.to_euler()


def setup_environment():
    """Sets up HDRI or background color."""
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
        except:
            print("Failed to load HDRI")
    else:
        # Default gray background
        node_bg.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1.0)


def setup_ground():
    """Sets up a ground plane at z=0."""
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground"

    # Create a simple material
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0

    plane.data.materials.append(mat)


def render_frame(obj_file, rigid_dir, frame_str, output_file):
    # 0. Load Assets File (Scene Setup)
    if os.path.exists(ASSETS_PATH):
        bpy.ops.wm.open_mainfile(filepath=ASSETS_PATH)
    else:
        print(f"Warning: {ASSETS_PATH} not found. Using default scene.")

    # Override Scene Setup
    cleanup_scene()
    setup_camera()
    setup_lighting()
    setup_environment()
    setup_ground()

    # 1. Load Rigid Bodies
    load_rigid_bodies(rigid_dir, frame_str)

    # 2. Load MPM Fluid
    load_mpm_mesh(obj_file)

    # 3. Render
    bpy.context.scene.render.filepath = output_file
    bpy.context.scene.render.engine = "CYCLES"

    # Optional: Speed up rendering
    bpy.context.scene.cycles.samples = 128

    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        cycles_prefs.refresh_devices()

        # Attempt to set a GPU device type
        gpu_found = False
        for device_type in ["OPTIX", "CUDA", "HIP", "METAL"]:
            cycles_prefs.compute_device_type = device_type

            # Check if we have any devices of this type
            devices_of_type = [d for d in cycles_prefs.devices if d.type == device_type]

            if devices_of_type:
                print(
                    f"Found {device_type} devices: {[d.name for d in devices_of_type]}"
                )

                # Enable these devices
                for device in cycles_prefs.devices:
                    if device.type == device_type:
                        device.use = True
                    else:
                        device.use = False  # Disable CPU and others

                gpu_found = True
                break

        if gpu_found:
            bpy.context.scene.cycles.device = "GPU"
        else:
            print("No GPU devices found. Using CPU.")
            bpy.context.scene.cycles.device = "CPU"

    except Exception as e:
        print(f"GPU setup failed or not supported: {e}")

    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    # CLI Args: ... -- <obj_path> <rigid_dir> <frame_str> <output_png>
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1 :]
        if len(args) >= 4:
            render_frame(args[0], args[1], args[2], args[3])
