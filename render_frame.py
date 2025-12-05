import bpy
import json
import sys
import os
import mathutils

# --- Configuration ---
ASSETS_PATH = "assets.blend"  # Path to your pre-made file
MPM_MATERIAL_NAME = "Jelly"  # Must match a material name in assets.blend
RB_MATERIAL_NAME = "RigidMat"  # Must match a material name in assets.blend

# Meshing Settings
PARTICLE_RADIUS = 1e-2
VOXEL_SIZE = 1.0 / 128.0
THRESHOLD = 1.0

# Scene Settings
CAMERA_POS = (3.0, -3.0, 2.5)
CAMERA_LOOKAT = (0.5, 0.5, 0.5)
LIGHT_POS = (5.0, 5.0, 10.0)
LIGHT_ENERGY = 5.0
HDRI_PATH = "background.exr"  # Set to None or invalid path to skip


def setup_geometry_nodes(obj):
    """Adds the Point-to-Mesh Geometry Nodes modifier."""
    mod = obj.modifiers.new(name="FluidMesher", type="NODES")
    group = bpy.data.node_groups.new("FluidMeshGroup", "GeometryNodeTree")
    mod.node_group = group

    # Setup Interface (Crucial for Blender 4.0+)
    if hasattr(group, "interface"):
        group.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        group.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
    else:
        # Fallback for older Blender versions
        group.inputs.new("NodeSocketGeometry", "Geometry")
        group.outputs.new("NodeSocketGeometry", "Geometry")

    # Create Nodes
    nodes = group.nodes
    links = group.links

    in_node = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")

    pts_to_vol = nodes.new("GeometryNodePointsToVolume")
    pts_to_vol.inputs["Radius"].default_value = PARTICLE_RADIUS
    pts_to_vol.inputs["Voxel Size"].default_value = VOXEL_SIZE
    pts_to_vol.inputs["Density"].default_value = 10.0

    vol_to_mesh = nodes.new("GeometryNodeVolumeToMesh")
    vol_to_mesh.inputs["Threshold"].default_value = THRESHOLD
    vol_to_mesh.inputs["Adaptivity"].default_value = 0.1

    smooth = nodes.new("GeometryNodeSetShadeSmooth")

    # Material Assignment Node (Crucial for GeoNodes)
    set_mat = nodes.new("GeometryNodeSetMaterial")
    mat = bpy.data.materials.get(MPM_MATERIAL_NAME)
    if mat:
        set_mat.inputs["Material"].default_value = mat

    # Link
    links.new(in_node.outputs["Geometry"], pts_to_vol.inputs["Points"])
    links.new(pts_to_vol.outputs["Volume"], vol_to_mesh.inputs["Volume"])
    links.new(vol_to_mesh.outputs["Mesh"], set_mat.inputs["Geometry"])
    links.new(set_mat.outputs["Geometry"], smooth.inputs["Geometry"])
    links.new(smooth.outputs["Geometry"], out_node.inputs["Geometry"])


def load_rigid_bodies(json_path):
    """Loads rigid body transforms from JSON."""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    mat = bpy.data.materials.get(RB_MATERIAL_NAME)

    for item in data:
        # Use mesh path from JSON if available, otherwise construct it
        mesh_path = f"rigid_meshes/rigid_{item['id']}.obj"

        if not os.path.exists(mesh_path):
            print(f"Warning: Mesh file not found: {mesh_path}")
            continue

        bpy.ops.object.select_all(action="DESELECT")

        # Use the new OBJ importer for Blender 4.0+
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=mesh_path)
        else:
            # Fallback for older Blender versions
            bpy.ops.import_scene.obj(filepath=mesh_path)

        selected_objs = bpy.context.selected_objects
        if not selected_objs:
            print(f"Warning: No objects imported from {mesh_path}")
            continue

        # If multiple objects are imported, join them or pick the first one
        bpy.context.view_layer.objects.active = selected_objs[0]
        if len(selected_objs) > 1:
            bpy.ops.object.join()

        obj = bpy.context.active_object
        obj.name = f"RB_{item['id']}"

        # Apply Transform
        obj.location = item["pos"]
        # Blender uses [w, x, y, z], ensure your JSON is compatible
        # If your JSON is [x, y, z, w], you need to reorder
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = item["rot"]

        if mat:
            obj.data.materials.append(mat)


def load_mpm_particles(ply_path):
    """Imports PLY and applies meshing."""
    if not os.path.exists(ply_path):
        return

    bpy.ops.object.select_all(action="DESELECT")

    # Use the new PLY importer for Blender 4.0+
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=ply_path)
    else:
        # Fallback for older Blender versions
        bpy.ops.import_mesh.ply(filepath=ply_path)

    obj = bpy.context.selected_objects[0]
    obj.name = "MPM_Fluid"

    # Apply Meshing (Geometry Nodes)
    setup_geometry_nodes(obj)

    # Add Smooth Modifier to reduce bumps
    smooth = obj.modifiers.new(name="Smooth", type="SMOOTH")
    smooth.factor = 1.0
    smooth.iterations = 50


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


def render_frame(ply_file, json_file, output_file):
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

    # 1. Load Rigid Bodies
    load_rigid_bodies(json_file)

    # 2. Load MPM Fluid
    load_mpm_particles(ply_file)

    # 3. Render
    bpy.context.scene.render.filepath = output_file
    bpy.context.scene.render.engine = "CYCLES"

    # Optional: Speed up rendering
    bpy.context.scene.cycles.samples = 128

    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        cycles_prefs.refresh_devices()

        # Attempt to set a GPU device type
        for device_type in ["OPTIX", "CUDA", "HIP", "METAL"]:
            try:
                cycles_prefs.compute_device_type = device_type
                break
            except:
                pass

        # Enable devices
        for device in cycles_prefs.devices:
            if device.type != "CPU":
                device.use = True

        bpy.context.scene.cycles.device = "GPU"
    except Exception as e:
        print(f"GPU setup failed or not supported: {e}")

    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    # CLI Args: ... -- <ply_path> <json_path> <output_png>
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1 :]
        if len(args) >= 3:
            render_frame(args[0], args[1], args[2])
