"""
Stage 2 - Primary Path-Traced Rendering Engine (Blender Cycles 5.x)
Author: Karthi (Person B - Render & Compositing Lead)

Two rendering modes:
  1. render_depth_pass()       -- renders Z-depth of empty scene (car hidden)
                                  used BEFORE background generation to give
                                  ControlNet the exact camera perspective.
  2. render_with_blender_cycles() -- full beauty render of car with HDR lighting.

Headless Python renderer using Blender Cycles Path Tracer.
Loads 3D car model (assets/Volvo S90.blend), assigns world EXR/HDR environment
texture, creates a Shadow Catcher ground plane, and renders path-traced
reflections & contact shadows.
"""

import sys
import os
import json


def find_blender_executable():
    """Auto-detects Blender executable path on Windows systems."""
    possible_paths = [
        r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        "blender"
    ]
    for p in possible_paths:
        if os.path.exists(p) or p == "blender":
            return p
    return "blender"


# ===========================================================================
# MODE 1 — DEPTH PASS  (no car, just ground + camera perspective)
# ===========================================================================

def render_depth_pass(
    camera_json_path: str,
    output_exr_path: str,
    car_model_path: str = "assets/Volvo S90.blend",
    resolution_x: int = 1920,
    resolution_y: int = 1080,
):
    """
    Renders a Z-depth pass of the scene with the car hidden.
    This captures the exact camera perspective so that ControlNet can
    generate a background that matches it perfectly.

    Outputs:
      - <output_exr_path>               32-bit float EXR depth map
      - <output_exr_path>.replace('.exr', '_depth_map.png')
                                        Normalised 8-bit PNG for ControlNet

    Called headlessly:
        blender -b <blend> --python blender_render.py -- depth <cam.json> <out.exr> <model>
    """
    camera_json_path = os.path.abspath(camera_json_path)
    output_exr_path  = os.path.abspath(output_exr_path)
    car_model_path   = os.path.abspath(car_model_path)

    try:
        import bpy
    except ImportError:
        blender_exe = find_blender_executable()
        cmd = (
            f'"{blender_exe}" -b --python "{__file__}" -- '
            f'depth "{camera_json_path}" "{output_exr_path}" "{car_model_path}"'
        )
        print(f"[Depth Pass] Invoking headless Blender: {cmd}")
        ret = os.system(cmd)
        return ret == 0

    print("[Depth Pass] Rendering Z-depth pass (car hidden)...")

    # ------------------------------------------------------------------ scene
    if car_model_path.endswith('.blend') and os.path.exists(car_model_path):
        bpy.ops.wm.open_mainfile(filepath=car_model_path)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y

    # ------------------------------------------------------------------ hide car
    # Hide every mesh object that is NOT the ground plane.
    GROUND_NAMES = {"Ground_Shadow_Catcher", "Ground", "Plane", "Floor"}
    hidden_objects = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name not in GROUND_NAMES:
            obj.hide_render = True
            hidden_objects.append(obj)
            print(f"[Depth Pass] Hidden from render: {obj.name}")

    # Also hide every collection that looks like it contains the car
    for col in bpy.data.collections:
        col_lower = col.name.lower()
        if any(k in col_lower for k in ["car", "vehicle", "volvo", "body"]):
            col.hide_render = True
            print(f"[Depth Pass] Hidden collection: {col.name}")

    # ------------------------------------------------------------------ z-pass
    scene.view_layers[0].use_pass_z = True

    # ------------------------------------------------------------------ camera
    import math
    bpy.ops.object.camera_add(location=(0, -4.5, 1.1))
    cam = bpy.context.active_object
    scene.camera = cam

    fov = 50.0
    if os.path.exists(camera_json_path):
        with open(camera_json_path, 'r') as f:
            cdata = json.load(f)
        loc = cdata.get('camera_location', [0, -450.0, 110.0])
        cam.location = (loc[0] / 100.0, loc[1] / 100.0, loc[2] / 100.0)
        fov = cdata.get('fov', 50.0)
        rot = cdata.get('camera_rotation', [-3.5, 0.0, 0.0])
        cam.rotation_euler = (
            math.radians(90 + rot[0]),
            math.radians(rot[2]),
            math.radians(rot[1]),
        )

    cam.data.angle = math.radians(fov)

    # ------------------------------------------------------------------ render
    os.makedirs(os.path.dirname(output_exr_path), exist_ok=True)
    scene.render.filepath = output_exr_path
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.film_transparent = True

    bpy.ops.render.render(write_still=True)
    print(f"[Depth Pass] EXR saved: {output_exr_path}")

    # ------------------------------------------------------------------ PNG
    _convert_exr_depth_to_png(output_exr_path)

    # ------------------------------------------------------------------ restore
    for obj in hidden_objects:
        obj.hide_render = False
    for col in bpy.data.collections:
        col.hide_render = False

    return output_exr_path


def _convert_exr_depth_to_png(exr_path: str) -> str:
    """
    Reads the Z-depth channel from the EXR and saves a normalised 8-bit PNG
    suitable for ControlNet conditioning.
    White = near, Black = far (inverted so near objects are prominent).
    """
    import numpy as np
    import cv2

    png_path = exr_path.replace('.exr', '_depth_map.png')

    img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[Depth Pass] WARNING: Could not read EXR for PNG conversion: {exr_path}")
        return png_path

    # EXR from Cycles Z-pass: single-channel or first channel = Z depth
    if len(img.shape) == 3:
        depth_ch = img[:, :, 0].astype(np.float32)
    else:
        depth_ch = img.astype(np.float32)

    # Clamp infinite/NaN values (sky = inf in Z-pass)
    finite_mask = np.isfinite(depth_ch)
    if finite_mask.any():
        max_finite = depth_ch[finite_mask].max()
        depth_ch[~finite_mask] = max_finite

    # Normalise 0→1, then invert so near = white
    d_min, d_max = depth_ch.min(), depth_ch.max()
    if d_max > d_min:
        depth_norm = 1.0 - (depth_ch - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_ch)

    depth_png = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(png_path, depth_png)
    print(f"[Depth Pass] ControlNet PNG saved: {png_path}")
    return png_path


# ===========================================================================
# MODE 2 — BEAUTY PASS  (full car render with HDR lighting)
# ===========================================================================

def render_with_blender_cycles(
    hdr_path: str,
    camera_json_path: str,
    output_render_path: str,
    car_model_path: str = "assets/Volvo S90.blend",
):
    """
    Renders vehicle under EXR environment using Blender Cycles path tracer.
    Executes headlessly via CLI:
        blender -b --python blender_render.py -- <hdr> <json> <out> <model>
    """
    hdr_path           = os.path.abspath(hdr_path)
    camera_json_path   = os.path.abspath(camera_json_path)
    output_render_path = os.path.abspath(output_render_path)
    car_model_path     = os.path.abspath(car_model_path)

    try:
        import bpy
    except ImportError:
        blender_exe = find_blender_executable()
        print(f"[Blender Engine] Invoking headless Blender via CLI: {blender_exe}")
        cmd = (
            f'"{blender_exe}" -b --python "{__file__}" -- '
            f'beauty "{hdr_path}" "{camera_json_path}" "{output_render_path}" "{car_model_path}"'
        )
        print(f"[Blender Engine] Running command: {cmd}")
        ret = os.system(cmd)
        return ret == 0

    print("[Blender Cycles] Initializing path-traced render scene...")

    # ------------------------------------------------------------------ scene
    if car_model_path.endswith('.blend') and os.path.exists(car_model_path):
        print(f"[Blender Cycles] Loading native vehicle scene file: {car_model_path}")
        bpy.ops.wm.open_mainfile(filepath=car_model_path)
        scene = bpy.context.scene
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene

    scene.render.engine = 'CYCLES'

    # ------------------------------------------------------------------ GPU
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.get_devices()
        for device_type in ['OPTIX', 'CUDA']:
            available = False
            for device in prefs.devices:
                if device.type == device_type:
                    device.use = True
                    available = True
            if available:
                prefs.compute_device_type = device_type
                break
        scene.cycles.device = 'GPU'
        print(f"[Blender Cycles] Configured GPU device type: {prefs.compute_device_type}")
    except Exception as e:
        print(f"[Blender Cycles Warning] GPU config failed: {e}. Using CPU.")
        scene.cycles.device = 'CPU'

    # ------------------------------------------------------------------ render settings
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = 'OPTIX'
    except Exception:
        pass
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    try:
        scene.cycles.transmission_bounces = 2
        scene.cycles.volume_bounces = 0
    except Exception:
        pass
    try:
        scene.cycles.tile_size = 256
    except Exception:
        pass

    # ------------------------------------------------------------------ color management
    try:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'AgX - High Contrast'
    except Exception:
        try:
            scene.view_settings.view_transform = 'AgX'
            scene.view_settings.look = 'High Contrast'
        except Exception:
            pass
    scene.view_settings.exposure = 0.0

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.film_transparent = True

    # ------------------------------------------------------------------ HDRI world
    world = bpy.data.worlds.new("EnvironmentWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    node_background  = nodes.new(type='ShaderNodeBackground')
    node_environment = nodes.new(type='ShaderNodeTexEnvironment')
    node_output      = nodes.new(type='ShaderNodeOutputWorld')

    actual_hdr_path = hdr_path
    if not os.path.exists(actual_hdr_path):
        fallback = actual_hdr_path.replace('.exr', '.hdr')
        if os.path.exists(fallback):
            actual_hdr_path = fallback

    if os.path.exists(actual_hdr_path):
        node_environment.image = bpy.data.images.load(actual_hdr_path)
        print(f"[Blender Cycles] Loaded HDRI world texture: {actual_hdr_path}")
    else:
        print(f"[Blender Cycles Warning] HDRI not found: {hdr_path}")

    links.new(node_environment.outputs['Color'],   node_background.inputs['Color'])
    links.new(node_background.outputs['Background'], node_output.inputs['Surface'])

    # ------------------------------------------------------------------ ground
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground_Shadow_Catcher"
    ground.is_shadow_catcher = True

    # ------------------------------------------------------------------ import model
    if car_model_path and os.path.exists(car_model_path):
        if car_model_path.endswith('.glb') or car_model_path.endswith('.gltf'):
            bpy.ops.import_scene.gltf(filepath=car_model_path)
        elif car_model_path.endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=car_model_path)
        elif car_model_path.endswith('.obj'):
            bpy.ops.wm.obj_import(filepath=car_model_path)
    else:
        # Fallback box car
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0.75))
        car = bpy.context.active_object
        car.scale = (2.1, 1.0, 0.6)
        mat = bpy.data.materials.new(name="MetallicCarPaint")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = (0.05, 0.2, 0.8, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.95
        bsdf.inputs['Roughness'].default_value = 0.1
        bsdf.inputs['Clearcoat'].default_value = 1.0
        car.data.materials.append(mat)

    # ------------------------------------------------------------------ camera
    import math
    bpy.ops.object.camera_add(location=(0, -4.5, 1.1))
    cam = bpy.context.active_object
    scene.camera = cam

    fov = 50.0
    if os.path.exists(camera_json_path):
        with open(camera_json_path, 'r') as f:
            cdata = json.load(f)
        loc = cdata.get('camera_location', [0, -450.0, 110.0])
        cam.location = (loc[0] / 100.0, loc[1] / 100.0, loc[2] / 100.0)
        fov = cdata.get('fov', 50.0)

    cam.data.angle = math.radians(fov)

    target = bpy.data.objects.new("CameraTarget", None)
    target.location = (0.0, 0.0, 0.55)
    scene.collection.objects.link(target)
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    # Ensure all objects and collections are visible for render
    for col in bpy.data.collections:
        col.hide_render = False
    for obj in bpy.data.objects:
        obj.hide_render = False

    scene.use_nodes = False

    # ------------------------------------------------------------------ render
    os.makedirs(os.path.dirname(output_render_path), exist_ok=True)
    scene.render.filepath = output_render_path
    bpy.ops.render.render(write_still=True)
    print(f"[Blender Cycles PASS] Saved path-traced render: {output_render_path}")
    return True


# ===========================================================================
# CLI entry-point
# ===========================================================================

if __name__ == "__main__":
    args = sys.argv
    if "--" in args:
        idx = args.index("--")
        cli_args = args[idx + 1:]
    else:
        cli_args = []

    if len(cli_args) >= 1 and cli_args[0] == "depth":
        # blender -b --python blender_render.py -- depth <cam.json> <out.exr> [model]
        cam_j    = cli_args[1] if len(cli_args) > 1 else "test_assets/test_camera.json"
        out_exr  = cli_args[2] if len(cli_args) > 2 else "outputs/depth_render.exr"
        model    = cli_args[3] if len(cli_args) > 3 else "assets/Volvo S90.blend"
        render_depth_pass(cam_j, out_exr, model)

    elif len(cli_args) >= 1 and cli_args[0] == "beauty":
        # blender -b --python blender_render.py -- beauty <hdr> <cam.json> <out> [model]
        hdr_p    = cli_args[1] if len(cli_args) > 1 else "test_assets/test_env.hdr"
        cam_j    = cli_args[2] if len(cli_args) > 2 else "test_assets/test_camera.json"
        out_r    = cli_args[3] if len(cli_args) > 3 else "outputs/car_render.png"
        model    = cli_args[4] if len(cli_args) > 4 else "assets/Volvo S90.blend"
        render_with_blender_cycles(hdr_p, cam_j, out_r, model)

    else:
        # Legacy fallback: positional args without mode keyword (backward compat)
        if len(cli_args) >= 3:
            car_path = cli_args[3] if len(cli_args) > 3 else "assets/Volvo S90.blend"
            render_with_blender_cycles(cli_args[0], cli_args[1], cli_args[2], car_path)
        else:
            render_with_blender_cycles(
                "test_assets/test_env.hdr",
                "test_assets/test_camera.json",
                "outputs/blender_car_render.png",
                "assets/Volvo S90.blend",
            )
