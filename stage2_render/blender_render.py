"""
Stage 2 - Primary Path-Traced Rendering Engine (Blender Cycles 5.x)
Author: Karthi (Person B - Render & Compositing Lead)

Headless Python renderer using Blender Cycles Path Tracer.
Loads 3D car model (`assets/car_model.obj`), assigns world EXR/HDR environment texture,
creates a Shadow Catcher ground plane, and renders path-traced reflections & contact shadows.
"""

import sys
import os
import json

def find_blender_executable():
    """
    Auto-detects Blender executable path on Windows systems.
    """
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

def render_with_blender_cycles(hdr_path: str, camera_json_path: str, output_render_path: str, car_model_path: str = "assets/car_model.obj"):
    """
    Renders vehicle under EXR environment using Blender Cycles path tracer.
    Executes headlessly via CLI: `blender -b -P blender_render.py -- <hdr> <json> <out> <model>`
    """
    hdr_path = os.path.abspath(hdr_path)
    camera_json_path = os.path.abspath(camera_json_path)
    output_render_path = os.path.abspath(output_render_path)
    car_model_path = os.path.abspath(car_model_path)
    
    try:
        import bpy
    except ImportError:
        blender_exe = find_blender_executable()
        print(f"[Blender Engine] Invoking headless Blender via CLI: {blender_exe}")
        cmd = f'"{blender_exe}" -b --python "{__file__}" -- "{hdr_path}" "{camera_json_path}" "{output_render_path}" "{car_model_path}"'
        print(f"[Blender Engine] Running command: {cmd}")
        ret = os.system(cmd)
        return ret == 0

    print(f"[Blender Cycles] Initializing path-traced render scene...")
    
    # 1. Reset factory scene or load native .blend
    if car_model_path.endswith('.blend') and os.path.exists(car_model_path):
        print(f"[Blender Cycles] Loading native vehicle scene file: {car_model_path}")
        bpy.ops.wm.open_mainfile(filepath=car_model_path)
        scene = bpy.context.scene
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        
    scene.render.engine = 'CYCLES'
    
    # Enable GPU device if available
    # Enable GPU device if available (Cycles / OptiX)
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.get_devices()
        
        # Prefer OptiX for RTX 3050 Laptop GPU
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
        print(f"[Blender Cycles Warning] GPU configuration failed: {e}. Falling back to default device.")
        scene.cycles.device = 'CPU'
        
    # Render Settings - Optimized for RTX 3050 4GB
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
        
    # Color management
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
    scene.render.film_transparent = True  # Transparent background for compositing
    
    # 2. Setup World Environment Lighting (HDRI)
    world = bpy.data.worlds.new("EnvironmentWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    
    node_background = nodes.new(type='ShaderNodeBackground')
    node_environment = nodes.new(type='ShaderNodeTexEnvironment')
    node_output = nodes.new(type='ShaderNodeOutputWorld')
    
    actual_hdr_path = hdr_path
    if not os.path.exists(actual_hdr_path):
        if actual_hdr_path.endswith('.exr') and os.path.exists(actual_hdr_path.replace('.exr', '.hdr')):
            actual_hdr_path = actual_hdr_path.replace('.exr', '.hdr')
            
    if os.path.exists(actual_hdr_path):
        node_environment.image = bpy.data.images.load(actual_hdr_path)
        print(f"[Blender Cycles] Loaded HDRI world texture: {actual_hdr_path}")
    else:
        print(f"[Blender Cycles Warning] HDRI file not found at: {hdr_path}")
        
    links.new(node_environment.outputs['Color'], node_background.inputs['Color'])
    links.new(node_background.outputs['Background'], node_output.inputs['Surface'])
    
    # 3. Add Ground Shadow Catcher Plane
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground_Shadow_Catcher"
    ground.is_shadow_catcher = True
    
    # 4. Import Vehicle Model (OBJ / FBX / GLTF)
    if car_model_path and os.path.exists(car_model_path):
        if car_model_path.endswith('.glb') or car_model_path.endswith('.gltf'):
            bpy.ops.import_scene.gltf(filepath=car_model_path)
        elif car_model_path.endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=car_model_path)
        elif car_model_path.endswith('.obj'):
            bpy.ops.wm.obj_import(filepath=car_model_path)
    else:
        # Create fallback vehicle chassis
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
        
    # 5. Position Camera from camera.json
    bpy.ops.object.camera_add(location=(0, -4.5, 1.1))
    cam = bpy.context.active_object
    scene.camera = cam
    
    fov = 50.0
    if os.path.exists(camera_json_path):
        with open(camera_json_path, 'r') as f:
            cdata = json.load(f)
            loc = cdata.get('camera_location', [0, -450.0, 110.0])
            cam.location = (loc[0]/100.0, loc[1]/100.0, loc[2]/100.0)
            fov = cdata.get('fov', 50.0)
            
    # Set camera field of view in radians
    import math
    cam.data.angle = math.radians(fov)
    
    # Create target empty for camera tracking at center height of Volvo (z=0.55m)
    target = bpy.data.objects.new("CameraTarget", None)
    target.location = (0.0, 0.0, 0.55)
    scene.collection.objects.link(target)
    
    # Add Track To constraint pointing at the target Empty
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
            
    # Force unhide all objects and collections for render
    for col in bpy.data.collections:
        col.hide_render = False
    for obj in bpy.data.objects:
        obj.hide_render = False
        
    # Disable compositing nodes in case they are set up to filter or override output
    scene.use_nodes = False

    # 6. Render & Save
    os.makedirs(os.path.dirname(output_render_path), exist_ok=True)
    scene.render.filepath = output_render_path
    bpy.ops.render.render(write_still=True)
    print(f"[Blender Cycles PASS] Successfully saved path-traced vehicle render: {output_render_path}")
    return True

if __name__ == "__main__":
    args = sys.argv
    if "--" in args:
        idx = args.index("--")
        cli_args = args[idx+1:]
        if len(cli_args) >= 3:
            car_path = cli_args[3] if len(cli_args) > 3 else "assets/car_model.obj"
            render_with_blender_cycles(cli_args[0], cli_args[1], cli_args[2], car_path)
    else:
        render_with_blender_cycles("test_assets/test_env.hdr", "test_assets/test_camera.json", "outputs/blender_car_render.png", "assets/car_model.obj")
