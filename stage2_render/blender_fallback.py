"""
Stage 2 - Headless Blender Cycles Path Tracer (Safety Fallback Engine)
Author: Karthi (Person B - UE5 & Rendering & Compositing)

Headless Python renderer using Blender Cycles Path Tracer.
Loads 3D car model, assigns world EXR environment texture, creates a Shadow Catcher ground plane,
and renders raytraced reflections & contact shadows.
"""

import sys
import os
import json

def render_with_blender_cycles(hdr_path: str, camera_json_path: str, output_render_path: str, car_model_path: str = "assets/car_model.obj"):
    """
    Renders vehicle under EXR environment using Blender Cycles path tracer.
    Executes headlessly via CLI: `blender -b -P blender_fallback.py -- <hdr> <json> <out>`
    """
    try:
        import bpy
    except ImportError:
        print("[Blender Engine] 'bpy' module not detected in standard system Python.")
        print("To run Blender rendering, execute via Blender executable:")
        print(f"  blender -b --python stage2_render/blender_fallback.py -- \"{hdr_path}\" \"{camera_json_path}\" \"{output_render_path}\" \"{car_model_path}\"")
        return False

    print(f"[Blender Cycles] Building path-traced scene...")
    
    # 1. Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Enable GPU if available
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.get_devices()
        for device in prefs.devices:
            device.use = True
        scene.cycles.device = 'GPU'
    except Exception:
        scene.cycles.device = 'CPU'
        
    scene.cycles.samples = 128
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.film_transparent = True  # Alpha background for transparent compositing
    
    # 2. Setup World HDRI Environment
    world = bpy.data.worlds.new("EnvironmentWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    
    node_background = nodes.new(type='ShaderNodeBackground')
    node_environment = nodes.new(type='ShaderNodeTexEnvironment')
    node_output = nodes.new(type='ShaderNodeOutputWorld')
    
    if os.path.exists(hdr_path):
        node_environment.image = bpy.data.images.load(hdr_path)
    
    links = world.node_tree.links
    links.new(node_environment.outputs['Color'], node_background.inputs['Color'])
    links.new(node_background.outputs['Background'], node_output.inputs['Surface'])
    
    # 3. Add Ground Shadow Catcher Plane
    bpy.ops.mesh.primitive_plane_add(size=25, location=(0, 0, 0))
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
        # Create sleek placeholder vehicle chassis if 3D file not present
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0.75))
        car = bpy.context.active_object
        car.scale = (2.1, 1.0, 0.6)
        
        mat = bpy.data.materials.new(name="MetallicCarPaint")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = (0.05, 0.2, 0.8, 1.0) # Metallic Blue
        bsdf.inputs['Metallic'].default_value = 0.95
        bsdf.inputs['Roughness'].default_value = 0.1
        bsdf.inputs['Clearcoat'].default_value = 1.0
        car.data.materials.append(mat)
        
    # 5. Setup Camera Pose from camera.json
    bpy.ops.object.camera_add(location=(0, -4.5, 1.1), rotation=(1.51, 0, 0))
    cam = bpy.context.active_object
    scene.camera = cam
    
    if os.path.exists(camera_json_path):
        with open(camera_json_path, 'r') as f:
            cdata = json.load(f)
            loc = cdata.get('camera_location', [0, -450.0, 110.0])
            rot = cdata.get('camera_rotation', [-3.5, 0, 0])
            cam.location = (loc[0]/100.0, loc[1]/100.0, loc[2]/100.0) # Scale to Blender meters
            
    # 6. Render Scene
    os.makedirs(os.path.dirname(output_render_path), exist_ok=True)
    scene.render.filepath = output_render_path
    bpy.ops.render.render(write_still=True)
    print(f"[Blender Cycles PASS] Render saved successfully to: {output_render_path}")
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
        # Default test run
        render_with_blender_cycles("test_assets/test_env.hdr", "test_assets/test_camera.json", "outputs/test_render.png")
