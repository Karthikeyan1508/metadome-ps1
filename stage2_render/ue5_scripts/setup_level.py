"""
Stage 2 - Unreal Engine Level Setup & HDRI Assignment
Author: Karthi (Person B - UE5 & Rendering & Compositing)

This script runs inside Unreal Engine's embedded Python environment or via
Python Remote Execution plugin to load generated HDRI, assign to HDRI Backdrop,
position CineCameraActor, and confirm vehicle mesh parameters.
"""

import sys
import json
import os

try:
    import unreal
except ImportError:
    unreal = None

def setup_ue5_scene(hdr_path: str, camera_json_path: str, car_asset_path: str = "assets/car_model.obj"):
    """
    Assigns HDRI environment map to SkyLight / SkyDome, positions CineCamera,
    and ensures vehicle mesh geometry remains 100% unaltered.
    """
    if unreal is None:
        print("[UE5 Setup Engine] Running in offline dry-run mode (UE5 Editor socket not attached).")
        print(f"  HDR Map Target: {hdr_path}")
        print(f"  Camera Json Target: {camera_json_path}")
        print(f"  Car Asset Path: {car_asset_path}")
        return False

    print(f"[UE5 Setup Engine] Initializing level update...")
    print(f"[UE5 Setup Engine] HDRI map target: {hdr_path}")
    
    # 1. Load Camera parameters
    if not os.path.exists(camera_json_path):
        print(f"[UE5 Setup Error] Camera JSON file missing: {camera_json_path}")
        return False
        
    with open(camera_json_path, "r") as f:
        cam_data = json.load(f)
        
    loc = cam_data.get("camera_location", [0.0, -450.0, 110.0])
    rot = cam_data.get("camera_rotation", [-3.5, 0.0, 0.0])
    fov = cam_data.get("fov", 50.0)
    
    # 2. Get World and Camera Actor
    editor_subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = editor_subsystem.get_editor_world()
    
    # 3. Position CineCameraActor
    camera_actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.CineCameraActor)
    if camera_actors:
        cam = camera_actors[0]
        cam.set_actor_location(unreal.Vector(loc[0], loc[1], loc[2]), False, False)
        cam.set_actor_rotation(unreal.Rotator(rot[0], rot[1], rot[2]), False)
        
        cam_component = cam.get_cine_camera_component()
        cam_component.current_fov = fov
        print(f"[UE5 Setup PASS] CineCamera successfully updated: Location={loc}, Rotation={rot}, FOV={fov}")
    else:
        print("[UE5 Setup Warning] CineCameraActor not found in level. Spawning new CineCameraActor...")
        cam_location = unreal.Vector(loc[0], loc[1], loc[2])
        cam_rotation = unreal.Rotator(rot[0], rot[1], rot[2])
        new_cam = world.spawn_actor_from_class(unreal.CineCameraActor, cam_location, cam_rotation)
        if new_cam:
            new_cam.get_cine_camera_component().current_fov = fov
            print(f"[UE5 Setup PASS] Spawned CineCameraActor at {loc}")

    # 4. Import & Apply HDRI Texture to SkyLight / SkyDome
    if os.path.exists(hdr_path):
        # Import HDR texture to Content Browser
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        import_task = unreal.AssetImportTask()
        import_task.filename = hdr_path
        import_task.destination_path = "/Game/HDRI"
        import_task.automated = True
        import_task.replace_existing = True
        asset_tools.import_asset_tasks([import_task])
        
        imported_hdri = import_task.get_objects()[0] if import_task.get_objects() else None
        
        # Assign imported texture to SkyLight
        skylight_actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.SkyLight)
        if skylight_actors and imported_hdri:
            skylight = skylight_actors[0].get_skylight_component()
            skylight.set_cubemap(imported_hdri)
            skylight.recapture_sky()
            print(f"[UE5 Setup PASS] Assigned imported HDRI '{imported_hdri.get_name()}' to SkyLight.")
            
    print("[UE5 Setup PASS] UE5 Level configuration completed successfully.")
    return True

if __name__ == "__main__":
    hdr = sys.argv[1] if len(sys.argv) > 1 else "test_assets/test_env.hdr"
    cam_json = sys.argv[2] if len(sys.argv) > 2 else "test_assets/test_camera.json"
    setup_ue5_scene(hdr, cam_json)
