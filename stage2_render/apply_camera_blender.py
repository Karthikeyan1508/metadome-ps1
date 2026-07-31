"""
Stage 2 - Apply camera.json to the Blender scene, fully automated.

Run headlessly (no UI, no manual dragging) with:

    blender assets/Volvo_S90.blend --background --python apply_camera_blender.py -- \
        --camera_json outputs/camera_02.json \
        --background_image outputs/prompt_02_forest.png \
        --render_output outputs/render_prompt_02.png

Notes on unit conversion:
- camera_location / units in camera.json are in CENTIMETERS (matches the
  original pipeline's convention, e.g. 550.0 = 5.5 meters).
  Blender's default scene unit is meters, so we divide by 100.
- camera_rotation is [pitch, yaw, roll] in DEGREES, matching Blender's
  X/Z/Y euler convention as used elsewhere in this pipeline: we map
  pitch -> rotation_euler.x, yaw -> rotation_euler.z, roll -> rotation_euler.y.
  Adjust the axis mapping below if your rig uses a different convention.
"""

import bpy
import json
import sys
import math
import argparse


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_json", required=True)
    parser.add_argument("--background_image", required=False, default=None)
    parser.add_argument("--render_output", required=False, default=None)
    parser.add_argument("--camera_name", default="Camera")
    return parser.parse_args(argv)


def apply_camera(camera_json_path, camera_name="Camera"):
    with open(camera_json_path, "r") as f:
        data = json.load(f)

    cam_obj = bpy.data.objects.get(camera_name)
    if cam_obj is None:
        raise RuntimeError(
            f"No object named '{camera_name}' found in the scene. "
            f"Available objects: {[o.name for o in bpy.data.objects]}"
        )

    loc_cm = data["camera_location"]
    rot_deg = data["camera_rotation"]
    fov_deg = data["fov"]

    # cm -> m
    cam_obj.location = (loc_cm[0] / 100.0, loc_cm[1] / 100.0, loc_cm[2] / 100.0)

    pitch, yaw, roll = rot_deg
    cam_obj.rotation_euler = (
        math.radians(90 + pitch),  # +90 so 0 pitch = camera pointing along -Y (Blender convention)
        math.radians(roll),
        math.radians(yaw),
    )

    if cam_obj.data.type == "PERSP":
        cam_obj.data.lens_unit = "FOV"
        cam_obj.data.angle = math.radians(fov_deg)

    # Sun direction + intensity, if a sun lamp named "Sun" exists
    sun_obj = bpy.data.objects.get("Sun")
    if sun_obj is not None and "sun_direction" in data:
        sx, sy, sz = data["sun_direction"]
        # Point the sun lamp opposite to the light direction vector
        direction = (sx, sy, sz)
        # Simple look-at: align lamp's -Z to the direction vector
        import mathutils
        vec = mathutils.Vector(direction).normalized()
        rot_quat = vec.to_track_quat('-Z', 'Y')
        sun_obj.rotation_euler = rot_quat.to_euler()
        if sun_obj.data.type == "SUN":
            sun_obj.data.energy = data.get("light_intensity_multiplier", 1.0) * sun_obj.data.energy

    print(f"[Blender] Camera '{camera_name}' set: location(m)={cam_obj.location[:]}, "
          f"rotation(deg)=pitch:{pitch} yaw:{yaw} roll:{roll}, fov(deg)={fov_deg}")
    return cam_obj


def set_background_image(image_path):
    """Set the world background / compositor backdrop to the generated plate,
    so the render preview matches the composite without manual setup."""
    scene = bpy.context.scene
    scene.render.film_transparent = True  # render car with alpha for compositing over the plate

    # Also load as a viewport camera background image for visual reference
    cam_obj = bpy.context.scene.camera
    if cam_obj and cam_obj.data.type == "PERSP":
        cam_obj.data.show_background_images = True
        bg = cam_obj.data.background_images.new()
        img = bpy.data.images.load(image_path)
        bg.image = img
        bg.display_depth = 'BACK'


def main():
    args = parse_args()
    apply_camera(args.camera_json, args.camera_name)

    if args.background_image:
        set_background_image(args.background_image)

    if args.render_output:
        scene = bpy.context.scene
        scene.render.filepath = args.render_output
        bpy.ops.render.render(write_still=True)
        print(f"[Blender] Rendered -> {args.render_output}")


if __name__ == "__main__":
    main()
