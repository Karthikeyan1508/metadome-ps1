"""
Mock Test Asset Generator
Author: Karthi & Ratish

Generates dummy background plate, HDRI map, and camera parameters in test_assets/
allowing Person B (Karthi) to test rendering & compositing pipelines immediately in Hour 1.
"""

import os
import json
import numpy as np
import cv2

def generate_mock_assets(dir_path: str = "test_assets"):
    os.makedirs(dir_path, exist_ok=True)
    
    bg_path = os.path.join(dir_path, "test_bg.png")
    hdr_path = os.path.join(dir_path, "test_env.exr")
    cam_path = os.path.join(dir_path, "test_camera.json")
    
    # 1. Create mock background plate (1920x1080)
    bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bg[:580, :] = [80, 50, 20]     # Sky
    bg[580:, :] = [40, 40, 40]     # Road
    # Add sun spot
    cv2.circle(bg, (1350, 250), 60, (255, 240, 210), -1)
    cv2.imwrite(bg_path, bg)
    print(f"[Test Generator] Created test background: {bg_path}")
    
    # 2. Create mock float32 HDR environment map
    hdr = (bg.astype(np.float32) / 255.0) ** 2.2
    # Boost sun spot
    cv2.circle(hdr, (1350, 250), 60, (15.0, 14.0, 10.0), -1)
    
    try:
        import imageio
        imageio.imwrite(hdr_path, hdr.astype(np.float32))
        print(f"[Test Generator] Created test EXR map using imageio: {hdr_path}")
    except Exception:
        fallback_hdr = hdr_path.replace(".exr", ".hdr")
        cv2.imwrite(fallback_hdr, hdr.astype(np.float32))
        print(f"[Test Generator] Created fallback test HDR map: {fallback_hdr}")
    
    # 3. Create mock camera.json
    cam_data = {
        "prompt_id": "test_prompt",
        "fov": 50.0,
        "camera_location": [0.0, -450.0, 110.0],
        "camera_rotation": [-3.5, 0.0, 0.0],
        "sun_direction": [0.5, 0.5, 0.707],
        "light_intensity_multiplier": 1.5
    }
    with open(cam_path, "w") as f:
        json.dump(cam_data, f, indent=2)
    print(f"[Test Generator] Created test camera JSON: {cam_path}")

if __name__ == "__main__":
    generate_mock_assets()
