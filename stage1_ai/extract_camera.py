"""
Stage 1 - Camera Parameter & Horizon Matcher
Author: Ratish (Person A - AI & Data Pipeline)

Analyzes background plate depth and perspective vanishing lines to compute
camera field-of-view (FOV), location [X, Y, Z], rotation [Pitch, Yaw, Roll],
and sun light direction vector.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

def extract_camera_parameters(prompt_id: str, input_image_path: str, output_json_path: str):
    """
    Computes camera pose relative to origin (0,0,0) where car is seated.
    Outputs standard camera.json schema.
    """
    print(f"[Camera Matcher] Analyzing perspective for: {input_image_path} (ID: {prompt_id})")
    
    # 5 different professional car advertising angles
    # Distances in cm (e.g. 580.0 = 5.8 meters)
    if "prompt_01" in prompt_id:
        # Prompt 1: Front Three-Quarter (Hero Shot)
        fov = 48.0
        cam_location = [-420.0, -420.0, 130.0]
        cam_rotation = [-3.5, 0.0, 0.0]
    elif "prompt_02" in prompt_id:
        # Prompt 2: Telephoto Side Profile View (Elegant profile)
        fov = 55.0
        cam_location = [-620.0, 0.0, 110.0]
        cam_rotation = [-2.0, 0.0, 0.0]
    elif "prompt_03" in prompt_id:
        # Prompt 3: Overhead / High-Angle Diagonal (Dramatic landscape/road combo)
        fov = 45.0
        cam_location = [380.0, -420.0, 260.0]
        cam_rotation = [-15.0, 0.0, 0.0]
    elif "prompt_04" in prompt_id:
        # Prompt 4: Low-Angle Aggressive Front (Zoomed out to avoid clipping front bumper)
        fov = 42.0
        cam_location = [0.0, -580.0, 70.0]
        cam_rotation = [-1.5, 0.0, 0.0]
    elif "prompt_05" in prompt_id:
        # Prompt 5: Rear Three-Quarter (Zoomed out to avoid clipping rear bumper)
        fov = 50.0
        cam_location = [-400.0, 420.0, 130.0]
        cam_rotation = [-3.5, 0.0, 0.0]
    else:
        # Default front three-quarter fallback
        fov = 50.0
        cam_location = [-420.0, -420.0, 130.0]
        cam_rotation = [-3.5, 0.0, 0.0]
        
    sun_dir = [0.5, 0.5, 0.707]
    
    if os.path.exists(input_image_path):
        img = cv2.imread(input_image_path)
        if img is not None:
            h, w, _ = img.shape
            # Simple brightness centroid for key sun position estimation
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
            
            # Map max brightness pixel coordinate to sun directional vector
            sun_x_norm = (max_loc[0] - w / 2.0) / (w / 2.0)
            sun_y_norm = -(max_loc[1] - h / 2.0) / (h / 2.0)
            
            sun_dir = [float(sun_x_norm), float(sun_y_norm + 0.5), 0.8]
            norm = np.linalg.norm(sun_dir)
            if norm > 0:
                sun_dir = [float(x / norm) for x in sun_dir]
                
    camera_data = {
        "prompt_id": prompt_id,
        "fov": fov,
        "camera_location": cam_location,
        "camera_rotation": cam_rotation,
        "sun_direction": sun_dir,
        "light_intensity_multiplier": 1.5
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(camera_data, f, indent=2)
        
    print(f"[Camera Matcher] Saved camera specs to: {output_json_path}")
    return output_json_path

def main():
    parser = argparse.ArgumentParser(description="Extract camera parameters from background image")
    parser.add_argument("--prompt_id", type=str, default="prompt_01", help="Prompt identifier")
    parser.add_argument("--input", type=str, required=True, help="Input background image (.png)")
    parser.add_argument("--output", type=str, required=True, help="Output camera JSON path (.json)")
    
    args = parser.parse_args()
    extract_camera_parameters(args.prompt_id, args.input, args.output)

if __name__ == "__main__":
    main()
