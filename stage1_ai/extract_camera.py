"""
Stage 1 - Camera Parameter & Horizon Matcher
Author: Ratish & Karthi

Analyzes background plate perspective to compute camera field-of-view (FOV),
location [X, Y, Z], rotation [Pitch, Yaw, Roll], and sun light direction.

Distance is pulled back (Y = -8.5m to -9.0m) so the entire Volvo S90 fits
comfortably in frame without extreme close-up zooming.
"""

import os
import json
import argparse
import numpy as np
import cv2


# ------------------------------------------------------------------
# Horizon + roll detection (color-gradient based)
# ------------------------------------------------------------------

def detect_horizon_row(img, search_band=(0.20, 0.80)):
    """
    Find the row with the sharpest average-brightness transition
    (sky -> ground boundary). Returns the row index and a 0-1 confidence score.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_means = gray.mean(axis=1)
    row_grad = np.abs(np.diff(row_means))

    lo, hi = int(h * search_band[0]), int(h * search_band[1])
    band = row_grad[lo:hi]
    if band.size == 0 or band.max() == 0:
        return h // 2, 0.0

    horizon_row = lo + int(np.argmax(band))
    confidence = float(band.max() / (band.mean() + 1e-6))
    confidence = float(np.clip((confidence - 1.0) / 5.0, 0.0, 1.0))
    return horizon_row, confidence


def detect_horizon_roll(img, horizon_row, half_width_frac=0.35):
    """
    Estimate camera roll by comparing horizon row on left vs right third of image.
    """
    h, w = img.shape[:2]
    band_px = max(int(h * 0.08), 10)
    y0 = max(horizon_row - band_px, 0)
    y1 = min(horizon_row + band_px, h)

    left = img[y0:y1, 0:int(w * half_width_frac)]
    right = img[y0:y1, int(w * (1 - half_width_frac)):w]

    left_row, left_conf = detect_horizon_row(left, search_band=(0.0, 1.0))
    right_row, right_conf = detect_horizon_row(right, search_band=(0.0, 1.0))

    if left_conf < 0.15 or right_conf < 0.15:
        return 0.0, 0.0

    dy = (y0 + right_row) - (y0 + left_row)
    dx = w * (1 - 2 * half_width_frac)
    roll_deg = float(np.degrees(np.arctan2(dy, dx)))
    roll_confidence = min(left_conf, right_conf)
    return roll_deg, roll_confidence


def pitch_from_horizon(horizon_row, img_h, fov_deg):
    """
    Pinhole camera model: convert horizon row position (in pixels)
    to a pitch angle.
    """
    half_h = img_h / 2.0
    focal_len_px = half_h / np.tan(np.radians(fov_deg) / 2.0)
    offset_px = horizon_row - half_h
    pitch_rad = np.arctan2(offset_px, focal_len_px)
    return float(np.degrees(pitch_rad))


def extract_camera_parameters(
    prompt_id: str,
    input_image_path: str,
    output_json_path: str,
    depth_path: str = None,
):
    print(f"[Camera Matcher] Processing: {input_image_path} (ID: {prompt_id})")

    # Camera location presets (in cm)
    # Pulled back to Y = -850cm to -950cm (-8.5m to -9.5m) so full car is framed nicely
    PRESETS = {
        "prompt_01": dict(fov=38.0, camera_location=[-400.0, -850.0, 150.0],
                           camera_rotation=[-5.0, 0.0, 0.0], road_based=False),
        "prompt_02": dict(fov=36.0, camera_location=[-350.0, -850.0, 130.0],
                           camera_rotation=[-3.0, 0.0, 0.0], road_based=True),
        "prompt_03": dict(fov=36.0, camera_location=[-300.0, -850.0, 220.0],
                           camera_rotation=[-12.0, 0.0, 0.0], road_based=True),
        "prompt_04": dict(fov=36.0, camera_location=[0.0, -900.0, 130.0],
                           camera_rotation=[-4.0, 0.0, 0.0], road_based=True),
        "prompt_05": dict(fov=38.0, camera_location=[-400.0, 850.0, 150.0],
                           camera_rotation=[-5.0, 0.0, 0.0], road_based=False),
    }

    key = next((k for k in PRESETS if k in prompt_id), "prompt_02")
    preset = PRESETS[key]

    fov = preset["fov"]
    camera_location = list(preset["camera_location"])
    camera_rotation = list(preset["camera_rotation"])
    detection_info = {"method": "preset_only", "confidence": None}

    # If background image exists, analyze horizon
    if input_image_path and os.path.exists(input_image_path):
        img = cv2.imread(input_image_path)
        if img is not None:
            h, w, _ = img.shape

            if preset["road_based"]:
                horizon_row, h_conf = detect_horizon_row(img)
                roll_deg, roll_conf = detect_horizon_roll(img, horizon_row)

                if h_conf >= 0.15:
                    measured_pitch = pitch_from_horizon(horizon_row, h, fov)
                    camera_rotation[0] = round(measured_pitch, 2)
                    detection_info["pitch_confidence"] = round(h_conf, 2)
                    detection_info["method"] = "image_derived"
                else:
                    detection_info["pitch_confidence"] = round(h_conf, 2)
                    detection_info["method"] = "preset_fallback"

                if roll_conf >= 0.15:
                    camera_rotation[2] = round(roll_deg, 2)
                    detection_info["roll_confidence"] = round(roll_conf, 2)

                detection_info["horizon_row"] = int(horizon_row)
                detection_info["image_height"] = int(h)

            # Brightest pixel -> approximate sun direction
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, _, _, max_loc = cv2.minMaxLoc(gray)
            sun_x = (max_loc[0] - w / 2) / (w / 2)
            sun_y = -(max_loc[1] - h / 2) / (h / 2)
            sun_direction = np.array([sun_x, sun_y + 0.5, 0.8], dtype=np.float32)
            sun_direction /= np.linalg.norm(sun_direction)
        else:
            sun_direction = np.array([0.5, 0.5, 0.8], dtype=np.float32)
    else:
        sun_direction = np.array([0.5, 0.5, 0.8], dtype=np.float32)

    light_multiplier = 1.5

    camera_data = {
        "prompt_id": prompt_id,
        "fov": round(fov, 2),
        "camera_location": camera_location,
        "camera_rotation": camera_rotation,
        "sun_direction": sun_direction.tolist(),
        "light_intensity_multiplier": light_multiplier,
        "detection_info": detection_info,
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(camera_data, f, indent=4)

    print(f"[Camera Matcher] Camera saved -> {output_json_path}")
    print(f"[Camera Matcher] Location: {camera_location}, FOV: {fov}, Pitch: {camera_rotation[0]}")
    return output_json_path


def main():
    parser = argparse.ArgumentParser(description="Camera Parameter Extraction")
    parser.add_argument("--prompt_id", default="prompt_01")
    parser.add_argument("--input", required=False, default="", help="Background image")
    parser.add_argument("--depth", required=False, default=None, help="Depth map")
    parser.add_argument("--output", required=True, help="camera.json")
    args = parser.parse_args()

    extract_camera_parameters(
        prompt_id=args.prompt_id,
        input_image_path=args.input,
        output_json_path=args.output,
        depth_path=args.depth,
    )


if __name__ == "__main__":
    main()