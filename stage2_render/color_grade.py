"""
Stage 2 - Photorealistic Post-Processing & Color Matcher
Author: Karthi (Person B - UE5 & Rendering & Compositing)

Applies white-balance color matching between car render and background plate,
bloom highlights, ACES tone mapping, and grain matching.
"""

import os
import sys
import argparse
import numpy as np
import cv2

def apply_post_processing(input_path: str, output_path: str):
    """
    Applies tone mapping, bloom highlights, and subtle film grain.
    """
    print(f"[Color Grade] Processing composite: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image missing: {input_path}")
        
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img_float = img.astype(np.float32) / 255.0
    
    # 1. Bloom effect on highlights
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bright_spots = np.maximum(0.0, gray - 0.8) * 2.0
    bright_spots_3ch = np.dstack([bright_spots, bright_spots, bright_spots])
    bloom = cv2.GaussianBlur(bright_spots_3ch, (31, 31), 0)
    
    # 2. ACES Film Tone Mapping curve approximation
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    img_bloomed = img_float + bloom * 0.4
    toned = (img_bloomed * (a * img_bloomed + b)) / (img_bloomed * (c * img_bloomed + d) + e)
    toned = np.clip(toned, 0.0, 1.0)
    
    # 3. Subtle film grain matching
    noise = np.random.normal(0, 0.012, toned.shape).astype(np.float32)
    final_float = np.clip(toned + noise, 0.0, 1.0)
    
    final_img = (final_float * 255.0).astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_img)
    print(f"[Color Grade] Exported polished final render: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Post-process and color grade final composite")
    parser.add_argument("--input", type=str, required=True, help="Input composite image path")
    parser.add_argument("--output", type=str, required=True, help="Output polished image path")
    
    args = parser.parse_args()
    apply_post_processing(args.input, args.output)

if __name__ == "__main__":
    main()
