"""
Stage 2 - Photorealistic Post-Processing & Color Matcher
Author: Karthi (Person B - UE5 & Rendering & Compositing)

Applies Reinhard color transfer to match car lighting temperature with the background,
plus specular bloom, ACES film tone mapping, and film grain matching.
"""

import os
import sys
import argparse
import numpy as np
import cv2

def color_transfer(source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """
    Performs Reinhard color transfer in LAB color space to match lighting temperature.
    Matches statistical distribution of source_img channels to target_img.
    """
    # Convert BGR to Lab color space
    src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Compute mean and standard deviation for source and target
    src_mean, src_std = cv2.meanStdDev(src_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)
    
    # Reshape stats for broadcasting
    src_mean = src_mean.reshape((1, 1, 3))
    src_std = src_std.reshape((1, 1, 3))
    tgt_mean = tgt_mean.reshape((1, 1, 3))
    tgt_std = tgt_std.reshape((1, 1, 3))
    
    # Normalize source, scale by target std, shift by target mean
    src_std = np.maximum(src_std, 1e-5) # Prevent divide-by-zero
    transfer = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    transfer = np.clip(transfer, 0.0, 255.0).astype(np.uint8)
    
    # Convert back to BGR
    result = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
    return result

def apply_post_processing(input_path: str, output_path: str, bg_path: str = None):
    """
    Applies tone mapping, bloom highlights, and subtle film grain.
    """
    print(f"[Color Grade] Processing composite: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image missing: {input_path}")
        
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    # 1. Apply Reinhard Color Temperature Transfer from Background
    if bg_path and os.path.exists(bg_path):
        print(f"[Color Grade] Transferring color temperature from background: {bg_path}")
        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        if bg is not None:
            # Resize background to match composite size for matching accuracy
            bg_resized = cv2.resize(bg, (img.shape[1], img.shape[0]))
            img = color_transfer(img, bg_resized)
            
    img_float = img.astype(np.float32) / 255.0
    
    # 2. Specular Bloom effect on highlights
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bright_spots = np.maximum(0.0, gray - 0.8) * 2.0
    bright_spots_3ch = np.dstack([bright_spots, bright_spots, bright_spots])
    bloom = cv2.GaussianBlur(bright_spots_3ch, (31, 31), 0)
    
    # 3. ACES Film Tone Mapping curve approximation
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    img_bloomed = img_float + bloom * 0.4
    toned = (img_bloomed * (a * img_bloomed + b)) / (img_bloomed * (c * img_bloomed + d) + e)
    toned = np.clip(toned, 0.0, 1.0)
    
    # 4. Subtle film grain matching
    noise = np.random.normal(0, 0.012, toned.shape).astype(np.float32)
    final_float = np.clip(toned + noise, 0.0, 1.0)
    
    final_img = (final_float * 255.0).astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_img)
    print(f"[Color Grade PASS] Exported polished final render: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Post-process and color grade final composite")
    parser.add_argument("--input", type=str, required=True, help="Input composite image path")
    parser.add_argument("--output", type=str, required=True, help="Output polished image path")
    parser.add_argument("--bg", type=str, default=None, help="Background image path for color matching")
    
    args = parser.parse_args()
    apply_post_processing(args.input, args.output, args.bg)

if __name__ == "__main__":
    main()
