"""
Stage 1 - HDR Light Map Estimation & EXR Converter
Author: Ratish (Person A - AI & Data Pipeline)

Converts 8-bit LDR background plates or 360 panoramas into 32-bit floating point 
.exr environment maps by isolating key light sources and boosting high-luminance peaks.
"""

import os
import sys
import argparse
import numpy as np
import cv2

def convert_ldr_to_hdr_exr(input_image_path: str, output_exr_path: str, sun_boost: float = 15.0):
    """
    Reads an 8-bit LDR image (PNG/JPG), applies inverse tone mapping and specular intensity boost,
    and exports a 32-bit float OpenEXR environment map.
    """
    print(f"[HDR Estimator] Processing input plate: {input_image_path}")
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
        
    # Read image in BGR, convert to float32 normalized [0.0, 1.0]
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {input_image_path}")
        
    img_float = img.astype(np.float32) / 255.0
    
    # Apply gamma linearisation (Gamma ~ 2.2)
    img_linear = np.power(img_float, 2.2)
    
    # Calculate luminance map
    luminance = 0.2126 * img_linear[:, :, 2] + 0.7152 * img_linear[:, :, 1] + 0.0722 * img_linear[:, :, 0]
    
    # Identify bright light spots (top 5% luminance)
    threshold = np.percentile(luminance, 95)
    bright_mask = np.maximum(0.0, (luminance - threshold) / (1.0 - threshold + 1e-5))
    bright_mask_3ch = np.dstack([bright_mask, bright_mask, bright_mask])
    
    # Apply exponential light boost to light sources (sun / lamps)
    hdr_map = img_linear + (img_linear * bright_mask_3ch * sun_boost)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_exr_path), exist_ok=True)
    
    # Write to 32-bit float OpenEXR or HDR format
    try:
        import imageio
        imageio.imwrite(output_exr_path, hdr_map.astype(np.float32))
        print(f"[HDR Estimator] Successfully exported EXR HDRI map: {output_exr_path}")
    except Exception:
        fallback_hdr = output_exr_path.replace(".exr", ".hdr")
        cv2.imwrite(fallback_hdr, hdr_map.astype(np.float32))
        print(f"[HDR Estimator] Exported fallback HDR map: {fallback_hdr}")
        
    return output_exr_path

def main():
    parser = argparse.ArgumentParser(description="Convert LDR background plate to EXR HDRI environment map")
    parser.add_argument("--input", type=str, required=True, help="Input LDR image path (.png/.jpg)")
    parser.add_argument("--output", type=str, required=True, help="Output HDR environment map path (.exr/.hdr)")
    parser.add_argument("--boost", type=float, default=15.0, help="Sun/Light intensity multiplier")
    
    args = parser.parse_args()
    convert_ldr_to_hdr_exr(args.input, args.output, args.boost)

if __name__ == "__main__":
    main()
