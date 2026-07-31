"""
Stage 2 - Multi-Pass OpenCV Compositor Engine
Author: Karthi (Person B - UE5 & Rendering & Compositing)

Takes:
1. GenAI Background Plate (PNG)
2. Car Render Beauty Pass (PNG with Alpha channel)
3. Optional Ground Shadow Pass (Grayscale / Alpha)
and composites them into a realistic photograph.
"""

import os
import sys
import argparse
import numpy as np
import cv2

def composite_layers(background_path: str, render_path: str, output_path: str, shadow_path: str = None, shadow_strength: float = 0.85):
    """
    Blends background plate and path-traced vehicle render with contact shadow math.
    """
    print(f"[Compositor] Loading background: {background_path}")
    print(f"[Compositor] Loading render pass: {render_path}")
    
    if not os.path.exists(background_path):
        raise FileNotFoundError(f"Background image missing: {background_path}")
        
    bg = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if bg is None:
        raise ValueError(f"Could not load background plate: {background_path}")
        
    h, w, _ = bg.shape
    composite = bg.copy()
    
    # 1. Apply Ground Shadow Pass if provided
    if shadow_path and os.path.exists(shadow_path):
        print(f"[Compositor] Processing ground shadow map: {shadow_path}")
        shadow_img = cv2.imread(shadow_path, cv2.IMREAD_GRAYSCALE)
        if shadow_img is not None:
            shadow_resized = cv2.resize(shadow_img, (w, h))
            # Blur the shadow map slightly for realistic soft edges
            shadow_blurred = cv2.GaussianBlur(shadow_resized, (9, 9), 0)
            # Normalize to [0, 1] range representing shadowing factor (0=full shadow, 1=no shadow)
            shadow_factor = shadow_blurred.astype(float) / 255.0
            
            # Apply shadow multiplier onto background plate
            for c in range(3):
                composite[:, :, c] = (composite[:, :, c] * (1.0 - (1.0 - shadow_factor) * shadow_strength)).astype(np.uint8)
    else:
        # Procedural fallback contact shadow if no pass exists
        print("[Compositor] No shadow pass detected. Generating procedural contact shadow.")
        car_w, car_h = int(w * 0.45), int(h * 0.22)
        car_y = int(h * 0.52)
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(shadow_mask, (int(w/2), car_y + car_h - 10), (int(car_w * 0.55), 25), 0, 0, 360, 255, -1)
        shadow_blurred = cv2.GaussianBlur(shadow_mask, (31, 31), 0)
        shadow_factor = 1.0 - (shadow_blurred.astype(float) / 255.0) * 0.65
        
        for c in range(3):
            composite[:, :, c] = (composite[:, :, c] * shadow_factor).astype(np.uint8)

    # 2. Overlay Transparent Car Beauty Pass
    if os.path.exists(render_path):
        fg = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
        if fg is not None and fg.shape[2] == 4:
            fg_resized = cv2.resize(fg, (w, h))
            alpha = (fg_resized[:, :, 3].astype(float) / 255.0)[:, :, np.newaxis]
            fg_bgr = fg_resized[:, :, :3]
            composite = (fg_bgr * alpha + composite * (1.0 - alpha)).astype(np.uint8)
        else:
            print("[Compositor Warning] Render pass is not 4-channel transparent PNG. Applying overlay blend.")
            fg = cv2.imread(render_path, cv2.IMREAD_COLOR)
            if fg is not None:
                fg_resized = cv2.resize(fg, (w, h))
                composite = cv2.addWeighted(composite, 0.3, fg_resized, 0.7, 0)
    else:
        print(f"[Compositor Warning] Render pass missing at: {render_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, composite)
    print(f"[Compositor PASS] Successfully exported final composite: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Composite Background and Car Render Pass")
    parser.add_argument("--bg", type=str, required=True, help="Background image path")
    parser.add_argument("--render", type=str, required=True, help="Car render pass image path")
    parser.add_argument("--shadow", type=str, default=None, help="Optional shadow map image path")
    parser.add_argument("--output", type=str, required=True, help="Output composite image path")
    
    args = parser.parse_args()
    composite_layers(args.bg, args.render, args.output, args.shadow)

if __name__ == "__main__":
    main()
