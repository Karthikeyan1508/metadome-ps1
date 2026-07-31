"""
Stage 1 - AI Environment Background Generation
Author: Ratish (Person A - AI & Data Pipeline)

Generates background environment plates using HuggingFace Diffusers (SDXL/FLUX)
or API fallbacks (Replicate / Skybox AI).
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

def generate_background_fallback(prompt: str, output_path: str, width: int = 1920, height: int = 1080):
    """
    Procedural fallback background generator when local GPU / API is unavailable.
    Creates a high-contrast gradient scene with realistic sky/ground horizon.
    """
    print(f"[Fallback Gen] Generating synthetic environment plate for: '{prompt[:30]}...'")
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    
    # Sky to ground gradient
    horizon_y = int(height * 0.55)
    
    # Sky gradient
    for y in range(horizon_y):
        ratio = y / horizon_y
        r = int(30 + ratio * 150)
        g = int(40 + ratio * 130)
        b = int(70 + ratio * 160)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
        
    # Sun / Key light spot
    sun_x, sun_y = int(width * 0.7), int(height * 0.25)
    for r in range(80, 0, -2):
        alpha = int(255 * (1.0 - r / 80))
        draw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=(255, 240, 200))
        
    # Ground asphalt gradient
    for y in range(horizon_y, height):
        ratio = (y - horizon_y) / (height - horizon_y)
        val = int(25 + ratio * 45)
        draw.line([(0, y), (width, y)], fill=(val, val, val + 5))
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    print(f"[Fallback Gen] Saved background plate to: {output_path}")
    return output_path

def generate_background_sdxl(prompt: str, output_path: str, width: int = 1920, height: int = 1080):
    """
    Generates background plate using local Diffusers SDXL pipeline.
    """
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
        
        print(f"[SDXL Gen] Initializing SDXL pipeline...")
        print(f"[SDXL Gen] Loading model 'stabilityai/stable-diffusion-xl-base-1.0' (Downloading ~6.6GB model weights if first run)...")
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe.to("cuda")
        
        print(f"[SDXL Gen] Generating AI background image for prompt: '{prompt[:40]}...'")
        negative_prompt = "car, vehicle, auto, traffic, text, watermark, blurred, low quality, distorted"
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"[SDXL Gen PASS] Successfully saved background plate: {output_path}")
        return output_path
    except Exception as e:
        print(f"[SDXL Gen Notice] Local SDXL GPU run unavailable or interrupted ({e}). Switching to procedural background generator.")
        return generate_background_fallback(prompt, output_path, width, height)

def main():
    parser = argparse.ArgumentParser(description="Generate AI Environment Background Plate")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for background")
    parser.add_argument("--output", type=str, required=True, help="Output image file path (.png)")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")
    parser.add_argument("--use_fallback", action="store_true", help="Force synthetic fallback generator")
    
    args = parser.parse_args()
    
    if args.use_fallback:
        generate_background_fallback(args.prompt, args.output, args.width, args.height)
    else:
        generate_background_sdxl(args.prompt, args.output, args.width, args.height)

if __name__ == "__main__":
    main()
