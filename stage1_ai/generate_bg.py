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
    Creates a high-contrast gradient scene tailored to the prompt keywords.
    """
    print(f"[Fallback Gen] Generating synthetic environment plate for: '{prompt[:40]}...'")
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    
    prompt_lower = prompt.lower()
    
    # Identify environment type from prompt keywords
    if "night" in prompt_lower or "neon" in prompt_lower:
        env_type = "night"
    elif "forest" in prompt_lower or "mist" in prompt_lower:
        env_type = "forest"
    elif "desert" in prompt_lower or "sand" in prompt_lower:
        env_type = "desert"
    elif "racetrack" in prompt_lower or "grid" in prompt_lower:
        env_type = "racetrack"
    elif "coastal" in prompt_lower or "sunset" in prompt_lower:
        env_type = "coastal"
    else:
        env_type = "default"
        
    horizon_y = int(height * 0.55)
    
    if env_type == "night":
        # Dark purple to dark blue sky
        for y in range(horizon_y):
            ratio = y / horizon_y
            r = int(10 + ratio * 15)
            g = int(10 + ratio * 15)
            b = int(25 + ratio * 35)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        # Draw some bright neon rectangles (representing buildings/neon signs)
        import random
        # Seed random for deterministic outputs based on prompt
        random.seed(hash(prompt) % 10000)
        for _ in range(8):
            sign_w = random.randint(30, 80)
            sign_h = random.randint(100, 300)
            sign_x = random.randint(100, width - 100)
            sign_y = horizon_y - sign_h - random.randint(10, 50)
            color = random.choice([(255, 0, 128), (0, 255, 255), (255, 255, 0), (0, 255, 0)])
            draw.rectangle([sign_x, sign_y, sign_x + sign_w, sign_y + sign_h], outline=color, width=3)
            
        # Wet ground with reflections
        for y in range(horizon_y, height):
            ratio = (y - horizon_y) / (height - horizon_y)
            val = int(15 + ratio * 15)
            draw.line([(0, y), (width, y)], fill=(val, val, val + 5))
            
        # Draw reflection blobs on road
        for _ in range(12):
            ref_w = random.randint(50, 200)
            ref_h = random.randint(4, 12)
            ref_x = random.randint(0, width - ref_w)
            ref_y = random.randint(horizon_y + 10, height - ref_h)
            color = random.choice([(100, 0, 50), (0, 100, 100), (100, 100, 0)])
            draw.ellipse([ref_x, ref_y, ref_x + ref_w, ref_y + ref_h], fill=color)

    elif env_type == "forest":
        # Misty grey-green sky
        for y in range(horizon_y):
            ratio = y / horizon_y
            r = int(90 + ratio * 30)
            g = int(105 + ratio * 25)
            b = int(100 + ratio * 20)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        # Draw soft white sun behind fog
        sun_x, sun_y = int(width * 0.5), int(height * 0.25)
        for r in range(150, 0, -4):
            alpha = int(30 * (1.0 - r / 150))
            draw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=(255, 255, 255))
            
        # Draw soft tree silhouettes (vertical lines with some width)
        import random
        random.seed(hash(prompt) % 10000)
        for _ in range(15):
            tree_w = random.randint(8, 25)
            tree_h = random.randint(150, 400)
            tree_x = random.randint(50, width - 50)
            tree_y = horizon_y - tree_h
            draw.rectangle([tree_x, tree_y, tree_x + tree_w, horizon_y], fill=(45, 60, 50))
            
        # Damp asphalt ground
        for y in range(horizon_y, height):
            ratio = (y - horizon_y) / (height - horizon_y)
            val = int(30 + ratio * 20)
            draw.line([(0, y), (width, y)], fill=(val, val + 5, val))

    elif env_type == "desert":
        # Golden orange/red sky
        for y in range(horizon_y):
            ratio = y / horizon_y
            r = int(240 - ratio * 40)
            g = int(110 + ratio * 50)
            b = int(20 + ratio * 40)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        # Bright sun on right
        sun_x, sun_y = int(width * 0.75), int(height * 0.2)
        for r in range(90, 0, -3):
            draw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=(255, 240, 180))
            
        # Warm desert highway ground
        for y in range(horizon_y, height):
            ratio = (y - horizon_y) / (height - horizon_y)
            val_road = int(35 + ratio * 25)
            draw.line([(0, y), (width, y)], fill=(val_road + 30, val_road + 15, val_road))
            
        # Draw central asphalt highway perspective lane
        draw.polygon([(int(width * 0.45), horizon_y), (int(width * 0.55), horizon_y), (int(width * 0.8), height), (int(width * 0.2), height)], fill=(40, 38, 38))

    elif env_type == "racetrack":
        # Steel grey moody overcast sky
        for y in range(horizon_y):
            ratio = y / horizon_y
            r = int(50 + ratio * 40)
            g = int(55 + ratio * 40)
            b = int(65 + ratio * 40)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        # Dark wet asphalt racetrack
        for y in range(horizon_y, height):
            ratio = (y - horizon_y) / (height - horizon_y)
            val = int(20 + ratio * 25)
            draw.line([(0, y), (width, y)], fill=(val, val, val + 2))
            
        # Draw white starting grid perspective lines
        draw.line([(int(width * 0.45), horizon_y), (int(width * 0.15), height)], fill=(200, 200, 200), width=4)
        draw.line([(int(width * 0.55), horizon_y), (int(width * 0.85), height)], fill=(200, 200, 200), width=4)
        for i in range(1, 5):
            y_pos = int(horizon_y + (height - horizon_y) * (i / 4.0))
            w_offset = int((width * 0.1) + (width * 0.2) * (i / 4.0))
            draw.line([(int(width * 0.5 - w_offset), y_pos), (int(width * 0.5 + w_offset), y_pos)], fill=(200, 200, 200), width=3)

    elif env_type == "coastal":
        # Pink/purple sunset sky
        for y in range(horizon_y):
            ratio = y / horizon_y
            r = int(230 - ratio * 60)
            g = int(70 + ratio * 60)
            b = int(120 + ratio * 50)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        # Sun setting at horizon center
        sun_x, sun_y = int(width * 0.5), horizon_y
        for r in range(100, 0, -4):
            draw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=(255, 210, 120))
            
        # Ground: Left side ocean, right side cliff road
        for y in range(horizon_y, height):
            ratio = (y - horizon_y) / (height - horizon_y)
            sea_val = int(30 + ratio * 50)
            road_val = int(25 + ratio * 35)
            draw.line([(0, y), (int(width * 0.45), y)], fill=(10, 20 + sea_val // 2, sea_val))
            draw.line([(int(width * 0.45), y), (width, y)], fill=(road_val, road_val, road_val))
            
    else:
        # Default gradient
        for y in range(horizon_y):
            ratio = y / horizon_y
            r = int(30 + ratio * 150)
            g = int(40 + ratio * 130)
            b = int(70 + ratio * 160)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        sun_x, sun_y = int(width * 0.7), int(height * 0.25)
        for r in range(80, 0, -2):
            draw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=(255, 240, 200))
            
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
