"""
Stage 1 - AI Environment Background Generation
FLUX.1-dev via HuggingFace Inference API — Optimized for photorealism.
Free tier. Each prompt = unique high-quality 1920x1080 background.
"""

import os
import sys
import argparse
import requests
import io
import time
from PIL import Image
import numpy as np

# ============================================================
# PASTE YOUR HUGGINGFACE TOKEN HERE
# Get it free from: https://huggingface.co/settings/tokens
# ============================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def generate_background(prompt: str, output_path: str, width: int = 1920, height: int = 1080):
    """
    Generate photorealistic background using FLUX with optimized prompts.
    """
    
    # ============================================================
    # QUALITY-OPTIMIZED PROMPT ENGINEERING
    # ============================================================
    
    # Negative prompt: what we DON'T want in the image
    negative_prompt = (
        "car, vehicle, automobile, truck, people, person, text, watermark, "
        "logo, signature, blurry, blur, low quality, low resolution, distorted, "
        "cartoon, 3d render, CGI, artificial, fake, painting, illustration, drawing"
    )
    
    # Enhanced positive prompt with quality boosters
    enhanced_prompt = (
        f"{prompt}, "
        "breathtaking, 8k ultra HD, photorealistic, professional landscape photography, "
        "award-winning photo, National Geographic, sharp focus, natural lighting, "
        "high contrast, rich colors, detailed texture, wide angle lens, 16:9 aspect ratio, "
        "DSLR, RAW, unedited, realistic, hyperrealistic"
    )
    
    print(f"[FLUX] Generating: '{prompt[:60]}...'")
    print(f"[FLUX] Resolution: {width}x{height}")
    
    # ============================================================
    # OPTIMIZED API PARAMETERS
    # ============================================================
    payload = {
        "inputs": enhanced_prompt,
        "parameters": {
            "width": width,
            "height": height,
            "num_inference_steps": 30,      # More steps = better quality (was 28)
            "guidance_scale": 5.0,           # Higher = follows prompt better (was 3.5)
            "negative_prompt": negative_prompt,
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[FLUX] Sending request (attempt {attempt+1}/{max_retries})...")
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=180)
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                
                # ============================================================
                # POST-PROCESSING: Enhance image quality
                # ============================================================
                image = enhance_image_quality(image)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path, "PNG", quality=100)
                
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"[FLUX] ✅ Saved: {output_path}")
                print(f"[FLUX] Resolution: {image.size[0]}x{image.size[1]}, Size: {file_size_mb:.1f}MB")
                return output_path
                
            elif response.status_code == 503:
                wait_time = (attempt + 1) * 20
                print(f"[FLUX] Model loading (cold start), waiting {wait_time}s...")
                time.sleep(wait_time)
                
            elif response.status_code == 429:
                print(f"[FLUX] Rate limited. Waiting 45s...")
                time.sleep(45)
                
            else:
                print(f"[FLUX] ❌ Error {response.status_code}")
                try:
                    error_msg = response.json()
                    print(f"[FLUX] {error_msg}")
                except:
                    print(f"[FLUX] {response.text[:300]}")
                    
                if "loading" in str(response.text).lower():
                    time.sleep(30)
                    continue
                break
                
        except requests.exceptions.Timeout:
            print(f"[FLUX] Request timed out. Retrying...")
            time.sleep(15)
            
        except Exception as e:
            print(f"[FLUX] ❌ Failed: {e}")
            break
    
    print(f"[FLUX] API failed. Using enhanced procedural fallback.")
    return generate_enhanced_fallback(prompt, output_path, width, height)


def enhance_image_quality(image):
    """
    Post-process image to improve sharpness, contrast, and color.
    """
    from PIL import ImageEnhance
    
    # Convert to array
    img_array = np.array(image, dtype=np.float32)
    
    # 1. Increase contrast slightly
    mean = np.mean(img_array, axis=(0, 1), keepdims=True)
    img_array = mean + (img_array - mean) * 1.1  # 10% contrast boost
    
    # 2. Clip values
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL
    image = Image.fromarray(img_array)
    
    # 3. Sharpen
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)  # 30% sharper
    
    # 4. Boost color saturation slightly
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.15)  # 15% more vibrant
    
    return image


def generate_enhanced_fallback(prompt: str, output_path: str, width: int = 1920, height: int = 1080):
    """
    Enhanced procedural fallback with gradients, noise texture, and vignette.
    Creates more realistic-looking scenes than flat gradients.
    """
    from PIL import ImageFilter
    
    print(f"[Fallback+] Generating enhanced procedural background...")
    
    prompt_lower = prompt.lower()
    
    # Environment detection
    if "night" in prompt_lower or "neon" in prompt_lower:
        env = "night"
    elif "forest" in prompt_lower or "mist" in prompt_lower or "pine" in prompt_lower:
        env = "forest"
    elif "desert" in prompt_lower or "sand" in prompt_lower or "golden hour" in prompt_lower:
        env = "desert"
    elif "racetrack" in prompt_lower or "grid" in prompt_lower or "tarmac" in prompt_lower:
        env = "racetrack"
    elif "coastal" in prompt_lower or "sunset" in prompt_lower or "ocean" in prompt_lower:
        env = "coastal"
    else:
        env = "default"
    
    # Create base image
    img_array = np.zeros((height, width, 3), dtype=np.float32)
    y_norm = np.linspace(0, 1, height).reshape(-1, 1)
    x_norm = np.linspace(0, 1, width).reshape(1, -1)
    
    # Generate random noise for texture (same seed per prompt = consistent)
    seed = hash(prompt) % 100000
    np.random.seed(seed)
    noise = np.random.randn(height, width, 3) * 8  # Subtle noise texture
    
    # Horizon line
    horizon = 0.5
    
    if env == "night":
        # Deep night sky gradient
        sky_r = 8 + y_norm * 18
        sky_g = 8 + y_norm * 15
        sky_b = 25 + y_norm * 35
        
        # Stars
        star_mask = (np.random.rand(height, width) > 0.997) & (y_norm < horizon)
        
        img_array[:, :, 0] = np.where(y_norm < horizon, sky_r, 12)
        img_array[:, :, 1] = np.where(y_norm < horizon, sky_g, 12)
        img_array[:, :, 2] = np.where(y_norm < horizon, sky_b, 18)
        
        # Stars
        img_array[star_mask, :] = [200, 220, 255]
        
        # Wet ground reflection
        ground = y_norm >= horizon
        img_array[ground, 0] = 10 + (y_norm[ground] - horizon) * 10
        img_array[ground, 1] = 10 + (y_norm[ground] - horizon) * 10
        img_array[ground, 2] = 15 + (y_norm[ground] - horizon) * 12
        
        # Neon glow spots
        for _ in range(5):
            nx = int(np.random.uniform(100, width - 100))
            ny = int(np.random.uniform(50, height * 0.35))
            glow_radius = int(np.random.uniform(20, 60))
            color = np.random.choice([[255, 0, 100], [0, 220, 255], [255, 180, 0]])
            
            y_min = max(0, ny - glow_radius)
            y_max = min(height, ny + glow_radius)
            x_min = max(0, nx - glow_radius)
            x_max = min(width, nx + glow_radius)
            
            for c in range(3):
                dist = np.sqrt((x_norm[:, x_min:x_max] * width - nx)**2 + 
                              (y_norm[y_min:y_max, :] * height - ny)**2)
                falloff = np.clip(1 - dist / glow_radius, 0, 1)
                img_array[y_min:y_max, x_min:x_max, c] += color[c] * falloff * 0.4
        
    elif env == "forest":
        # Misty atmosphere
        sky_r = 90 + y_norm * 35
        sky_g = 100 + y_norm * 30
        sky_b = 95 + y_norm * 25
        
        img_array[:, :, 0] = np.where(y_norm < horizon, sky_r, 22)
        img_array[:, :, 1] = np.where(y_norm < horizon, sky_g, 28)
        img_array[:, :, 2] = np.where(y_norm < horizon, sky_b, 25)
        
        # Fog effect
        fog = np.random.randn(height, width) * 5
        img_array[:, :, 0] += fog * 0.3
        img_array[:, :, 1] += fog * 0.4
        img_array[:, :, 2] += fog * 0.3
        
        # Tree silhouettes
        for _ in range(25):
            tx = int(np.random.uniform(50, width - 50))
            th = int(np.random.uniform(80, 350))
            ty_start = max(0, int(height * horizon) - th)
            tree_width = int(np.random.uniform(4, 20))
            
            if ty_start > 0:
                img_array[ty_start:int(height * horizon), 
                         tx - tree_width:tx + tree_width, :] = [20, 28, 22]
        
    elif env == "desert":
        # Golden hour sky
        sky_r = 235 - y_norm * 55
        sky_g = 100 + y_norm * 65
        sky_b = 25 + y_norm * 45
        
        img_array[:, :, 0] = np.where(y_norm < horizon, sky_r, 185)
        img_array[:, :, 1] = np.where(y_norm < horizon, sky_g, 135)
        img_array[:, :, 2] = np.where(y_norm < horizon, sky_b, 65)
        
        # Sun
        sun_x, sun_y = int(width * 0.72), int(height * 0.22)
        sun_dist = np.sqrt((x_norm * width - sun_x)**2 + (y_norm * height - sun_y)**2)
        sun_glow = np.exp(-sun_dist / 100)
        img_array[:, :, 0] += sun_glow * 30
        img_array[:, :, 1] += sun_glow * 15
        img_array[:, :, 2] -= sun_glow * 5
        
        # Heat shimmer effect
        heat = np.sin(y_norm * 80 + x_norm * 20) * 3
        img_array[:, :, 0] += heat * 0.5
        
    elif env == "racetrack":
        # Moody overcast
        sky_r = 50 + y_norm * 50
        sky_g = 55 + y_norm * 50
        sky_b = 65 + y_norm * 48
        
        img_array[:, :, 0] = np.where(y_norm < horizon, sky_r, 20)
        img_array[:, :, 1] = np.where(y_norm < horizon, sky_g, 20)
        img_array[:, :, 2] = np.where(y_norm < horizon, sky_b, 22)
        
        # Track perspective lines
        cx = width // 2
        for line_offset in [-1, 1]:
            x1 = int(cx + line_offset * width * 0.02)
            x2 = int(cx + line_offset * width * 0.35)
            for y in range(int(height * horizon), height):
                t = (y - height * horizon) / (height * (1 - horizon))
                x = int(x1 + (x2 - x1) * t)
                img_array[y, max(0, x-2):min(width, x+2), :] = [180, 180, 180]
        
        # Horizontal grid lines
        for i in range(1, 6):
            y_pos = int(height * horizon + (height * (1 - horizon)) * (i / 5))
            line_width = int(40 + width * 0.15 * (i / 5))
            img_array[y_pos-1:y_pos+1, cx - line_width:cx + line_width, :] = [200, 200, 200]
    
    elif env == "coastal":
        # Sunset sky
        sky_r = 225 - y_norm * 80
        sky_g = 55 + y_norm * 75
        sky_b = 105 + y_norm * 65
        
        img_array[:, :, 0] = sky_r
        img_array[:, :, 1] = sky_g
        img_array[:, :, 2] = sky_b
        
        # Ocean (left) + road (right)
        ground = y_norm >= horizon
        left = x_norm < 0.45
        
        for i in range(height):
            if ground[i, 0]:
                for j in range(width):
                    if left[0, j]:
                        # Ocean with wave variation
                        wave = np.sin(j * 0.05 + i * 0.1) * 8
                        img_array[i, j] = [12, 25 + wave, 55 + wave]
                    else:
                        # Road
                        img_array[i, j] = [28, 28, 30]
        
        # Sun reflection on water
        sun_x = width // 2
        for y in range(int(height * horizon), height):
            for j in range(width):
                if left[0, j]:
                    dist_from_sun = abs(j - sun_x)
                    if dist_from_sun < 40:
                        reflection = (1 - dist_from_sun / 40) * 80
                        img_array[y, j, 0] += reflection * 0.6
                        img_array[y, j, 1] += reflection * 0.3
    else:
        # Generic outdoor scene
        img_array[:, :, 0] = 40 + y_norm * 180
        img_array[:, :, 1] = 50 + y_norm * 160
        img_array[:, :, 2] = 80 + y_norm * 200
        
        ground = y_norm >= horizon
        img_array[ground, 0] = 55
        img_array[ground, 1] = 55
        img_array[ground, 2] = 60
    
    # Add noise texture
    img_array += noise
    
    # Apply vignette
    center_x, center_y = width / 2, height / 2
    dist_from_center = np.sqrt((x_norm * width - center_x)**2 + (y_norm * height - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = 1 - (dist_from_center / max_dist) * 0.4  # Darken edges by 40%
    vignette = np.clip(vignette, 0.6, 1.0)
    
    for c in range(3):
        img_array[:, :, c] *= vignette
    
    # Clip and convert
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_array)
    
    # Slight blur to mimic atmospheric haze
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, "PNG", quality=100)
    print(f"[Fallback+] ✅ Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    
    args = parser.parse_args()
    
    if HF_TOKEN and HF_TOKEN != "hf_" and len(HF_TOKEN) > 10:
        generate_background(args.prompt, args.output, args.width, args.height)
    else:
        print("[WARNING] No valid HuggingFace token. Using enhanced procedural fallback.")
        generate_enhanced_fallback(args.prompt, args.output, args.width, args.height)


if __name__ == "__main__":
    main()
