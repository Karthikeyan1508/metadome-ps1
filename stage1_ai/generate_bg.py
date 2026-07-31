"""
Stage 1 - AI Environment Background Generation
Uses Pollinations.ai (free, no API key required) as the primary generator,
with Replicate SDXL as an optional paid fallback, and a procedural
generator as the final offline fallback.
Each prompt produces a unique photorealistic 1920x1080 background.
"""

import os
import sys
import argparse
import requests
import io
import time
from urllib.parse import quote
from PIL import Image, ImageEnhance
import numpy as np

# ============================================================
# Optional: only needed if you want the paid Replicate fallback.
# Leave blank to rely on Pollinations.ai (free) + procedural fallback.
# Get it free: https://replicate.com/account/api-tokens
# NEVER hardcode a real token here — load it from the environment
# (e.g. `export REPLICATE_API_TOKEN=...` before running the script).
# ============================================================
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")


def generate_background_pollinations(prompt: str, output_path: str,
                                      width: int = 1920, height: int = 1080,
                                      negative_prompt: str = "",
                                      model: str = "flux",
                                      seed: int = None,
                                      enhance: bool = True):
    """
    Generate an image using Pollinations.ai's free image API.
    No API key required. Uses the 'flux' model by default (photorealistic,
    good general-purpose quality). There is no native negative_prompt
    parameter in this API, so exclusions are folded into the main prompt
    text instead (e.g. "no people, no text, no watermark").
    """
    full_prompt = prompt
    if negative_prompt:
        exclusions = ", ".join(
            f"no {term.strip()}" for term in negative_prompt.split(",") if term.strip()
        )
        full_prompt = f"{prompt}, {exclusions}"

    encoded_prompt = quote(full_prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    params = {
        "width": width,
        "height": height,
        "model": model,
        "nologo": "true",
        "enhance": "true" if enhance else "false",
        "private": "true",
    }
    if seed is not None:
        params["seed"] = seed

    print(f"[Pollinations] Prompt: '{prompt[:80]}...'")
    print(f"[Pollinations] Requesting {width}x{height} image (model={model})...")

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))

        # Light post-processing to match the original pipeline's look
        image = ImageEnhance.Sharpness(image).enhance(1.15)
        image = ImageEnhance.Contrast(image).enhance(1.05)
        image = ImageEnhance.Color(image).enhance(1.05)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, "PNG", quality=100)

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Pollinations] Saved: {output_path}")
        print(f"[Pollinations] Size: {image.size[0]}x{image.size[1]}, {file_size:.1f}MB")
        return output_path

    except Exception as e:
        print(f"[Pollinations] Failed: {e}")
        if REPLICATE_API_TOKEN:
            print("[Pollinations] Falling back to Replicate...")
            return generate_background(prompt, output_path, width, height)
        print("[Pollinations] Falling back to procedural generator...")
        return generate_procedural_background(prompt, output_path, width, height)


def generate_background(prompt: str, output_path: str, width: int = 1920, height: int = 1080):
    """
    Generate photorealistic background using SDXL on Replicate.
    Optional, paid fallback — only used if REPLICATE_API_TOKEN is set.
    Post-processes for sharpness and contrast.
    """
    import replicate

    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

    enhanced_prompt = (
        f"{prompt}, photorealistic, professional landscape photography, "
        f"8k ultra HD, sharp focus, natural lighting, rich colors, wide angle, "
        f"National Geographic style, highly detailed"
    )

    negative_prompt = (
        "car, vehicle, automobile, truck, people, person, text, watermark, "
        "logo, blurry, low quality, distorted, cartoon, CGI, 3d render, "
        "painting, illustration, drawing, artificial"
    )

    print(f"[Replicate] Prompt: '{prompt[:80]}...'")
    print(f"[Replicate] Generating (3-10 seconds)...")

    try:
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_outputs": 1,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "refine": "expert_ensemble_refiner",
                "refine_steps": 15,
            }
        )

        image_url = output[0] if isinstance(output, list) else output

        print(f"[Replicate] Downloading image...")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))

        image = ImageEnhance.Sharpness(image).enhance(1.2)
        image = ImageEnhance.Contrast(image).enhance(1.1)
        image = ImageEnhance.Color(image).enhance(1.1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, "PNG", quality=100)

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Replicate] Saved: {output_path}")
        print(f"[Replicate] Size: {image.size[0]}x{image.size[1]}, {file_size:.1f}MB")
        return output_path

    except Exception as e:
        print(f"[Replicate] Failed: {e}")
        print(f"[Replicate] Falling back to procedural generator...")
        return generate_procedural_background(prompt, output_path, width, height)


def generate_procedural_background(prompt: str, output_path: str,
                                    width: int = 1920, height: int = 1080):
    """
    Procedural fallback — creates realistic gradient + texture backgrounds.
    Fully offline, no network calls.
    """
    import numpy as np
    from PIL import ImageFilter

    print(f"[Fallback] Generating procedural background...")

    prompt_lower = prompt.lower()

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

    img_array = np.zeros((height, width, 3), dtype=np.float32)
    y_norm = np.linspace(0, 1, height).reshape(-1, 1)
    x_norm = np.linspace(0, 1, width).reshape(1, -1)

    seed = hash(prompt) % 100000
    np.random.seed(seed)
    noise = np.random.randn(height, width, 3) * 6

    horizon = 0.5

    color_schemes = {
        "night": {
            "sky": [8, 8, 28],
            "sky_grad": [18, 15, 35],
            "ground": [12, 12, 18],
            "ground_grad": [8, 8, 10]
        },
        "forest": {
            "sky": [90, 100, 95],
            "sky_grad": [35, 30, 25],
            "ground": [22, 28, 25],
            "ground_grad": [5, 5, 5]
        },
        "desert": {
            "sky": [235, 100, 25],
            "sky_grad": [-55, 65, 45],
            "ground": [185, 135, 65],
            "ground_grad": [10, 10, 10]
        },
        "racetrack": {
            "sky": [50, 55, 65],
            "sky_grad": [50, 50, 48],
            "ground": [20, 20, 22],
            "ground_grad": [5, 5, 5]
        },
        "coastal": {
            "sky": [225, 55, 105],
            "sky_grad": [-80, 75, 65],
            "ground": [28, 28, 30],
            "ground_grad": [5, 5, 5]
        },
        "default": {
            "sky": [40, 50, 80],
            "sky_grad": [180, 160, 200],
            "ground": [55, 55, 60],
            "ground_grad": [10, 10, 10]
        }
    }

    scheme = color_schemes[env]

    for c in range(3):
        img_array[:, :, c] = scheme["sky"][c] + y_norm * scheme["sky_grad"][c]

    ground_mask = (y_norm >= horizon).flatten()  # shape (height,)
    for c in range(3):
        ground_color = scheme["ground"][c] + (y_norm - horizon) * scheme["ground_grad"][c] * 5
        # ground_color has shape (height, 1); broadcast across width, then apply row mask
        img_array[ground_mask, :, c] = np.broadcast_to(ground_color, (height, width))[ground_mask]

    img_array += noise

    cx, cy = width / 2, height / 2
    dist = np.sqrt((x_norm * width - cx)**2 + (y_norm * height - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1 - (dist / max_dist) * 0.35
    vignette = np.clip(vignette, 0.65, 1.0)
    img_array *= vignette[:, :, np.newaxis]

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_array)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, "PNG", quality=100)
    print(f"[Fallback] Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="",
                         help="Comma-separated list of things to avoid (folded into the prompt for Pollinations)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default="flux", help="Pollinations model, e.g. flux or turbo")
    parser.add_argument("--engine", type=str, default="pollinations",
                         choices=["pollinations", "replicate", "procedural"],
                         help="Which backend to use (default: pollinations, free)")

    args = parser.parse_args()

    if args.engine == "pollinations":
        generate_background_pollinations(
            args.prompt, args.output, args.width, args.height,
            negative_prompt=args.negative_prompt, model=args.model, seed=args.seed
        )
    elif args.engine == "replicate":
        if REPLICATE_API_TOKEN:
            generate_background(args.prompt, args.output, args.width, args.height)
        else:
            print("[WARNING] No REPLICATE_API_TOKEN set. Using Pollinations instead.")
            generate_background_pollinations(
                args.prompt, args.output, args.width, args.height,
                negative_prompt=args.negative_prompt, model=args.model, seed=args.seed
            )
    else:
        generate_procedural_background(args.prompt, args.output, args.width, args.height)


if __name__ == "__main__":
    main()