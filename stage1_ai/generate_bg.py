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
                                      camera_json_path: str = None,
                                      enhance: bool = True):
    """
    Generate an image using Pollinations.ai's free image API.
    No API key required. Inject perspective constraints from camera.json
    so the generated landscape aligns with Blender's camera vanishing point.
    """
    perspective_tags = "photorealistic automotive background plate, wide angle 35mm, crisp asphalt road in lower half, straight horizon line"
    if camera_json_path and os.path.exists(camera_json_path):
        try:
            with open(camera_json_path, 'r') as f:
                cdata = json.load(f)
            rot = cdata.get('camera_rotation', [-3.5, 0.0, 0.0])
            pitch = rot[0]
            if pitch < -8.0:
                perspective_tags = "overhead high angle landscape view, receding road stretching to high horizon"
            elif pitch < 0.0:
                perspective_tags = "eye-level automotive photography, asphalt road receding to central vanishing point at horizon"
            else:
                perspective_tags = "low angle ground view, road surface filling lower frame"
        except Exception:
            pass

    full_prompt = f"{prompt}, {perspective_tags}"
    if negative_prompt:
        exclusions = ", ".join(
            f"no {term.strip()}" for term in negative_prompt.split(",") if term.strip()
        )
        full_prompt = f"{full_prompt}, {exclusions}"

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

        # Upscale to full target resolution (1920x1080) if needed
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)

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


def generate_background_controlnet_depth(
    prompt: str,
    depth_map_path: str,
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    negative_prompt: str = "",
    condition_scale: float = 0.8,
):
    """
    Generate a background conditioned on a Blender depth pass using
    Replicate's SDXL ControlNet Depth model.

    This guarantees the AI background has the same camera perspective,
    horizon line, and vanishing point as the 3D scene — eliminating the
    'pasted on' look from perspective mismatch.

    Falls back to Pollinations if ControlNet fails or no token is set.

    Parameters
    ----------
    prompt : str
        Scene description prompt.
    depth_map_path : str
        Path to the normalised 8-bit depth PNG rendered by Blender.
    output_path : str
        Destination for the generated background PNG.
    condition_scale : float
        How strongly the depth map controls the output (0.0 – 1.0).
        0.8 = strong structural adherence while allowing creative freedom.
    """
    if not REPLICATE_API_TOKEN:
        print("[ControlNet] No REPLICATE_API_TOKEN set — falling back to Pollinations")
        return generate_background_pollinations(
            prompt, output_path, width, height, negative_prompt=negative_prompt
        )

    if not os.path.exists(depth_map_path):
        print(f"[ControlNet] Depth map not found: {depth_map_path} — falling back to Pollinations")
        return generate_background_pollinations(
            prompt, output_path, width, height, negative_prompt=negative_prompt
        )

    import replicate
    import base64

    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

    enhanced_prompt = (
        f"{prompt}, photorealistic, professional landscape photography, "
        f"8k ultra HD, sharp focus, natural lighting, rich colors, wide angle, "
        f"highly detailed, no car, no vehicle"
    )
    full_negative = (
        f"car, vehicle, automobile, people, person, text, watermark, logo, "
        f"blurry, low quality, distorted, cartoon, CGI, 3d render, {negative_prompt}"
    )

    # Read depth map as base64 for Replicate API
    with open(depth_map_path, "rb") as f:
        depth_b64 = base64.b64encode(f.read()).decode("utf-8")
    depth_data_uri = f"data:image/png;base64,{depth_b64}"

    print(f"[ControlNet] Generating depth-conditioned background...")
    print(f"[ControlNet] Prompt: '{prompt[:80]}...'")
    print(f"[ControlNet] Depth map: {depth_map_path}")
    print(f"[ControlNet] Condition scale: {condition_scale}")

    try:
        # Open depth map as a binary file — replicate client uploads it automatically
        with open(depth_map_path, "rb") as depth_file:
            output = replicate.run(
                # Pinned to latest stable version (2023-09-12)
                "lucataco/sdxl-controlnet-depth:465fb41789dc2203a9d7158be11d1d2570606a039c65e0e236fd329b5eecb10c",
                input={
                    "prompt": enhanced_prompt,
                    "negative_prompt": full_negative,
                    "image": depth_file,
                    "condition_scale": condition_scale,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                },
            )

        image_url = output[0] if isinstance(output, list) else str(output)

        print(f"[ControlNet] Downloading result...")
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))

        # Upscale to target resolution if needed
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)

        image = ImageEnhance.Sharpness(image).enhance(1.15)
        image = ImageEnhance.Contrast(image).enhance(1.05)
        image = ImageEnhance.Color(image).enhance(1.05)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, "PNG", quality=100)

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ControlNet] Saved: {output_path} ({image.size[0]}x{image.size[1]}, {file_size:.1f}MB)")
        return output_path

    except Exception as e:
        print(f"[ControlNet] Failed: {e}")
        # Mark this output with _fallback suffix and use Pollinations
        fallback_path = output_path.replace(".png", "_fallback.png")
        print(f"[ControlNet] Falling back to Pollinations -> {fallback_path}")
        result = generate_background_pollinations(
            prompt, fallback_path, width, height, negative_prompt=negative_prompt
        )
        # Copy to the expected output path so the pipeline can continue
        import shutil
        shutil.copy(fallback_path, output_path)
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
                         choices=["pollinations", "replicate", "procedural", "controlnet"],
                         help="Backend: pollinations (free), replicate (paid SDXL), "
                              "controlnet (depth-conditioned SDXL), procedural (offline)")
    parser.add_argument("--depth_map", type=str, default=None,
                         help="Path to Blender depth map PNG for ControlNet conditioning. "
                              "When provided with --engine controlnet, generates a background "
                              "that matches the exact camera perspective of the 3D scene.")
    parser.add_argument("--condition_scale", type=float, default=0.8,
                         help="ControlNet conditioning strength (0.0-1.0, default 0.8)")
    parser.add_argument("--camera_json", type=str, default=None,
                         help="Path to camera.json to inject perspective constraints")

    args = parser.parse_args()
    negative = getattr(args, 'negative_prompt', '')

    if args.engine == "controlnet" or (args.depth_map and os.path.exists(args.depth_map or '')):
        generate_background_controlnet_depth(
            args.prompt, args.depth_map, args.output, args.width, args.height,
            negative_prompt=negative, condition_scale=args.condition_scale,
        )
    elif args.engine == "pollinations":
        generate_background_pollinations(
            args.prompt, args.output, args.width, args.height,
            negative_prompt=negative, model=args.model, seed=args.seed,
            camera_json_path=args.camera_json
        )
    elif args.engine == "replicate":
        if REPLICATE_API_TOKEN:
            generate_background(args.prompt, args.output, args.width, args.height)
        else:
            print("[WARNING] No REPLICATE_API_TOKEN set. Using Pollinations instead.")
            generate_background_pollinations(
                args.prompt, args.output, args.width, args.height,
                negative_prompt=negative, model=args.model, seed=args.seed
            )
    else:
        generate_procedural_background(args.prompt, args.output, args.width, args.height)


if __name__ == "__main__":
    main()