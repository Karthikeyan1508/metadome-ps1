"""
Master Pipeline Orchestrator
Author: Karthi & Ratish

Runs Stage 1 (AI background + HDR estimation + camera match) and Stage 2 (UE5/Blender render + compositing + color grading)
end-to-end for all configured prompts in shared/config.json.
"""

import os
import sys
import json
import argparse
import subprocess


def run_pipeline(config_path: str = "shared/config.json", use_fallback: bool = False):
    print("=" * 70)
    print("  PS1 AUTOMOTIVE SCENE GENERATION PIPELINE")
    print("  Orchestrating GenAI + Path-Traced Reflections + Compositing")
    print("=" * 70)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    prompts = config.get("prompts", [])
    output_base_dir = config.get("paths", {}).get("output_dir", "outputs")

    print(f"[Main Orchestrator] Loaded {len(prompts)} prompts from config.")

    for i, pitem in enumerate(prompts):

        pid = pitem["id"]
        pname = pitem["name"]
        text_prompt = pitem["text"]

        print("\n" + "-" * 60)
        print(f"[{i+1}/{len(prompts)}] Processing: {pname} ({pid})")
        print(f"Prompt: {text_prompt}")
        print("-" * 60)

        p_dir = os.path.join(output_base_dir, pid)
        os.makedirs(p_dir, exist_ok=True)

        # Output paths
        bg_path         = os.path.join(p_dir, "background.png")
        depth_path      = os.path.join(p_dir, "depth.png")
        hdr_path        = os.path.join(p_dir, "environment.exr")
        cam_path        = os.path.join(p_dir, "camera.json")
        depth_exr_path  = os.path.join(p_dir, "depth_render.exr")
        depth_ctrl_path = os.path.join(p_dir, "depth_render_depth_map.png")

        render_path     = os.path.join(p_dir, "car_render.png")
        composite_path  = os.path.join(p_dir, "composite.png")
        final_path      = os.path.join(p_dir, "final_postprocessed.png")

        # ======================================================
        # STAGE 1 — CAMERA FIRST
        # Camera parameters must be derived BEFORE background
        # generation so we can pass the depth map to ControlNet.
        # ======================================================

        print("\n[Stage 1] Camera Parameter Extraction (pre-background)")

        # We can't derive pitch from the background yet (it doesn't exist),
        # so extract_camera runs on a blank/placeholder pass first, using
        # the per-prompt PRESET values.  It will run again after background
        # generation if the background is available, to refine pitch.
        # For the first run we pass no --input, so the script uses presets.
        subprocess.run(
            [
                sys.executable,
                "stage1_ai/extract_camera.py",
                "--prompt_id", pid,
                "--input", bg_path if os.path.exists(bg_path) else "",
                "--output", cam_path,
            ],
            check=False,  # soft fail — preset values are fine for depth pass
        )

        # ======================================================
        # STAGE 1.5 — BLENDER DEPTH PASS (car hidden)
        # Renders the exact camera perspective as a Z-depth map
        # so ControlNet can generate a matching background.
        # ======================================================

        blender_exe = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
        car_model_path = config.get("paths", {}).get("car_model_path", "assets/Volvo S90.blend")
        use_controlnet = bool(os.environ.get("REPLICATE_API_TOKEN", ""))

        if os.path.exists(blender_exe) and use_controlnet:
            print("\n[Stage 1.5] Blender Depth Pass (for ControlNet conditioning)")
            cmd_depth = [
                blender_exe, "-b", "--python",
                "stage2_render/blender_render.py",
                "--",
                "depth",
                cam_path,
                depth_exr_path,
                car_model_path,
            ]
            try:
                subprocess.run(cmd_depth, check=True)
                print(f"[Stage 1.5] Depth map ready: {depth_ctrl_path}")
            except subprocess.CalledProcessError as e:
                print(f"[Stage 1.5] Depth pass failed ({e}), skipping ControlNet conditioning")
                use_controlnet = False
        else:
            if not use_controlnet:
                print("[Stage 1.5] No REPLICATE_API_TOKEN set — skipping depth pass, using Pollinations")
            else:
                print("[Stage 1.5] Blender not found — skipping depth pass")
            use_controlnet = False

        # ======================================================
        # STAGE 1 — BACKGROUND GENERATION
        # If ControlNet: depth-conditioned, perspective-matched.
        # Otherwise: Pollinations free Flux (standard flow).
        # ======================================================

        print("\n[Stage 1] Background Generation")

        negative_prompt = pitem.get("negative_prompt", "")
        cmd_bg = [
            sys.executable,
            "stage1_ai/generate_bg.py",
            "--prompt", text_prompt,
            "--output", bg_path,
            "--camera_json", cam_path,
        ]

        if negative_prompt:
            cmd_bg.extend(["--negative-prompt", negative_prompt])

        if use_controlnet and os.path.exists(depth_ctrl_path):
            cmd_bg.extend([
                "--engine", "controlnet",
                "--depth_map", depth_ctrl_path,
                "--condition_scale", "0.8",
            ])
            print("[Stage 1] Mode: ControlNet depth-conditioned")
        elif use_fallback:
            cmd_bg.append("--use_fallback")
            print("[Stage 1] Mode: Procedural fallback")
        else:
            print("[Stage 1] Mode: Pollinations Flux (free)")

        subprocess.run(cmd_bg, check=True)

        # ======================================================
        # STAGE 1 — DEPTH ESTIMATION FROM GENERATED BACKGROUND
        # ======================================================

        print("[Stage 1] Depth Estimation (Depth Anything V2)")

        subprocess.run(
            [
                sys.executable,
                "stage1_ai/estimate_depth.py",
                "--input", bg_path,
                "--output", depth_path,
            ],
            check=True,
        )

        # ======================================================
        # STAGE 1 — HDR ESTIMATION
        # ======================================================

        print("[Stage 1] HDR Light Estimation")

        subprocess.run(
            [
                sys.executable,
                "stage1_ai/estimate_hdr.py",
                "--input", bg_path,
                "--depth", depth_path,
                "--output", hdr_path,
            ],
            check=True,
        )

        # ======================================================
        # STAGE 1 — CAMERA REFINEMENT
        # Now that the background exists, re-run camera extraction
        # to refine pitch/roll from the actual image horizon.
        # ======================================================

        print("[Stage 1] Camera Parameter Refinement (image-derived horizon)")

        subprocess.run(
            [
                sys.executable,
                "stage1_ai/extract_camera.py",
                "--prompt_id", pid,
                "--input", bg_path,
                "--depth", depth_path,
                "--output", cam_path,
            ],
            check=True,
        )

        # ======================================================
        # STAGE 2 — BEAUTY RENDER (Karthi)
        # ======================================================

        print("\n[Stage 2] Path-Traced Car Rendering")

        if os.path.exists(blender_exe):
            cmd_render = [
                blender_exe, "-b", "--python",
                "stage2_render/blender_render.py",
                "--",
                "beauty",
                hdr_path,
                cam_path,
                render_path,
                car_model_path,
            ]
            subprocess.run(cmd_render, check=True)
        else:
            subprocess.run(
                [
                    sys.executable,
                    "stage2_render/blender_render.py",
                    hdr_path,
                    cam_path,
                    render_path,
                ],
                check=True,
            )

        # ------------------------------------------------------

        print("[Stage 2] Compositing")

        subprocess.run(
            [
                sys.executable,
                "stage2_render/compositor.py",
                "--bg",
                bg_path,
                "--render",
                render_path,
                "--output",
                composite_path,
            ],
            check=True,
        )

        # ------------------------------------------------------

        print("[Stage 2] Color Grading")

        subprocess.run(
            [
                sys.executable,
                "stage2_render/color_grade.py",
                "--input",
                composite_path,
                "--output",
                final_path,
            ],
            check=True,
        )

        print(f"\n[SUCCESS] Completed: {pname}")
        print(f"Output: {final_path}")

    print("\n" + "=" * 70)
    print(" ALL ENVIRONMENTS SUCCESSFULLY GENERATED")
    print("=" * 70)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PS1 Automotive Scene Pipeline")

    parser.add_argument(
        "--config",
        default="shared/config.json",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use fallback procedural background generation",
    )

    args = parser.parse_args()

    run_pipeline(args.config, args.fallback)
