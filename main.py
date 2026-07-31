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
        bg_path = os.path.join(p_dir, "background.png")
        depth_path = os.path.join(p_dir, "depth.png")
        hdr_path = os.path.join(p_dir, "environment.exr")
        cam_path = os.path.join(p_dir, "camera.json")

        render_path = os.path.join(p_dir, "car_render.png")
        composite_path = os.path.join(p_dir, "composite.png")
        final_path = os.path.join(p_dir, "final_postprocessed.png")

        # ======================================================
        # STAGE 1 - AI & DATA PIPELINE (Ratish)
        # ======================================================

        print("\n[Stage 1] Environment Generation")

        negative_prompt = pitem.get("negative_prompt", "")

        cmd_bg = [
            sys.executable,
            "stage1_ai/generate_bg.py",
            "--prompt",
            text_prompt,
            "--output",
            bg_path,
        ]

        if negative_prompt:
            cmd_bg.extend(["--negative-prompt", negative_prompt])

        if use_fallback:
            cmd_bg.append("--use_fallback")

        subprocess.run(cmd_bg, check=True)

        # ------------------------------------------------------

        print("[Stage 1] Depth Estimation")

        subprocess.run(
            [
                sys.executable,
                "stage1_ai/estimate_depth.py",
                "--input",
                bg_path,
                "--output",
                depth_path,
            ],
            check=True,
        )

        # ------------------------------------------------------

        print("[Stage 1] HDR Estimation")

        subprocess.run(
            [
                sys.executable,
                "stage1_ai/estimate_hdr.py",
                "--input",
                bg_path,
                "--depth",
                depth_path,
                "--output",
                hdr_path,
            ],
            check=True,
        )

        # ------------------------------------------------------

        print("[Stage 1] Camera Parameter Extraction")

        subprocess.run(
            [
                sys.executable,
                "stage1_ai/extract_camera.py",
                "--prompt_id",
                pid,
                "--input",
                bg_path,
                "--depth",
                depth_path,
                "--output",
                cam_path,
            ],
            check=True,
        )

        # ======================================================
        # STAGE 2 - RENDERING (Karthi)
        # ======================================================

        print("\n[Stage 2] Path-Traced Rendering")

        blender_exe = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
        car_model_path = config.get("paths", {}).get("car_model_path", "assets/Volvo S90.blend")

        if os.path.exists(blender_exe):

            cmd_render = [
                blender_exe,
                "-b",
                "--python",
                "stage2_render/blender_render.py",
                "--",
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
                    car_model_path,
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
