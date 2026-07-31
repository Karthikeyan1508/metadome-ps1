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
        
        print("\n" + "-" * 50)
        print(f"[{i+1}/{len(prompts)}] Processing Environment: '{pname}' ({pid})")
        print(f"Prompt: \"{text_prompt}\"")
        print("-" * 50)
        
        p_dir = os.path.join(output_base_dir, pid)
        os.makedirs(p_dir, exist_ok=True)
        
        bg_path = os.path.join(p_dir, "background.png")
        hdr_path = os.path.join(p_dir, "environment.exr")
        cam_path = os.path.join(p_dir, "camera.json")
        render_path = os.path.join(p_dir, "car_render.png")
        composite_path = os.path.join(p_dir, "composite.png")
        final_path = os.path.join(p_dir, "final_postprocessed.png")
        
        # STAGE 1: AI Background Generation (Ratish)
        print(f"[Stage 1] Generating background plate...")
        cmd_bg = [sys.executable, "stage1_ai/generate_bg.py", "--prompt", text_prompt, "--output", bg_path]
        if use_fallback:
            cmd_bg.append("--use_fallback")
        subprocess.run(cmd_bg, check=True)
        
        # STAGE 1: HDR Light Estimation (Ratish)
        print(f"[Stage 1] Estimating EXR light probe...")
        subprocess.run([sys.executable, "stage1_ai/estimate_hdr.py", "--input", bg_path, "--output", hdr_path], check=True)
        
        # STAGE 1: Camera Pose Extraction (Ratish)
        print(f"[Stage 1] Extracting camera perspective...")
        subprocess.run([sys.executable, "stage1_ai/extract_camera.py", "--prompt_id", pid, "--input", bg_path, "--output", cam_path], check=True)
        
        # STAGE 2: Path-Traced Render Execution (Karthi)
        print(f"[Stage 2] Rendering path-traced vehicle pass using Blender Cycles...")
        car_model_path = config.get("paths", {}).get("car_model_path", "assets/car_model.obj")
        blender_exe = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
        if os.path.exists(blender_exe):
            cmd_render = [blender_exe, "-b", "--python", "stage2_render/blender_render.py", "--", hdr_path, cam_path, render_path, car_model_path]
            subprocess.run(cmd_render, check=True)
        else:
            subprocess.run([sys.executable, "stage2_render/blender_render.py", hdr_path, cam_path, render_path, car_model_path], check=True)
            
        # STAGE 2: Multi-Pass Compositing (Karthi)
        print(f"[Stage 2] Compositing vehicle render onto background plate...")
        subprocess.run([sys.executable, "stage2_render/compositor.py", "--bg", bg_path, "--render", render_path, "--output", composite_path], check=True)
        
        # STAGE 2: Post-Processing & Color Grade (Karthi)
        print(f"[Stage 2] Polishing final composite image...")
        subprocess.run([sys.executable, "stage2_render/color_grade.py", "--input", composite_path, "--output", final_path, "--bg", bg_path], check=True)
        
        print(f"[SUCCESS] Final render completed for '{pname}': {final_path}")
        
    print("\n" + "=" * 70)
    print("  ALL ENVIRONMENTS SUCCESSFULLY GENERATED AND RENDERED!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PS1 Main Automation Pipeline")
    parser.add_argument("--config", type=str, default="shared/config.json", help="Path to config.json")
    parser.add_argument("--fallback", action="store_true", help="Use synthetic procedural fallbacks")
    
    args = parser.parse_args()
    run_pipeline(args.config, args.fallback)
