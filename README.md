# PS1 — Automotive Scene Generation with Physically Plausible Reflections

## Overview
This project presents an automated pipeline that takes a single 3D vehicle model and places it in diverse GenAI-generated environments (**urban night, forest mist, desert sunset, wet racetrack, coastal highway**) while guaranteeing **100% vehicle mesh & material consistency** and **physically plausible raytraced reflections**.

---

## Architecture

```
[ Text Prompt ] ──► Stage 1: GenAI Background Generator (SDXL / Skybox)
                           │
                           ▼
                    Stage 1: Inverse Tone Mapping & Light Probe (OpenEXR)
                           │
                           ▼
                    Stage 2: UE5 Path Tracer / Blender Cycles Render Pass
                           │
                           ▼
                    Stage 2: Multi-Pass OpenCV Compositor + ACES Color Grade
                           │
                           ▼
                     [ Final Photorealistic Automotive Image ]
```

---

## Team Split & Roles
- **Karthi (Person B)**: UE5 / Blender Path Tracer Automation, OpenCV Multi-Pass Compositing, ACES Color Grading & Post-Processing.
- **Ratish (Person A)**: GenAI Environment Pipeline (SDXL/Flux), LDR-to-EXR HDR Light Estimation, Camera Parameter Extractor.

---

## Setup & Running Instructions

### Prerequisites
- Python 3.10 / 3.11 with CUDA support
- Unreal Engine 5.4 (or Blender 4.x as path-tracing fallback)

### Installation
```bash
pip install -r requirements.txt
```

### Pre-Hackathon Mock Test
Generate test assets to verify rendering and compositing:
```bash
python test_assets/generate_mock_assets.py
```

### Run Full Pipeline
```bash
python main.py --config shared/config.json
```

### Verify Vehicle Consistency (Judges' Proof)
```bash
python consistency_check.py
```

---

## Technical Highlights
1. **Unchanged Vehicle Model**: Geometry and materials remain locked in 3D scene files (`Constraint 1`).
2. **GenAI Environment**: Backgrounds generated dynamically via AI text prompts (`Constraint 2`).
3. **High Dynamic Range (HDR) Lighting**: 32-bit float OpenEXR environment maps power path-traced reflections on car clearcoat paint & glass (`Constraint 3`).
4. **Shadow Catcher Grounding**: Contact shadows ground the car seamlessly into background asphalt.
