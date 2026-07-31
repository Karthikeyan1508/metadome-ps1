"""
Stage 1 - Depth Estimation
Author: Ratish (Person A - AI & Data Pipeline)

Generates a dense depth map from a background image using
Depth Anything V2.

Input:
    background.png

Output:
    depth.png
"""

import os
import argparse
import numpy as np
import cv2
import torch
from transformers import pipeline


class DepthEstimator:

    def __init__(self):

        device = 0 if torch.cuda.is_available() else -1

        print("[Depth] Loading Depth Anything V2 model...")

        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )

        print("[Depth] Model loaded successfully.")

    def estimate(self, input_image, output_image):

        if not os.path.exists(input_image):
            raise FileNotFoundError(input_image)

        print(f"[Depth] Reading {input_image}")

        image = cv2.imread(input_image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)

        print("[Depth] Estimating depth...")

        result = self.pipe(pil_image)

        depth = np.array(result["depth"])

        depth = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        depth = depth.astype(np.uint8)

        os.makedirs(os.path.dirname(output_image), exist_ok=True)

        cv2.imwrite(output_image, depth)

        print(f"[Depth] Saved depth map -> {output_image}")

        return output_image


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Background image",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Depth image",
    )

    args = parser.parse_args()

    estimator = DepthEstimator()

    estimator.estimate(
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()