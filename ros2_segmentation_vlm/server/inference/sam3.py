import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from inference.sam3_multi_prompt import Sam3MultiPromptProcessor
from utils.utils import load_colorcode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths & threshold (adjust as needed)
IMAGE_PATH = "/home/jaime/repos/SAM_labeler/data/ETSII/rgb/color_20250604_163126.png"
COLORCODE_PATH = "colorcodes/demo.json"
THRESHOLD = 0.2

# -------------------------------------------------------------------------
# Model & prompts setup (similar to CLIPSeg script style)
# -------------------------------------------------------------------------

print("Loading SAM3 model...")
sam3_model = build_sam3_image_model().to(device)

processor = Sam3MultiPromptProcessor(
    model=sam3_model,
    device=device,
    confidence_threshold=THRESHOLD,
)

# Load prompts and their colors from the JSON colormap
prompts, colors = load_colorcode(COLORCODE_PATH)
print(f"Loaded {len(prompts)} prompts from colorcode:")
print(prompts)

print("Encoding text prompts for SAM3...")
processor.set_text_prompts(prompts)


# -------------------------------------------------------------------------
# SAM3 segmentation that mirrors CLIPSeg strategy
# -------------------------------------------------------------------------

def segment_image(image: Image.Image) -> np.ndarray:
    """
    Segment an image using SAM3 with given prompts and colors.
    Mirrors the CLIPSeg fusion strategy:
      - For each pixel, keep only the class (prompt) with the highest score.
      - Colors come from the colormap JSON.

    Args:
        image: PIL Image (RGB)

    Returns:
        segm_colors_rgba: numpy array (H, W, 4) with RGBA segmentation.
    """
    image = image.convert("RGB")
    width, height = image.size

    # Run SAM3 multi-prompt segmentation
    with torch.inference_mode(), autocast(device_type="cuda", dtype=torch.float16):
        results = processor.segment_image_with_text_prompts(image)

    # Initialize per-pixel best scores and colors
    segm_colors = 255*np.ones((height, width, 3), dtype=np.uint8)
    segm_scores = np.zeros((height, width), dtype=np.float32)

    # For each prompt, we may have multiple instance masks with their own scores.
    # We treat per-pixel value as "score if inside mask, else 0",
    # and keep only the best-scoring prompt per pixel.
    for prompt_idx, res in enumerate(results):
        prompt = res["prompt"]
        masks = res["masks"]          # [N, H, W] or [N, 1, H, W]
        scores = res["scores"]        # [N]

        if masks.numel() == 0 or scores.numel() == 0:
            print(f"No detections for prompt '{prompt}'")
            continue

        # Ensure masks shape is [N, H, W]
        if masks.ndim == 4:  # [N, 1, H, W]
            masks_tensor = masks[:, 0, :, :]
        else:                # [N, H, W]
            masks_tensor = masks

        masks_np = masks_tensor.bool().cpu().numpy()   # [N, H, W]
        scores_np = scores.cpu().numpy()               # [N]

        print(f"Prompt '{prompt}': {masks_np.shape[0]} detections")
        print(f"Scores: {scores_np}")

        for inst_idx in range(masks_np.shape[0]):
            score = float(scores_np[inst_idx])
            if score < THRESHOLD:
                continue  # extra safety threshold, even though SAM3 already filters

            mask_k = masks_np[inst_idx]  # (H, W) boolean

            # For the pixels where this mask is present, update only if
            # this instance's score is higher than the current best.
            better_pixels = mask_k & (score > segm_scores)

            # Apply this prompt's color to those pixels
            if np.any(better_pixels):
                segm_colors[better_pixels] = colors[prompt_idx]
                segm_scores[better_pixels] = score

    # Add fully-opaque alpha channel
    alpha = np.full((height, width), 255, dtype=np.uint8)
    segm_colors_rgba = np.dstack((segm_colors, alpha))

    return segm_colors_rgba


# -------------------------------------------------------------------------
# Script entry point: mirror CLIPSeg visualization workflow
# -------------------------------------------------------------------------

if __name__ == "__main__":
    image = Image.open(IMAGE_PATH).convert("RGB")

    t0 = time.time()
    segmented_colors = segment_image(image)
    print(f"Segmentation took {time.time() - t0:.2f} seconds.")

    # Save or display results
    segmented_image = Image.fromarray(segmented_colors)
    segmented_image.save("sam3_segmented_output.png")

    # Display original and segmented images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(segmented_image)
    ax2.set_title("SAM3 Segmented Image")
    ax2.axis("off")

    # Create a separate figure for the legend (same style as CLIPSeg)
    legend_fig, legend_ax = plt.subplots(figsize=(8, 2))
    legend_ax.axis("off")

    legend_elements = []
    for prompt, color in zip(prompts, colors):
        # color is already [R, G, B] in 0–255
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=np.array(color) / 255.0,
                label=prompt,
            )
        )

    legend_ax.legend(
        handles=legend_elements,
        loc="center",
        ncol=min(len(prompts), 3),
        frameon=False,
    )

    legend_fig.tight_layout()
    legend_fig.savefig("sam3_legend.png", bbox_inches="tight", dpi=150)

    plt.tight_layout()
    plt.show()
