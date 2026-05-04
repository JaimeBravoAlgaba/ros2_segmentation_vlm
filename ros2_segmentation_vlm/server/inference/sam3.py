from __future__ import annotations

from contextlib import nullcontext
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast

from sam3.model_builder import build_sam3_image_model

from inference.sam3_multi_prompt import Sam3MultiPromptProcessor


UNKNOWN_CLASS_ID = np.uint8(255)
THRESHOLD = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading SAM3 model...")
sam3_model = build_sam3_image_model().to(device)
processor = Sam3MultiPromptProcessor(
    model=sam3_model,
    device=device,
    confidence_threshold=THRESHOLD,
)

_cached_prompts: List[str] = []
_cached_class_ids = np.empty((0,), dtype=np.uint8)


def configure_prompts(prompts: Sequence[str], class_ids: Sequence[int] | None = None) -> None:
    global _cached_prompts, _cached_class_ids

    prompts = [prompt.strip() for prompt in prompts if isinstance(prompt, str) and prompt.strip()]
    if not prompts:
        raise ValueError("La lista de prompts no puede estar vacía.")
    if len(prompts) > 255:
        raise ValueError("SAM3 solo admite hasta 255 prompts en un class_map uint8.")

    if class_ids is None:
        class_ids_array = np.arange(len(prompts), dtype=np.uint8)
    else:
        class_ids_array = np.asarray(class_ids, dtype=np.uint8)
        if class_ids_array.ndim != 1 or class_ids_array.shape[0] != len(prompts):
            raise ValueError("'class_ids' debe tener la misma longitud que 'prompts'.")

    if len(np.unique(class_ids_array)) != len(class_ids_array):
        raise ValueError("'class_ids' contiene IDs duplicados.")
    if int(UNKNOWN_CLASS_ID) in class_ids_array.tolist():
        raise ValueError("El valor 255 está reservado para unknown/background.")

    print(f"Configuring SAM3 with {len(prompts)} prompts...")
    processor.set_text_prompts(prompts)
    _cached_prompts = list(prompts)
    _cached_class_ids = class_ids_array.copy()


def segment_image(image: Image.Image) -> np.ndarray:
    if not _cached_prompts:
        raise RuntimeError("SAM3 no tiene prompts configurados. Llama antes a configure_prompts().")

    image = image.convert("RGB")
    width, height = image.size

    if device.type == "cuda":
        autocast_context = autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_context = nullcontext()

    with torch.inference_mode(), autocast_context:
        results = processor.segment_image_with_text_prompts(image)

    segm_scores = np.zeros((height, width), dtype=np.float32)
    class_map = np.full((height, width), UNKNOWN_CLASS_ID, dtype=np.uint8)

    for prompt_idx, res in enumerate(results):
        prompt = res["prompt"]
        masks = res["masks"]
        scores = res["scores"]

        if masks.numel() == 0 or scores.numel() == 0:
            print(f"No detections for prompt '{prompt}'")
            continue

        masks_tensor = masks[:, 0, :, :] if masks.ndim == 4 else masks
        masks_np = masks_tensor.bool().cpu().numpy()
        scores_np = scores.cpu().numpy()

        print(f"Prompt '{prompt}': {masks_np.shape[0]} detections")
        print(f"Scores: {scores_np}")

        for inst_idx in range(masks_np.shape[0]):
            score = float(scores_np[inst_idx])
            if score < THRESHOLD:
                continue

            mask_k = masks_np[inst_idx]
            better_pixels = mask_k & (score > segm_scores)
            if np.any(better_pixels):
                segm_scores[better_pixels] = score
                class_map[better_pixels] = _cached_class_ids[prompt_idx]

    return class_map
