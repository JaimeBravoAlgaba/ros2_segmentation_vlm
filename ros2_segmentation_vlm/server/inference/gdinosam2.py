import argparse
argparse = argparse.ArgumentParser(description="Zero-shot segmentation using GroundingDINO and SAM2")
argparse.add_argument("--image_path", type=str, default="input.png", help="Path to the input image")
argparse.add_argument("--output_path", type=str, default="output.png", help="Path to the output image")
argparse.add_argument("--colorcode_path", type=str, default="colorcodes/gdinosam2_demo.json", help="Path to the colorcode JSON file")
args = argparse.parse_args()

import os

import torch
import numpy as np
import cv2
from PIL import Image

from groundingdino.util.inference import load_image
from groundingdino.util.inference import Model as GDINOModel
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils import load_colorcode, color_from_class

# Initial setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

DEMO_IMAGE_PATH = args.image_path
DEMO_OUTPUT_PATH = args.output_path

COLORCODE_PATH = args.colorcode_path

prompts, colors = load_colorcode(COLORCODE_PATH)

# Load models
model_gdino = GDINOModel("configs/gdino/GroundingDINO_SwinT_OGC.py",
                         "weights/weights_gdino/groundingdino_swint_ogc.pth",
                         device=device)

model_sam2 = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml",
                        "weights/weights_sam2/sam2.1_hiera_tiny.pt",
                        device=device)

predictor_sam2 = SAM2ImagePredictor(model_sam2)

def gdino_detect_objects(model_gdino, image, box_threshold=0.1, text_threshold=0.2):
    """
    Perform object detection using GroundingDINO model.
    
    Args:
        model_gdino: GroundingDINO model instance
        image: Already loaded image (numpy array)
        box_threshold: Box confidence threshold
        text_threshold: Text confidence threshold
    
    Returns:
        tuple: (boxes_xyxy, logits, class_ids, phrases, xyxy_tensor)
    """
    gdino_detections = model_gdino.predict_with_classes(
        image=image,
        classes=prompts,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    boxes_xyxy = gdino_detections.xyxy
    logits = gdino_detections.confidence
    class_ids = gdino_detections.class_id

    xyxy = torch.tensor(boxes_xyxy)
    
    return xyxy, logits, class_ids

def segment_with_sam2(predictor_sam2, image, xyxy, class_ids, logits):
    """
    Perform instance segmentation using SAM2 model.
    
    Args:
        predictor_sam2: SAM2 predictor instance
        image: Input image (numpy array)
        xyxy: Bounding boxes tensor
        class_ids: List of class IDs for each detection
        colors: List of colors for each class
        logits: GDINO confidence scores
    
    Returns:
        numpy.ndarray: Segmented image with colored masks
    """
    predictor_sam2.set_image(image)

    masks, scores, _ = predictor_sam2.predict(
        point_coords=None,
        point_labels=None,
        box=xyxy,
        multimask_output=False
    )

    mask_list = [m.squeeze(0) if m.ndim == 3 else m for m in masks]
    logits_arr = np.array(logits)    # GDINO confidences (one per box)
    idx_sorted = np.argsort(logits_arr)  # Indices of boxes sorted by GDINO confidence (ascending)
    logits_sorted = logits_arr[idx_sorted]

    # Draw masks corresponding to the lower GDINO logits first (ascending)
    image_segmented = 255*np.ones_like(image)
    
    if len(idx_sorted) == 0:
        return image_segmented
    else:
        for idx in idx_sorted:
            if scores[idx] < 0.9:  # skip low-confidence SAM2 masks
                continue
            else:
                m = mask_list[int(idx)]
                if m is None:
                    continue
                class_id = class_ids[int(idx)]
                if class_id is None:
                    continue
                class_name = prompts[class_id]
                if class_name is None:
                    continue

                color = np.array(color_from_class(class_name, prompts, colors)) / 255.0
                alpha = np.array([1.0])
                color = np.concatenate([color, alpha], axis=0)

                image_segmented[m==1] = (color[:3]*255).astype(np.uint8)
    
    return image_segmented

def segment_image(image: Image.Image) -> np.ndarray:
    """
    Perform zero-shot segmentation on the input image.

    This function:
      * Accepts either a PIL.Image or a NumPy array.
      * Optionally downsizes very large images to reduce GPU memory.
      * Runs GroundingDINO + SAM2 under torch.no_grad().
      * Uses autocast(float16) on CUDA for lower VRAM usage.
    """
    # Convert PIL -> numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")

    # OPTIONAL: resize large images to reduce VRAM pressure
    max_size = 240  # lower this (e.g. 768/512) if you still get OOM
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    with torch.no_grad():
        if device.type == "cuda":
            # Mixed precision on GPU to save memory
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                xyxy, logits, class_ids = gdino_detect_objects(model_gdino, image)
                image_segmented = segment_with_sam2(
                    predictor_sam2, image, xyxy, class_ids, logits
                )
        else:
            xyxy, logits, class_ids = gdino_detect_objects(model_gdino, image)
            image_segmented = segment_with_sam2(
                predictor_sam2, image, xyxy, class_ids, logits
            )

    return image_segmented


if __name__ == "__main__":
    image, _ = load_image(DEMO_IMAGE_PATH)
    image_segmented = segment_image(image)
    cv2.imwrite(DEMO_OUTPUT_PATH, cv2.cvtColor(image_segmented, cv2.COLOR_RGB2BGR))