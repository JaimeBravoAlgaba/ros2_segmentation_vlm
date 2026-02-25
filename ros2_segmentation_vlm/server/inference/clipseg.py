import time
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

from utils import load_colorcode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIPSEG_LOCAL_DIR = "weights/clipseg_local"
COLORCODE_PATH = "colorcodes/gdinosam2_demo.json"
THRESHOLD = 0.2

processor = CLIPSegProcessor.from_pretrained(
    CLIPSEG_LOCAL_DIR,
    local_files_only=True,   # never go online
    use_fast=True            # avoid the slow-processor warning
)

model = CLIPSegForImageSegmentation.from_pretrained(
    CLIPSEG_LOCAL_DIR,
    local_files_only=True    # never go online
).to(device)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

# Load image
prompts, colors = load_colorcode(COLORCODE_PATH)

def segment_image(image: Image.Image) -> np.ndarray:
    """
    Segment an image using CLIPSeg with given prompts and colors.
    
    Args:
        image: PIL Image
    
    Returns:
        segmented_colors: numpy array of segmented image (RGBA)
    """

    # Prepare text inputs
    text_inputs = processor.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Prepare image inputs
    image_inputs = processor.image_processor(
        [image] * len(prompts),
        return_tensors="pt"
    )

    # Merge
    inputs = {**text_inputs, **image_inputs}
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits).cpu().numpy()  # (num_prompts, 1, H, W)

    segm_colors = 255*np.ones((image.height, image.width, 3), dtype=np.uint8)
    #segm_colors = np.array(image)[:, :, :3].copy()
    segm_probs = np.zeros((image.height, image.width, 1), dtype=np.float32)

    for i in range(len(prompts)):
        # squeeze channel dimension -> (H, W)
        mask = probs[i, :, :].astype(np.float32)

        # resize to (width, height)
        mask = cv2.resize(mask, (image.width, image.height))

        segm_mask = (mask > segm_probs[:, :, 0]) & (mask > THRESHOLD)
        segm_colors[segm_mask] = colors[i]
        segm_probs[segm_mask, 0] = mask[segm_mask]

    # Add alpha channel (fully opaque)
    alpha = np.full((image.height, image.width), 255, dtype=np.uint8)
    segm_colors = np.dstack((segm_colors, alpha))

    return segm_colors

if __name__ == "__main__":
    image_path = "input.png"
    print(f"Using device: {device}")

    image = Image.open(image_path).convert("RGB")

    t0 = time.time()
    segmented_colors = segment_image(image)
    print(f"Segmentation took {time.time() - t0:.2f} seconds.")

    # Save or display results
    segmented_image = Image.fromarray(segmented_colors)
    segmented_image.save("segmented_output.png")

    # Display original and segmented images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(segmented_image)
    ax2.set_title('Segmented Image')
    ax2.axis('off')

    # Create a separate figure for the legend
    legend_fig, legend_ax = plt.subplots(figsize=(8, 2))
    legend_ax.axis('off')

    # Create legend with colors and classes
    legend_elements = []
    for i, (prompt, color) in enumerate(zip(prompts, colors)):
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=np.array(color)/255.0, label=prompt))

    legend_ax.legend(handles=legend_elements, loc='center', 
                    ncol=min(len(prompts), 3), frameon=False)

    # Save the legend as a separate image
    legend_fig.tight_layout()
    legend_fig.savefig("legend.png", bbox_inches='tight', dpi=150)

    plt.tight_layout()
    plt.show()