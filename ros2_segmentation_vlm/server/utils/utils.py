import json
import numpy as np
from skimage.measure import label, regionprops, find_contours

def load_colorcode(path: str):
    dict = json.load(open(path))
    classes = list(dict.keys())
    colors = list(dict.values())    
    return classes, colors

def class_from_color(color, classes, colors):
    idx = colors.index(color) if color in colors else -1
    if idx >= 0:
        return classes[idx]
    print(f"Color {color} not found in colorcode.")
    return None

def color_from_class(cls, classes, colors):
    idx = classes.index(cls) if cls in classes else -1
    if idx >= 0:
        return colors[idx]
    print(f"Class {cls} not found in colorcode.")
    return None

def simplify_class(cls, colorcode_map):
    cls_simp = colorcode_map.get(cls, None)
    return cls_simp

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask