import numpy as np
from PIL import Image
import time

import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.segmentation import find_boundaries
from torchange.models.segment_any_change import AnyChange, show_change_masks
from torchange.models.segment_any_change.segment_anything.utils.amg import (
    area_from_rle,
    box_xyxy_to_xywh,
    rle_to_mask,
    MaskData
)
from pathlib import Path
from typing import Optional, Tuple

def show_mask(mask_data, ax=None):
    assert isinstance(mask_data, MaskData)
    anns = []
    
    if len(mask_data["rles"]) == 0:
        return None, 0.0
    first_mask = rle_to_mask(mask_data["rles"][0])
    mask_change = np.zeros_like(first_mask)
    height, width = first_mask.shape
    total_pixels = height * width

    for idx in range(len(mask_data["rles"])):
        mask = rle_to_mask(mask_data["rles"][idx])
        area = area_from_rle(mask_data["rles"][idx])
        mask_change += mask
        ann_i = {
            "segmentation": mask,
            "area": area,
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)
    total_changed_pixels = (mask_change > 0).sum().item()

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        boundary = find_boundaries(m)
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        color_boundary = np.array([0., 1., 1., 0.8])
        img[m] = color_mask
        img[boundary] = color_boundary

        if 'label' in ann:
            x, y, w, h = ann['bbox']
            ax.text(
                x + w / 2,
                y + h / 2,
                ann['label'],
                bbox={
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color='red',
                fontsize=4,
                verticalalignment='top',
                horizontalalignment='left'
            )
    return img, total_changed_pixels/total_pixels*100.0

def load_and_resize(url_or_path, size=(1024, 1024)):
    """_summary_

    Args:
        url_or_path (str): _description_
        size (tuple, optional): Defaults to (1024, 1024).
    Returns:
        _type_: _description_
    """
    img = imread(url_or_path)
    pil_img = Image.fromarray(img).convert("RGB")
    pil_img = pil_img.resize(size, Image.BILINEAR)
    return np.array(pil_img)


def infer(
    img1_path: str,
    img2_path: str,
    overlay_path: Optional[str] = "outputs/change_mask_overlay.png",
    model_type: str = "vit_l",
    checkpoint: str = "weights/sam_vit_l_0b3195.pth",
    points_per_side: int = 32,
    stability_thresh: float = 0.95,
    conf_thresh: int = 145,
    normalized_feature: bool = True,
    bitemporal_match: bool = True,
    percent_path: str = "temp_percent.txt") -> float:
    """
    Run AnyChange on two images and (optionally) save an overlay.

    Returns
    -------
    float
        Percentage of pixels that changed.
    """
    m = AnyChange(model_type, sam_checkpoint=checkpoint)
    m.make_mask_generator(
        points_per_side=points_per_side,
        stability_score_thresh=stability_thresh,
    )
    m.set_hyperparameters(
        change_confidence_threshold=conf_thresh,
        use_normalized_feature=normalized_feature,
        bitemporal_match=bitemporal_match,
    )

    img1 = load_and_resize(img1_path)
    img2 = load_and_resize(img2_path)

    changemasks, *_ = m.forward(img1, img2)  # automatic mode
    img, percent = show_mask(changemasks)

    # Path(percent_path).write_text(f"{percent:.2f}%")gheps

    if percent > 0.0 and overlay_path:
        fig, _ = show_change_masks(img1, img2, changemasks)
        fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0.1)

    return f"{percent:.2f}%"


if __name__ == "__main__":
    current_time = time.time()
    percent = infer(
        "./test_samples/test1_1.png",
        "./test_samples/test1_2.png",
        model_type = "vit_h",
        checkpoint = "weights/sam_vit_h_4b8939.pth",
        points_per_side = 32,
        stability_thresh = 0.95,
        conf_thresh = 145,
        normalized_feature = True,
        bitemporal_match = True,
        overlay_path=f"outputs/{current_time}_mask_overlay.png",
    )
    print(f"Diff: {percent:.2f}%")