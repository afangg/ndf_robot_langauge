import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import build_sam, sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download
from IPython import embed

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth")
predictor = SamPredictor(build_sam(checkpoint=chkpt_path))

checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=chkpt_path)
# sam.to(device='cuda')
predictor = SamPredictor(sam)


def get_masks(image, all_obj_bbs):
    predictor.set_image(image)

    all_obj_masks = {}
    captions = list(all_obj_bbs.keys())
    for caption in captions:
        all_obj_masks[caption] = []
        for bb in all_obj_bbs[caption]:
            print(bb)

            bb_inpt = np.array([np.array(bb)])
            masks_np, iou_predictions_np, low_res_masks_np = predictor.predict(box = bb_inpt)
            if len(masks_np) == 0:
                continue
            combined_mask = np.array(np.sum(masks_np, axis=0), dtype=bool)
            all_obj_masks[caption].append(combined_mask)

    return all_obj_masks
