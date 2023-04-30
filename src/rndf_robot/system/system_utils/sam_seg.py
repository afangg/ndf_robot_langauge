from copy import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from segment_anything import build_sam, sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download
from IPython import embed

class Annotate(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release_draw(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.fig.canvas.draw()
        time.sleep(0.3)
        plt.close()

    def select_bb(self, image, message):
        print(f'{message}')
        self.ax.add_patch(self.rect)
        cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release_draw)
        self.ax.imshow(image)
        plt.show(block=True)
        self.fig.canvas.mpl_disconnect(cid_press)
        self.fig.canvas.mpl_disconnect(cid_release)
        return np.array([self.x0, self.y0, self.x1, self.y1])

    def select_pt(self, image, message):
        print(f'{message}')
        cid = self.fig.canvas.mpl_connect('button_release_event', self.on_press)
        self.ax.imshow(image)
        plt.show(block=True)
        self.fig.canvas.mpl_disconnect(cid)
        plt.close()
        return np.array([self.x0, self.y0])



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
sam.to(device='cuda')
predictor = SamPredictor(sam)

def get_masks(image, all_obj_bbs):
    predictor.set_image(image)

    all_obj_masks = {}
    captions = list(all_obj_bbs.keys())
    for caption in captions:
        all_obj_masks[caption] = []
        for bb in all_obj_bbs[caption]:
            combined_mask = get_mask_from_bb(bb)
            if combined_mask is None:
                continue
            all_obj_masks[caption].append(combined_mask)

    return all_obj_masks

def get_mask_from_bb(bb, image=None, show=False):
    if image is not None:
        predictor.set_image(image)

    bb_inpt = np.array([np.array(bb)])
    masks_np, _, _ = predictor.predict(box = bb_inpt)
    if len(masks_np) == 0:
        return []
    combined_mask = np.array(np.sum(masks_np, axis=0), dtype=bool)
    if show:
        show_mask(combined_mask, plt)
    return combined_mask

def get_mask_from_pt(input_pt, pt_label=1, image=None, show=False):
    if image is not None:
        predictor.set_image(image)

    pt_inpt = np.array([np.array(input_pt)])
    pt_label = np.array([pt_label])
    masks_np, _, _ = predictor.predict(point_coords=pt_inpt, point_labels=pt_label)
    if len(masks_np) == 0:
        return []
    combined_mask = np.array(np.sum(masks_np, axis=0), dtype=bool)
    if show:
        show_mask(combined_mask, plt)
    return combined_mask