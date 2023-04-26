# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import sys
import cv2
import pickle

from detectron2.config import get_cfg
import shutil

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config

from detic.config import add_detic_config
from detic.predictor import VisualizationDemo
from argparse import ArgumentParser


from collections import namedtuple
Args = namedtuple("Args", ["vocabulary", "custom_vocabulary"])
# args = Args(vocabulary="lvis", custom_vocabulary=["bowl"])
# args_custom = Args(vocabulary="custom", custom_vocabulary=["bowl", "mug", "shelf", "cardboard_box"])


def setup_cfg():
    cfg = get_cfg()
    # cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.merge_from_list(['MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.3
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg

def get_predictions(rgbs, _args):
    cfg = setup_cfg()
    demo = VisualizationDemo(cfg, _args)

    pred_lst = []
    output_dir = "detic_predictions"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)
    

    for idx, im in enumerate(rgbs):
        print(f"finding objects in {idx}")

        # converting the input image to BGR format 
        # (the image needs to be BGR, and 0-255, np.uint8)

        img = im.astype(np.uint8)
        predictions, visualized_output = demo.run_on_image(img)
        pred_lst.append(predictions)
        visualized_output.save(os.path.join(output_dir, f"predictions_{idx}.png"))

    names = demo.metadata.get("thing_classes", None)
    return pred_lst, names

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--image-dir")
    parser.add_argument("--vocabulary")
    parser.add_argument("--custom-vocabulary")
    args = parser.parse_args()
    
    vocab_args = Args(vocabulary=args.vocabulary, custom_vocabulary=args.custom_vocabulary)

    rgbs = []
    image_fnames = os.listdir(args.image_dir)
    print("here", (image_fnames))

    tmp = []
    for f in image_fnames:
        if f.endswith('.png'):
            print(f); tmp.append(f)
            rgb = cv2.imread(os.path.join(args.image_dir, f))
            rgbs.append(rgb)

    preds_lst, names = get_predictions(rgbs, vocab_args)

    info_dict = {tmp[idx]: preds_lst[idx] for idx in range(len(tmp))}

    with open("predictions_summary.pkl", 'wb') as f:
        pickle.dump((info_dict, names), f)
