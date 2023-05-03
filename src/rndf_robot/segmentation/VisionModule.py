import numpy as np

from .Annotate import Annotate
from .SAM import SAMSeg
from .pcd_utils import pcds_from_masks, filter_pcds, extend_pcds
from rndf_robot.cameras.CameraSys import CameraSys
from rndf_robot.utils import util

from airobot import log_warn, log_debug
from IPython import embed

METHODS = {'bbox', 'point', 'owl', 'pb_seg'}
class VisionModule:
    
    def __init__(self, seg_method, mc_vis,) -> None:
        assert seg_method in METHODS, f'Please choose from {METHODS}'
        self.seg_method = seg_method

        self.mc_vis = mc_vis
        self.seg = SAMSeg()

        if seg_method == 'owl':
            from .OWLViT import OWLViTDetect
            self.obj_detector = OWLViTDetect()

    def initialize_cameras(self, camera_sys: CameraSys):
        self.camera_sys = camera_sys

    def segment_all_scenes(self, input_label):
        if self.seg_method == 'pb_seg':
            return self.pybullet_seg(input_label)
        else:
            return self.language_seg(input_label)    
    
    def pybullet_seg(self, obj_id_to_class):
        assert self.seg_method == 'pb_seg', 'Seg method does not use PyBullet'
        return self.camera_sys.get_pb_seg(obj_id_to_class)

    def language_seg(self, captions, centroid_thresh=0.1, detect_thresh=0.15):
        pcd_2ds, rgb_imgs, valid_depths = self.camera_sys.get_all_real_views(crop_show=True)
 
        label_to_pcds = {}
        label_to_scores = {}

        for i, rgb, pcd_2d, valid in zip(range(len(rgb_imgs)), rgb_imgs, pcd_2ds, valid_depths):
            all_obj_masks = {}
            all_obj_bb_scores = {}       
            if self.seg_method == 'bbox' or self.seg_method == 'point':

                for caption in captions:
                    a = Annotate()
                    mask = None
                    if self.seg_method == 'bbox':
                        selected = a.select_bb(rgb, f'Select {caption} in scene')
                        from IPython import embed
                        if None not in selected:
                            mask = self.seg.mask_from_bb(selected, image=rgb, show=False)
                    elif self.seg_method == 'point':
                        selected = a.select_pt(rgb, f'Select {caption} in scene')
                        if None not in selected:
                            mask = self.seg.mask_from_pt(selected, image=rgb, show=False)                    

                    if mask is not None:
                        if caption not in all_obj_masks:
                            all_obj_masks[caption] = []
                            all_obj_bb_scores[caption] = []

                        all_obj_masks[caption].append(mask)
                        all_obj_bb_scores[caption].append([1.0])
            elif self.seg_method == 'owl':
                detector_bboxes, detector_scores = self.obj_detector.detect_captions(
                                                        rgb, 
                                                        captions, 
                                                        max_count=1, 
                                                        score_threshold=detect_thresh)
                log_debug(f'Detected the following captions {detector_scores.keys()}')

                if not detector_bboxes:
                    continue
                all_obj_masks = self.seg.mask_from_bb(detector_bboxes, rgb)
            else:
                raise NotImplementedError('This segmentation method does not exist')

            for obj_label, obj_masks in all_obj_masks.items():
                log_debug(f'Mask count for {obj_label}: {len(obj_masks)}')
                obj_pcds, obj_scores = pcds_from_masks(pcd_2d, valid, obj_masks, all_obj_bb_scores[obj_label])
                log_debug(f'{obj_label} after filtering is now {len(obj_pcds)}')
                obj_pcds, obj_scores = filter_pcds(obj_pcds, obj_scores, mean_inliers=True, downsample=True)

                if not obj_pcds:
                    continue
                for j in range(len(obj_pcds)):
                    util.meshcat_pcd_show(self.mc_vis, obj_pcds[j], color=(100, 0, 100), name=f'scene/cam_{i}_{obj_label}_region_{j}')

                if obj_label not in label_to_pcds:
                    label_to_pcds[obj_label], label_to_scores[obj_label] = obj_pcds, obj_scores
                else:
                    new_pcds, new_lables = extend_pcds(obj_pcds, 
                                                       label_to_pcds[obj_label], 
                                                       obj_scores, 
                                                       label_to_scores[obj_label], 
                                                       threshold=centroid_thresh)
                    label_to_pcds[obj_label], label_to_scores[obj_label] = new_pcds, new_lables
                log_debug(f'{obj_label} size is now {len(label_to_pcds[obj_label])}')
        
        pcds_output = {}

        for obj_label in captions:
            if obj_label not in label_to_pcds:
                log_warn(f'WARNING: COULD NOT FIND {obj_label} OBJ')
                continue
            obj_pcd_sets = label_to_pcds[obj_label]
            for i, target_obj_pcd_obs in enumerate(obj_pcd_sets):
                score = np.average(label_to_scores[obj_label][i])
                if obj_label not in pcds_output:
                    pcds_output[obj_label] = []
                pcds_output[obj_label].append((score, target_obj_pcd_obs, -1))
        return pcds_output      
