from pipeline import Pipeline 
from vizServer import VizServer 


import argparse
import os, os.path as osp

import sys
from rndf_robot.system import vizServer
# sys.path.append('/home/afo/repos/relational_ndf/src/')

import time
import torch
from rndf_robot.utils import path_util
from rndf_robot.utils import util
import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network

from airobot import Robot, log_info, set_log_level, log_warn

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_util

random = True
def main(pipeline):
    # torch.manual_seed(args.seed)

    if random and pipeline.state == -1:
        config = dict(
            objects={'container': 1, 'bottle': 1}
        )
        pipeline.setup_random_scene(config)

    test_objs, concept = pipeline.prompt_query()
    pipeline.assign_classes(test_objs)
    pipeline.cfg = pipeline.get_env_cfgs()
    
    pipeline.set_initial_paths(concept)
    
    pipeline.load_demos(concept, pipeline.args.n_demos)
    pipeline.load_models()

    if not random:
        ids = pipeline.setup_scene_objs()
    else:
        ids = pipeline.find_relevant_objs()

    pipeline.segment_scene(ids)

    ee_poses = pipeline.find_correspondence()
    pipeline.execute(ee_poses)

    # pipeline.teleport_obj(obj_id, obj_end_pose_list)

    # current_scene = dict(
    #     final_ee_pos = ee_poses[-1],
    #     obj_pcd=target_obj_pcd,
    #     obj_pose=obj_pose_world,
    #     obj_id=obj_id
    # )

    pipeline.step(ee_poses[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--query_text', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=5)

    parser.add_argument('--pybullet_viz', action='store_true')
    parser.add_argument('--opt_visualize', action='store_true')
    parser.add_argument('--grasp_viz', action='store_true')

    parser.add_argument('--parent_model_path', type=str, default='')
    parser.add_argument('--child_model_path', type=str, default='')

    # parser.add_argument('--random', action='store_true', help='utilize random weights')
    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--opt_iterations', type=int, default=500)

    parser.add_argument('--relation_method', type=str, default='intersection', help='either "intersection", "ebm"')

    parser.add_argument('--pc_reference', type=str, default='parent', help='either "parent" or "child"')
    parser.add_argument('--skip_alignment', action='store_true')
    parser.add_argument('--new_descriptors', action='store_true')
    parser.add_argument('--create_descriptors', action='store_true')
    parser.add_argument('--n_demos', type=int, default=15)
    parser.add_argument('--target_idx', type=int, default=-1)
    parser.add_argument('--query_scale', type=float, default=0.025)
    parser.add_argument('--target_rounds', type=int, default=3)

    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--noise_idx', type=int, default=0)
    args = parser.parse_args()

    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    # all_objs_dirs = [path for path in path_util.get_ndf_obj_descriptions() if '_centered_obj_normalized' in path] 
    # all_demos_dirs = osp.join(path_util.get_ndf_data(), 'demos')

    pipeline = Pipeline(args)

    pipeline.load_demos_dict()
    pipeline.setup_client()

    server = VizServer(pipeline.robot.pb_client)
    pipeline.register_vizServer(server)

    pipeline.setup_table()
    pipeline.reset_robot()
    log_info('Loaded new table')
    for iter in range(args.iterations):
        main(pipeline)
