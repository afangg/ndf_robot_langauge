from pipeline import Pipeline 
from vizServer import VizServer 


import argparse
import os, os.path as osp

import sys

from rndf_robot.system import vizServer
sys.path.append('/home/afo/repos/relational_ndf/src/')

import time
import torch
from rndf_robot.utils import path_util
from rndf_robot.utils import util
import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network

from airobot import Robot, log_info, set_log_level, log_warn

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_util

def main_teleport(pipeline):
    # torch.manual_seed(args.seed)

    pipeline.prompt_query()
    pipeline.get_env_cfgs()
    pipeline.load_demos()
    pipeline.process_demos(True)

    pipeline.setup_table()
    log_info('Loaded new table')

    pipeline.setup_scene_objs()
    pipeline.segment_scene()

    ee_poses = pipeline.find_correspondence_rndf()

    # target_obj_pcd, obj_pose_world = pipeline.segment_pcd(obj_id)
    # obj_pose_world_list = util.pose_stamped2list(obj_pose_world)
    # pos, ori = obj_pose_world_list[:3], obj_pose_world_list[3:]

    # print('Object at pose:', util.pose_stamped2list(obj_pose_world))
    # optimizer = pipeline.load_optimizer(pipeline.demos)

    # ee_poses = pipeline.find_correspondence(optimizer, target_obj_pcd, obj_pose_world)
    # obj_end_pose_list = ee_poses[-1]

    # pipeline.teleport_obj(obj_id, obj_end_pose_list)

    # current_scene = dict(
    #     final_ee_pos = ee_poses[-1],
    #     obj_pcd=target_obj_pcd,
    #     obj_pose=obj_pose_world,
    #     obj_id=obj_id
    # )
    pipeline.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--query_text', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--pybullet_viz', action='store_true')
    parser.add_argument('--grasp_viz', action='store_true')
    parser.add_argument('--parent_model_path', type=str, default='ndf_vnn/rndf_weights/multi_category_weights.pth')
    parser.add_argument('--child_model_path', type=str, default='ndf_vnn/rndf_weights/multi_category_weights.pth')

    # parser.add_argument('--random', action='store_true', help='utilize random weights')
    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--teleport', action='store_true')
    parser.add_argument('--opt_iterations', type=int, default=100)

    args = parser.parse_args()
    # query_text = args.query_text

    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    # all_objs_dirs = [path for path in path_util.get_ndf_obj_descriptions() if '_centered_obj_normalized' in path] 
    # all_demos_dirs = osp.join(path_util.get_ndf_data(), 'demos')

    pipeline = Pipeline(args)

    pipeline.load_models()
    pipeline.load_meshes_dict()
    pipeline.load_demos_dict()
    pipeline.setup_client()

    server = VizServer(pipeline.robot.pb_client)
    pipeline.register_vizServer(server)

    for iter in range(args.iterations):
        main_teleport(pipeline)
