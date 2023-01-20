from pipeline import Pipeline 
import argparse
import os, os.path as osp

import time
import torch
from ndf_robot.utils import path_util
from ndf_robot.utils import util
from airobot import Robot, log_info, set_log_level, log_warn

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_util
import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network


def main(pipeline, args):
    # torch.manual_seed(args.seed)
    pipeline.prompt_query()

    if pipeline.scene_obj is None:
        pipeline.load_table()

        # load a test object
        test_obj_ids = pipeline.get_test_objs()
        obj_id, pos, ori = pipeline.add_object(test_obj_ids)
        target_obj_pcd, obj_pose_world = pipeline.segment_pcd(obj_id)
    else:
        target_obj_pcd, obj_pose_world, obj_id = pipeline.scene_obj
        obj_pose_world_list = util.pose_stamped2list(obj_pose_world)
        pos, ori = obj_pose_world_list[:3], obj_pose_world_list[3:]
    print('Object at pose:', util.pose_stamped2list(obj_pose_world))
    optimizer = pipeline.load_optimizer(pipeline.demos)

    ee_poses = pipeline.find_correspondence(optimizer, args, target_obj_pcd, obj_pose_world)
    obj_end_pose_list = ee_poses[-1]

    pipeline.pre_execution(obj_id, pos, ori, obj_end_pose_list)
    jnt_poses = pipeline.get_iks(ee_poses)

    prev_pos = pipeline.robot.arm.get_jpos()
    for i, jnt_pos in enumerate(jnt_poses):
        if jnt_pos is None:
            log_warn('No IK for jnt', i)
            break
        
        plan = pipeline.plan_motion(prev_pos, jnt_pos)
        if plan is None:
            log_warn('FAILED TO FIND A PLAN. STOPPING')
            break
        pipeline.execute_plan(plan)
        prev_pos = jnt_pos
        input('Press enter to continue')

    pipeline.post_execution(obj_id, pos, ori)

    time.sleep(1.0)
    current_scene = dict(
        final_ee_pos = ee_poses[-1],
        obj_pcd=target_obj_pcd,
        obj_pose=obj_pose_world,
        obj_id=obj_id
    )
    pipeline.step(current_scene)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--query_text', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--pybullet_viz', action='store_true')
    parser.add_argument('--grasp_viz', action='store_true')
    parser.add_argument('--weights', type=str, default='multi_category_weights')
    parser.add_argument('--random', action='store_true', help='utilize random weights')
    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)

    args = parser.parse_args()
    # query_text = args.query_text

    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    all_objs_dirs = [path for path in path_util.get_ndf_obj_descriptions() if '_centered_obj_normalized' in path] 
    all_demos_dirs = osp.join(path_util.get_ndf_data(), 'demos')

    vnn_model_path = osp.join(path_util.get_ndf_model_weights(), args.weights + '.pth')
    # ee_mesh = trimesh.load('../floating/panda_gripper.obj')
    # ee_mesh.show()

    ll_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    ndf_model = vnn_occupancy_network.VNNOccNet(
                latent_dim=256, 
                model_type='pointnet',
                return_features=True, 
                sigmoid=True).cuda()
    if not args.random:
        ndf_model.load_state_dict(torch.load(vnn_model_path))


    global_dict = dict(
        all_objs_dirs=all_objs_dirs,
        all_demos_dirs=all_demos_dirs,
        ndf_model=ndf_model,
        ll_model=ll_model,
    )

    pipeline = Pipeline(global_dict, args)
    
    for iter in range(args.iterations):
        main(pipeline, args)
