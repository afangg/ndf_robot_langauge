import os, os.path as osp
import random
import numpy as np
# import torch
import argparse
import time

import pybullet as p

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.utils import util
from ndf_robot.utils import path_util

from ndf_robot.utils.franka_ik import FrankaIK
from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from ndf_robot.utils.new_eval_utils import (
    safeCollisionFilterPair,
    process_xq_data,
    process_demo_data,
    post_process_grasp,
    get_ee_offset
)

from airobot import Robot, log_info
from airobot.utils import common
from airobot.utils.common import euler2quat

class Pipeline():

    def __init__(self):
        self.cfg = self.get_env_cfgs()
        self.obj_cfgs = self.get_obj_cfgs()
        self.robot = Robot('franka', pb_cfg={'gui': args.pybullet_viz}, arm_cfg={'self_collision': False, 'seed': args.seed})
        self.iterations = 3
        self.random_pos = False
        self.ee_pose = None

    def get_demo_dict(self, demo_dic, demo_class):
        obj_class = demo_class.split('/')[-1]
        for demo_dir in os.listdir(demo_class):
            demos_path = osp.join(demo_class, demo_dir)
            concept = obj_class + ' '
            if 'shelf' in demo_dir:
                concept += 'shelf '

            for fname in os.listdir(demos_path):
                if '_demo_' not in fname: continue
                verb = fname.split('_demo_')[0]

                if concept+verb not in demo_dic:
                    demo_dic[concept+verb] = []
                file_path = osp.join(demos_path, fname)
                demo_dic[concept+verb] = file_path
        return demo_dic

    def choose_demos(self, query_text):
        demo_dic = {}
        for demo_class in os.listdir(all_demos_dirs):
            class_path = osp.join(all_demos_dirs, demo_class)
            demo_dic = self.get_demo_dict(demo_dic, class_path)

        return demo_dic[query_text] if query_text in demo_dic else []

    def get_env_cfgs(self):
        # general experiment + environment setup/scene generation configs
        cfg = get_eval_cfg_defaults()
        config_fname = osp.join(path_util.get_ndf_config(), 'eval_cfgs', 'base_config.yaml')
        cfg.merge_from_file(config_fname)
        cfg.freeze()
        return cfg

    def get_obj_cfgs(self):
        # object specific configs
        obj_cfg = get_obj_cfg_defaults()
        # change hardcoded bottle
        obj_config_name = osp.join(path_util.get_ndf_config(), 'bottle_obj_cfg.yaml')
        obj_cfg.merge_from_file(obj_config_name)
        obj_cfg.freeze()

    def setup_sim(self):
        ik_helper = FrankaIK(gui=False)
        
        finger_joint_id = 9
        left_pad_id = 9
        right_pad_id = 10

        p.changeDynamics(self.robot.arm.robot_id, left_pad_id, lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, right_pad_id, lateralFriction=1.0)

        preplace_horizontal_tf_list = self.cfg.PREPLACE_HORIZONTAL_OFFSET_TF
        preplace_horizontal_tf = util.list2pose_stamped(self.cfg.PREPLACE_HORIZONTAL_OFFSET_TF)
        preplace_offset_tf = util.list2pose_stamped(self.cfg.PREPLACE_OFFSET_TF)

        # reset
        self.robot.arm.reset(force_reset=True)
        self.robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, self.cfg.TABLE_Z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)

        self.cams = MultiCams(self.cfg.CAMERA, self.robot.pb_client, n_cams=self.cfg.N_CAMERAS)
        cam_info = {}
        cam_info['pose_world'] = []
        for cam in self.cams.cams:
            cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.demos[0]['table_urdf'].item())
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
                                self.cfg.TABLE_POS,
                                table_ori,
                                scaling=self.cfg.TABLE_SCALING)
        if self.load_shelf:
            self.placement_link_id = 0
        

    def load_optimizer(self, model, demos):
        demo_target_info = []
        demo_shapenet_ids = []
        gripper_pts = None
        for fname in demos:
            print('Loading demo from fname: %s' % fname)
            data = np.load(fname, allow_pickle=True)
            ee_pose = data['ee_pose_world'].tolist()
            if not gripper_pts:
                gripper_pts = process_xq_data(data, shelf=self.load_shelf)        
            # do i need handle another case for rack and shelf or does rndf take care of that

            target_info, shapenet_id = process_demo_data(data)
            demo_target_info.append(target_info)
            demo_shapenet_ids.append(shapenet_id)

        optimizer = OccNetOptimizer(model, query_pts=gripper_pts, query_pts_real_shape=gripper_pts)
        optimizer.set_demo_info(demo_target_info)
        return optimizer, demo_shapenet_ids

    def get_test_objs(self, demo_objs, query_text):
        #either replace with random objects (manipulation can fail) or label object in demo
        obj_classes = ['bottle', 'mug', 'bowl']
        test_obj = [obj for obj in obj_classes if obj in query_text][0]
        if test_obj == 'mug':
            avoid_shapenet_ids = bad_shapenet_mug_ids_list + self.cfg.MUG.AVOID_SHAPENET_IDS
        elif test_obj == 'bowl':
            avoid_shapenet_ids = bad_shapenet_bowls_ids_list + self.cfg.BOWL.AVOID_SHAPENET_IDS
        elif test_obj == 'bottle':
            avoid_shapenet_ids = bad_shapenet_bottles_ids_list + self.cfg.BOTTLE.AVOID_SHAPENET_IDS 
        else:
            test_shapenet_ids = []

        # get objects that we can use for testing
        test_object_ids = {}
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), test_obj + '_centered_obj_normalized')
        shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)] if test_obj == 'mug' else os.listdir(shapenet_obj_dir)
        for s_id in shapenet_id_list:
            valid = s_id not in demo_objs and s_id not in avoid_shapenet_ids
            if args.only_test_ids:
                valid = valid and (s_id in test_shapenet_ids)
            
            if valid:
                # for testing, use the "normalized" object
                obj_obj_file = osp.join(shapenet_obj_dir, s_id, 'models/model_normalized.obj')
                obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'
                test_object_ids[s_id] = obj_obj_file_dec
        self.test_obj = test_obj
        return test_object_ids

    def add_object(self, test_obj_ids):
        x_low, x_high = self.cfg.OBJ_SAMPLE_X_HIGH_LOW
        y_low, y_high = self.cfg.OBJ_SAMPLE_Y_HIGH_LOW

        obj_shapenet_id = random.sample(test_obj_ids.keys(), 1)[0]
        id_str = 'Shapenet ID: %s' % obj_shapenet_id
        print(id_str)

        upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()
        # for testing, use the "normalized" object
        obj_file = test_obj_ids[obj_shapenet_id]
    
        scale_high, scale_low = self.cfg.MESH_SCALE_HIGH, self.cfg.MESH_SCALE_LOW
        scale_default = self.cfg.MESH_SCALE_DEFAULT
        if args.rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_high - scale_low) + scale_low] * 3
        else:
            mesh_scale=[scale_default] * 3

        if self.random_pos:
            if self.test_obj in ['bowl', 'bottle']:
                rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            else:
                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()
            pos = [
                np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                self.cfg.TABLE_Z]
            pose = pos + ori
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(util.list2pose_stamped(pose), util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]

        else:
            pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, table_z]
            pose = util.list2pose_stamped(pos + upright_orientation)
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]

        return obj_file, pos, ori, mesh_scale

    def segment_pcd(self, obj_id):
        depth_imgs = []
        seg_idxs = []
        obj_pcd_pts = []
        table_pcd_pts = []

        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
        for i, cam in enumerate(self.cams.cams): 
            # get image and raw point cloud
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)

            # flatten and find corresponding pixels in segmentation mask
            flat_seg = seg.flatten()
            flat_depth = depth.flatten()
            obj_inds = np.where(flat_seg == obj_id)
            table_inds = np.where(flat_seg == self.table_id)
            seg_depth = flat_depth[obj_inds[0]]  
            
            obj_pts = pts_raw[obj_inds[0], :]
            obj_pcd_pts.append(util.crop_pcd(obj_pts))
            table_pts = pts_raw[table_inds[0], :][::int(table_inds[0].shape[0]/500)]
            table_pcd_pts.append(table_pts)

            depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)

        target_obj_pcd_obs = np.concatenate(obj_pcd_pts, axis=0)  # object shape point cloud
        target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
        inliers = np.where(np.linalg.norm(target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
        target_obj_pcd_obs = target_obj_pcd_obs[inliers]

        return target_obj_pcd_obs

    def main(self, args, global_dict):
        # torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        all_objs_dirs = global_dict['all_objs_dirs']
        all_demos_dirs = global_dict['all_demos_dirs']
        query_text =  global_dict['query_text']
        if 'shelf' in query_text:
            self.load_shelf = True
        else:
            self.load_shelf = False

        self.demos = self.choose_demos(query_text)

        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256, 
            model_type='pointnet',
            return_features=True, 
            sigmoid=True).cuda()
        optimizer, demo_shapenet_ids = self.load_optimizer(model, self.demos)
        test_obj_ids = self.get_test_objs(demo_shapenet_ids, query_text)


        for iter in self.iterations:
            # load a test object
            obj_file, pos, ori, scale = self.add_object(test_obj_ids)
            obj_id = self.robot.pb_client.load_geom(
                'mesh',
                mass=0.01,
                mesh_scale=scale,
                visualfile=obj_file,
                collifile=obj_file,
                base_pos=pos,
                base_ori=ori)
            p.changeDynamics(obj_id, -1, lateralFriction=0.5)

            self.robot.arm.go_home(ignore_physics=True)
            self.robot.arm.move_ee_xyz([0, 0, 0.2])

            safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
            p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
            time.sleep(1.5)

            target_obj_pcd = self.segment_pcd(obj_id)

            # optimize ee pose
            ee_pose_mats, best_idx = optimizer.optimize_transform_implicit(target_obj_pcd, ee=True)
            ee_pose = util.pose_stamped2list(util.pose_from_matrix(ee_pose_mats[best_idx]))

            if not self.ee_pose:
                grasp_pt = post_process_grasp(ee_pose, target_obj_pcd, thin_feature=(not args.non_thin_feature), grasp_viz=args.grasp_viz, grasp_dist_thresh=args.grasp_dist_thresh)
                ee_pose[:3] = grasp_pt
                pregrasp_offset_tf = get_ee_offset(ee_pose=ee_pose)
                pre_ee_pose = util.pose_stamped2list(
                util.transform_pose(pose_source=util.list2pose_stamped(ee_pose), pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))
            else:
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_text', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    query_text = args.query_text

    all_objs_dirs = [path for path in path_util.get_ndf_obj_descriptions() if '_centered_obj_normalized' in path] 
    all_demos_dirs = osp.join(path_util.get_ndf_data(), 'demos')

    vnn_model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')

    global_dict = dict(
        all_objs_dirs=all_objs_dirs,
        all_demos_dirs=all_demos_dirs,
        vnn_checkpoint_path=vnn_model_path,
        query_text=query_text
    )

    eval = Pipeline()
    eval.main(args, global_dict)