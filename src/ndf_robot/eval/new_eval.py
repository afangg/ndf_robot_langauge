import os, os.path as osp
import random
import numpy as np
# import torch
import argparse
import time
import sys
print(sys.path)
sys.path.append('/home/afo/repos/ndf_robot_language/src/')
import pybullet as p

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.utils import util
from ndf_robot.utils import path_util

from ndf_robot.utils.franka_ik import FrankaIK
from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
# from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from ndf_robot.utils.new_eval_utils import (
    safeCollisionFilterPair,
    safeRemoveConstraint,
    soft_grasp_close,
    object_is_still_grasped,
    constraint_grasp_close,
    process_xq_data,
    process_demo_data,
    post_process_grasp,
    get_ee_offset,
    constraint_obj_world
)

from airobot import Robot, log_info
from airobot.utils import common
from airobot.utils.common import euler2quat

class Pipeline():

    def __init__(self, global_dict):
        self.global_dict = global_dict
        self.cfg = self.get_env_cfgs()
        self.obj_cfgs = self.get_obj_cfgs()
        self.robot = Robot('franka', pb_cfg={'gui': args.pybullet_viz}, arm_cfg={'self_collision': False, 'seed': args.seed})
        self.random_pos = False
        self.ee_pose = None
        self.table_model = None

        self.all_objs_dirs = self.global_dict['all_objs_dirs']
        self.all_demos_dirs = self.global_dict['all_demos_dirs']
        demo_dic = {}
        for demo_class in os.listdir(self.all_demos_dirs):
            print("Demo:", demo_class)
            class_path = osp.join(self.all_demos_dirs, demo_class)
            demo_dic = self.get_demo_dict(demo_dic, class_path)
        print('All demo labels:', demo_dic.keys())
        self.demo_dic = demo_dic
        

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
                # print('file', fname, 'with label', concept+verb)

                if concept+verb not in demo_dic:
                    demo_dic[concept+verb] = []
                file_path = osp.join(demos_path, fname)
                if self.table_model is None:
                    self.table_model = np.load(file_path, allow_pickle=True)['table_urdf'].item()
                demo_dic[concept+verb].append(file_path)
        return demo_dic

    def choose_demos(self, query_text):
        # print(self.demo_dic[query_text])
        return self.demo_dic[query_text] if query_text in self.demo_dic else []

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
        self.ik_helper = FrankaIK(gui=False)
        
        self.finger_joint_id = 9
        self.left_pad_id = 9
        self.right_pad_id = 10

        p.changeDynamics(self.robot.arm.robot_id, self.left_pad_id, lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, self.right_pad_id, lateralFriction=1.0)

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
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_model)
        print('Table file path:', tmp_urdf_fname)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
                                self.cfg.TABLE_POS,
                                table_ori,
                                scaling=self.cfg.TABLE_SCALING)
        if self.load_shelf:
            self.placement_link_id = 0
        print("DONE SETTING UP")


    def load_optimizer(self, model, demos):
        demo_target_info = []
        demo_shapenet_ids = []
        gripper_pts = None
        for fname in demos:
            print('Loading demo from fname: %s' % fname)
            data = np.load(fname, allow_pickle=True)
            if gripper_pts is None:
                gripper_pts = process_xq_data(data, shelf=self.load_shelf)        
            # do i need handle another case for rack and shelf or does rndf take care of that

            target_info, shapenet_id = process_demo_data(data)
            demo_target_info.append(target_info)
            demo_shapenet_ids.append(shapenet_id)

        optimizer = OccNetOptimizer(model, query_pts=gripper_pts, query_pts_real_shape=gripper_pts)
        optimizer.set_demo_info(demo_target_info)
        print("OPTIMIZER LOADED")
        return optimizer, demo_shapenet_ids

    def get_test_objs(self, demo_objs, query_text):
        #either replace with random objects (manipulation can fail) or label object in demo
        obj_classes = ['bottle', 'mug', 'bowl']
        test_obj = [obj for obj in obj_classes if obj in query_text][0]
        # if test_obj == 'mug':
        #     avoid_shapenet_ids = bad_shapenet_mug_ids_list + self.cfg.MUG.AVOID_SHAPENET_IDS
        # elif test_obj == 'bowl':
        #     avoid_shapenet_ids = bad_shapenet_bowls_ids_list + self.cfg.BOWL.AVOID_SHAPENET_IDS
        # elif test_obj == 'bottle':
        #     avoid_shapenet_ids = bad_shapenet_bottles_ids_list + self.cfg.BOTTLE.AVOID_SHAPENET_IDS 
        # else:
        #     test_shapenet_ids = []
        avoid_shapenet_ids = []
        # get objects that we can use for testing
        test_object_ids = {}
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), test_obj + '_centered_obj_normalized')
        print('ALL TEST OBJECTS DIRECTORY:', shapenet_obj_dir)
        shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)] if test_obj == 'mug' else os.listdir(shapenet_obj_dir)
        for s_id in shapenet_id_list:
            valid = s_id not in demo_objs and s_id not in avoid_shapenet_ids
            if valid:
                # for testing, use the "normalized" object
                obj_obj_file = osp.join(shapenet_obj_dir, s_id, 'models/model_normalized.obj')
                test_object_ids[s_id] = obj_obj_file
        self.test_obj = test_obj
        return test_object_ids

    def add_object(self, test_obj_ids):
        x_low, x_high = self.cfg.OBJ_SAMPLE_X_HIGH_LOW
        y_low, y_high = self.cfg.OBJ_SAMPLE_Y_HIGH_LOW
        table_z = self.cfg.TABLE_Z

        obj_shapenet_id = random.sample(test_obj_ids.keys(), 1)[0]
        id_str = 'Loading Shapenet ID: %s' % obj_shapenet_id
        print(id_str)

        upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()
        # for testing, use the "normalized" object
        obj_file = test_obj_ids[obj_shapenet_id]
        obj_file_dec = obj_file.split('.obj')[0] + '_dec.obj'

        scale_high, scale_low = self.cfg.MESH_SCALE_HIGH, self.cfg.MESH_SCALE_LOW
        scale_default = self.cfg.MESH_SCALE_DEFAULT
        # if args.rand_mesh_scale:
        #     mesh_scale = [np.random.random() * (scale_high - scale_low) + scale_low] * 3
        # else:
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

        # convert mesh with vhacd
        if not osp.exists(obj_file_dec):
            p.vhacd(
                obj_file,
                obj_file_dec,
                'log.txt',
                concavity=0.0025,
                alpha=0.04,
                beta=0.05,
                gamma=0.00125,
                minVolumePerCH=0.0001,
                resolution=1000000,
                depth=20,
                planeDownsampling=4,
                convexhullDownsampling=4,
                pca=0,
                mode=0,
                convexhullApproximation=1
            )

        obj_id = self.robot.pb_client.load_geom(
            'mesh',
            mass=0.01,
            mesh_scale=mesh_scale,
            visualfile=obj_file,
            collifile=obj_file,
            base_pos=pos,
            base_ori=ori)
        p.changeDynamics(obj_id, -1, lateralFriction=0.5)
        self.o_cid = constraint_obj_world(obj_id, pos, ori)
        self.robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        return obj_id, pos, ori

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

        return target_obj_pcd_obs, obj_pose_world

    def find_correspondence(self, optimizer, target_obj_pcd, obj_pose_world):
        ee_poses, obj_end_pose = [], None
        if self.ee_pose is None:
            #grasping
            ee_pose_mats, best_idx = optimizer.optimize_transform_implicit(target_obj_pcd, ee=True)
            ee_end_pose = util.pose_stamped2list(util.pose_from_matrix(ee_pose_mats[best_idx]))
            
            # grasping requires post processing to find anti-podal point
            grasp_pt = post_process_grasp(ee_end_pose, target_obj_pcd, thin_feature=True, grasp_viz=False, grasp_dist_thresh=0.0025)
            ee_end_pose[:3] = grasp_pt
            pregrasp_offset_tf = get_ee_offset(ee_pose=ee_end_pose)
            pre_ee_pose = util.pose_stamped2list(
            util.transform_pose(pose_source=util.list2pose_stamped(ee_end_pose), pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))
            ee_poses.append(pre_ee_pose)
            ee_poses.append(ee_end_pose)
        else:
            #placement
            pose_mats, best_idx = optimizer.optimize_transform_implicit(target_obj_pcd, ee=False)
            relative_pose = util.pose_stamped2list(util.pose_from_matrix(pose_mats[best_idx]))
            
            ee_pose = util.transform_pose(pose_source=util.list2pose_stamped(self.ee_pose), pose_transform=util.list2pose_stamped(relative_pose))
            ee_end_pose = ee_pose
            obj_start_pose = obj_pose_world
            obj_end_pose = util.transform_pose(pose_source=obj_start_pose, pose_transform=util.list2pose_stamped(relative_pose))
            
            ee_poses.append(obj_end_pose)
        return ee_poses, obj_end_pose

    def main(self, args):
        # torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        iterations = args.iterations

        query_text =  self.global_dict['query_text']
        
        if 'shelf' in query_text:
            self.load_shelf = True
            placement_link_id = 0
        else:
            self.load_shelf = False
            placement_link_id = None

        self.demos = self.choose_demos(query_text)
        print('Number of Demos', len(self.demos))
        print('Examples', self.demos[:5])
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256, 
            model_type='pointnet',
            return_features=True, 
            sigmoid=True).cuda()
        optimizer, demo_shapenet_ids = self.load_optimizer(model, self.demos)
        test_obj_ids = self.get_test_objs(demo_shapenet_ids, query_text)
        print('Number of Objects', len(test_obj_ids))

        self.setup_sim()

        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])
        self.robot.arm.eetool.open()
        time.sleep(1.5)

        for _ in range(iterations):
            # load a test object
            obj_id, pos, ori = self.add_object(test_obj_ids)

            target_obj_pcd, obj_pose_world = self.segment_pcd(obj_id)
            ee_poses, obj_end_pose = self.find_correspondence(optimizer, target_obj_pcd, obj_pose_world)

            if obj_end_pose:
                # reset object to placement pose to detect placement success
                obj_end_pose_list = util.pose_stamped2list(obj_end_pose)
                safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
                safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=False)
                self.robot.pb_client.set_step_sim(True)
                # safeRemoveConstraint(o_cid)
                self.robot.pb_client.reset_body(obj_id, obj_end_pose_list[:3], obj_end_pose_list[3:])

                time.sleep(1.0)
                safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=True)
                self.robot.pb_client.set_step_sim(False)
                time.sleep(1.0)

                safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
                self.robot.pb_client.reset_body(obj_id, pos, ori)
            
            jnt_pos = grasp_jnt_pos = grasp_plan = None
            place_success = grasp_success = False
            jnt_poses = [None]*len(ee_poses)
                
            # reset everything
            self.robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
            # safeRemoveConstraint(o_cid)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)
            print(p.getBasePositionAndOrientation(obj_id))
            time.sleep(0.5)

            # turn OFF collisions between robot and object / table, and move to pre-grasp pose
            for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                safeCollisionFilterPair(
                    bodyUniqueIdA=self.robot.arm.robot_id, 
                    bodyUniqueIdB=self.table_id, 
                    linkIndexA=i, 
                    linkIndexB=-1, 
                    enableCollision=False, 
                    physicsClientId=self.robot.pb_client.get_client_id())
                safeCollisionFilterPair(
                    bodyUniqueIdA=self.robot.arm.robot_id, 
                    bodyUniqueIdB=obj_id, 
                    linkIndexA=i, 
                    linkIndexB=-1, 
                    enableCollision=False, 
                    physicsClientId=self.robot.pb_client.get_client_id())
    
            for i, pose in enumerate(ee_poses):
                jnt_pos = jnt_poses[i]
                if jnt_pos is None:
                    jnt_pos = self.ik_helper.get_feasible_ik(pose)
                    if jnt_pos is None:
                        jnt_pos = self.ik_helper.get_ik(pose)
                        if jnt_pos is None:
                            jnt_pos = self.robot.arm.compute_ik(pose[:3], pose[3:]) 
                jnt_poses[i] = jnt_pos

            # ee_plans = []
            prev_pos = self.robot.arm.get_jpos()
            success = True
            for jnt_pos in jnt_poses:
                plan = self.ik_helper.plan_joint_motion(prev_pos, jnt_pos)
                if plan is None:
                    print('FAILED TO FIND A PLAN. STOPPING')
                    success = False
                    break
                for jnt in plan:
                    self.robot.arm.set_jpos(jnt, wait=False)
                    time.sleep(0.025)
                prev_pos = jnt_pos
            
            if success:
                
                # # get pose that's straight up
                # offset_pose = util.transform_pose(
                #     pose_source=util.list2pose_stamped(np.concatenate(self.robot.arm.get_ee_pose()[:2]).tolist()),
                #     pose_transform=util.list2pose_stamped([0, 0, 0.15, 0, 0, 0, 1])
                # )
                # offset_pose_list = util.pose_stamped2list(offset_pose)
                # offset_jnts = self.ik_helper.get_feasible_ik(offset_pose_list)

                # turn ON collisions between robot and object, and close fingers
                for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=True, physicsClientId=self.robot.pb_client.get_client_id())
                    # safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=rack_link_id, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())

                if self.ee_pose is None:
                    time.sleep(0.8)
                    obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[0]
                    jnt_pos_before_grasp = self.robot.arm.get_jpos()
                    soft_grasp_close(self.robot, self.finger_joint_id, force=50)
                    safeRemoveConstraint(self.o_cid)
                    time.sleep(0.8)
                    safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
                    time.sleep(0.8)
                else:
                    grasp_success = object_is_still_grasped(self.robot, obj_id, self.right_pad_id, self.left_pad_id) 

                    if grasp_success:
                        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
                        self.robot.arm.eetool.open()
                        # p.resetBasePositionAndOrientation(obj_id, obj_pos_before_grasp, ori)
                        # soft_grasp_close(self.robot, self.finger_joint_id, force=40)
                        # self.robot.arm.set_jpos(jnt_pos_before_grasp, ignore_physics=True)
                        # cid = constraint_grasp_close(self.robot, obj_id)
            
            # # observe and record outcome
            # obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, placement_link_id)
            # touching_surf = len(obj_surf_contacts) > 0
            # obj_floor_contacts = p.getContactPoints(obj_id, self.robot.arm.floor_id, -1, -1)
            # touching_floor = len(obj_floor_contacts) > 0
            # place_success = touching_surf and not touching_floor

            # if offset_jnts is not None:
            #     offset_plan = self.ik_helper.plan_joint_motion(self.robot.arm.get_jpos(), offset_jnts)

            #     if offset_plan is not None:
            #         for jnt in offset_plan:
            #             self.robot.arm.set_jpos(jnt, wait=False)
            #             time.sleep(0.04)
            #         self.robot.arm.set_jpos(offset_plan[-1], wait=True)

            # # turn OFF collisions between object / table and object / rack, and move to pre-place pose
            # safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
            # safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=False)
            time.sleep(1.0)
            while True:
                x = input("Press 1 to continue or 2 to use a new object")
                if x == 1:
                    query_text = input('Please enter the new query')
                    break
                elif x == 2:
                    self.robot.pb_client.remove_body(obj_id)
                    break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_text', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--pybullet_viz', action='store_true')

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

    eval = Pipeline(global_dict)
    eval.main(args)
