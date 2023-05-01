import os, os.path as osp
import random
import copy
import numpy as np
import torch
import time
import sys
import meshcat
import trimesh
import open3d
import pyrealsense2 as rs
from polymetis import GripperInterface, RobotInterface
from matplotlib import pyplot as plt

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

sys.path.append(os.environ['SOURCE_DIR'])

import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from rndf_robot.utils import util, path_util, trimesh_util

from rndf_robot.utils.franka_ik_ndf import FrankaIK
from rndf_robot.opt.optimizer import OccNetOptimizer
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rndf_robot.utils.pipeline_util import (
    process_xq_data,
    process_xq_rs_data,
    process_demo_data,
    post_process_grasp_point,
    get_ee_offset,
)

from rndf_robot.data.rndf_utils import infer_relation_intersection, create_target_descriptors
from system_utils.language import chunk_query, create_keyword_dic
from system_utils.language import query_correspondance
from system_utils.demos import all_demos, get_concept_demos, create_target_desc_subdir, get_model_paths
import system_utils.objects as objects

from rndf_robot.utils.visualize import PandaHand, Robotiq2F140Hand
from rndf_robot.segmentation.pcd_utils import filter_pcds, pcds_from_masks, extend_pcds, manually_segment_pcd

from rndf_robot.robot.franka_ik import FrankaIK #, PbPlUtils
from rndf_robot.cameras.simple_multicam import MultiRealsenseLocal

from rndf_robot.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rndf_robot.utils.real.traj_util import PolymetisTrajectoryUtil
from rndf_robot.utils.real.plan_exec_util import PlanningHelper
from rndf_robot.utils.real.perception_util import enable_devices
from rndf_robot.utils.real.polymetis_util import PolymetisHelper

from rndf_robot.cameras.simple_multicam import MultiRealsenseLocal
from rndf_robot.cameras.realsense import RealsenseLocal, enable_devices, pipeline_stop

from IPython import embed;

poly_util = PolymetisHelper()

class Pipeline():

    def __init__(self, args):
        self.args = args
        self.ee_pose = None

        self.state = -1 #pick = 0, place = 1, teleport = 2
        self.table_obj = -1 #shelf = 0, rack = 1

        self.cfg = self.get_env_cfgs()
        self.current_panda_plan = []
        self.gripper_is_open = True

        self.ranked_objs = {}
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        self.mc_vis = meshcat.Visualizer(zmq_url=f'tcp://127.0.0.1:{self.args.port_vis}')
        self.mc_vis['scene'].delete()
        self.mc_vis['optimizer'].delete()
        self.mc_vis['ee'].delete()

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        if args.debug:
            set_log_level('debug')
        else:
            set_log_level('info')

    def next_iter(self):
        while True:
            i = input(
                '''What should we do
                    [h]: Move to home
                    [o]: Open gripper
                    [n]: Continue to next iteration
                    [i] to set into low stiffness mode
                    [l] to lock into current configuration with high stiffness
                    [em]: Launch interactive mode
                    [clear]: Clears the env variables - will segment everything again

                ''')
            
            if i == 'h':
                self.reset_robot()
                continue
            elif i == 'o':
                self.planning.gripper_open()
                self.gripper_is_open = True
            elif i == 'n':
                self.mc_vis['scene'].delete()
                self.mc_vis['optimizer'].delete()
                self.mc_vis['ee'].delete()
                break
            elif i == 'i':
                print('\n\nSetting low stiffness in current pose, you can now move the robot')
                # panda.set_cart_impedance_pose(panda.endpoint_pose(), stiffness=[0]*6)
                self.panda.start_cartesian_impedance(Kx=torch.zeros(6), Kxd=torch.zeros(6))
                continue
            elif i == 'l':
                print('\n\nSetting joint positions to current value\n\n')
                self.panda.start_joint_impedance()
                # panda.start_joint_impedance(Kq=Kq_new, Kqd=Kqd_new)
                continue
            elif i =='em':
                embed()
            elif i == 'clear':
                self.ranked_objs = {}     
                self.state = -1
                continue
            else:
                print('Unknown command')
                continue
        torch.cuda.empty_cache()

    def setup_client(self):
        #TODO: Setup real robot - hopefully interfacing isn't too different than Pybullet robot
        self.ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], occnet=False, robotiq=(self.args.gripper_type=='2f140'), mc_vis=self.mc_vis)
        franka_ip = "173.16.0.1" 
        panda = RobotInterface(ip_address=franka_ip)
        gripper = GripperInterface(ip_address=franka_ip)

        traj_helper = PolymetisTrajectoryUtil(robot=panda)

        tmp_obstacle_dir = osp.join(path_util.get_rndf_obj_descriptions(), 'tmp_planning_obs')
        planning = PlanningHelper(
            mc_vis=self.mc_vis,
            robot=panda,
            gripper=gripper,
            ik_helper=self.ik_helper,
            traj_helper=traj_helper,
            tmp_obstacle_dir=tmp_obstacle_dir
        )

        gripper_speed = planning.gripper_speed
        gripper_force = 40.0
        gripper_open_pos = 0.0 if self.args.gripper_type == 'panda' else gripper.get_state().max_width
        default_2f140_open_width = 0
        planning.set_gripper_speed(gripper_speed)
        planning.set_gripper_force(gripper_force)
        planning.set_gripper_open_pos(gripper_open_pos)
        if not self.gripper_is_open:
            planning.gripper_open()
            self.gripper_is_open = True

        self.planning = planning
        self.panda = panda

    def setup_cams(self):
        rs_cfg = get_default_multi_realsense_cfg()
        serials = rs_cfg.SERIAL_NUMBERS

        prefix = rs_cfg.CAMERA_NAME_PREFIX
        camera_names = [f'{prefix}{i}' for i in range(len(serials))]
        cam_list = [camera_names[int(idx)] for idx in self.args.cam_index]       
        serials = [serials[int(idx)] for idx in self.args.cam_index]

        calib_dir = osp.join(path_util.get_rndf_src(), 'robot/camera_calibration_files')
        calib_filenames = [osp.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in self.args.cam_index]

        self.cams = MultiRealsenseLocal(cam_names=cam_list, calib_filenames=calib_filenames)
        ctx = rs.context() # Create librealsense context for managing devices

        # Define some constants
        resolution_width = 640 # pixels
        resolution_height = 480 # pixels
        frame_rate = 30  # fps

        # pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)
        self.pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)
        self.cam_interface = RealsenseLocal()

    def setup_table(self):   
        tmp_obstacle_dir = osp.join(path_util.get_rndf_obj_descriptions(), 'tmp_planning_obs')
        util.safe_makedirs(tmp_obstacle_dir)
        table_obs = trimesh.creation.box([0.77, 1.22, 0.001]) #.apply_transform(util.matrix_from_list([0.15 + 0.77/2.0, 0.0015, 0.0, 0.0, 0.0, 0.0, 1.0]))
        cam_obs1 = trimesh.creation.box([0.2, 0.1, 0.2]) #.apply_transform(util.matrix_from_list([0.135, 0.55, 0.1, 0.0, 0.0, 0.0, 1.0]))
        cam_obs2 = trimesh.creation.box([0.2, 0.1, 0.5]) #.apply_transform(util.matrix_from_list([0.135, -0.525, 0.25, 0.0, 0.0, 0.0, 1.0]))

        table_obs_fname = osp.join(tmp_obstacle_dir, 'table.obj')
        cam1_obs_fname = osp.join(tmp_obstacle_dir, 'cam1.obj')
        cam2_obs_fname = osp.join(tmp_obstacle_dir, 'cam2.obj')
        table_obs.export(table_obs_fname)
        cam_obs1.export(cam1_obs_fname)
        cam_obs2.export(cam2_obs_fname)

        self.ik_helper.register_object(
        table_obs_fname,
        pos=[0.15 + 0.77/2.0, 0.0, 0.0015],
        ori=[0, 0, 0, 1],
        name='table')

        self.ik_helper.register_object(
        cam1_obs_fname,
        pos=[0.0, -0.525, 0.25],  # pos=[0.135, -0.525, 0.25],
        ori=[0, 0, 0, 1],
        name='cam2')
        
    def reset_robot(self):
        current_panda_plan = self.planning.plan_home()
        self.planning.execute_pb_loop(current_panda_plan)            
        i = input('Take it home (y/n)?')
        if i == 'y':
            self.planning.execute_loop(current_panda_plan)            
         

    #################################################################################################
    # Loading config settings and files

    def get_env_cfgs(self):
        # general experiment + environment setup/scene generation configs
        cfg = get_eval_cfg_defaults()
        # if 0 not in self.ranked_objs: return

        # config = 'base_config.yaml' if 1 in self.ranked_objs else 'eval_'+self.obj_info[self.ranked_objs[0]]['class']+'_gen.yaml'
        config = 'base_config.yaml'
        config_fname = osp.join(path_util.get_rndf_config(), 'eval_cfgs', config)
        if osp.exists(config_fname):
            cfg.merge_from_file(config_fname)
            log_debug('Config file loaded')
        else:
            log_info('Config file %s does not exist, using defaults' % config_fname)
        cfg.freeze()
        return cfg

    #################################################################################################
    # Segment the scene

    def assign_pcds(self, labels_to_pcds, re_seg=False):
        # pick the pcd with the highest score
        for label in labels_to_pcds:
            labels_to_pcds[label].sort(key=lambda x: x[0])

        for obj_rank in self.ranked_objs:
            if not re_seg and 'pcd' in self.ranked_objs[obj_rank]: continue
            description = self.ranked_objs[obj_rank]['description']
            obj_class = self.ranked_objs[obj_rank]['potential_class']

            if description in labels_to_pcds:
                pcd_key = description
            elif obj_class in labels_to_pcds:
                pcd_key = obj_class
            else:
                log_warn(f'Could not find pcd for ranked obj {obj_rank}')
                continue
            # just pick the last pcd
            new_pcds = []
            for pcd_tup in labels_to_pcds[pcd_key]:
                score, pcd = pcd_tup

                new_pcds.append(pcd)
            self.ranked_objs[obj_rank]['pcd'] = new_pcds.pop(-1)
            if self.args.show_pcds:
                log_debug(f'Best score was {score}')

        for rank, obj in self.ranked_objs.items():
            label = f'scene/initial_{rank}_pcd'
            color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
            util.meshcat_pcd_show(self.mc_vis, obj['pcd'], color=color, name=label)

    def get_real_pcd(self):
        pcd_pts = []
        pcd_dict_list = []
        cam_int_list = []
        cam_poses_list = []
        rgb_imgs = []
        depth_imgs = []
        for idx, cam in enumerate(self.cams.cams):
            cam_intrinsics = self.cam_interface.get_intrinsics_mat(self.pipelines[idx])
            rgb, depth = self.cam_interface.get_rgb_and_depth_image(self.pipelines[idx])

            cam.cam_int_mat = cam_intrinsics
            cam._init_pers_mat()
            cam_pose_world = cam.cam_ext_mat
            cam_int_list.append(cam_intrinsics)
            cam_poses_list.append(cam_pose_world)

            valid = depth < cam.depth_max
            valid = np.logical_and(valid, depth > cam.depth_min)
            depth_valid = copy.deepcopy(depth)
            depth_valid[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth
            depth_imgs.append(depth_valid)

            pcd_cam = cam.get_pcd(in_world=False, filter_depth=False, rgb_image=rgb, depth_image=depth_valid)[0]
            pcd_cam_img = pcd_cam.reshape(depth.shape[0], depth.shape[1], 3)
            pcd_world = util.transform_pcd(pcd_cam, cam_pose_world)
            pcd_world_img = pcd_world.reshape(depth.shape[0], depth.shape[1], 3)
            pcd_dict = {
                'world': pcd_world,
                'cam': pcd_cam_img,
                'cam_img': pcd_cam,
                'world_img': pcd_world_img,
                'cam_pose_mat': cam_pose_world
            }
            util.meshcat_pcd_show(self.mc_vis, pcd_world, color=(0, 255, 0), name=f'scene/scene_{idx}')

            rgb_imgs.append(rgb)
            pcd_pts.append(pcd_world)
            pcd_dict_list.append(pcd_dict)
        pcd_full = np.concatenate(pcd_pts, axis=0)
        # util.meshcat_pcd_show(self.mc_vis, pcd_full, color=(0, 255, 0), name='scene/full_scene')
        return pcd_full, rgb_imgs
    

    def get_real_pcd_cam(self, idx, rgb, depth):
        cam = self.cams.cams[idx]
        cam_intrinsics = self.cam_interface.get_intrinsics_mat(self.pipelines[idx])

        cam.cam_int_mat = cam_intrinsics
        cam._init_pers_mat()
        cam_pose_world = cam.cam_ext_mat

        valid = depth < cam.depth_max
        valid = np.logical_and(valid, depth > cam.depth_min)
        depth_valid = copy.deepcopy(depth)
        depth_valid[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth

        pcd_cam = cam.get_pcd(in_world=False, filter_depth=False, rgb_image=rgb, depth_image=depth_valid)[0]
        pcd_cam_img = pcd_cam.reshape(depth.shape[0], depth.shape[1], 3)
        pcd_world = util.transform_pcd(pcd_cam, cam_pose_world)
        pcd_world_img = pcd_world.reshape(depth.shape[0], depth.shape[1], 3)
    
        cropx, cropy, cropz, crop_note = [0.2, 0.75], [-0.4, 0.0], [0.01, 0.35], 'table_right'
        proc_pcd = manually_segment_pcd(pcd_world, x=cropx, y=cropy, z=cropz, note=crop_note)
        util.meshcat_pcd_show(self.mc_vis, proc_pcd, name=f'scene/pcd_world_cam_{idx}')

        return pcd_world_img, depth_valid
    
    def segment_scene(self, captions=None):
        '''
        @obj_captions: list of object captions to have CLIP detect
        @sim_seg: use pybullet gt segmentation or not 
        '''
        if not captions:
            captions = []
            for obj_rank in self.ranked_objs:
                if 'pcd' not in self.ranked_objs[obj_rank]:
                    captions.append(self.ranked_objs[obj_rank]['description'])
        
        rgb_imgs = []
        pcds_2d = []
        valid_depths = []
        for i, cam in enumerate(self.cams.cams): 
            # get image and raw point cloud
            rgb, depth = self.cam_interface.get_rgb_and_depth_image(self.pipelines[i])
            pts_2d, valid = self.get_real_pcd_cam(i, rgb, depth)
            rgb_imgs.append(rgb)
            pcds_2d.append(pts_2d)
            valid_depths.append(valid)

        label_to_pcds = {}
        label_to_scores = {}
        centroid_thresh = 0.1
        detect_thresh = 0.15
        for i, rgb, pcd_2d, valid in zip(range(len(rgb)), rgb_imgs, pcds_2d, valid_depths):
            # Object Detection
            # all_obj_bbs, all_obj_bb_scores = detect_bbs(rgb, 
            #                                             captions, 
            #                                             max_count=1, 
            #                                             score_threshold=detect_thresh)
            # log_debug(f'Detected the following captions {all_obj_bb_scores.keys()}')

            # if not all_obj_bbs:
            #     continue
            # all_obj_masks = get_masks(rgb, all_obj_bbs)

            # all_obj_masks = {}
            # all_obj_bb_scores = {}
            # for caption in captions:
            #     selected_pt = a.select_pt(rgb, f'Select {caption} in scene')
            #     if selected_pt is not None:
            #         mask = get_mask_from_pt(selected_pt, image=rgb, show=True)
            #         # partial_pcd = pts_2d[mask].reshape((-1,3))
            #         if caption not in all_obj_masks:
            #             all_obj_masks[caption] = []
            #             all_obj_bb_scores[caption] = []

            #         all_obj_masks[caption].append(mask)
            #         all_obj_bb_scores[caption].append([1.0])

            all_obj_masks = {}
            all_obj_bb_scores = {}

            for caption in captions:
                a = Annotate()

                selected_bb = a.select_bb(rgb, f'Select {caption} in scene')
                if selected_bb is not None:
                    mask = get_mask_from_bb(selected_bb, image=rgb, show=False)
                    # partial_pcd = pts_2d[mask].reshape((-1,3))
                    if caption not in all_obj_masks:
                        all_obj_masks[caption] = []
                        all_obj_bb_scores[caption] = []

                    all_obj_masks[caption].append(mask)
                    all_obj_bb_scores[caption].append([1.0])


            for obj_label, obj_masks in all_obj_masks.items():
                log_debug(f'Region count for {obj_label}: {len(obj_masks)}')
                obj_pcds, obj_scores = pcds_from_masks(pcd_2d, valid, obj_masks, all_obj_bb_scores[obj_label], is_bbox=False)
                log_debug(f'{obj_label} after filtering is now {len(obj_pcds)}')
                obj_pcds, obj_scores = filter_pcds(obj_pcds, obj_scores, mean_inliers=True)
                for j in range(len(obj_pcds)):
                    util.meshcat_pcd_show(self.mc_vis, obj_pcds[j], color=(0, 255, 0), name=f'scene/cam_{i}_{obj_label}_region_{j}')

                # this was not commented out?
                # cam_pcds, cam_scores = obj_pcds, obj_scores
                if not obj_pcds:
                    continue
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
                pcds_output[obj_label].append((score, target_obj_pcd_obs))
        return pcds_output        

    #################################################################################################
    # Optimization 
    def get_initial_model_paths(self, concept):
        target_class = self.ranked_objs[0]['potential_class']
        target_model_path = self.args.child_model_path
        if not target_model_path:
            target_model_path = 'ndf_vnn/rndf_weights/ndf_'+target_class+'.pth'

        if 1 in self.ranked_objs:
            relational_class = self.ranked_objs[1]['potential_class']
            relational_model_path = self.args.parent_model_path
            if not relational_model_path:
                relational_model_path = 'ndf_vnn/rndf_weights/ndf_'+relational_class+'.pth'

            demo_path = osp.join(path_util.get_rndf_data(), 'release_real_demos', concept)
            target_desc_subdir = create_target_desc_subdir(demo_path, relational_model_path, target_model_path, create=False)
            target_desc_fname = osp.join(demo_path, target_desc_subdir, 'target_descriptors.npz')
            if not osp.exists(target_desc_fname):
                alt_descs = os.listdir(demo_path)
                assert len(alt_descs) > 0, 'There are no descriptors for this concept. Please generate descriptors first'
                for alt_desc in alt_descs:
                    if not alt_desc.endswith('.npz'):
                        relational_model_path, target_model_path = get_model_paths(alt_desc)
                        break
                log_warn('Using the first set of descriptors found because descriptors not specified')
            self.ranked_objs[1]['model_path'] = relational_model_path
        self.ranked_objs[0]['model_path'] = target_model_path
        log_debug('Using model %s' %target_model_path)

    def load_models(self):
        for obj_rank in self.ranked_objs:
            if 'model' in self.ranked_objs[obj_rank]: continue
            model_path = osp.join(path_util.get_rndf_model_weights(), self.ranked_objs[obj_rank]['model_path'])
            model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
            model.load_state_dict(torch.load(model_path))
            self.ranked_objs[obj_rank]['model'] = model
            self.load_optimizer(obj_rank)
            log_debug('Model for %s is %s'% (obj_rank, model_path))

    def load_optimizer(self, obj_rank):
        query_pts_rs = self.ranked_objs[obj_rank]['query_pts'] if 'query_pts_rs' not in self.ranked_objs[obj_rank] else self.ranked_objs[obj_rank]['query_pts_rs']
        log_debug(f'Now loading optimizer for {obj_rank}')
        optimizer = OccNetOptimizer(
            self.ranked_objs[obj_rank]['model'],
            query_pts=self.ranked_objs[obj_rank]['query_pts'],
            query_pts_real_shape=query_pts_rs,
            opt_iterations=self.args.opt_iterations,
            cfg=self.cfg.OPTIMIZER)
        optimizer.setup_meshcat(self.mc_vis)
        self.ranked_objs[obj_rank]['optimizer'] = optimizer

    def find_correspondence(self):
        relational_rank = 1 if 1 in self.ranked_objs else 0
        if self.state == 0:
            ee_poses = self.find_pick_transform(0)
        elif self.state == 1:
            current_ee_pose = poly_util.polypose2np(self.panda.get_ee_pose())
            ee_poses = self.find_place_transform(0, relational_rank=relational_rank, ee=current_ee_pose)
        elif self.state == 2:
            ee_poses = self.find_place_transform(0, relational_rank=relational_rank)

        ee_file = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/robotiq_2f140/full_hand_2f140.obj')
        for i, ee_pose in enumerate(ee_poses):
            pose = util.body_world_yaw(util.list2pose_stamped(ee_pose), theta=-1.5708)
            pose = util.matrix_from_pose(pose)
            util.meshcat_obj_show(self.mc_vis, ee_file, pose, 1.0, name=f'ee/ee_{i}')
        return ee_poses

    def find_pick_transform(self, target_rank):
        optimizer = self.ranked_objs[target_rank]['optimizer']
        target_pcd = self.ranked_objs[target_rank]['pcd']

        ee_poses = []
        log_debug('Solve for pre-grasp coorespondance')
        optimizer.set_demo_info(self.ranked_objs[target_rank]['demo_info'])
        pre_ee_pose_mats, best_idx = optimizer.optimize_transform_implicit(target_pcd, ee=True, visualize=self.args.opt_visualize)
        pre_ee_pose = util.pose_stamped2list(util.pose_from_matrix(pre_ee_pose_mats[best_idx]))
        # grasping requires post processing to find anti-podal point
        grasp_pt = post_process_grasp_point(pre_ee_pose, target_pcd, thin_feature=(not self.args.non_thin_feature), grasp_viz=self.args.grasp_viz, grasp_dist_thresh=self.args.grasp_dist_thresh)
        pre_ee_pose[:3] = grasp_pt
        pre_ee_offset_tf = get_ee_offset(ee_pose=pre_ee_pose)
        pre_pre_ee_pose = util.pose_stamped2list(
            util.transform_pose(pose_source=util.list2pose_stamped(pre_ee_pose), pose_transform=util.list2pose_stamped(pre_ee_offset_tf)))

        # ee_poses.append(pre_pre_ee_pose)
        ee_poses.append(pre_ee_pose)
        return ee_poses
    
    def find_place_transform(self, target_rank, relational_rank=None, ee=None):
        #placement
        log_debug('Solve for placement coorespondance')
        optimizer = self.ranked_objs[target_rank]['optimizer']
        target_pcd = self.ranked_objs[target_rank]['pcd']
        ee_poses = []

        if relational_rank:
            relational_optimizer = self.ranked_objs[relational_rank]['optimizer']
            relational_pcd = self.ranked_objs[relational_rank]['pcd']
            relational_target_desc, target_desc = self.ranked_objs[relational_rank]['target_desc'], self.ranked_objs[target_rank]['target_desc']
            relational_query_pts, target_query_pcd = self.ranked_objs[relational_rank]['query_pts'], self.ranked_objs[target_rank]['query_pts']

            final_pose_mat = infer_relation_intersection(
                self.mc_vis, relational_optimizer, optimizer, 
                relational_target_desc, target_desc, 
                relational_pcd, target_pcd, relational_query_pts, target_query_pcd, opt_visualize=self.args.opt_visualize)
        else:
            optimizer.set_demo_info(self.ranked_objs[target_rank]['demo_info'])
            pose_mats, best_idx = optimizer.optimize_transform_implicit(target_pcd, ee=False, visualize=self.args.opt_visualize)
            final_pose_mat = pose_mats[best_idx]
        final_pose = util.pose_from_matrix(final_pose_mat)

        if ee is not None:
            ee_end_pose = util.transform_pose(pose_source=util.list2pose_stamped(ee), pose_transform=final_pose)
            preplace_offset_tf = util.list2pose_stamped(self.cfg.PREPLACE_OFFSET_TF)
            # preplace_direction_tf = util.list2pose_stamped(self.cfg.PREPLACE_HORIZONTAL_OFFSET_TF)

            preplace_direction_tf = util.list2pose_stamped(self.cfg.PREPLACE_VERTICAL_OFFSET_TF)

            pre_ee_end_pose2 = util.transform_pose(pose_source=ee_end_pose, pose_transform=preplace_offset_tf)
            pre_ee_end_pose1 = util.transform_pose(pose_source=pre_ee_end_pose2, pose_transform=preplace_direction_tf)        

            # get pose that's straight up
            current_ee_pose = poly_util.polypose2np(self.panda.get_ee_pose())
            offset_pose = util.transform_pose(
                pose_source=util.list2pose_stamped(current_ee_pose),
                pose_transform=util.list2pose_stamped([0, 0, 0.15, 0, 0, 0, 1])
            )
            ee_poses.append(util.pose_stamped2list(offset_pose)) 
            # ee_poses.append(util.pose_stamped2list(pre_ee_end_pose1))
            # ee_poses.append(util.pose_stamped2list(pre_ee_end_pose2))
            ee_poses.append(util.pose_stamped2list(ee_end_pose))
        else:
            ee_poses.append(util.pose_stamped2list(final_pose))
        return ee_poses

    #########################################################################################################
    # Motion Planning 
    def execute(self, ee_poses):
        jnt_poses = [self.cascade_ik(pose) for pose in ee_poses]

        i = 0

        start_pose = None
        joint_traj = []
        for jnt_pose in jnt_poses:
            input('Press enter to show next plan')
            # if not execute:
            if start_pose is None:
                resulting_traj = self.planning.plan_joint_target(joint_position_desired=jnt_pose, 
                            from_current=True, 
                            start_position=None, 
                            execute=False)
            else:
                resulting_traj = self.planning.plan_joint_target(joint_position_desired=jnt_pose, 
                                                from_current=False, 
                                                start_position=start_pose, 
                                                execute=False)
            start_pose = jnt_pose
            if resulting_traj is None:
                break
            else:
                joint_traj += resulting_traj
        else:
            while True:
                i = input(
                    '''What should we do
                        [s]: Run in sim
                        [e]: Execute on robot
                        [b]: Exit
                    ''')
                
                if i == 's':
                    self.planning.execute_pb_loop(joint_traj)
                    continue
                elif i == 'e':
                    confirm = input('Should we execute (y/n)???')
                    if confirm == 'y':
                        self.pre_execution()
                        self.planning.execute_loop(joint_traj)
                        self.post_execution()

                    continue
                elif i == 'b':
                    break
                else:
                    print('Unknown command')
                    continue

    def cascade_ik(self, ee_pose):
        jnt_pos = None
        if jnt_pos is None:
            jnt_pos = self.ik_helper.get_feasible_ik(ee_pose, verbose=False)
            if jnt_pos is None:
                jnt_pos = self.ik_helper.get_ik(ee_pose)
        if jnt_pos is None:
            log_warn('Failed to find IK')
        return jnt_pos

    def teleport(self, obj_pcd, relative_pose):
        transform = util.matrix_from_pose(util.list2pose_stamped(relative_pose[0]))

        final_pcd = util.transform_pcd(obj_pcd, transform)
        util.meshcat_pcd_show(self.mc_vis, final_pcd, color=(255, 0, 255), name=f'scene/teleported_obj')

    def pre_execution(self):
        if self.state == 0:
            if not self.gripper_is_open:
                self.planning.gripper_open()
                self.gripper_is_open = True

    def post_execution(self):
        if self.state == 0:
            time.sleep(0.8)
            if self.gripper_is_open:
                self.planning.gripper_grasp()
                self.gripper_is_open = False
            

        else:
            release = input('Press o to open end effector or Enter to continue')
            if release == 'o':
                self.planning.gripper_open()
                self.gripper_is_open = True
                self.state = -1
                time.sleep(1.0)
            else:
                self.state = 0
                if 1 in self.ranked_objs:
                    del self.ranked_objs[1]

