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

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

sys.path.append(os.environ['SOURCE_DIR'])

import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from rndf_robot.utils import util, trimesh_util
from rndf_robot.utils import path_util

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

from rndf_robot.utils.rndf_utils import infer_relation_intersection, create_target_descriptors
from system_utils.segmentation import detect_bbs, apply_pcd_mask, apply_bb_mask, extend_pcds, filter_pcds
from system_utils.sam_seg import get_masks
from system_utils.language import query_correspondance, chunk_query, create_keyword_dic
from system_utils.demos import all_demos, get_concept_demos, create_target_desc_subdir, get_model_paths
import system_utils.objects as objects

from rndf_robot.utils.visualize import PandaHand, Robotiq2F140Hand
from rndf_robot.utils.record_demo_utils import manually_segment_pcd

from rndf_robot.utils.franka_ik import FrankaIK #, PbPlUtils
from rndf_robot.robot.simple_multicam import MultiRealsenseLocal

from rndf_robot.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rndf_robot.utils.real.traj_util import PolymetisTrajectoryUtil
from rndf_robot.utils.real.plan_exec_util import PlanningHelper
from rndf_robot.utils.real.perception_util import enable_devices
from rndf_robot.utils.real.polymetis_util import PolymetisHelper

from rndf_robot.robot.simple_multicam import MultiRealsenseLocal
from rndf_robot.system.system_utils.realsense import RealsenseLocal, enable_devices, pipeline_stop

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

        self.ranked_objs = {}
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        if args.debug:
            set_log_level('debug')
        else:
            set_log_level('info')
        
        self.mc_vis = meshcat.Visualizer(zmq_url=f'tcp://127.0.0.1:{self.args.port_vis}')
        self.mc_vis['scene'].delete()
        log_debug('Done with init')

    def reset(self, new_scene=False):
        if new_scene:
            time.sleep(1.5)

            self.last_ee = None
            self.obj_info = {}
            self.class_to_id = {}
            self.mc_vis['scene'].delete()

        self.mc_vis['optimizer'].delete()
        self.ranked_objs = {}
        self.reset_robot()
        self.state = -1
        time.sleep(1.5)

    def step(self, obj_grasped=None):
        while True:
            x = input('Press 1 to continue or 2 to use a new object\n')
            if x == '1':
                if not obj_grasped:
                    self.reset(False)
                return False
            elif x == '2':
                self.reset(True)
                return True

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
        planning.gripper_open()

        # if self.args.gripper_type == '2f140':
        #     grasp_pose_viz = Robotiq2F140Hand(grasp_frame=False)
        #     place_pose_viz = Robotiq2F140Hand(grasp_frame=False)
        # else:
        #     grasp_pose_viz = PandaHand(grasp_frame=True)
        #     place_pose_viz = PandaHand(grasp_frame=True)
        # grasp_pose_viz.reset_pose()
        # place_pose_viz.reset_pose()
        # grasp_pose_viz.meshcat_show(self.mc_vis, name_prefix='grasp_pose')
        # place_pose_viz.meshcat_show(self.mc_vis, name_prefix='place_pose')

        self.planning = planning
        self.panda = panda

    def setup_cams(self):
        rs_cfg = get_default_multi_realsense_cfg()
        serials = rs_cfg.SERIAL_NUMBERS

        prefix = rs_cfg.CAMERA_NAME_PREFIX
        camera_names = [f'{prefix}{i}' for i in range(len(serials))]
        cam_list = [camera_names[int(idx)] for idx in self.args.cam_index]       

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
        self.planning.execute_loop(current_panda_plan)            
        self.planning.gripper_open()

    # def test_execution(self, current_panda_plan):
    #     if len(current_panda_plan) == 0:
    #         print('\n\nCurrent panda plan is empty!')

    #     self.planning.execute_pb_loop(current_panda_plan)
    #     confirm = input('Confirm execution (y/n)' )
    #     if confirm == 'y':
    #         self.planning.execute_loop(current_panda_plan)            

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
    # Language
    
    def prompt_user(self):
        '''
        Prompts the user to input a command and finds the demos that relate to the concept.
        Moves to the next state depending on the input

        return: the concept most similar to their query and their input text
        '''
        log_debug('All demo labels: %s' %all_demos.keys())
        while True:
            # query = self.ask_query()
            query = "grasp mug_handle", "grab the mug by the handle"
            if not query: return
            corresponding_concept, query_text = query
            demos = get_concept_demos(corresponding_concept)
            if not len(demos):
                log_warn('No demos correspond to the query! Try a different prompt')
                continue
            else:
                log_debug('Number of Demos %s' % len(demos)) 
                break
        
        self.skill_demos = demos
        if corresponding_concept.startswith('grasp'):
            self.state = 0
        elif corresponding_concept.startswith('place') and self.state == 0:
            self.state = 1
        elif corresponding_concept.startswith('place') and self.state == -1:
            self.state = 2
        log_debug('Current State is %s' %self.state)
        return corresponding_concept, query_text

    def ask_query(self):
        '''
        Prompts the user to input a command and identifies concept

        return: the concept most similar to their query and their input text
        '''
        concepts = list(all_demos.keys())
        while True:
            query_text = input('Please enter a query or \'reset\' to reset the scene\n')
            if not query_text: continue
            if query_text.lower() == "reset": return
            ranked_concepts = query_correspondance(concepts, query_text)
            corresponding_concept = None
            for concept in ranked_concepts:
                print('Corresponding concept:', concept)
                correct = input('Corrent concept? (y/n)\n')
                if correct == 'n':
                    continue
                elif correct == 'y':
                    corresponding_concept = concept
                    break
            
            if corresponding_concept:
                break
        return corresponding_concept, query_text


    def identify_classes_from_query(self, query, corresponding_concept):
        '''
        Takes a query and skill concept and identifies the relevant object classes to execute the skill.
        
        @query: english input for the skill
        @corresponding_concept: concept in the form 'grasp/place {concept}'

        returns: the key for the set of demos relating to the concept (just the {concept} part)
        '''
        all_obj_classes = set(objects.mesh_data_dirs.keys())
        concept_key = corresponding_concept[corresponding_concept.find(' ')+1:]
        concept_language = frozenset(concept_key.lower().replace('_', ' ').split(' '))
        relevant_classes = concept_language.intersection(all_obj_classes)
        chunked_query = chunk_query(query)
        keywords = create_keyword_dic(relevant_classes, chunked_query)
        self.assign_classes(keywords)
        return concept_key, keywords

    def assign_classes(self, keywords):
        '''
        @test_objs: list of relevant object classes to determine rank for
        @keywords: list of associated obj class, noun phrase, and verb flag as pairs of tuples in form (class, NP, True/False)
        '''

        # what's the best way to determine which object should be manipulated and which is stationary automatically?
        if self.state == 0:
            # only one noun phrase mentioned, probably the object to be moved
            keyword = keywords.pop()
            self.ranked_objs[0] = {}
            self.ranked_objs[0]['description'] = keyword[1]
            self.ranked_objs[0]['potential_class'] = keyword[0]
        else:
            if self.state == 1:
                if len(keywords) >= 1:
                    for pair in keywords:
                        # check if the obj class mentioned in noun phrase same as object to be moved
                        if pair[0] == self.ranked_objs[0]['potential_class']:
                            keywords.remove(pair)   
                if len(keywords) == 0:
                    return
                else:
                    if len(keywords) > 1:
                        log_warn('There is more than one noun mentioned in the query, just choosing one')

                    priority_rank = 0 if 0 not in self.ranked_objs else 1
                    keyword = keywords.pop()
                    self.ranked_objs[priority_rank] = {}
                    self.ranked_objs[priority_rank]['description'] = keyword[1]
                    self.ranked_objs[priority_rank]['potential_class'] = keyword[0]
            else:
                if self.state != 2 and len(keywords) > 1:
                    log_warn('There is more than one noun mentioned in the query and unsure what to do')
                    return
                if len(keywords) == 2:
                    pair_1, pair_2 = keywords
                    if pair_1[2]:
                        classes_to_assign = [pair_1, pair_2]
                    elif pair_2[2]:
                        classes_to_assign = [pair_2, pair_1]
                    else:
                        log_warn("Unsure which object to act on")
                else:
                    classes_to_assign = keywords
                for i in range(len(classes_to_assign)):
                    self.ranked_objs[i] = {}
                    self.ranked_objs[i]['description'] = classes_to_assign[i][1]
                    self.ranked_objs[i]['potential_class'] = classes_to_assign[i][0]

        target = 'Target - class:%s, descr: %s'% (self.ranked_objs[0]['potential_class'], self.ranked_objs[0]['description'])
        log_debug(target)
        if 1 in self.ranked_objs:
            relation = 'Relational - class:%s, descr: %s'% (self.ranked_objs[1]['potential_class'], self.ranked_objs[1]['description'])
            log_debug(relation)

    #################################################################################################
    # Segment the scene

    def assign_pcds(self, labels_to_pcds):
        assigned_centroids = []
        for obj_rank, obj in self.ranked_objs.items():
            if 'pcd' in obj:
                assigned_centroids.append(np.average(obj['pcd'], axis=0))
        assigned_centroids = np.array(assigned_centroids)
        # pick the pcd with the most pts
        for label in labels_to_pcds:
            labels_to_pcds[label].sort(key=lambda x: x[0])

        for obj_rank in self.ranked_objs:
            if 'pcd' in self.ranked_objs[obj_rank]: continue
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
                if assigned_centroids.any():
                    diff = assigned_centroids-np.average(pcd, axis=0)
                    centroid_dists = np.sqrt(np.sum(diff**2,axis=-1))
                    if min(centroid_dists) <= 0.05:
                        continue
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
        util.meshcat_pcd_show(self.mc_vis, pcd_world, name=f'scene/pcd_world_cam_{idx}')

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
                    
        label_to_pcds = {}
        label_to_scores = {}
        centroid_thresh = 0.1
        detect_thresh = 0.2
        for i, cam in enumerate(self.cams.cams): 
            # get image and raw point cloud
            rgb, depth = self.cam_interface.get_rgb_and_depth_image(self.pipelines[i])
            pts_2d, depth_valid = self.get_real_pcd_cam(i, rgb, depth)
            all_obj_bbs, all_obj_bb_scores = detect_bbs(rgb, 
                                                        captions, 
                                                        max_count=1, 
                                                        score_threshold=detect_thresh)
            log_debug(f'Detected the following captions {all_obj_bb_scores.keys()}')

            if not all_obj_bbs:
                continue
            all_obj_masks = get_masks(rgb, all_obj_bbs)
            # all_obj_masks = all_obj_bbs
            for obj_label, obj_masks in all_obj_masks.items():
                log_debug(f'Region count for {obj_label}: {len(obj_masks)}')
                obj_pcds, obj_scores = apply_pcd_mask(pts_2d, depth_valid, obj_masks, all_obj_bb_scores[obj_label])
                log_debug(f'{obj_label} after filtering is now {len(obj_pcds)}')
                cam_pcds, cam_scores = filter_pcds(obj_pcds, obj_scores)
                for j in range(len(cam_pcds)):
                    util.meshcat_pcd_show(self.mc_vis, cam_pcds[j], color=(0, 255, 0), name=f'scene/cam_{i}_{obj_label}_region_{j}')

                cam_pcds, cam_scores = obj_pcds, obj_scores
                if not cam_pcds:
                    continue
                if obj_label not in label_to_pcds:
                    label_to_pcds[obj_label], label_to_scores[obj_label] = cam_pcds, cam_scores
                else:
                    new_pcds, new_lables = extend_pcds(cam_pcds, 
                                                       label_to_pcds[obj_label], 
                                                       cam_scores, 
                                                       label_to_scores[obj_label], 
                                                       threshold=centroid_thresh)
                    label_to_pcds[obj_label], label_to_scores[obj_label] = new_pcds, new_lables
                log_debug(f'{obj_label} size is now {len(label_to_pcds[obj_label])}')
            
        pcds_output = {}

        for obj_label in captions:
            if obj_label not in label_to_pcds:
                log_warn('WARNING: COULD NOT FIND RELEVANT OBJ')
                break
            obj_pcd_sets = label_to_pcds[obj_label]
            for i, target_obj_pcd_obs in enumerate(obj_pcd_sets):
                # target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
                # inliers = np.where(np.linalg.norm(target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
                # target_obj_pcd_obs = target_obj_pcd_obs[inliers]
                # if not target_obj_pcd_obs.any(): continue
                score = np.average(label_to_scores[obj_label][i])
                if obj_label not in pcds_output:
                    pcds_output[obj_label] = []
                pcds_output[obj_label].append((score, target_obj_pcd_obs))
                # if self.args.debug:
                #     trimesh_util.trimesh_show([target_obj_pcd_obs])
        return pcds_output        

    #################################################################################################
    # Process demos

    def load_demos(self, concept):
        n = self.args.n_demos
        if 1 in self.ranked_objs:
            self.get_relational_descriptors(concept, n)
        else:
            self.get_single_descriptors(n)

    def get_single_descriptors(self, n=None):
        self.ranked_objs[0]['demo_info'] = []
        self.ranked_objs[0]['demo_ids'] = []
        # don't re-init from pick to place? might not matter for relational descriptors because it saves it
        if self.state == 0:
            self.ranked_objs[0]['demo_start_poses'] = {}

        for fname in self.skill_demos[:min(n, len(self.skill_demos))]:
            demo = np.load(fname, allow_pickle=True)
            if 'shapenet_id' in demo:
                obj_id = demo['shapenet_id'].item()

            if self.state == 0:
                target_info, initial_pose = process_demo_data(demo, initial_pose = None, table_obj=None)
                self.ranked_objs[0]['demo_start_poses'][obj_id]  = initial_pose
            else:
                grasp_pose = util.pose_stamped2list(util.unit_pose())
                grasp_pose = None
                target_info, _ = process_demo_data(demo, grasp_pose, table_obj=self.table_obj)
                if not target_info: continue

            if target_info is not None:
                self.ranked_objs[0]['demo_info'].append(target_info)
                self.ranked_objs[0]['demo_ids'].append(obj_id)
            else:
                log_debug('Could not load demo')

        # if self.relevant_objs['grasped']:
        #     scene = trimesh_util.trimesh_show([target_info['demo_obj_pts'],target_info['demo_query_pts_real_shape']], show=True)

        self.ranked_objs[0]['query_pts'] = process_xq_data(demo, table_obj=self.table_obj)
        self.ranked_objs[0]['query_pts_rs'] = process_xq_rs_data(demo, table_obj=self.table_obj)
        self.ranked_objs[0]['demo_ids'] = frozenset(self.ranked_objs[0]['demo_ids'])
        log_debug('Finished loading single descriptors')
        return True

        # self.initial_poses = initial_poses

    def get_relational_descriptors(self, concept, n=None):
        for obj_rank in self.ranked_objs.keys():
            self.ranked_objs[obj_rank]['demo_start_pcds'] = []
            self.ranked_objs[obj_rank]['demo_final_pcds'] = []
            self.ranked_objs[obj_rank]['demo_ids'] = []
            self.ranked_objs[obj_rank]['demo_start_poses'] = []

        for obj_rank in self.ranked_objs.keys():
            if obj_rank == 0:
                demo_key = 'child'
            elif obj_rank == 1:
                demo_key == 'parent'
            else:
                log_warn('This obj rank should not exist')
                continue
            for demo_path in self.skill_demos[:min(n, len(self.skill_demos))]:
                demo = np.load(demo_path, allow_pickle=True)
                s_pcd = demo['multi_obj_start_pcd'].item()[demo_key]
                if 'multi_obj_final_pcd' in demo:
                    f_pcd = demo['multi_obj_final_pcd'].item()[demo_key]
                else:
                    f_pcd = s_pcd
                self.ranked_objs[obj_rank]['demo_start_pcds'].append(s_pcd)
                self.ranked_objs[obj_rank]['demo_final_pcds'].append(f_pcd)
                
        demo_path = osp.join(path_util.get_rndf_data(), 'release_demos', concept)
        relational_model_path, target_model_path = self.ranked_objs[1]['model_path'], self.ranked_objs[0]['model_path']
        target_desc_subdir = create_target_desc_subdir(demo_path, relational_model_path, target_model_path, create=False)
        target_desc_fname = osp.join(demo_path, target_desc_subdir, 'target_descriptors.npz')
        if osp.exists(target_desc_fname):
            log_debug(f'Loading target descriptors from file:\n{target_desc_fname}')
            target_descriptors_data = np.load(target_desc_fname)
        else:
            log_warn('Descriptor file not found')
            return

        target_descriptors_data = np.load(target_desc_fname)
        relational_overall_target_desc = target_descriptors_data['parent_overall_target_desc']
        target_overall_target_desc = target_descriptors_data['child_overall_target_desc']
        self.ranked_objs[1]['target_desc'] = torch.from_numpy(relational_overall_target_desc).float().cuda()
        self.ranked_objs[0]['target_desc'] = torch.from_numpy(target_overall_target_desc).float().cuda()
        self.ranked_objs[1]['query_pts'] = target_descriptors_data['parent_query_points']
        #slow - change deepcopy to shallow copy? 
        self.ranked_objs[0]['query_pts'] = copy.deepcopy(target_descriptors_data['parent_query_points'])
        log_debug('Finished loading relational descriptors')

    def prepare_new_descriptors(self):
        log_info(f'\n\n\nCreating target descriptors for this parent model + child model, and these demos\nSaving to {target_desc_fname}\n\n\n')
        # MAKE A NEW DIR FOR TARGET DESCRIPTORS
        demo_path = osp.join(path_util.get_rndf_data(), 'targets', self.concept)

        # TODO: get default model_path for the obj_class?
        # parent_model_path, child_model_path = self.args.parent_model_path, self.args.child_model_path
        relational_model_path = self.ranked_objs[1]['model_path']
        target_model_path = self.ranked_objs[0]['model_path']

        target_desc_subdir = create_target_desc_subdir(demo_path, relational_model_path, target_model_path, create=False)
        target_desc_fname = osp.join(demo_path, target_desc_subdir, 'target_descriptors.npz') 
        
        assert not osp.exists(target_desc_fname) or self.args.new_descriptor, 'Descriptor exists and/or did not specify creation of new descriptors'
        create_target_desc_subdir(demo_path, relational_model_path, target_model_path, create=True)
        self.prepare_new_descriptors(target_desc_fname)
       
        n_demos = 'all' if self.args.n_demos < 1 else self.args.n_demos
        
        if self.ranked_objs[0]['potential_class'] == 'container' and self.ranked_objs[1]['potential_class'] == 'bottle':
            use_keypoint_offset = True
            keypoint_offset_params = {'offset': 0.025, 'type': 'bottom'}
        else:
            use_keypoint_offset = False 
            keypoint_offset_params = None

        self.set_initial_models()
        self.load_models()
        # bare minimum settings
        create_target_descriptors(
            self.ranked_objs[1]['model'], self.ranked_objs[0]['model'], self.ranked_objs, target_desc_fname, 
            self.cfg, query_scale=self.args.query_scale, scale_pcds=False, 
            target_rounds=self.args.target_rounds, pc_reference=self.args.pc_reference,
            skip_alignment=self.args.skip_alignment, n_demos=n_demos, manual_target_idx=self.args.target_idx, 
            add_noise=self.args.add_noise, interaction_pt_noise_std=self.args.noise_idx,
            use_keypoint_offset=use_keypoint_offset, keypoint_offset_params=keypoint_offset_params, visualize=True, mc_vis=self.mc_vis)
       

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

            demo_path = osp.join(path_util.get_rndf_data(), 'release_demos', concept)
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

        ee_poses.append(pre_pre_ee_pose)
        ee_poses.append(pre_ee_pose)
        return ee_poses
    
    def find_place_transform(self, target_rank, relational_rank=None, ee=False):
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

        if ee:
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
            ee_poses.append(util.pose_stamped2list(pre_ee_end_pose1))
            ee_poses.append(util.pose_stamped2list(pre_ee_end_pose2))
            ee_poses.append(util.pose_stamped2list(ee_end_pose))
        else:
            ee_poses.append(util.pose_stamped2list(final_pose))
        return ee_poses

    #########################################################################################################
    # Motion Planning 
    def execute(self, ee_poses, execute=False):
        jnt_poses = [self.cascade_ik(pose) for pose in ee_poses]
        self.planning.plan_joint_target(joint_position_desired=jnt_poses,
                      execute=execute)
        if execute:
            self.post_execution()

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

    def post_execution(self):
        if self.state == 0:
            time.sleep(0.8)
            self.planning.gripper_grasp()

        else:
            release = input('Press o to open end effector or Enter to continue')
            if release == 'o':
                self.planning.gripper_open()
                self.state = -1

                time.sleep(1.0)
            else:
                self.state = 0
                if 1 in self.ranked_objs:
                    del self.ranked_objs[1]

