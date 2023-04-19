import os, os.path as osp
import random
import copy
import numpy as np
import torch
import time
import sys
sys.path.append(os.environ['SOURCE_DIR'])

import pybullet as p
import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from rndf_robot.utils import util, trimesh_util
from rndf_robot.utils import path_util

from rndf_robot.utils.franka_ik_ndf import FrankaIK
from rndf_robot.opt.optimizer import OccNetOptimizer
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rndf_robot.utils.pipeline_util import (
    safeCollisionFilterPair,
    safeRemoveConstraint,
    soft_grasp_close,
    object_is_still_grasped,
    process_xq_data,
    process_xq_rs_data,
    process_demo_data,
    post_process_grasp_point,
    get_ee_offset,
    constraint_obj_world,
)
from rndf_robot.utils.rndf_utils import infer_relation_intersection, create_target_descriptors
from system_utils.segmentation import detect_bbs, get_label_pcds, extend_pcds, get_region
from system_utils.language import query_correspondance, chunk_query, create_keyword_dic
from system_utils.demos import all_demos, get_concept_demos, create_target_desc_subdir, get_model_paths
import system_utils.objects as objects

from airobot import Robot, log_info, set_log_level, log_warn, log_debug
from airobot.utils import common

import pyrealsense2 as rs
from rndf_robot.robot.simple_multicam import MultiRealsenseLocal
from rndf_robot.system.system_utils.realsense import RealsenseLocal, enable_devices, pipeline_stop

from IPython import embed;

class Pipeline():

    def __init__(self, args):
        self.args = args

        self.random_pos = False
        self.ee_pose = None

        self.table_model = None
        self.state = -1 #pick = 0, place = 1, teleport = 2
        self.table_obj = -1 #shelf = 0, rack = 1

        # self.cfg = get_eval_cfg_defaults()
        self.cfg = self.get_env_cfgs()
        self.meshes_dic = objects.load_meshes_dict(self.cfg)

        self.ranked_objs = {}
        self.obj_info = {}
        self.class_to_id = {}

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        if args.debug:
            set_log_level('debug')
        else:
            set_log_level('info')

    def reset(self, new_scene=False):
        if new_scene:
            for obj_id in self.obj_info:
                self.robot.pb_client.remove_body(obj_id)
            time.sleep(1.5)

            self.last_ee = None
            self.obj_info = {}
            self.class_to_id = {}

        self.ranked_objs = {}
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])
        self.robot.arm.eetool.open()
        self.state = -1
        time.sleep(1.5)

    def register_vizServer(self, vizServer):
        self.viz = vizServer

    def step(self, last_ee=None):
        while True:
            x = input('Press 1 to continue or 2 to use a new object\n')

            if x == '1':
                # self.ee_pose = scene['final_ee_pos']
                # robot.arm.get_ee_pose()
                # self.scene_obj = scene['obj_pcd'], scene['obj_pose'], scene['obj_id']
                if last_ee:
                    self.last_ee = last_ee
                else:
                    self.reset(False)
                return False
            elif x == '2':
                self.reset(True)
                return True

    def setup_client(self):
        #TODO: Setup real robot - hopefully interfacing isn't too different than Pybullet robot
        self.robot = Robot('franka', pb_cfg={'gui': self.args.pybullet_viz}, arm_cfg={'self_collision': False, 'seed': self.args.seed})
        self.ik_helper = FrankaIK(gui=False)
        
        self.finger_joint_id = 9
        self.left_pad_id = 9
        self.right_pad_id = 10

        p.changeDynamics(self.robot.arm.robot_id, self.left_pad_id, lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, self.right_pad_id, lateralFriction=1.0)

        # reset
        self.robot.arm.reset(force_reset=True)
        self.setup_cams()

    def setup_cams(self):
        # self.robot.cam.setup_camera(
        #     focus_pt=[0.4, 0.0, self.cfg.TABLE_Z],
        #     dist=0.9,
        #     yaw=45,
        #     pitch=-25,
        #     roll=0)
        serials = [
            '143122065292',
            '840412060551',
            '215122255998',
            '843112073228'
        ]

        prefix = 'cam_'
        camera_names = [f'{prefix}{i}' for i in range(len(serials))]
        cam_index = [int(idx) for idx in self.args.cam_index]
        cam_list = [camera_names[int(idx)] for idx in cam_index]        

        calib_dir = osp.join(path_util.get_rndf_src(), 'robot/camera_calibration_files')
        calib_filenames = [osp.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in cam_index]

        self.cams = MultiRealsenseLocal(cam_names=cam_list, calib_filenames=calib_filenames)
        
        ctx = rs.context() # Create librealsense context for managing devices

        # Define some constants
        resolution_width = 640 # pixels
        resolution_height = 480 # pixels
        frame_rate = 30  # fps

        # pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)
        self.pipelines = enable_devices(np.array(serials)[cam_index], ctx, resolution_width, resolution_height, frame_rate)
        self.cam_interface = RealsenseLocal()

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

            pcd_pts.append(pcd_world)
            pcd_dict_list.append(pcd_dict)

        pcd_full = np.concatenate(pcd_pts, axis=0)
        return pcd_full

    def get_real_pcd_cam(self, idx, rgb, depth):
        embed()
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
        # pcd_dict = {
        #     'world': pcd_world,
        #     'cam': pcd_cam_img,
        #     'cam_img': pcd_cam,
        #     'world_img': pcd_world_img,
        #     'cam_pose_mat': cam_pose_world
        # }

        return pcd_world
    
    def setup_table(self):            
        table_fname = osp.join(path_util.get_rndf_descriptions(), 'hanging/table')
        table_urdf_file = 'table_manual.urdf'
        if self.table_model:    
            if self.cfg.PLACEMENT_SURFACE == 'shelf':
                self.table_obj = 0
                table_urdf_file = 'table_shelf.urdf'
                log_debug('Shelf loaded')
            else:
                log_debug('Rack loaded')
                self.table_obj = 1
                table_urdf_file = 'table_rack.urdf'
        table_fname = osp.join(table_fname, table_urdf_file)

        self.table_id = self.robot.pb_client.load_urdf(table_fname,
                                self.cfg.TABLE_POS, 
                                self.cfg.TABLE_ORI,
                                scaling=1.0)
        self.viz.recorder.register_object(self.table_id, table_fname)
        self.viz.pause_mc_thread(False)
        safeCollisionFilterPair(self.robot.arm.robot_id, self.table_id, -1, -1, enableCollision=True)

        time.sleep(3.0)
        
    def reset_robot(self):
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])
        self.robot.arm.eetool.open()

    def setup_random_scene(self, config_dict):
        '''
        @config_dict: Key are ['objects': {'class': #}]
        '''
        for obj_class, colors in config_dict['objects'].items():
            for color, n in colors.items():
                for _ in range(n):
                    obj_id, obj_pose_world, o_cid = self.add_obj(obj_class,color=color)
                    obj = {
                        'class': obj_class,
                        'pose': obj_pose_world,
                        'o_cid': o_cid,
                        'rank': -1
                    }
                    self.obj_info[obj_id] = obj

    #TODO: Remove simulator object calls
    def add_obj(self, obj_class, scale_default=None, color=None):
        obj_file = objects.choose_obj(self.meshes_dic, obj_class)
        x_low, x_high = self.cfg.OBJ_SAMPLE_X_HIGH_LOW
        y_low, y_high = self.cfg.OBJ_SAMPLE_Y_HIGH_LOW

        self.placement_link_id = 0 

        if not scale_default: scale_default = objects.scale_default[obj_class] 
        mesh_scale=[scale_default] * 3

        pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, self.cfg.TABLE_Z]
        log_debug('original: %s' %pos)
        for obj_id, obj in self.obj_info.items():
            existing_pos = util.pose_stamped2list(obj['pose'])

            if abs(pos[1]-existing_pos[1]) < self.cfg.OBJ_SAMPLE_PLACE_Y_DIST:
                if abs(pos[1]+self.cfg.OBJ_SAMPLE_PLACE_Y_DIST-existing_pos[1]) > abs(pos[1]-self.cfg.OBJ_SAMPLE_PLACE_Y_DIST-existing_pos[1]):
                    pos[1] += self.cfg.OBJ_SAMPLE_PLACE_Y_DIST 
                else:
                    pos[1] -= self.cfg.OBJ_SAMPLE_PLACE_Y_DIST                         
                log_debug('obj too close, moved Y')
                continue

            if abs(pos[0]-existing_pos[0]) < self.cfg.OBJ_SAMPLE_PLACE_X_DIST:
                if abs(pos[0]+self.cfg.OBJ_SAMPLE_PLACE_X_DIST-existing_pos[0]) > abs(pos[0]-self.cfg.OBJ_SAMPLE_PLACE_X_DIST-existing_pos[0]):
                    pos[0] += self.cfg.OBJ_SAMPLE_PLACE_X_DIST 
                else:
                    pos[0] -= self.cfg.OBJ_SAMPLE_PLACE_X_DIST                         
                log_debug('obj too close, moved x')
            
        log_debug('final: %s' %pos)

        if self.random_pos:
            if self.test_obj in ['bowl', 'bottle']:
                rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            else:
                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()
        else:
            ori = objects.upright_orientation_dict[obj_class]

        pose = util.list2pose_stamped(pos + ori)
        rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
        pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
        pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
        log_debug('OBJECT POSE: %s'% util.pose_stamped2list(pose))

        obj_file_dec = obj_file.split('.obj')[0] + '_dec.obj'
        log_debug(f'{obj_file_dec} exists: {osp.exists(obj_file_dec)}')

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
            visualfile=obj_file_dec,
            collifile=obj_file_dec,
            base_pos=pos,
            base_ori=ori,
            rgba=color)
        self.ik_helper.add_collision_bodies({obj_id: obj_id})

        # register the object with the meshcat visualizer
        self.viz.recorder.register_object(obj_id, obj_file_dec, scaling=mesh_scale)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)

        p.changeDynamics(obj_id, -1, lateralFriction=0.5, linearDamping=5, angularDamping=5)

        # depending on the object/pose type, constrain the object to its world frame pose
        o_cid = None
        # or (load_pose_type == 'any_pose' and pc == 'child')
        if (obj_class in ['syn_rack_easy', 'syn_rack_hard', 'syn_rack_med', 'rack']):
            o_cid = constraint_obj_world(obj_id, pos, ori)
            self.robot.pb_client.set_step_sim(False)

        time.sleep(1.5)

        obj_pose_world_list = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world_list[0]) + list(obj_pose_world_list[1]))
        # with self.viz.recorder.meshcat_scene_lock:
        #     pose = util.matrix_from_pose(obj_pose_world)
        #     util.meshcat_obj_show(self.viz.mc_vis, obj_file_dec, pose, mesh_scale, name='scene/%s'%obj_class)
        # self.ik_helper.register_object(obj_file_dec, pos, ori, mesh_scale, name=obj_id)
        return obj_id, obj_pose_world, o_cid

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
            query = self.ask_query()
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

    #################################################################################################
    # Misc?

    def associate_ranked_objs(self, relevent_objs):
        '''
        @relevent_objs: {keywords: obj_class}
        '''
        for obj_rank, obj_class in relevent_objs.items():
            if obj_rank in self.ranked_objs: continue
            obj_class = self.obj_info[obj_id]['class']
            # class might not be enough - keywords are more helpful
            if obj_class in self.class_to_id:
                obj_id = next(iter(self.obj_id[obj_class]))
                self.ranked_objs[obj_rank] = {'obj_id': obj_id}
                self.obj_info[obj_class]['rank'] = obj_rank
            else:
                # TODO: Generate the objects? Ask for feedback?
                log_warn('NO RELEVANT OBJECT IN THE SCENE. EXITING')
                return
            
        for keyword, obj_class in relevent_objs.items():
            # for obj_rank, obj in self.ranked_objs.items():
            #     if 'obj_id' in obj
            obj_class = self.obj_info[obj_id]['class']
            # class might not be enough - keywords are more helpful
            if obj_class in self.class_to_id:
                obj_id = next(iter(self.obj_id[obj_class]))
                self.ranked_objs[obj_rank] = {'obj_id': obj_id}
                self.obj_info[obj_class]['rank'] = obj_rank
            else:
                # TODO: Generate the objects? Ask for feedback?
                log_warn('NO RELEVANT OBJECT IN THE SCENE. EXITING')
                return


    #################################################################################################
    # Segment the scene

    def identify_classes_from_query(self, query, corresponding_concept):
        '''
        Takes a query and skill concept and identifies the relevant object classes to execute the skill.
        
        @query: english input for the skill
        @corresponding_concept: concept in the form 'grasp/place {concept}'

        returns: the key for the set of demos relating to the concept (just the {concept} part)
        '''
        all_obj_classes = set(objects.mesh_data_dirs.keys())
        concept_key = corresponding_concept.split(' ')[-1]
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
            if len(keywords) > 1 and self.state == 1:
                for pair in keywords:
                    # check if the obj class mentioned in noun phrase same as object to be moved
                    if pair[0] == self.ranked_objs[0]['potential_class']:
                        keywords.remove(pair)                
            if len(keywords) == 1:
                priority_rank = 0 if 0 not in self.ranked_objs else 1
                keyword = keywords.pop()
                self.ranked_objs[priority_rank] = {}
                self.ranked_objs[priority_rank]['description'] = keyword[1]
                self.ranked_objs[priority_rank]['potential_class'] = keyword[0]
            else:
                if self.state != 2 and len(keywords) > 2:
                    log_warn('There is more than one noun mentioned in the query and unsure what to do')
                    return
                pair_1, pair_2 = keywords

                if pair_1[2]:
                    pairs = [pair_1, pair_2]
                elif pair_2[2]:
                    pairs = [pair_2, pair_1]
                else:
                    log_warn("Unsure which object to act on")

                for i in range(2):
                    self.ranked_objs[i] = {}
                    self.ranked_objs[1]['description'] = pairs[i][1]
                    self.ranked_objs[0]['potential_class'] = pairs[i][0]

        target = 'Target - class:%s, descr: %s'% (self.ranked_objs[0]['potential_class'], self.ranked_objs[0]['description'])
        log_debug(target)
        if 1 in self.ranked_objs:
            relation = 'Relational - class:%s, descr: %s'% (self.ranked_objs[1]['potential_class'], self.ranked_objs[1]['description'])
            log_debug(relation)

    def assign_pcds(self, labels_to_pcds, obj_ranks=None):
        embed()
        assigned_centroids = []
        for obj_rank, obj in self.ranked_objs.items():
            if 'pcd' in obj:
                assigned_centroids.append(np.average(obj['pcd'], axis=0))
        assigned_centroids = np.array(assigned_centroids)
        # pick the pcd with the most pts
        for label in labels_to_pcds:
            labels_to_pcds[label].sort(key=lambda x: x[0])

        for obj_rank in obj_ranks:
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
            for score, pcd in labels_to_pcds[pcd_key]:
                if assigned_centroids.any():
                    diff = assigned_centroids-np.average(pcd, axis=0)
                    centroid_dists = np.sqrt(np.sum(diff**2,axis=-1))
                    if min(centroid_dists) <= 0.05:
                        continue
                new_pcds.append(pcd)
            self.ranked_objs[obj_rank]['pcd'] = new_pcds.pop(-1)
            if self.args.show_pcds:
                log_debug(f'Best score was {score}')
                trimesh_util.trimesh_show([self.ranked_objs[obj_rank]['pcd']])

        with self.viz.recorder.meshcat_scene_lock:
            for _, obj in self.ranked_objs.items():
                label = 'scene/%s_pcd' % obj['description']
                color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                util.meshcat_pcd_show(self.viz.mc_vis, obj['pcd'], color=color, name=label)

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
        threshold = 0.08

        for i, cam in enumerate(self.cams.cams): 
            # get image and raw point cloud
            # rgb, depth, _ = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            rgb, depth = self.cam_interface.get_rgb_and_depth_image(self.pipelines[i])
            # pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)
            pts_raw = self.get_real_pcd_cam(i, rgb, depth)
            height, width, _ = rgb.shape
            pts_2d = pts_raw.reshape((height, width, 3))
            obj_bbs, cam_scores = detect_bbs(rgb, captions)
            log_debug(f'Detected the following captions {obj_bbs.keys()}')
            # embed()
            for obj_label, regions in obj_bbs.items():
                log_debug(f'Region count for {obj_label}: {len(regions)}')

                cam_pcds = get_label_pcds(regions, pts_2d)
                if obj_label not in label_to_pcds:
                    label_to_pcds[obj_label], label_to_scores[obj_label] = cam_pcds, cam_scores[obj_label]
                else:
                    label_to_pcds[obj_label], label_to_scores[obj_label] = extend_pcds(cam_pcds, label_to_pcds[obj_label], 
                                                                                        cam_scores[obj_label], label_to_scores[obj_label], threshold=threshold-0.01*i)
                log_debug(f'{obj_label} size is now {len(label_to_pcds[obj_label])}')
            
        pcds_output = {}
        for obj_label in captions:
            if obj_label not in label_to_pcds:
                log_warn('WARNING: COULD NOT FIND RELEVANT OBJ')
                break
            obj_pcd_sets = label_to_pcds[obj_label]
            for i, target_obj_pcd_obs in enumerate(obj_pcd_sets):
                target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
                inliers = np.where(np.linalg.norm(target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
                target_obj_pcd_obs = target_obj_pcd_obs[inliers]
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
            self.ranked_objs[0]['demo_start_poses'] = []

        for fname in self.skill_demos[:min(n, len(self.skill_demos))]:
            demo = np.load(fname, allow_pickle=True)
            if 'shapenet_id' not in demo:
                continue
            obj_id = demo['shapenet_id'].item()

            if self.state == 0:
                target_info, initial_pose = process_demo_data(demo, table_obj=self.table_obj)
                self.ranked_objs[0]['demo_start_poses'].append(initial_pose)
            elif obj_id in self.ranked_objs[0]['demo_start_poses']:
                target_info, _ = process_demo_data(demo, self.initial_poses[obj_id], table_obj=self.table_obj)
                # del self.initial_poses[obj_id]
            else:
                continue
            
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
                f_pcd = demo['multi_obj_final_pcd'].item()[demo_key]
                obj_ids = demo['multi_object_ids'].item()[demo_key]
                start_pose = demo['multi_obj_start_obj_pose'].item()[demo_key]

                self.ranked_objs[obj_rank]['demo_start_pcds'].append(s_pcd)
                self.ranked_objs[obj_rank]['demo_final_pcds'].append(f_pcd)
                self.ranked_objs[obj_rank]['demo_ids'].append(obj_ids)
                self.ranked_objs[obj_rank]['demo_start_poses'].append(start_pose)
                
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
            use_keypoint_offset=use_keypoint_offset, keypoint_offset_params=keypoint_offset_params, visualize=True, mc_vis=self.viz.mc_vis)
       

    #################################################################################################
    # Optimization 
    def get_intial_model_paths(self, concept):
        target_class = self.ranked_objs[0]['potential_class']
        target_model_path = self.args.child_model_path
        if not target_model_path:
            target_model_path = 'ndf_vnn/rndf_weights/ndf_'+target_class+'.pth'

        if self.state > 0:
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
        optimizer.setup_meshcat(self.viz.mc_vis)
        self.ranked_objs[obj_rank]['optimizer'] = optimizer

    def find_correspondence(self):
        if self.state == 0:
            ee_poses = self.find_pick_transform(0)
        elif self.state == 1:
            ee_poses = self.find_place_transform(0, relational_rank=1, ee=self.last_ee)
        elif self.state == 2:
            ee_poses = self.find_place_transform(0, relational_rank=1)
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

            self.viz.pause_mc_thread(True)
            final_pose_mat = infer_relation_intersection(
                self.viz.mc_vis, relational_optimizer, optimizer, 
                relational_target_desc, target_desc, 
                relational_pcd, target_pcd, relational_query_pts, target_query_pcd, opt_visualize=self.args.opt_visualize)
            self.viz.pause_mc_thread(False)
        else:
            pose_mats, best_idx = optimizer.optimize_transform_implicit(target_pcd, ee=False)
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
            offset_pose = util.transform_pose(
                pose_source=util.list2pose_stamped(np.concatenate(self.robot.arm.get_ee_pose()[:2]).tolist()),
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
    def execute(self, obj_rank, ee_poses):
        obj_id = None
        if 'obj_id' in self.ranked_objs[obj_rank]:
            obj_id = self.ranked_objs[obj_rank]['obj_rd']

        if self.state == 2:
            self.teleport(obj_id, ee_poses[0])
        else:
            self.pre_execution(obj_id)
            jnt_poses = self.get_iks(ee_poses)

            prev_pos = self.robot.arm.get_jpos()
            for i, jnt_pos in enumerate(jnt_poses):
                if jnt_pos is None:
                    log_warn('No IK for jnt', i)
                    break
                
                plan = self.plan_motion(prev_pos, jnt_pos)
                if plan is None:
                    log_warn('FAILED TO FIND A PLAN. STOPPING')
                    break
                self.move_robot(plan)
                prev_pos = jnt_pos
                # input('Press enter to continue')

            self.post_execution(obj_id)
            time.sleep(1.0)

    def get_iks(self, ee_poses):
        return [self.cascade_ik(pose) for pose in ee_poses]
    
    def cascade_ik(self, ee_pose):
        jnt_pos = None
        if jnt_pos is None:
            jnt_pos = self.ik_helper.get_feasible_ik(ee_pose, verbose=False)
            if jnt_pos is None:
                jnt_pos = self.ik_helper.get_ik(ee_pose)
                if jnt_pos is None:
                    #TODO: Remove pybullet IK computer
                    jnt_pos = self.robot.arm.compute_ik(ee_pose[:3], ee_pose[3:])
        if jnt_pos is None:
            log_warn('Failed to find IK')
        return jnt_pos

    def teleport(self, obj_id, relative_pose):
        transform = util.matrix_from_pose(util.list2pose_stamped(relative_pose))
        start_pose = np.concatenate(self.robot.pb_client.get_body_state(obj_id)[:2]).tolist()
        start_pose_mat = util.matrix_from_pose(util.list2pose_stamped(start_pose))
        final_pose_mat = np.matmul(transform, start_pose_mat)
        self.robot.pb_client.set_step_sim(True)
        final_pose_list = util.pose_stamped2list(util.pose_from_matrix(final_pose_mat))
        final_pos, final_ori = final_pose_list[:3], final_pose_list[3:]

        self.robot.pb_client.reset_body(obj_id, final_pos, final_ori)

        if self.obj_info[obj_id]['class'] not in ['syn_rack_easy', 'syn_rack_med', 'rack']:
            obj_rank = self.obj_info[obj_id]['rank']
            safeRemoveConstraint(self.ranked_objs[obj_rank]['o_cid'])

        final_pcd = util.transform_pcd(self.obj_info[obj_id]['pcd'], transform)
        with self.viz.recorder.meshcat_scene_lock:
            util.meshcat_pcd_show(self.viz.mc_vis, final_pcd, color=[255, 0, 255], name=f'scene/{obj_id}_pcd')

        safeCollisionFilterPair(obj_id, self.table_id, -1, 0, enableCollision=False)
        time.sleep(5.0)
        # turn on the physics and let things settle to evaluate success/failure
        self.robot.pb_client.set_step_sim(False)

    def plan_motion(self, start_pos, goal_pos):
        return self.ik_helper.plan_joint_motion(start_pos, goal_pos)

    def move_robot(self, plan):
        for jnt in plan:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan[-1], wait=True)

    def pre_execution(self, obj_id):
        #TODO: Remove turning on/off collisions
        if self.state == 0:
            if obj_id:
                time.sleep(0.5)
                # turn OFF collisions between robot and object / table, and move to pre-grasp pose
                for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
            self.robot.arm.eetool.open()
            time.sleep(0.5)
        else:
            if obj_id:
                safeCollisionFilterPair(self.ranked_objs[1]['obj_id'], self.ranked_objs[0]['obj_id'], -1, -1, enableCollision=True)

    def post_execution(self, obj_id):
        #TODO: Remove turning on/off collisions
        if self.state == 0:
            # turn ON collisions between robot and object, and close fingers
            if obj_id:
                for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=True, physicsClientId=self.robot.pb_client.get_client_id())
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=self.placement_link_id, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())

            time.sleep(0.8)
            jnt_pos_before_grasp = self.robot.arm.get_jpos()
            soft_grasp_close(self.robot, self.finger_joint_id, force=40)
            time.sleep(1.5)

            if obj_id:
                grasp_success = object_is_still_grasped(self.robot, obj_id, self.right_pad_id, self.left_pad_id) 
                if grasp_success:
                    log_debug("It is grasped")
                    self.robot.arm.set_jpos(jnt_pos_before_grasp, ignore_physics=True)
                # self.o_cid = constraint_grasp_close(self.robot, obj_id)
                else:
                    log_debug('Not grasped')
        else:
            self.robot.arm.eetool.open()
            if obj_id:
                # turn ON collisions between object and rack, and open fingers
                safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
                safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=True)
                
                p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
            # constraint_grasp_open(self.o_cid)
                grasp_success = object_is_still_grasped(self.robot, obj_id, self.right_pad_id, self.left_pad_id) 

            self.state = -1
            if obj_id:
                time.sleep(0.2)
                for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
                safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)

            self.robot.arm.move_ee_xyz([0, 0.075, 0.075])
            time.sleep(1.0)

