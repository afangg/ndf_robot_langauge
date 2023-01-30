import os, os.path as osp
import random
import copy
import numpy as np
import torch
import argparse
import time
import sys
sys.path.append('/home/afo/repos/relational_ndf/src/')
import pybullet as p
import trimesh
import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from rndf_robot.utils import util, trimesh_util
from rndf_robot.utils import path_util

# from rndf_robot.utils.franka_ik import FrankaIK
from rndf_robot.opt.optimizer import OccNetOptimizer
from rndf_robot.robot.multicam import MultiCams
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from rndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from rndf_robot.utils.pipeline_util import (
    safeCollisionFilterPair,
    safeRemoveConstraint,
    soft_grasp_close,
    object_is_still_grasped,
    constraint_grasp_close,
    process_xq_data,
    process_xq_rs_data,
    process_demo_data,
    post_process_grasp_point,
    get_ee_offset,
    constraint_obj_world,
    constraint_grasp_open
)
from rndf_robot.eval.relation_tools.multi_ndf import infer_relation_intersection, create_target_descriptors

from airobot import Robot, log_info, set_log_level, log_warn
from airobot.utils import common
from airobot.utils.common import euler2quat
from airobot.utils.pb_util import create_pybullet_client

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_util
import rndf_utils

def show_link(obj_id, link_id, color):
    if link_id is not None:
        p.changeVisualShape(obj_id, link_id, rgbaColor=color)

class Pipeline():

    def __init__(self, args):
        self.tables = ['table_rack.urdf', 'table_shelf.urdf']
        self.ll_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.all_demos_dirs = osp.join(path_util.get_rndf_data(), 'release_demos')
        self.args = args

        self.random_pos = False
        self.ee_pose = None
        self.scene_obj = None
        self.table_model = None
        self.scene_dict = dict(parent={}, child={})

        self.cfg = get_eval_cfg_defaults()

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

    def register_vizServer(self, vizServer):
        self.viz = vizServer

    def step(self, scene=None):
        while True:
            x = input('Press 1 to continue or 2 to use a new object\n')

            if x == '1':
                self.ee_pose = scene['final_ee_pos']
                # robot.arm.get_ee_pose()
                self.scene_obj = scene['obj_pcd'], scene['obj_pose'], scene['obj_id']
                break
            elif x == '2':
                self.robot.pb_client.remove_body(scene['obj_id'])
                self.robot.pb_client.remove_body(self.table_id)

                self.robot.arm.go_home(ignore_physics=True)
                self.robot.arm.move_ee_xyz([0, 0, 0.2])
                self.robot.arm.eetool.open()

                self.scene_obj = None
                self.ee_pose = None
                self.initial_poses = None
                self.table_model = None
                time.sleep(1.5)
                break

    def setup_client(self):
        self.robot = Robot('franka', pb_cfg={'gui': self.args.pybullet_viz}, arm_cfg={'self_collision': False, 'seed': self.args.seed})
        # self.ik_helper = FrankaIK(gui=False)
        
        self.finger_joint_id = 9
        self.left_pad_id = 9
        self.right_pad_id = 10

        p.changeDynamics(self.robot.arm.robot_id, self.left_pad_id, lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, self.right_pad_id, lateralFriction=1.0)

        # reset
        self.robot.arm.reset(force_reset=True)
        self.setup_cams()

    def setup_cams(self):
        self.robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, self.cfg.TABLE_Z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)
        self.cams = MultiCams(self.cfg.CAMERA, self.robot.pb_client, n_cams=self.cfg.N_CAMERAS)
        log_info('Number of cameras: %s' % len(self.cams.cams))
        cam_info = {}
        cam_info['pose_world'] = []
        for cam in self.cams.cams:
            cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

    def setup_table(self):            
        table_fname = osp.join(path_util.get_rndf_descriptions(), 'hanging/table')
        table_urdf_file = 'table_manual.urdf'
        print('TABLE MODEL', self.table_model)
        if self.table_model:    
            if self.cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
                self.load_shelf = True
                table_urdf_file = 'table_shelf.urdf'
                log_info('Shelf loaded')
            else:
                log_info('Rack loaded')
                self.load_shelf = False
                table_urdf_file = 'table_rack.urdf'
        table_fname = osp.join(table_fname, table_urdf_file)
        print(table_fname)
        
        self.table_id = self.robot.pb_client.load_urdf(table_fname,
                                self.cfg.TABLE_POS, 
                                self.cfg.TABLE_ORI,
                                scaling=1.0)
        self.viz.recorder.register_object(self.table_id, table_fname)
        self.viz.pause_mc_thread(False)

        time.sleep(1.5)
        log_info("DONE SETTING UP")

    #################################################################################################
    # Loading config settings and files

    def get_env_cfgs(self):
        # general experiment + environment setup/scene generation configs
        cfg = get_eval_cfg_defaults()
        config = 'base_config.yaml' if len(self.test_objs) > 1 else 'eval_'+self.test_objs[0]+'_gen.yaml'
        config_fname = osp.join(path_util.get_rndf_config(), 'eval_cfgs', config)
        if osp.exists(config_fname):
            cfg.merge_from_file(config_fname)
            log_info('Config file loaded')
        else:
            log_info('Config file %s does not exist, using defaults' % config_fname)
        cfg.freeze()
        return cfg

    def get_obj_cfgs(self):
        # object specific configs
        obj_cfg = get_obj_cfg_defaults()
        config = 'base_config' if len(self.test_objs) > 1 else self.test_objs[0]
        obj_config_name = osp.join(path_util.get_rndf_config(), config+'_obj_cfg.yaml')
        if osp.exists(obj_config_name):
            obj_cfg.merge_from_file(obj_config_name)
            log_info("Set up config settings for %s" % self.test_obj)
        else:
            log_warn(f'Config file {obj_config_name} does not exist, using defaults')
        # obj_cfg.freeze()
        return obj_cfg

    def load_demos_dict(self):
        demo_dic = {}
        for class_pair in os.listdir(self.all_demos_dirs):
            print(class_pair)
            class_pair_path = osp.join(self.all_demos_dirs, class_pair)
            for fname in os.listdir(class_pair_path):
                if '_demo_' not in fname: continue
                if not fname.endswith('npz'): continue
                verb = fname.split('_demo_')[0]
                demo_path = osp.join(class_pair_path, fname)
                concept = class_pair + ' ' + verb
                if concept not in demo_dic:
                    demo_dic[concept] = []
                demo_dic[concept].append(demo_path)
        self.demo_dic = demo_dic

    def load_meshes_dict(self):
        mesh_names = {}
        for k, v in rndf_utils.mesh_data_dirs.items():
            # get train samples
            objects_raw = os.listdir(v)
            objects_filtered = [fn for fn in objects_raw if (fn.split('/')[-1] not in rndf_utils.bad_ids[k] and '_dec' not in fn)]
            # objects_filtered = objects_raw
            total_filtered = len(objects_filtered)
            train_n = int(total_filtered * 0.9); test_n = total_filtered - train_n

            train_objects = sorted(objects_filtered)[:train_n]
            test_objects = sorted(objects_filtered)[train_n:]

            log_info('\n\n\nTest objects: ')
            log_info(test_objects)
            # log_info('\n\n\n')

            mesh_names[k] = objects_filtered
        self.obj_meshes = mesh_names

    def load_models(self):
        # LOAD EACH MODEL
        parent_model_path = osp.join(path_util.get_rndf_model_weights(), self.args.parent_model_path)
        child_model_path = osp.join(path_util.get_rndf_model_weights(), self.args.child_model_path)
        parent_model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
        self.scene_dict['parent']['model_path'] = parent_model_path
        self.scene_dict['child']['model_path'] = child_model_path

        child_model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
        parent_model.load_state_dict(torch.load(parent_model_path))
        child_model.load_state_dict(torch.load(child_model_path))

        self.scene_dict['parent']['model'] = parent_model
        self.scene_dict['child']['model'] = child_model
        
    #################################################################################################
    # Language

    def prompt_query(self):
        print('All demo labels:', self.demo_dic.keys())
        self.demos, concept = self.process_query()
        print(self.demos)
        concept = frozenset(concept.lower().replace('_', ' ').split(' '))
        if not len(self.demos):
            log_warn('No demos correspond to the query!')
        test_obj_classes = set(rndf_utils.mesh_data_dirs.keys())
        self.test_objs = concept.intersection(test_obj_classes)
        
        self.scene_dict['child']['class'], self.scene_dict['parent']['class'] = self.test_objs
        log_info('Parent: %s'% self.scene_dict['parent']['class'])
        log_info('Child: %s'% self.scene_dict['child']['class'])


    def process_query(self):
        concepts = list(self.demo_dic.keys())
        n = len(concepts)
        concept_embeddings = self.ll_model.encode(concepts, convert_to_tensor=True)

        while True:
            query_text = input('Please enter a query\n')
            target_embedding= self.ll_model.encode(query_text, convert_to_tensor=True)
            scores = sentence_util.pytorch_cos_sim(target_embedding, concept_embeddings)
            sorted_scores, idx = torch.sort(scores, descending=True)
            sorted_scores, idx = sorted_scores.flatten(), idx.flatten()
            corresponding_concept = None
            for i in range(n):
                print('Best matches:', concepts[idx[i]])
                query_text = input('Corrent? (y/n)\n')
                if query_text == 'n':
                    continue
                elif query_text == 'y':
                    corresponding_concept = concepts[idx[i]]
                    break
            
            if corresponding_concept:
                break

        demos = self.demo_dic[corresponding_concept] if corresponding_concept in self.demo_dic else []
        if not len(demos):
            log_warn('No demos correspond to the query!')
            
        log_info('Number of Demos %s' % len(demos)) 

        if demos is not None and self.table_model is None:
            demo_file = np.load(demos[0], allow_pickle=True)
            if 'table_urdf' in demo_file:
                self.table_model = demo_file['table_urdf'].item()
            log_info('Found new table model')
        return demos, corresponding_concept

    #################################################################################################
    # Set up scene

    def setup_scene_objs(self):
        # scene_objs = []
        # for test_obj in [self.child_class, self.parent_class]:
        #     obj_shapenet_id = random.sample(self.obj_meshes[test_obj].keys(), 1)[0]
        #     id_str = 'Loading Shapenet ID: %s' % obj_shapenet_id
        #     print(id_str)
        #     scene_objs.append(obj_shapenet_id)

        for obj_key in ['parent', 'child']:
            self.scene_dict[obj_key]['shapenet_id'] = random.sample(self.obj_meshes[self.scene_dict[obj_key]['class']], 1)[0]
            log_info('Loading %s shape: %s' % (obj_key, self.scene_dict[obj_key]['shapenet_id']))
            obj_id, obj_pose_world = self.add_test_objs(self.scene_dict[obj_key]['shapenet_id'], self.scene_dict[obj_key]['class'], obj_key)
            self.scene_dict[obj_key]['obj_id'] = obj_id
            self.scene_dict[obj_key]['pose'] = obj_pose_world
        
    def preprocess_obj(self, obj_class):
        parent_extent = rndf_utils.reshape_bottle()
        # TODO

    def add_test_objs(self, shapenet_id, obj_class, obj_key):
        x_low, x_high = self.cfg.OBJ_SAMPLE_X_HIGH_LOW
        y_low, y_high = self.cfg.OBJ_SAMPLE_Y_HIGH_LOW

        self.placement_link_id = 0 
        upright_orientation = rndf_utils.upright_orientation_dict[obj_class]

        scale_default = self.cfg.MESH_SCALE_DEFAULT
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
            pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, self.cfg.TABLE_Z]
            pose = util.list2pose_stamped(pos + upright_orientation)
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]


        # for testing, use the "normalized" object
        # obj_file = self.all_demos_dirs[obj_class][shapenet_id]
        if obj_class in ['bottle', 'bowl', 'mug']:
            obj_file = osp.join(rndf_utils.mesh_data_dirs[obj_class], shapenet_id, 'models/model_normalized.obj')
            obj_file_dec = obj_file.split('.obj')[0] + '_dec.obj'
        # IF IT'S NOT SHAPENET NO NESTED FOLDERS
        else:
            obj_file = osp.join(rndf_utils.mesh_data_dirs[obj_class], shapenet_id + '.obj')
            obj_file_dec = obj_file.split('.obj')[0] + '_dec.obj'
        obj_file = osp.join(rndf_utils.mesh_data_dirs[obj_class], shapenet_id, 'models/model_normalized.obj')

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
            base_ori=ori)

        # register the object with the meshcat visualizer
        self.viz.recorder.register_object(obj_id, obj_file_dec, scaling=mesh_scale)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        safeCollisionFilterPair(self.robot.arm.robot_id, self.table_id, -1, -1, enableCollision=True)

        p.changeDynamics(obj_id, -1, lateralFriction=0.5, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # depending on the object/pose type, constrain the object to its world frame pose
        o_cid = None
        # or (load_pose_type == 'any_pose' and pc == 'child')
        if (obj_class in ['syn_rack_easy', 'syn_rack_hard', 'syn_rack_med', 'rack']):
            o_cid = constraint_obj_world(obj_id, pos, ori)
            self.robot.pb_client.set_step_sim(False)
        self.scene_dict[obj_key]['o_cid'] = o_cid

        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
        return obj_id, obj_pose_world


    def segment_scene(self, obj_ids=None):
        depth_imgs = []
        seg_idxs = []

        pc_obs_info = {}
        pc_obs_info['pcd'] = {}
        pc_obs_info['pcd_pts'] = {}

        if not obj_ids:
            obj_ids = [self.scene_dict[obj_key]['obj_id'] for obj_key in self.scene_dict]

        for obj_id in obj_ids:
            pc_obs_info['pcd_pts'][obj_id] = []

        for i, cam in enumerate(self.cams.cams): 
            # get image and raw point cloud
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)

            # flatten and find corresponding pixels in segmentation mask
            flat_seg = seg.flatten()
            flat_depth = depth.flatten()

            for obj_id in obj_ids:
                obj_inds = np.where(flat_seg == obj_id)
                seg_depth = flat_depth[obj_inds[0]]  
                
                obj_pts = pts_raw[obj_inds[0], :]
                pc_obs_info['pcd_pts'][obj_id].append(util.crop_pcd(obj_pts))

            depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)

        # merge point clouds from different views, and filter weird artifacts away from the object
        for obj_id, obj_pcd_pts in pc_obs_info['pcd_pts'].items():
            target_obj_pcd_obs = np.concatenate(obj_pcd_pts, axis=0)  # object shape point cloud
            target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
            inliers = np.where(np.linalg.norm(target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
            target_obj_pcd_obs = target_obj_pcd_obs[inliers]

            #Debt: key should be obj_id not obj_key but whatever
            for obj_key in self.scene_dict:
                if self.scene_dict[obj_key]['obj_id'] == obj_id:
                    self.scene_dict[obj_key]['pcd'] = target_obj_pcd_obs 

        with self.viz.recorder.meshcat_scene_lock:
            for obj_key in self.scene_dict:
                label = 'scene/%s_pcd' % obj_key
                color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                util.meshcat_pcd_show(self.viz.mc_vis, self.scene_dict[obj_key]['pcd'], color=color, name=label)

    #################################################################################################
    # Process demos

    def load_demos(self, n=None):
        for obj_key in ['parent', 'child']:
            self.scene_dict[obj_key]['demo_start_pcds'] = []
            self.scene_dict[obj_key]['demo_final_pcds'] = []
            self.scene_dict[obj_key]['demo_ids'] = []
            self.scene_dict[obj_key]['demo_start_poses'] = []

            for demo_path in self.demos:
                demo = np.load(demo_path, allow_pickle=True)
                s_pcd = demo['multi_obj_start_pcd'].item()[obj_key]
                f_pcd = demo['multi_obj_final_pcd'].item()[obj_key]
                obj_ids = demo['multi_object_ids'].item()[obj_key]
                start_pose = demo['multi_obj_start_obj_pose'].item()[obj_key]

                self.scene_dict[obj_key]['demo_start_pcds'].append(s_pcd)
                self.scene_dict[obj_key]['demo_final_pcds'].append(f_pcd)
                self.scene_dict[obj_key]['demo_ids'].append(obj_ids)
                self.scene_dict[obj_key]['demo_start_poses'].append(start_pose)
                
    def process_demos(self, relational=False):
        if relational:            
            # MAKE A NEW DIR FOR TARGET DESCRIPTORS
            demo_path = osp.join(path_util.get_rndf_data(), 'targets')
            parent_model_path, child_model_path = self.scene_dict['parent']['model_path'], self.scene_dict['child']['model_path']
            target_desc_subdir = rndf_utils.create_target_desc_subdir(demo_path, parent_model_path, child_model_path)
            target_desc_fname = osp.join(demo_path, target_desc_subdir, 'target_descriptors.npz')
            self.prepare_new_descriptors(target_desc_fname)

        if osp.exists(target_desc_fname):
            log_info(f'Loading target descriptors from file:\n{target_desc_fname}')
            target_descriptors_data = np.load(target_desc_fname)
            parent_overall_target_desc = target_descriptors_data['parent_overall_target_desc']
            child_overall_target_desc = target_descriptors_data['child_overall_target_desc']
            parent_overall_target_desc = torch.from_numpy(parent_overall_target_desc).float().cuda()
            child_overall_target_desc = torch.from_numpy(child_overall_target_desc).float().cuda()
            parent_query_points = target_descriptors_data['parent_query_points']
            child_query_points = copy.deepcopy(parent_query_points)

            log_info(f'Making a copy of the target descriptors in eval folder')

            parent_optimizer = OccNetOptimizer(
                self.model,
                query_pts=parent_query_points,
                query_pts_real_shape=parent_query_points,
                opt_iterations=self.args.opt_iterations,
                cfg=self.cfg.OPTIMIZER)

            child_optimizer = OccNetOptimizer(
                self.model,
                query_pts=child_query_points,
                query_pts_real_shape=child_query_points,
                opt_iterations=self.args.opt_iterations,
                cfg=self.cfg.OPTIMIZER)

            self.scene_dict['parent']['optimizer'] = parent_optimizer
            self.scene_dict['child']['optimizer'] = child_optimizer

    def prepare_new_descriptors(self, target_desc_fname):
        # add arg for just wanting to make a new descriptor 
        if not osp.exists(target_desc_fname):
            print(f'\n\n\nCreating target descriptors for this parent model + child model, and these demos\nSaving to {target_desc_fname}\n\n\n')
            # n_demos = 'all' if args.n_demos < 1 else args.n_demos
            
            if self.scene_dict['parent']['class'] == 'syn_container' and self.scene_dict['child']['class'] == 'bottle':
                use_keypoint_offset = True
                keypoint_offset_params = {'offset': 0.025, 'type': 'bottom'}
            else:
                use_keypoint_offset = False 
                keypoint_offset_params = None

            # bare minimum settings
            create_target_descriptors(
                self.scene_dict['parent']['model'], self.scene_dict['child']['model'], self.scene_dict, target_desc_fname, 
                self.cfg, use_keypoint_offset=use_keypoint_offset, keypoint_offset_params=keypoint_offset_params, visualize=True, mc_vis=self.viz.mc_vis)
       

    #################################################################################################
    # Optimization 

    def load_optimizer_ndf(self, demos, n=None):
        if n is None:
            n = len(demos)
        else:
            demos[:n]
        demo_target_info = []
        demo_shapenet_ids = []
        initial_poses = {}
        query_pts, query_pts_rs = None, None
        for fname in demos:
            target_info = None
            data = np.load(fname, allow_pickle=True)
            if query_pts is None:
                query_pts  = process_xq_data(data, shelf=self.load_shelf)
            if query_pts_rs is None:
                query_pts_rs = process_xq_rs_data(data, shelf=self.load_shelf)               
            demo_shapenet_id = data['shapenet_id'].item()

            if self.scene_obj is None:
                target_info, initial_pose = process_demo_data(data, shelf=self.load_shelf)
                initial_poses[demo_shapenet_id] = initial_pose
            elif self.scene_obj is not None and demo_shapenet_id in self.initial_poses:
                target_info, _ = process_demo_data(data, self.initial_poses[demo_shapenet_id], shelf=self.load_shelf)
                del self.initial_poses[demo_shapenet_id]
            else:
                continue
            
            if target_info is not None:
                demo_target_info.append(target_info)
                demo_shapenet_ids.append(demo_shapenet_id)
            else:
                log_info('Could not load demo')

        if self.scene_obj:
            scene = trimesh_util.trimesh_show([target_info['demo_obj_pts'],target_info['demo_query_pts_real_shape']], show=True)
            
        self.initial_poses = initial_poses
        optimizer = OccNetOptimizer(self.model, query_pts=query_pts, query_pts_real_shape=query_pts_rs, opt_iterations=500)
        optimizer.set_demo_info(demo_target_info)
        log_info("OPTIMIZER LOADED")
        return optimizer

    def find_correspondence_rndf(self):
        pass
    def find_correspondence(self, optimizer, target_obj_pcd, obj_pose_world):
        ee_poses = []
        if self.scene_obj is None:
            #grasping
            log_info('Solve for pre-grasp coorespondance')
            pre_ee_pose_mats, best_idx = optimizer.optimize_transform_implicit(target_obj_pcd, ee=True)
            pre_ee_pose = util.pose_stamped2list(util.pose_from_matrix(pre_ee_pose_mats[best_idx]))
            # grasping requires post processing to find anti-podal point
            grasp_pt = post_process_grasp_point(pre_ee_pose, target_obj_pcd, thin_feature=(not self.args.non_thin_feature), grasp_viz=self.args.grasp_viz, grasp_dist_thresh=self.args.grasp_dist_thresh)
            pre_ee_pose[:3] = grasp_pt
            pre_ee_offset_tf = get_ee_offset(ee_pose=pre_ee_pose)
            pre_pre_ee_pose = util.pose_stamped2list(
                util.transform_pose(pose_source=util.list2pose_stamped(pre_ee_pose), pose_transform=util.list2pose_stamped(pre_ee_offset_tf)))

            ee_poses.append(pre_pre_ee_pose)
            ee_poses.append(pre_ee_pose)
        else:
            #placement
            log_info('Solve for placement coorespondance')
            pose_mats, best_idx = optimizer.optimize_transform_implicit(target_obj_pcd, ee=False)
            relative_pose = util.pose_stamped2list(util.pose_from_matrix(pose_mats[best_idx]))
            ee_end_pose = util.transform_pose(pose_source=util.list2pose_stamped(self.ee_pose), pose_transform=util.list2pose_stamped(relative_pose))
            # obj_start_pose = obj_pose_world
            # obj_end_pose = util.transform_pose(pose_source=obj_start_pose, pose_transform=util.list2pose_stamped(rack_relative_pose))
            # obj_end_pose_list = util.pose_stamped2list(obj_end_pose)
            # ee_end_pose = util.transform_pose(pose_source=obj_start_pose, pose_transform=util.list2pose_stamped(relative_pose))
            # final_pcd = util.transform_pcd(target_obj_pcd, pose_mats[best_idx])
            # trimesh_util.trimesh_show([target_obj_pcd, final_pcd], show=True)
            preplace_offset_tf = util.list2pose_stamped(self.cfg.PREPLACE_OFFSET_TF)
            preplace_horizontal_tf = util.list2pose_stamped(self.cfg.PREPLACE_HORIZONTAL_OFFSET_TF)

            pre_ee_end_pose2 = util.transform_pose(pose_source=ee_end_pose, pose_transform=preplace_offset_tf)
            pre_ee_end_pose1 = util.transform_pose(pose_source=pre_ee_end_pose2, pose_transform=preplace_horizontal_tf)        
    
                # get pose that's straight up
            offset_pose = util.transform_pose(
                pose_source=util.list2pose_stamped(np.concatenate(self.robot.arm.get_ee_pose()[:2]).tolist()),
                pose_transform=util.list2pose_stamped([0, 0, 0.15, 0, 0, 0, 1])
            )
            ee_poses.append(util.pose_stamped2list(offset_pose)) 
            ee_poses.append(util.pose_stamped2list(pre_ee_end_pose1))
            ee_poses.append(util.pose_stamped2list(pre_ee_end_pose2))
            ee_poses.append(util.pose_stamped2list(ee_end_pose))

        log_info('Found correspondence')
        return ee_poses


    #########################################################################################################
    # Motion Planning 

    def pre_execution(self, obj_id, pos, ori, final_pos=None):
        if self.scene_obj is None:
            # reset everything
            self.robot.pb_client.set_step_sim(False)
            time.sleep(0.5)

            # turn OFF collisions between robot and object / table, and move to pre-grasp pose
            for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
            self.robot.arm.eetool.open()
            time.sleep(0.25)

        else:
            # # reset object to placement pose to detect placement success
            # safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
            # safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=False)
            # self.robot.pb_client.set_step_sim(True)
            # self.robot.pb_client.reset_body(obj_id, final_pos[:3], final_pos[3:])

            # time.sleep(0.2)
            # safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=True)
            # self.robot.pb_client.set_step_sim(False)
            # time.sleep(0.2)

            # safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
            # self.robot.pb_client.reset_body(obj_id, pos, ori)
            pass

    def get_iks(self, ee_poses):
        # jnt_poses = [None]*len(ee_poses)
        # for i, pose in enumerate(ee_poses):
        #     log_info('Attempt to find IK %i' % i)
        #     jnt_poses[i] = self.cascade_ik(pose)
        return [self.cascade_ik(pose) for pose in ee_poses]
    
    def cascade_ik(self, ee_pose):
        jnt_pos = None
        if jnt_pos is None:
            jnt_pos = self.ik_helper.get_feasible_ik(ee_pose, verbose=True)
            if jnt_pos is None:
                jnt_pos = self.ik_helper.get_ik(ee_pose)
                if jnt_pos is None:
                    jnt_pos = self.robot.arm.compute_ik(ee_pose[:3], ee_pose[3:])
        if jnt_pos is None:
            log_warn('Failed to find IK')
        return jnt_pos

    def teleport_obj(self, obj_id, pose):
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=False)
        self.robot.pb_client.set_step_sim(True)
        self.robot.pb_client.reset_body(obj_id, pose[:3], pose[3:])

        time.sleep(1.0)
        safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=True)
        self.robot.pb_client.set_step_sim(False)
        time.sleep(1.0)

        # obj_surf_contacts = p.getContactPoints(obj_id, table_id, -1, placement_link_id)
        # touching_surf = len(obj_surf_contacts) > 0
        # place_success_teleport = touching_surf
        # place_success_teleport_list.append(place_success_teleport)

        # time.sleep(1.0)
        # safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        # robot.pb_client.reset_body(obj_id, pos, ori)

    def plan_motion(self, start_pos, goal_pos):
        return self.ik_helper.plan_joint_motion(start_pos, goal_pos)

    def execute_plan(self, plan):
        for jnt in plan:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan[-1], wait=True)

    def allow_pregrasp_collision(self, obj_id):
        log_info('Turning off collision between gripper and object for pre-grasp')
        # turn ON collisions between robot and object, and close fingers
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=True, physicsClientId=self.robot.pb_client.get_client_id())
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=self.placement_link_id, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
    
    def is_pregrasp_state(self):
        return self.scene_obj is None

    def post_execution(self, obj_id, pos, ori):
        if self.scene_obj is None:
            # turn ON collisions between robot and object, and close fingers
            for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=True, physicsClientId=self.robot.pb_client.get_client_id())
                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=self.placement_link_id, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())

            time.sleep(0.8)
            jnt_pos_before_grasp = self.robot.arm.get_jpos()
            soft_grasp_close(self.robot, self.finger_joint_id, force=40)
            time.sleep(1.5)
            grasp_success = object_is_still_grasped(self.robot, obj_id, self.right_pad_id, self.left_pad_id) 
            time.sleep(1.5)

            if grasp_success:
                log_info("It is grasped")
                self.robot.arm.set_jpos(jnt_pos_before_grasp, ignore_physics=True)
                # self.o_cid = constraint_grasp_close(self.robot, obj_id)
            else:
                log_info('Not grasped')
        else:
            grasp_success = object_is_still_grasped(self.robot, obj_id, self.right_pad_id, self.left_pad_id) 
            if grasp_success:
                # turn ON collisions between object and rack, and open fingers
                safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
                safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=True)
                
                p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
                # constraint_grasp_open(self.o_cid)
                self.robot.arm.eetool.open()

                time.sleep(0.2)
                for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
                self.robot.arm.move_ee_xyz([0, 0.075, 0.075])
                safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
                time.sleep(4.0)
            else:
                log_warn('No object in hand')
