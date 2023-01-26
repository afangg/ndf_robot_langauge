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

mesh_data_dirs = {
    'mug': 'mug_centered_obj_normalized', 
    'bottle': 'bottle_centered_obj_normalized', 
    'bowl': 'bowl_centered_obj_normalized',
    'rack': 'syn_racks_easy_obj',
    'container': 'box_containers_unnormalized'
}
mesh_data_dirs = {k: osp.join(path_util.get_rndf_obj_descriptions(), v) for k, v in mesh_data_dirs.items()}

bad_ids = {
    'rack': [],
    'bowl': bad_shapenet_bowls_ids_list,
    'mug': bad_shapenet_mug_ids_list,
    'bottle': bad_shapenet_bottles_ids_list,
    'container': []
}

upright_orientation_dict = {
    'mug': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
    'bottle': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
    'bowl': common.euler2quat([np.pi/2, 0, 0]).tolist(),
    'rack': common.euler2quat([0, 0, 0]).tolist(),
    'container': common.euler2quat([0, 0, 0]).tolist(),
}

def show_link(obj_id, link_id, color):
    if link_id is not None:
        p.changeVisualShape(obj_id, link_id, rgbaColor=color)

def create_target_desc_subdir(demo_path, parent_model_path, child_model_path):
    parent_model_name_full = parent_model_path.split('ndf_vnn/')[-1]
    child_model_name_full = child_model_path.split('ndf_vnn/')[-1]

    parent_model_name_specific = parent_model_name_full.split('.pth')[0].replace('/', '--')
    child_model_name_specific = child_model_name_full.split('.pth')[0].replace('/', '--')
    
    subdir_name = f'parent_model--{parent_model_name_specific}_child--{child_model_name_specific}'
    dirname = osp.join(demo_path, subdir_name)
    util.safe_makedirs(dirname)
    return dirname

class Pipeline():

    def __init__(self, global_dict, args):
        self.tables = ['table_rack.urdf', 'table_shelf.urdf']
        self.ll_model = global_dict['ll_model']
        self.model = global_dict['ndf_model']
        self.all_demos_dirs = global_dict['all_demos_dirs']
        self.args = args

        self.random_pos = False
        self.ee_pose = None
        self.scene_obj = None
        self.table_model = None
        self.scene_dict = dict(parent={}, child={})

        self.cfg = get_eval_cfg_defaults()
    
        self.load_robot()
        self.load_cams()
    
        self.demo_dic = self.get_demo_dict()

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

                # self.robot.arm.go_home(ignore_physics=True)
                # self.robot.arm.move_ee_xyz([0, 0, 0.2])
                # self.robot.arm.eetool.open()

                self.scene_obj = None
                self.ee_pose = None
                self.initial_poses = None
                self.table_model = None
                time.sleep(1.5)
                break

    def load_robot(self):
        self.robot = Robot('franka', pb_cfg={'gui': self.args.pybullet_viz}, arm_cfg={'self_collision': False, 'seed': self.args.seed})
        # self.ik_helper = FrankaIK(gui=False)
        
        self.finger_joint_id = 9
        self.left_pad_id = 9
        self.right_pad_id = 10

        p.changeDynamics(self.robot.arm.robot_id, self.left_pad_id, lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, self.right_pad_id, lateralFriction=1.0)

        # reset
        self.robot.arm.reset(force_reset=True)
        self.robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, self.cfg.TABLE_Z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)

    def load_cams(self):
        self.cams = MultiCams(self.cfg.CAMERA, self.robot.pb_client, n_cams=self.cfg.N_CAMERAS)
        log_info('Number of cameras: %s' % len(self.cams.cams))
        cam_info = {}
        cam_info['pose_world'] = []
        for cam in self.cams.cams:
            cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

    def load_table(self):            
        self.cfg = self.get_env_cfgs()
        self.obj_cfgs = self.get_obj_cfgs()

        # this is the URDF that was used in the demos -- make sure we load an identical one
        # tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_tmp.urdf')
        # open(tmp_urdf_fname, 'w').write(self.table_model)
        # self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
        #                         self.cfg.TABLE_POS,
        #                         table_ori,
        #                         scaling=self.cfg.TABLE_SCALING)
        # print('TABLE ID', self.table_id, open(tmp_urdf_fname, 'r').read())

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

        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])
        self.robot.arm.eetool.open()
        time.sleep(1.5)
        log_info("DONE SETTING UP")

    def get_env_cfgs(self):
        # general experiment + environment setup/scene generation configs
        cfg = get_eval_cfg_defaults()
        config = 'base_cfg.yaml' if len(self.test_objs) > 1 else 'eval_'+self.test_objs[0]+'_gen.yaml'
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
        config = 'base_cfg' if len(self.test_objs) > 1 else self.test_objs[0]
        obj_config_name = osp.join(path_util.get_rndf_config(), config+'_obj_cfg.yaml')
        if osp.exists(obj_config_name):
            obj_cfg.merge_from_file(obj_config_name)
            log_info("Set up config settings for %s" % self.test_obj)
        else:
            log_warn(f'Config file {obj_config_name} does not exist, using defaults')
        # obj_cfg.freeze()
        return obj_cfg

    def get_demo_dict(self):
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
        return demo_dic

    def prompt_query(self):
        print('All demo labels:', self.demo_dic.keys())
        self.demos, concept = self.choose_demos()
        concept = frozenset(concept.lower().replace('_', ' ').split(' '))
        if not len(self.demos):
            log_warn('No demos correspond to the query!')
        test_obj_classes = set(mesh_data_dirs.keys())
        self.test_objs = concept.intersection(test_obj_classes)
        print(concept, test_obj_classes, self.test_objs)
        self.scene_dict['child']['class'], self.scene_dict['parent']['class'] = self.test_objs
        log_info('Parent: %s'% self.scene_dict['parent']['class'])
        log_info('Child: %s'% self.scene_dict['child']['class'])

    def choose_demos(self):
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

    # def get_test_objs(self):
    #     #either replace with random objects (manipulation can fail) or label object in demo
    #     test_obj = self.test_obj
    #     avoid_shapenet_ids = []
    #     if test_obj == 'mug':
    #         avoid_shapenet_ids = bad_shapenet_mug_ids_list + self.cfg.MUG.AVOID_SHAPENET_IDS
    #     elif test_obj == 'bowl':
    #         avoid_shapenet_ids = bad_shapenet_bowls_ids_list + self.cfg.BOWL.AVOID_SHAPENET_IDS
    #     elif test_obj == 'bottle':
    #         avoid_shapenet_ids = bad_shapenet_bottles_ids_list + self.cfg.BOTTLE.AVOID_SHAPENET_IDS 
    #     else:
    #         test_shapenet_ids = []
    #     # get objects that we can use for testing
    #     test_object_ids = {}
    #     shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), test_obj + '_centered_obj_normalized')
    #     shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)] if test_obj == 'mug' else os.listdir(shapenet_obj_dir)
    #     for s_id in shapenet_id_list:
    #         valid = s_id not in avoid_shapenet_ids
    #         if valid:
    #             # for testing, use the "normalized" object
    #             obj_obj_file = osp.join(shapenet_obj_dir, s_id, 'models/model_normalized.obj')
    #             test_object_ids[s_id] = obj_obj_file
    #     self.test_obj = test_obj
    #     return test_object_ids

    def load_all_obj_files(self):
        mesh_names = {}
        for k, v in mesh_data_dirs.items():
            # get train samples
            objects_raw = os.listdir(v)
            objects_filtered = [fn for fn in objects_raw if (fn.split('/')[-1] not in bad_ids[k] and '_dec' not in fn)]
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

    def generate_scene_objs(self):
        # scene_objs = []
        # for test_obj in [self.child_class, self.parent_class]:
        #     obj_shapenet_id = random.sample(self.obj_meshes[test_obj].keys(), 1)[0]
        #     id_str = 'Loading Shapenet ID: %s' % obj_shapenet_id
        #     print(id_str)
        #     scene_objs.append(obj_shapenet_id)
        self.scene_dict['parent']['shapenet_id'] = random.sample(self.obj_meshes[self.scene_dict['parent']['class']], 1)[0]
        self.scene_dict['child']['shapenet_id'] = random.sample(self.obj_meshes[self.scene_dict['child']['class']], 1)[0]
        log_info('Loading child shape: %s' % self.scene_dict['child']['shapenet_id'])
        log_info('Loading parent shape: %s' % self.scene_dict['parent']['shapenet_id'])

        parent_obj_id = self.add_test_objs(self.scene_dict['parent']['shapenet_id'], self.scene_dict['parent']['class'])
        child_obj_id = self.add_test_objs(self.scene_dict['child']['shapenet_id'], self.scene_dict['child']['class'])
        return [parent_obj_id, child_obj_id]

    def add_test_objs(self, shapenet_id, obj_class):
        x_low, x_high = self.cfg.OBJ_SAMPLE_X_HIGH_LOW
        y_low, y_high = self.cfg.OBJ_SAMPLE_Y_HIGH_LOW

        # if self.test_obj == 'mug':
        #     rack_link_id = 0
        #     shelf_link_id = 1
        # elif self.test_obj in ['bowl', 'bottle']:
        #     rack_link_id = None
        #     shelf_link_id = 0

        # if self.cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
        #     self.placement_link_id = shelf_link_id
        # elif self.cfg.DEMOS.PLACEMENT_SURFACE == 'rack':
        #     self.placement_link_id = rack_link_id 
        # else:
        #     self.placement_link_id = rack_link_id 
        self.placement_link_id = 0 

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
            upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()
            pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, self.cfg.TABLE_Z]
            pose = util.list2pose_stamped(pos + upright_orientation)
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]


        # for testing, use the "normalized" object
        # obj_file = self.all_demos_dirs[obj_class][shapenet_id]
        obj_file = osp.join(mesh_data_dirs[obj_class], shapenet_id, 'models/model_normalized.obj')
        print(obj_file)

        obj_file_dec = obj_file.split('.obj')[0] + '_dec.obj'
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
        p.changeDynamics(obj_id, -1, lateralFriction=0.5)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        safeCollisionFilterPair(self.robot.arm.robot_id, self.table_id, -1, -1, enableCollision=True)

        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # if self.test_obj == 'mug':
        #     rack_color = p.getVisualShapeData(self.table_id)[rack_link_id][7]
        #     show_link(self.table_id, rack_link_id, rack_color)
        # return obj_id, pos, ori
        return obj_id

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

    def load_optimizer_rndf(self, demos, n=None):
        for pc in ['parent', 'child']:
            for idx, demo in enumerate(demos):
                s_pcd = demo['multi_obj_start_pcd'].item()[pc]
                f_pcd = demo['multi_obj_final_pcd'].item()[pc]

                self.scene_dict[pc]['demo_start_pcds'].append(s_pcd)
                self.scene_dict[pc]['demo_final_pcds'].append(f_pcd)
        for pc in ['parent', 'child']:
            self.scene_dict[pc]['demo_ids'] = [dat['multi_object_ids'].item()[pc] for dat in demos] 
            self.scene_dict[pc]['demo_start_poses'] = [dat['multi_obj_start_obj_pose'].item()[pc] for dat in demos] 
        model_path = osp.join(path_util.get_rndf_model_weights(), 'ndf_vnn/rndf_weights', self.args.weights)
        
        demo_path = osp.join(path_util.get_rndf_data(), 'targets')
        target_desc_subdir = create_target_desc_subdir(demo_path, model_path, model_path)
        target_desc_fname = osp.join(demo_path, model_path, target_desc_subdir, 'target_descriptors.npz')

    def prepare_target_descriptors(self, model_path, target_desc_fname, relation_method=None):
        if relation_method == 'intersection':
            if not osp.exists(target_desc_fname):
                print(f'\n\n\nCreating target descriptors for this parent model + child model, and these demos\nSaving to {target_desc_fname}\n\n\n')
                # n_demos = 'all' if args.n_demos < 1 else args.n_demos
                
                if self.test_objs[0] == 'syn_container' and self.test_objs[1] == 'bottle':
                    use_keypoint_offset = True
                    keypoint_offset_params = {'offset': 0.025, 'type': 'bottom'}
                else:
                    use_keypoint_offset = False 
                    keypoint_offset_params = None
                create_target_descriptors(
                    self.model, self.model, self.scene_dict, target_desc_fname, 
                    self.cfg, use_keypoint_offset=use_keypoint_offset, keypoint_offset_params=keypoint_offset_params,)
       
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

    # def plan_and_execute(self, poses, obj_id):
    #     prev_pos = self.robot.arm.get_jpos()
    #     for i, jnt_pos in enumerate(poses):
    #         if jnt_pos is None:
    #             log_warn('No IK for jnt', i)
    #             break
            
    #         plan = self.plan_motion(prev_pos, jnt_pos)
    #         self.execute_plan(plan)
    #         prev_pos = jnt_pos

    #         if i == 0:
    #             # turn ON collisions between robot and object, and close fingers
    #             for _ in range(p.getNumJoints(self.robot.arm.robot_id)):
    #                 safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=True, physicsClientId=self.robot.pb_client.get_client_id())
    #                 safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=self.placement_link_id, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
            
    #         input('Press enter to continue')

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
