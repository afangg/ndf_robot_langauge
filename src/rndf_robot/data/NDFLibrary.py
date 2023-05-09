import os, os.path as osp
import numpy as np
# import torch
import copy

import sys
sys.path.append(os.getenv('SOURCE_DIR'))
from rndf_robot.utils import util, path_util
from rndf_robot.utils.pipeline_util import (
    process_xq_data,
    process_xq_rs_data,
    process_demo_data,
    post_process_grasp_point,
    get_ee_offset
)

from rndf_robot.opt.optimizer import OccNetOptimizer
import rndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from rndf_robot.data.rndf_utils import infer_relation_intersection, create_target_descriptors
from rndf_robot.data.demo_data_processing import *
from rndf_robot.system.garbage import free_memory

from airobot import log_debug, log_warn, log_info
from IPython import embed

DELIM = '--rndf_weights--'
TABLE_OBJ = {'shelf': 0, 'rack': 1}
RELATIONS = {'on the', 'by the', 'in the'}
DOCSTRINGS = {  
                'grasp':  
                    '''
                        Skill to grasp an object conditioned on its pointcloud, its object class and 
                        an optional argument if there's a certain geometric feature to grasp at. Skill is executed

                        Args:
                            target (target_pcds, target_class):
                                target_pcd (np.ndarray): N x 3 array representing the 3D point cloud of the 
                                target object to be grasped, expressed in the world coordinate system
                                target_class (str): Class of the object to be grasped. Must be in self.obj_classes 
                            geometry (str): Description of a geometric feature on the object to execute grasp upon
                                ex. "handle" of a mug class object
                        
                        Return:
                            new_position (Array-like, length 3): Final position of the EE, after performing the move action.
                                If grasp fails, returns None
                    ''',
                'place_position': 
                    '''
                        Skill to place an object conditioned on its pointcloud, its object class, and 
                            a certain geometric feature to place at. Skill is executed

                        Args:
                            target (target_pcds, target_class):
                                target_pcd (np.ndarray): N x 3 array representing the 3D point cloud of the 
                                target object to be placed, expressed in the world coordinate system
                                target_class (str): Class of the object to be placed. Must be in self.obj_classes 
                            geometry (str): Description of a geometric feature on the scene to execute place upon
                                ex. "shelf" on the table
                        
                        Return:
                            new_position (Array-like, length 3): Final position of the EE, after performing the move action.
                                If grasp fails, returns None
                    ''',
                'place_relative':
                    '''
                        Skill to place an object relative to another relational object, and a certain geometric feature 
                            to place at. Skill is executed

                        Args:
                            target (np.ndarray, str): N x 3 array representing the 3D point cloud of the target object 
                                to be placed, expressed in the world coordinate system and the class of the object to be placed. 
                                Must be in self.obj_classes 
                            relational (np.ndarray, str): N x 3 array representing the 3D point cloud of the relational 
                                object to be placed in relation to, expressed in the world coordinate system, and the class of 
                                the object. Must be in self.obj_classes 
                            geometry (str): Description of a geometric feature on the scene to execute place upon
                                ex. "shelf" on the table or "on the bowl" of another bowl
                        
                        Return:
                            new_position (Array-like, length 3): Final position of the EE, after performing the move action.
                                If grasp fails, returns None
                    ''',
                'find':
                    '''
                        Skill to find pointclouds of object instances that match a language description

                        Args:
                            language_description (str): English language description of object to find
                                n (optional int): Number of instances of the object to find
                        
                        Return:
                            list of tuples (confidence score, pointcloud) of length n that match the language 
                                description of the object to find. If n isn't specified, return all that match above
                                a threshold.
                    ''',
                'learn_skill':
                    '''
                        Skill to learn a new skill from a human demonstration

                        Args:
                            skill_name (str): Short name for the skill to be learned
                            target (np.ndarray, str): (pcd, obj_class) N x 3 array representing the 3D point cloud of 
                                the target object that the new skill is executed on, expressed in the world coordinate 
                                system
                            obj_class (str): Class of the object to be executed on. Must be in self.obj_classes 
                        
                        Return:
                            the function skill_name that can perform the skill given by "skil_name" and takes as input
                                the mentioned parameters. This skill is added permanently to skill library.
                    '''
              }
class NDFLibrary:

    def __init__(self, args, mc_vis, cfg, folder, obj_classes=set(), obj_models={}) -> None:
        '''
        Library of NDF skills

        Args:
            (optional)
                folder (str): Folder name containing NDF demos to load from. 'sim_demos' by default
                obj_classes (set): Set of object classes to load primitives for. If None, load all
                obj_models (dict): Map of object class to model paths to use

        args (Environement): reference to the environment args
        folder (dict): Maps the concept name to a list of the saved NDF demos specified 
            folder 
        obj_classes (set): Set of all object classes that have primitives associated with them
        default_models (dict): Maps object class to pre-defined models
        '''
        self.args = args
        self.mc_vis = mc_vis
        self.cfg = cfg
        self.folder = folder
        self.ndf_primitives = {'grasp': {'obj_classes': set(), 'geometry': set()}, 
                               'place': {'obj_classes': set(), 'geometry': set()},
                               'place_relative': {'obj_classes': set(), 'geometry': set()}}
        self.obj_classes = obj_classes
        self.default_demos = self.load_default_demos()
        self.default_models = obj_models
        self.inital_poses = {}

        #{action: {param1: set(), param2: set()}}
        self.FUNCTIONS = {'grasp': self.grasp, 'place_position': self.place_position, 'place_relative': self.place_relative, 'find': self.find}
        self.FUNCTION_PARAMS = {'grasp': ('obj_class', 'geometry'), 'place_position': ('obj_class', 'geometry'), 'place_relative': ('obj_class', 'geometry')}


    def load_default_demos(self):
        '''
        Load in the primitives from the NDF demos saved in rndf_robot/demos/sim_demos

        '''
        all_demos = osp.join(path_util.get_rndf_demos(), self.folder)
        demo_dic = {}
        for demo_type in os.listdir(all_demos):
            demo_type_path = osp.join(all_demos, demo_type)
            obj_class, geometry = demo_type.split('_')

            for demo_npz in os.listdir(demo_type_path):
                if not demo_npz.endswith('npz'): continue

                if demo_npz.startswith('grasp') or demo_npz.startswith('place'):
                    verb = demo_npz.split('_demo_')[0]
                    if verb == 'place' and set(geometry.split(' ')).intersection(self.obj_classes):
                        verb += '_relative'
                    demo_path = osp.join(demo_type_path, demo_npz)
                    concept = verb + ' ' + demo_type
                    if concept not in demo_dic:
                        demo_dic[concept] = []
                    demo_dic[concept].append(demo_path)

                    self.ndf_primitives[verb]['obj_classes'].add(obj_class)
                    self.ndf_primitives[verb]['geometry'].add((obj_class, geometry))
                else:
                    raise NotImplementedError("Unsure how to store these demos")
                
        return demo_dic
    
    def add_demo_dir(self, skill_name, demo_dir_path):
        demos = []
        for demo_npz in os.listdir(demo_dir_path):
            if not demo_npz.endswith('npz'): continue
            demo_path = osp.join(demo_dir_path, demo_npz)
            demos.append(demo_path)
        if not demos and skill_name not in demo_dir_path:
            self.default_demos[skill_name] = demos

    def get_model_path_for_class(self, obj_class):
        if obj_class in self.default_models:
            obj_weight = self.default_models[obj_class]
            default_model_path = osp.join('ndf_vnn', obj_weight)
            return default_model_path
        else:
            model_path = input('Please specify the model path from model_weights')
            return model_path
        
    ###########################################################################################################
    # Demo Reading

    def get_ndf_demo_info(self, demo_name, geometry):
        demos = self.default_demos.get(demo_name, [])
        if not demos:
            print(f'There are no demos for {demo_name}')
            return
        
        table_obj = TABLE_OBJ.get(geometry, None)

        demo_dic = {'demo_info': [], 'demo_ids': [], 'demo_start_poses': []}
        for demo_npz in demos:
            demo = np.load(demo_npz, allow_pickle=True)
            obj_id = None
            if 'shapenet_id' in demo:
                obj_id = demo['shapenet_id'].item()

            if demo_name.startswith('grasp'):
                target_info, initial_pose = process_demo_data(demo, initial_pose = None, table_obj=table_obj)
                demo_dic['demo_start_poses'].append(initial_pose)
            elif demo_name.startswith('place'):
                grasp_pose = util.pose_stamped2list(util.unit_pose())
                grasp_pose = None
                target_info, _ = process_demo_data(demo, grasp_pose, table_obj=table_obj)
                if not target_info: continue
            else:
                raise NotImplementedError("Unsure how to read these demos")
            
            if target_info is not None:
                demo_dic['demo_info'].append(target_info)
                if obj_id is not None:
                    demo_dic['demo_ids'].append(obj_id)
            else:
                log_debug('Could not load demo')
        
        demo_dic['query_pts'] = process_xq_data(demo, table_obj=table_obj)
        demo_dic['query_pts_rs'] = process_xq_rs_data(demo, table_obj=table_obj)
        demo_dic['demo_ids'] = frozenset(demo_dic['demo_ids'])

        log_debug('Finished loading single descriptors')
        return demo_dic
    
    def get_rndf_demo_info(self, demo_name):
        demo_path = osp.join(path_util.get_rndf_demos(), self.folder, demo_name)
        demo_dic = {0: {}, 1: {}}

        log_warn('Using the first set of descriptors found')
        descriptor_models = find_rndf_descriptor_models(demo_path)
        relational_model_path, target_model_path = descriptor_models[0]
        desc_subdir = get_rndf_desc_subdir(demo_path, relational_model_path, target_model_path)
        desc_demo_npz = osp.join(demo_path, desc_subdir, 'target_descriptors.npz')
        if not osp.exists(desc_demo_npz):
            log_warn('Descriptor file not found')
            return
        
        log_debug(f'Loading target descriptors from file:\n{desc_demo_npz}')
        target_descriptors_data = np.load(desc_demo_npz)
        relational_overall_target_desc = target_descriptors_data['parent_overall_target_desc']
        target_overall_target_desc = target_descriptors_data['child_overall_target_desc']
        demo_dic[1]['target_desc'] = torch.from_numpy(relational_overall_target_desc).float().cuda()
        demo_dic[0]['target_desc'] = torch.from_numpy(target_overall_target_desc).float().cuda()
        demo_dic[1]['query_pts'] = target_descriptors_data['parent_query_points']
        #slow - change deepcopy to shallow copy? 
        demo_dic[0]['query_pts'] = copy.deepcopy(target_descriptors_data['parent_query_points'])
        log_debug('Finished loading relational descriptors')
        return demo_dic, relational_model_path, target_model_path
    
    ###########################################################################################################
    # Template
    def get_primitive_templates(self):
        output = []
        for primitive in DOCSTRINGS:
            primitive_desc = {'description': DOCSTRINGS[primitive],'fn': primitive }

            if primitive in self.ndf_primitives:
                obj_classes_set = self.ndf_primitives[primitive]['obj_classes']
                geometry_set = self.ndf_primitives[primitive]['geometry']

                geometry_map = {}
                for obj_class, part in geometry_set:
                    if obj_class not in geometry_map:
                        geometry_map[obj_class] = []
                    geometry_map[obj_class].append(part)

                params = {'obj_class': list(obj_classes_set), 'geometry': geometry_map}
                for param in self.FUNCTION_PARAMS.get(primitive, ()):
                    primitive_desc[param] = params.get(param, set())
            
            output.append(primitive_desc)
        return output
    
    ###########################################################################################################
    #Optimization
    
    def get_optimizer(self, obj_class, query_pts, query_pts_rs=None, model_path=None):
        if model_path is None:
            model_path = self.get_model_path_for_class(obj_class)
        model = load_model(model_path)
        # query_pts_rs = query_pts if query_pts_rs is None else query_pts_rs
        query_pts_rs = query_pts
        optimizer = OccNetOptimizer(
            model,
            query_pts=query_pts,
            query_pts_real_shape=query_pts_rs,
            opt_iterations=self.args.opt_iterations,
            cfg=self.cfg.OPTIMIZER)
        optimizer.setup_meshcat(self.mc_vis)
        return optimizer

    ###########################################################################################################
    # Primitives

    def grasp(self, target_pcd, obj_class, geometry):
        '''
        Skill to grasp an object conditioned on its pointcloud, its object class and 
        an optional argument if there's a certain geometric feature to grasp at. Skill is executed

        Args:
            target_pcd (np.ndarray): N x 3 array representing the 3D point cloud of the 
                target object to be grasped, expressed in the world coordinate system
            obj_class (str): Class of the object to be grasped. Must be in self.obj_classes 
            geometry (str): Description of a geometric feature on the object to execute grasp upon
                ex. "handle" of a mug class object
        
        Return:
            list of ee poses to move to
        '''
        assert obj_class in self.obj_classes, "Object class doesn't have skills"

        demo_name = f'grasp {obj_class}_{geometry}'
        demo_dic = self.get_ndf_demo_info(demo_name, geometry)

        optimizer = self.get_optimizer(obj_class, demo_dic['query_pts'])
        optimizer.set_demo_info(demo_dic['demo_info'])
        
        pose_mats, best_idx, losses = optimizer.optimize_transform_implicit(target_pcd, ee=True, opt_visualize=True, return_score_list=True)
        
        #TODO: Add something to determine the offset - if object z is much lower that target pose, offset from top vs if xy is greater, come from the side
        
        for pose_idx in torch.argsort(torch.stack(losses)):
            potential_mat = pose_mats[pose_idx]
            corresponding_pose = util.pose_from_matrix(potential_mat)
            log_debug(f'EE Pose {pose_idx}: {corresponding_pose.pose}')
            if corresponding_pose.pose.position.z > 0:
                corresponding_pose = util.pose_from_matrix(pose_mats[pose_idx])
                break
        else:
            log_debug('Failed to find a reasonable grasp')
            return
        
        
        # corresponding_pose = util.pose_from_matrix(pose_mats[best_idx])

        # grasping requires post processing to find anti-podal point
        corresponding_pose_list = util.pose_stamped2list(corresponding_pose)
        corresponding_pose_list[:3] = post_process_grasp_point(corresponding_pose_list, 
                                                               target_pcd, 
                                                               thin_feature=(not self.args.non_thin_feature), 
                                                               grasp_viz=self.args.grasp_viz, 
                                                               grasp_dist_thresh=self.args.grasp_dist_thresh)
        corresponding_pose = util.list2pose_stamped(corresponding_pose_list)
        corresponding_pose_list = util.pose_stamped2list(corresponding_pose)
        corresponding_pose_list[:3] = post_process_grasp_point(corresponding_pose_list, 
                                                               target_pcd, 
                                                               thin_feature=(not self.args.non_thin_feature), 
                                                               grasp_viz=self.args.grasp_viz, 
                                                               grasp_dist_thresh=self.args.grasp_dist_thresh)
        corresponding_pose = util.list2pose_stamped(corresponding_pose_list)

        offset_pose = util.transform_pose(
            pose_source=corresponding_pose,
            pose_transform=util.list2pose_stamped([0, 0, 0.15, 0, 0, 0, 1])
        )
        ee_poses = [*self.get_grasp_pre_poses(corresponding_pose), corresponding_pose, offset_pose]
        free_memory([optimizer], debug=False)
        return [util.pose_stamped2list(pose) for pose in ee_poses]
    
    def get_grasp_pre_poses(self, corresponding_pose):
        corresponding_pose_list = util.pose_stamped2list(corresponding_pose)
        offset_tf1 = util.list2pose_stamped(get_ee_offset(ee_pose=corresponding_pose_list))
        offset_tf2 = util.list2pose_stamped(self.cfg.PREPLACE_VERTICAL_OFFSET_TF)

        pre_ee_end_pose2 = util.transform_pose(pose_source=util.list2pose_stamped(corresponding_pose_list), pose_transform=offset_tf1)
        pre_ee_end_pose1 = util.transform_pose(pose_source=pre_ee_end_pose2, pose_transform=offset_tf2)
        # pre_ee_end_pose1 = util.transform_pose(pose_source=pre_ee_end_pose2, pose_transform=offset_tf2) 
        return [pre_ee_end_pose1, pre_ee_end_pose2]
    
    def place_position(self, target_pcd, obj_class, geometry, position, ee_pose):        
        '''
        Skill to place an object conditioned on its pointcloud, its object class, and 
            a certain geometric feature to place at. Skill is executed

        Args:
            target_pcd (np.ndarray): N x 3 array representing the 3D point cloud of the 
                target object to be placed, expressed in the world coordinate system
            obj_class (str): Class of the object to be placed. Must be in self.obj_classes 
            geometry (str): Description of a geometric feature of object that is placed on
                ex. "shelf" on the table
            position (3x1np.array): 3D position of where object should be placed
            ee_pose (Array-like, length 3): Last pose of the EE
        
        Return:
            list of ee poses to move to
        '''
        assert obj_class in self.obj_classes, "Object class doesn't have skills"
        assert self.inital_poses, "Need to load the grasp demo first"
        
        demo_name = f'place {obj_class}_{geometry}'
        demo_dic = self.get_ndf_demo_info(demo_name, geometry)

        optimizer = self.get_optimizer(obj_class, demo_dic['query_pts'])
        optimizer.set_demo_info(demo_dic['demo_info'])
        
        pose_mats, best_idx = optimizer.optimize_transform_implicit(target_pcd, ee=False, opt_visualize=True,)
        # pose_mats, best_idx, losses = optimizer.optimize_transform_implicit(target_pcd, ee=False, opt_visualize=True, return_score_list=True)
        corresponding_pose = util.pose_from_matrix(pose_mats[best_idx])

        placement_pose = util.list2pose_stamped([position[0], position[1], position[2], 0, 0, 0, 1])
        placement2target_feat_pose_mat = np.linalg.inv(pose_mats[best_idx])
        target_pose_mat = np.matmul(placement2target_feat_pose_mat, util.matrix_from_pose(placement_pose))
        transformed_target_pose_mat = np.matmul(placement_pose, target_pose_mat)
        transformed_target_pose = util.pose_from_matrix(transformed_target_pose_mat)

        corresponding_pose = util.transform_pose(pose_source=util.list2pose_stamped(ee_pose), pose_transform=transformed_target_pose)
        ee_poses = [*self.get_place_pre_poses(corresponding_pose), corresponding_pose]
        return [util.pose_stamped2list(pose) for pose in ee_poses]
    
    def place_relative(self, target, relational, geometry, ee_pose):
        '''
        Skill to place an object relative to another relational object, and a certain geometric feature 
            to place at. Skill is executed

        Args:
            target (np.ndarray, str): (pcd, obj_class) N x 3 array representing the 3D point cloud of 
                the target object to be placed, expressed in the world coordinate system and the class of the object 
                to be placed. Must be in self.obj_classes 

            relational (np.ndarray, str): (pcd, obj_class) N x 3 array representing the 3D point cloud 
                of the relational object to be placed in relation to, expressed in the world coordinate system, 
                and the class of the object. Must be in self.obj_classes 
            geometry (str): Description of a geometric feature on the scene to execute place upon
                ex. "shelf" on the table or "on the bowl" of another bowl
            ee_pose (Array-like, length 3): Last pose of the EE
        
        Return:
            list of ee poses to move to
        '''
        target_pcd, target_class = target
        relational_pcd, relational_class = relational
        assert (target_class in self.obj_classes and relational_class in self.obj_classes), "Object class doesn't have skills"
        demo_name = f'{target_class}_{geometry}'
        demo_dic, relational_model_path, target_model_path = self.get_rndf_demo_info(demo_name)
        target_qps = demo_dic[0]['query_pts']
        relational_qps = demo_dic[1]['query_pts']
        target_optimizer = self.get_optimizer(target_class, target_qps, model_path=target_model_path)
        rel_optimizer = self.get_optimizer(relational_class, relational_qps, model_path=relational_model_path)

        target_desc, relational_desc = demo_dic[0]['target_desc'], demo_dic[1]['target_desc']
        target_query_pts, relational_query_pts = demo_dic[0]['query_pts'], demo_dic[1]['query_pts']
        
        target_ndf = (target_desc, target_pcd, target_query_pts)
        relational_ndf = (relational_desc, relational_pcd, relational_query_pts)

        corresponding_mat = infer_relation_intersection(
            self.mc_vis, rel_optimizer, target_optimizer, relational_ndf, target_ndf, 
            current_ee_pose=ee_pose, opt_visualize=self.args.opt_visualize, visualize=True)

        corresponding_pose = util.pose_from_matrix(corresponding_mat)
        #try making transform relative to object?
        corresponding_pose = util.transform_pose(pose_source=util.list2pose_stamped(ee_pose), pose_transform=corresponding_pose)
        print(f'best place pose is: {util.pose_stamped2list(corresponding_pose)}')

        ee_poses = [*self.get_place_pre_poses(corresponding_pose), corresponding_pose]
        # ee_poses = [corresponding_pose]
        # ee_poses = [*self.get_place_pre_poses(corresponding_pose)]

        free_memory([target_optimizer, rel_optimizer], debug=False)
        return [util.pose_stamped2list(pose) for pose in ee_poses]

    def get_place_pre_poses(self, corresponding_pose):
        offset_tf1 = util.list2pose_stamped(self.cfg.PREPLACE_OFFSET_TF)
        # offset_tf1 = util.list2pose_stamped(self.cfg.PREPLACE_VERTICAL_OFFSET_TF)

        offset_tf2 = util.list2pose_stamped(self.cfg.PREPLACE_VERTICAL_OFFSET_TF)
        pre_ee_end_pose2 = util.transform_pose(pose_source=corresponding_pose, pose_transform=offset_tf1)
        pre_ee_end_pose1 = util.transform_pose(pose_source=pre_ee_end_pose2, pose_transform=offset_tf2) 
        return [pre_ee_end_pose1, pre_ee_end_pose2]
        # return [pre_ee_end_pose2]
    
    def learn_skill(self, skill_name, target_pcd, obj_class, **kwargs):
        '''
        Skill to learn a new skill from a human demonstration

        Args:
            skill_name (str): Short name for the skill to be learned ideally in the form 
            'action [obj class]_[obj geometry]' where action is in {grasp, place} and geometry 
            is any additional description of the skill
            target (np.ndarray, str): (pcd, obj_class) N x 3 array representing the 3D point cloud of 
                the target object that the new skill is executed on, expressed in the world coordinate 
                system
            obj_class (str): Class of the object to be executed on. Must be in self.obj_classes 
        
        Return:
            the function skill_name that can perform the skill given by "skil_name" and takes as input
                the mentioned parameters. This skill is added permanently to skill library.
        '''
        skill_type = None
        if 'grasp' in skill_name:
            skill_type = 'grasp'
        elif 'relational_pcd' in kwargs:
            skill_type = 'place_relative'
        elif 'position' in kwargs:
            skill_type = 'place_position'
        else:
            raise NotImplementedError(f'Unsure what skill type {skill_name} is')
        
        while True:
            i = input(
                '''How to learn a new skill
                    [p]: Get demos from path
                    [r]: Record a demo live
                    [b]: Exit - Fail to learn a skill
                ''')
            if i == 'p':
                demo_dir_path = input('Please enter the path to the directory')
                break
            elif i == 'r':
                raise NotImplementedError('Recording a new demo live has not been implemented')
            elif i == 'b':
                return 
        
        if not osp.exists(demo_dir_path):
            log_warn(f'{demo_dir_path} path does not exist, failed to create skill')
            return
        
        self.add_demo_dir(skill_name, demo_dir_path)
        geometry = skill_name.split('_')[-1]
        self.ndf_primitives[skill_type]['obj_classes'].add(obj_class)
        self.ndf_primitives[skill_type]['geometry'].add((obj_class, geometry))
        
        skill_func = self.ndf_primitives[skill_type]
        if skill_type == 'grasp':
            return skill_func(target_pcd, obj_class, geometry)
        elif skill_type == 'place_relative':
            return skill_func((target_pcd, obj_class), kwargs['relational'], geometry, kwargs['ee_pose'])
        elif skill_type == 'place_position':
            return skill_func((target_pcd, obj_class), geometry, kwargs['position'], kwargs['ee_pose'])
        else:
            raise NotImplementedError('This skill type does not exist, learning failed')

###########################################################################################################
# RNDF Descriptor Creation

NOISE_VALUE_LIST = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.16, 0.24, 0.32, 0.4]

def read_rndf_demos(self, demo_name, n=None):
    demo_path = [osp.join(path_util.get_rndf_demos(), self.folder, demo_name)]
    assert osp.exists(demo_path), f'{demo_path} does not exist'
    demo_dic = {0: {}, 1: {}}

    for obj_rank in demo_dic.keys():
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
            if 'multi_object_ids' in demo:
                obj_ids = demo['multi_object_ids'].item()[demo_key]
                start_pose = demo['multi_obj_start_obj_pose'].item()[demo_key]
                demo_dic[obj_rank]['demo_ids'].append(obj_ids)
                demo_dic[obj_rank]['demo_start_poses'].append(start_pose)

            demo_dic[obj_rank]['demo_start_pcds'].append(s_pcd)
            demo_dic[obj_rank]['demo_final_pcds'].append(f_pcd)
    return demo_dic

def create_rndf_descriptior(self, demo_name, demo_dic, parent_model, child_model, args):
    demo_path = osp.join(path_util.get_rndf_demos(), self.folder, demo_name)
    parent_model_path =  osp.join(path_util.get_rndf_model_weights(), parent_model)
    child_model_path =  osp.join(path_util.get_rndf_model_weights(), child_model)
    
    target_desc_subdir = get_rndf_desc_subdir(demo_path, parent_model_path, child_model_path)
    target_desc_fname = osp.join(demo_path, target_desc_subdir, demo_name)
    if args.relation_method == 'intersection':
        if not osp.exists(target_desc_fname) or args.new_descriptors:
            util.safe_makedirs(target_desc_subdir)

            print(f'\n\n\nCreating target descriptors for this parent model + child model, and these demos\nSaving to {target_desc_fname}\n\n\n')
            n_demos = 'all' if args.n_demos < 1 else args.n_demos
            if args.add_noise:
                add_noise = True
                noise_value = NOISE_VALUE_LIST[args.noise_idx]
            else:
                add_noise = False
                noise_value = 0.0001

            create_target_descriptors(
                parent_model, child_model, demo_dic, target_desc_fname, 
                self.cfg, query_scale=args.query_scale, scale_pcds=False, 
                target_rounds=args.target_rounds, pc_reference=args.pc_reference,
                skip_alignment=args.skip_alignment, n_demos=n_demos, manual_target_idx=args.target_idx, 
                add_noise=add_noise, interaction_pt_noise_std=noise_value,
                use_keypoint_offset=args.use_keypoint_offset, keypoint_offset_params=args.keypoint_offset_params,
                visualize=True, mc_vis=args.mc_vis)

###########################################################################################################
# Private helpers

def load_model(model_path):
    full_model_path = osp.join(path_util.get_rndf_model_weights(), model_path)
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    model.load_state_dict(torch.load(full_model_path))
    return model

def get_model_paths_for_dir(descriptor_dirname):
    _, parent_model_path, child_model_path = descriptor_dirname.split(DELIM)
    parent_model_path = parent_model_path.split('_child')[0]
    parent_model_path = 'ndf_vnn/' + parent_model_path + '.pth'
    child_model_path = 'ndf_vnn/' + child_model_path + '.pth'
    return parent_model_path, child_model_path

def find_rndf_descriptor_models(demo_path):
    alt_descs = os.listdir(demo_path)
    if not alt_descs: 
        print('There are no descriptors for this concept. Please generate descriptors first')
        return
    
    all_model_paths = []
    for alt_desc in alt_descs:
        if not alt_desc.endswith('.npz'):
            model_paths = get_model_paths_for_dir(alt_desc)
            all_model_paths.append(model_paths)
    return all_model_paths

def get_rndf_desc_subdir(demo_path, parent_model_path, child_model_path):
    parent_model_name_full = parent_model_path.split('ndf_vnn/')[-1]
    child_model_name_full = child_model_path.split('ndf_vnn/')[-1]

    parent_model_name_specific = parent_model_name_full.split('.pth')[0].replace('/', '--')
    child_model_name_specific = child_model_name_full.split('.pth')[0].replace('/', '--')
    
    subdir_name = f'parent_model{DELIM}{parent_model_name_specific}_child{DELIM}{child_model_name_specific}'
    dirname = osp.join(demo_path, subdir_name)
    return dirname

if __name__ == "__main__":
    path = osp.join(path_util.get_rndf_demos())
    library = NDFLibrary(None, None, None, path_util.get_rndf_demos())
    library.learn_skill('grasp mug_body', None, 'mug')