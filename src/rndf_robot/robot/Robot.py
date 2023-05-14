import os, os.path as osp
from rndf_robot.data.NDFLibrary import NDFLibrary
from rndf_robot.utils import path_util, util
from airobot import log_debug, log_warn, log_info
import numpy as np
from IPython import embed

class Robot:
    def __init__(self, args, skill_library: NDFLibrary, mc_vis, cfg=None) -> None:
        self.args = args
        self.skill_library = skill_library
        self.mc_vis = mc_vis
        self.cfg = cfg
        self.table_id = None
        self.use_diff = False

        if self.args.gripper_type == 'panda':
            self.ee_file = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/panda_hand_full.obj')
        elif self.args.gripper_type == '2f140':
            self.ee_file = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/robotiq_2f140/full_hand_2f140.obj')

        self.gripper_is_open = False
        self.state =-1 # -1 home, 0 to grasp, 1 to place, # 2 to teleport 

        self.last_ee_pose = None

        self.setup_robot()
        self.setup_table()

    #################################################################################################
    # Setup
    
    def setup_robot(self):
        pass

    def setup_table(self):   
        pass

    def delete_scene(self, obj_ids):
        pass

    # Controls
    def gripper_state(self, open=True):
        pass

    def get_ee_pose(self):
        pass

    def get_jpos(self):
        pass

    def gravity_comp(self, on=True):
        raise NotImplementedError('Gravity comp mode not available')

    # Motion Plan

    def resample_place(self, corresponding_pose):
        corresponding_pose_list = util.pose_stamped2list(corresponding_pose)
        T = util.rand_body_yaw_transform(corresponding_pose_list[:3], min_theta=0.78)
        new_corresponding = util.transform_pose(corresponding_pose, util.pose_from_matrix(T))
        new_ee_poses = [*self.get_place_pre_poses(new_corresponding), new_corresponding]
        return [util.pose_stamped2list(pose) for pose in new_ee_poses]
    
    def get_place_pre_poses(self, corresponding_pose):
        offset_tf1 = util.list2pose_stamped(self.cfg.PREPLACE_OFFSET_TF)
        # offset_tf1 = util.list2pose_stamped(self.cfg.PREPLACE_VERTICAL_OFFSET_TF)

        # offset_tf2 = util.list2pose_stamped(self.cfg.PREPLACE_VERTICAL_OFFSET_TF)
        pre_ee_end_pose2 = util.transform_pose(pose_source=corresponding_pose, pose_transform=offset_tf1)
        # pre_ee_end_pose1 = util.transform_pose(pose_source=pre_ee_end_pose2, pose_transform=offset_tf2) 
        # return [pre_ee_end_pose1, pre_ee_end_pose2]
        return [pre_ee_end_pose2]
    
    def cascade_ik(self, ee_pose, place=False):
        jnt_pos = None

        if jnt_pos is None:
            jnt_pos = self.ik_helper.get_feasible_ik(ee_pose, verbose=False)
            if jnt_pos is None:
                jnt_pos = self.ik_helper.get_ik(ee_pose)
        if jnt_pos is not None:
            return jnt_pos
        
        if not place and jnt_pos is None:
            rotated_ee_mat = util.rotate_grasp(util.matrix_from_pose(util.list2pose_stamped(ee_pose)), np.pi)
            rotated_ee = util.pose_stamped2list(util.pose_from_matrix(rotated_ee_mat))
            if jnt_pos is None:
                jnt_pos = self.ik_helper.get_feasible_ik(rotated_ee, verbose=False)
                if jnt_pos is None:
                    jnt_pos = self.ik_helper.get_ik(rotated_ee)

        if jnt_pos is None:
            log_warn('Failed to find IK')
        return jnt_pos

    def go_home(self):
        pass

    def execute(self, ee_poses, place=False):
        pass

    #################################################################################################
    # Primitive Skills

    def grasp(self, target, geometry):
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
        '''
        target_pcd, target_class = target
        ee_poses = self.skill_library.grasp(target_pcd, target_class, geometry)
        if ee_poses is None:
            return
        self.execute(ee_poses)
        self.last_ee_pose = ee_poses[-1]
        return self.last_ee_pose
    
    def place_position(self, target, geometry, position):
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
        '''
        #TODO: Use custom query points!
        target_pcd, target_class = target

        if self.last_ee_pose is None:
            self.last_ee_pose = self.get_ee_pose()
        ee_poses = self.skill_library.place_position(target_pcd, target_class, geometry, position, self.last_ee_pose)
        self.execute(ee_poses, place=True)
        # self.last_ee_pose = ee_poses[-1] if sucess else None
        return self.last_ee_pose
    
    def place_relative(self, target, relational, geometry):
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
        '''
        if self.last_ee_pose is None:
            self.last_ee_pose = self.get_ee_pose()
        ee_poses = self.skill_library.place_relative(target, relational, geometry, self.last_ee_pose)
        if ee_poses is None:
            return
        self.execute(ee_poses, place=True)
        # self.last_ee_pose = ee_poses[-1] if sucess else None
        return self.last_ee_pose

    def learned_skill(self, skill_name, target, **kwargs):
        target_pcd, target_class = target
        if self.last_ee_pose is None:
            self.last_ee_pose = self.get_ee_pose()
        kwargs['ee_pose'] = self.last_ee_pose
        ee_poses, skill = self.skill_library.learn_skill(skill_name, target_pcd, target_class, kwargs)
        if ee_poses is None:
            return
        self.execute(ee_poses)
        return skill
