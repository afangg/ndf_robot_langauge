import os, os.path as osp
import time

from rndf_robot.robot.franka_ik import FrankaIK
from rndf_robot.data.NDFLibrary import NDFLibrary
from rndf_robot.utils import util, path_util

from airobot import log_debug, log_warn, log_info

class Robot:
    def __init__(self, skill_library: NDFLibrary, mc_vis) -> None:
        self.skill_library = skill_library
        self.mc_vis = mc_vis
        #self.ik_helper = FrankaIK(gui=False)
        self.robot = None #AIRobot Client?
        self.last_ee_pose = None
        self.FUNCTIONS = {'grasp': self.grasp, 'place': self.place, 'place_relative': self.place_relative, 'find': self.find}

    def get_jpos():
        pass

    def get_iks(self, ee_poses):
        return [self.cascade_ik(pose) for pose in ee_poses]
    
    def cascade_ik(self, ee_pose):
        jnt_pos = None
        if jnt_pos is None:
            jnt_pos = self.ik_helper.get_feasible_ik(ee_pose, verbose=False)
            if jnt_pos is None:
                jnt_pos = self.ik_helper.get_ik(ee_pose)
                if jnt_pos is None:
                    jnt_pos = self.robot.arm.compute_ik(ee_pose[:3], ee_pose[3:])
        if jnt_pos is None:
            log_warn('Failed to find IK')
        return jnt_pos
    
    def move_robot(self, plan):
        for jnt in plan:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan[-1], wait=True)

    def execute_pre_step(self):
        # probably open gripper
        pass

    def execute_post_step(self):
        # close gripper
        pass

    def execute_traj(self, ee_poses):
        ee_file = osp.join(path_util.get_llm_descriptions(), 'franka_panda/meshes/robotiq_2f140/full_hand_2f140.obj')
        for i, ee_pose in enumerate(ee_poses):
            pose = util.body_world_yaw(util.list2pose_stamped(ee_pose), theta=-1.5708)
            pose = util.matrix_from_pose(pose)
            util.meshcat_obj_show(self.mc_vis, ee_file, pose, 1.0, name=f'ee/ee_{i}')

        jnt_poses = self.get_iks(ee_poses)
        prev_pos = self.get_jpos()
        for i, jnt_pos in enumerate(jnt_poses):
            if jnt_pos is None:
                log_warn('No IK for jnt', i)
                break
            
            plan = self.ik_helper.plan_joint_motion(prev_pos, jnt_pos)
            if plan is None:
                log_warn('FAILED TO FIND A PLAN. STOPPING')
                break
            self.move_robot(plan)
            prev_pos = jnt_pos

    def grasp(self, target_pcd, target_class, geometry):
        '''
        Skill to grasp an object conditioned on its pointcloud, its object class and 
        an optional argument if there's a certain geometric feature to grasp at. Skill is executed

        Args:
            target_pcd (np.ndarray): N x 3 array representing the 3D point cloud of the 
            target object to be grasped, expressed in the world coordinate system
            target_class (str): Class of the object to be grasped. Must be in self.obj_classes 
            geometry (str): Description of a geometric feature on the object to execute grasp upon
                ex. "handle" of a mug class object
        
        Return:
            new_position (Array-like, length 3): Final position of the EE, after performing the move action.
                If grasp fails, returns None
        '''
        ee_poses = self.skill_library.grasp(target_pcd, target_class, geometry)
        self.execute_pre_step()
        sucess = self.execute_traj(ee_poses)
        self.execute_post_step()
        # self.ee_poses = ee_poses[-1] if sucess else None
        return self.last_ee_pose
    
    def place(self, target_pcd, target_class, geometry):
        '''
        Skill to place an object conditioned on its pointcloud, its object class, and 
            a certain geometric feature to place at. Skill is executed

        Args:
            target_pcd (np.ndarray): N x 3 array representing the 3D point cloud of the 
            target object to be placed, expressed in the world coordinate system
            target_class (str): Class of the object to be placed. Must be in self.obj_classes 
            geometry (str): Description of a geometric feature on the scene to execute place upon
                ex. "shelf" on the table
        
        Return:
            new_position (Array-like, length 3): Final position of the EE, after performing the move action.
                If grasp fails, returns None
        '''
        ee_poses = self.skill_library.place(target_pcd, target_class, geometry)
        self.execute_pre_step()
        sucess = self.execute_traj(ee_poses)
        self.execute_post_step()
        # self.last_ee_pose = ee_poses[-1] if sucess else None
        return self.last_ee_pose
    
    def place_relative(self, target, relational, geometry):
        '''
        Skill to place an object relative to another relational object, and a certain geometric feature 
            to place at. Skill is executed

        Args:
            target (np.ndarray, str): N x 3 array representing the 3D point cloud of the  arget object 
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
        ee_poses = self.skill_library.place_relative(target, relational, geometry, self.ee_pose)
        self.execute_pre_step()
        sucess = self.execute_traj(ee_poses)
        self.execute_post_step()
        # self.last_ee_pose = ee_poses[-1] if sucess else None
        return self.last_ee_pose
    