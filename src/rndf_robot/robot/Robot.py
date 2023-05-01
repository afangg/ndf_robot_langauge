import torch

from rndf_robot.data.NDFLibrary import NDFLibrary
from IPython import embed

class Robot:
    def __init__(self, args, skill_library: NDFLibrary, mc_vis) -> None:
        self.args = args
        self.skill_library = skill_library
        self.mc_vis = mc_vis
        self.gripper_is_open = False
        self.state =-1 # -1 home, 0 to grasp, 1 to place, # 2 to teleport 

        self.last_ee_pose = None
        self.FUNCTIONS = {'grasp': self.grasp, 'place': self.place, 'place_relative': self.place_relative, 'find': self.find}
        self.go_home()

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

    def gravity_comp(self, on=True):
        raise NotImplementedError('Gravity comp mode not available')

    # Motion Plan

    def go_home(self):
        pass

    def execute_pre_step(self):
        pass

    def execute_post_step(self):
        pass

    def execute_traj(self, ee_poses):
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
        self.execute_pre_step()
        sucess = self.execute_traj(ee_poses)
        self.execute_post_step()
        # self.ee_poses = ee_poses[-1] if sucess else None
        return self.last_ee_pose
    
    def place(self, target, geometry):
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
        target_pcd, target_class = target
        ee_poses = self.skill_library.place(target_pcd, target_class, geometry, self.get_ee_pose())
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
        ee_poses = self.skill_library.place_relative(target, relational, geometry, self.ee_pose)
        self.execute_pre_step()
        sucess = self.execute_traj(ee_poses)
        self.execute_post_step()
        # self.last_ee_pose = ee_poses[-1] if sucess else None
        return self.last_ee_pose