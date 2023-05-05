import os, os.path as osp
import time
import numpy as np
import pybullet as p
from rndf_robot.data.NDFLibrary import NDFLibrary

from rndf_robot.robot.franka_ik import FrankaIK
from rndf_robot.utils import util, path_util
from rndf_robot.utils.pipeline_util import (
    safeCollisionFilterPair, 
    soft_grasp_close, 
    object_is_still_grasped, 
    constraint_obj_world
)
from .Robot import Robot as RobotParent

from airobot import Robot, log_debug, log_warn, log_info
from IPython import embed

class SimRobot(RobotParent):

    #################################################################################################
    # Setup

    def setup_robot(self):
        self.robot = Robot('franka', pb_cfg={'gui': self.args.pybullet_viz}, arm_cfg={'self_collision': False, 'seed': self.args.seed})
        self.ik_helper = FrankaIK(gui=False)
        self.finger_joint_id = 9
        self.left_pad_id = 9
        self.right_pad_id = 10

        p.changeDynamics(self.robot.arm.robot_id, self.left_pad_id, lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, self.right_pad_id, lateralFriction=1.0)

        # reset
        self.robot.arm.reset(force_reset=True)

    def setup_table(self):            
        table_fname = osp.join(path_util.get_rndf_descriptions(), 'hanging/table')
        table_urdfs = {'default': 'table_manual.urdf', 'rack': 'table_rack_manual.urdf', 'shelf': 'table_shelf_manual.urdf', 'peg': 'table_peg_manual.urdf'}
        table_type = self.args.table_type
        table_urdf_file = table_urdfs[table_type]
        if table_type == 'shelf':
            self.table_obj = 0
        elif table_type == 'rack' or table_type == 'peg':
            self.table_obj = 1
        table_fname = osp.join(table_fname, table_urdf_file)

        self.table_id = self.robot.pb_client.load_urdf(table_fname,
                                self.cfg.TABLE_POS, 
                                self.cfg.TABLE_ORI,
                                scaling=1.0)
        if table_type != 'manual':
            self.table_obj_link_id = 0
            place_color = p.getVisualShapeData(self.table_id)[0][7]
        safeCollisionFilterPair(self.robot.arm.robot_id, self.table_id, -1, -1, enableCollision=True)

        time.sleep(3.0)

    # Controls
    def gripper_state(self, open=True):
        if open:
            self.robot.arm.eetool.open()
            self.gripper_is_open = True
        else:
            soft_grasp_close(self.robot, self.finger_joint_id, force=40)
            self.gripper_is_open = False
    
    def get_ee_pose(self):
        return np.concatenate(self.robot.arm.get_ee_pose()[:2]).tolist()

    def get_jpos(self):
        return self.robot.arm.get_jpos()
    #################################################################################################
    # PyBullet Things
    def delete_scene(self, obj_ids):
        for obj_id in obj_ids:
            self.robot.pb_client.remove_body(obj_id)

    def add_obj(self, obj_file, scale_default, upright_ori, existing_objs=[], color=None):
        x_low, x_high = self.cfg.OBJ_SAMPLE_X_HIGH_LOW
        y_low, y_high = self.cfg.OBJ_SAMPLE_Y_HIGH_LOW

        self.placement_link_id = 0 

        mesh_scale=[scale_default] * 3

        pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, self.cfg.TABLE_Z]
        log_debug('original: %s' %pos)
        for obj_id, obj in existing_objs.items():
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

        if False:
            if self.test_obj in ['bowl', 'bottle']:
                rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            else:
                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()
        else:
            ori = upright_ori

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
        safeCollisionFilterPair(obj_id, self.robot.arm.robot_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, lateralFriction=0.5, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        obj_pose_world_list = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world_list[0]) + list(obj_pose_world_list[1]))
        return obj_id, obj_pose_world


    #################################################################################################
    # Motion Planning
    
    def go_home(self):
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

    def execute(self, ee_poses):
        self.execute_pre_step()
        self.execute_traj(ee_poses)
        if self.state != 0:
            self.execute_post_step()
        
    def execute_pre_step(self, obj_id=None):
        if obj_id:
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=self.table_id, linkIndexA=-1, linkIndexB=self.table_obj_link_id, enableCollision=True)
            if self.state == 0:
                time.sleep(0.5)
                # turn OFF collisions between robot and object / table, and move to pre-grasp pose
                for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=self.table_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=self.robot.pb_client.get_client_id())
        
        if self.state == 0:
            self.gripper_state(open=True)

            time.sleep(0.5)

    def execute_post_step(self, obj_id=None):
        if self.state == 0:
            # turn ON collisions between robot and object, and close fingers
            if obj_id:
                for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=True, physicsClientId=self.robot.pb_client.get_client_id())

            time.sleep(0.8)
            jnt_pos_before_grasp = self.get_jpos()
            self.gripper_state(open=False)
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
            release = input('Press o to open end effector or Enter to continue')
            if release == 'o':
                if obj_id:
                    # turn ON collisions between object and rack, and open fingers
                    safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
                    safeCollisionFilterPair(obj_id, self.table_id, -1, self.placement_link_id, enableCollision=True)
                    
                    p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
                    grasp_success = object_is_still_grasped(self.robot, obj_id, self.right_pad_id, self.left_pad_id) 
                self.gripper_state(open=True)


                self.state = -1

                self.robot.arm.move_ee_xyz([0, 0.075, 0.075])
                time.sleep(1.0)
            else:
                self.state = 0

    def execute_traj(self, ee_poses):
        for i, ee_pose in enumerate(ee_poses):
            pose = util.body_world_yaw(util.list2pose_stamped(ee_pose), theta=-1.5708)
            pose = util.matrix_from_pose(pose)
            util.meshcat_obj_show(self.mc_vis, self.ee_file, pose, 1.0, name=f'ee/ee_{i}')

        jnt_poses = [self.cascade_ik(pose) for pose in ee_poses]
        prev_pos = self.get_jpos()
        for i, jnt_pos in enumerate(jnt_poses):
            if jnt_pos is None:
                log_warn('No IK for jnt', i)
                break
            
            plan = self.ik_helper.plan_joint_motion(prev_pos, jnt_pos)
            if plan is None:
                log_warn('FAILED TO FIND A PLAN. STOPPING')
                break
            if i == len(jnt_poses)-1 and self.state == 0:
                self.execute_post_step()
                time.sleep(0.2)
                
            self.move_robot(plan)
            prev_pos = jnt_pos
    
    #################################################################################################
    # Other Helpers

    def teleport(self, obj_id, pcd, relative_pose):
        transform = util.matrix_from_pose(util.list2pose_stamped(relative_pose[0]))
        start_pose = np.concatenate(self.robot.pb_client.get_body_state(obj_id)[:2]).tolist()
        start_pose_mat = util.matrix_from_pose(util.list2pose_stamped(start_pose))
        final_pose_mat = np.matmul(transform, start_pose_mat)
        self.robot.pb_client.set_step_sim(True)
        final_pose_list = util.pose_stamped2list(util.pose_from_matrix(final_pose_mat))
        final_pos, final_ori = final_pose_list[:3], final_pose_list[3:]
        time.sleep(1.0)

        self.robot.pb_client.reset_body(obj_id, final_pos, final_ori)

        final_pcd = util.transform_pcd(pcd, transform)
        util.meshcat_pcd_show(self.mc_vis, final_pcd, color=[255, 0, 255], name=f'scene/{obj_id}_pcd')
        time.sleep(3.0)

        # turn on the physics and let things settle to evaluate success/failure
        self.robot.pb_client.set_step_sim(False)

    # def cascade_ik(self, ee_pose):
    #     jnt_pos = None
    #     if jnt_pos is None:
    #         jnt_pos = self.ik_helper.get_feasible_ik(ee_pose, verbose=False)
    #         if jnt_pos is None:
    #             jnt_pos = self.ik_helper.get_ik(ee_pose)
    #             if jnt_pos is None:
    #                 jnt_pos = self.robot.arm.compute_ik(ee_pose[:3], ee_pose[3:])
    #     if jnt_pos is None:
    #         log_warn('Failed to find IK')
    #     return jnt_pos
    
    def move_robot(self, plan):
        for jnt in plan:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan[-1], wait=True)