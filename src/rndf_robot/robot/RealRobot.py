import os, os.path as osp
import time
import torch
import trimesh

from polymetis import GripperInterface, RobotInterface

from rndf_robot.robot.franka_ik import FrankaIK
from rndf_robot.utils import util, path_util
from rndf_robot.utils.real.traj_util import PolymetisTrajectoryUtil
from rndf_robot.utils.real.plan_exec_util import PlanningHelper
from rndf_robot.utils.real.perception_util import enable_devices
from rndf_robot.utils.real.polymetis_util import PolymetisHelper

from .Robot import Robot

from airobot import log_debug, log_warn, log_info
poly_util = PolymetisHelper()

class RealRobot(Robot):

    #################################################################################################
    # Setup

    def setup_robot(self):
        self.ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], robotiq=(self.args.gripper_type=='2f140'), mc_vis=self.mc_vis)
        # self.ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0])
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
        self.ik_helper.register_object(
            cam2_obs_fname,
            pos=[0.55, -0.525, 0.25],  # pos=[0.135, -0.525, 0.25],
            ori=[0, 0, 0, 1],
            name='cam1')

    #################################################################################################
    # Controls

    def gripper_state(self, open=True):
        if open:
            self.planning.gripper_open()
            self.gripper_is_open = True
        else:
            self.planning.gripper_close()
            self.gripper_is_open = False

    def get_ee_pose(self):
        return poly_util.polypose2np(self.panda.get_ee_pose())

    def get_jpos(self):
        return poly_util.polypose2np(self.panda.get_joint_positions())
    
    def gravity_comp(self, on=True):
        if on:
            print('\n\nSetting low stiffness in current pose, you can now move the robot')
            # panda.set_cart_impedance_pose(panda.endpoint_pose(), stiffness=[0]*6)
            self.panda.start_cartesian_impedance(Kx=torch.zeros(6), Kxd=torch.zeros(6))
        else:
            print('\n\nSetting joint positions to current value\n\n')
            self.panda.start_joint_impedance()
            # panda.start_joint_impedance(Kq=Kq_new, Kqd=Kqd_new)

    # Motion Planning

    def go_home(self):
        current_panda_plan = self.planning.plan_home()
        self.planning.execute_pb_loop(current_panda_plan)            
        i = input('Take it home (y/n)?')
        if i == 'y':
            self.planning.execute_loop(current_panda_plan)    

    def execute_pre_step(self):
        if self.state == 0:
            if not self.gripper_is_open:
                self.planning.gripper_open()
                self.gripper_is_open = True

    def execute_post_step(self):
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

    def find_iks(self, ee_poses, j, place):
        colors = [16751514, 6207853, 15503192, 10264062, 14064455, 8415593, 48797, 16416176, 8785443, 16757683, 16751514]

        for i, ee_pose in enumerate(ee_poses):
            pose = util.body_world_yaw(util.list2pose_stamped(ee_pose), theta=0)
            pose = util.matrix_from_pose(pose)
            util.meshcat_obj_show(self.mc_vis, self.ee_file, pose, 1.0, name=f'ee/ee_{i}_alt.{j}', color=colors[j])

        jnt_poses = []
        for i, ee_pose in enumerate(ee_poses):
            jnt_pose = self.cascade_ik(ee_pose, place=place)
            if jnt_pose is None:
                jnt_poses.append(None)
                return jnt_poses
            jnt_poses.append(jnt_pose)
        return jnt_poses

    def execute(self, ee_poses, place=False):
        jnt_poses = self.find_iks(ee_poses, 0, place)
        if None in jnt_poses:
            rotated_ee_poses = self.resample_place(util.list2pose_stamped(ee_poses[-1]))

            for i in range(6):
                jnt_poses = self.find_iks(rotated_ee_poses, i, place)
                if None not in jnt_poses:
                    break

        if None in jnt_poses:
            log_warn('Could not find IKs therefore stopping motion plan')
            return
        
        start_pose = None
        joint_traj = []
        for jnt_pose in jnt_poses:
            if start_pose is None:
                resulting_traj = self.planning.plan_joint_target(joint_position_desired=jnt_pose, 
                            from_current=True, 
                            start_position=None, 
                            show_pb=False,
                            execute=False)
            else:
                resulting_traj = self.planning.plan_joint_target(joint_position_desired=jnt_pose, 
                                                from_current=False, 
                                                start_position=start_pose, 
                                                show_pb=False,
                                                execute=False)
            start_pose = jnt_pose
            if resulting_traj is None:
                break
            else:
                joint_traj.append(resulting_traj)
        else:
            self.execution_prompt(joint_traj)

    #################################################################################################
    # Other Helpers

    def teleport(self, obj_pcd, relative_pose):
        transform = util.matrix_from_pose(util.list2pose_stamped(relative_pose[0]))

        final_pcd = util.transform_pcd(obj_pcd, transform)
        util.meshcat_pcd_show(self.mc_vis, final_pcd, color=(255, 0, 255), name=f'scene/teleported_obj')

    def move_robot(self, plan):
        for jnt in plan:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan[-1], wait=True)

    def execution_prompt(self, joint_traj):
        while True:
            i = input(
                '''What should we do
                    [s]: Run in sim
                    [e]: Execute on robot
                    [b]: Exit
                ''')
            
            if i == 's':
                for traj in joint_traj:
                    self.planning.execute_pb_loop(traj)
                continue
            elif i == 'e':
                confirm = input('Should we execute (y/n)???')
                if confirm == 'y':
                    self.execute_pre_step()
                    for i, traj in enumerate(joint_traj):
                        if i == len(joint_traj)-1 and self.state == 0:
                            self.execute_post_step()
                            time.sleep(0.2)

                        self.planning.execute_loop(traj)
                    if self.state != 0:
                        self.execute_post_step()
                    break
            elif i == 'b':
                break
            else:
                print('Unknown command')
                continue

    