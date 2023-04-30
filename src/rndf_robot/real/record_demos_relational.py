import os.path as osp
import time
import sys
import signal
import random
import cv2
import numpy as np
import torch
import argparse
import copy
import meshcat
import trimesh
import open3d
from matplotlib import pyplot as plt

import pyrealsense2 as rs

from polymetis import GripperInterface, RobotInterface

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

from rndf_robot.utils import util, path_util
from rndf_robot.utils.plotly_save import plot3d, plotly_scene_dict
from rndf_robot.utils.visualize import PandaHand, Robotiq2F140Hand
from rndf_robot.utils.record_demo_utils import DefaultQueryPoints, manually_segment_pcd, convert_wrist2tip
from rndf_robot.utils.relational_utils import ParentChildObjectManager

from rndf_robot.utils.franka_ik import FrankaIK #, PbPlUtils
from rndf_robot.robot.simple_multicam import MultiRealsenseLocal

from rndf_robot.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rndf_robot.utils.real.traj_util import PolymetisTrajectoryUtil
from rndf_robot.utils.real.plan_exec_util import PlanningHelper
from rndf_robot.utils.real.perception_util import RealsenseInterface, enable_devices
from rndf_robot.utils.real.polymetis_util import PolymetisHelper

poly_util = PolymetisHelper()

from rndf_robot.system.system_utils.SAM import get_mask_from_pt
from IPython import embed

click_x, click_y = None, None
def select_pt(image, child=True):
    global click_x, click_y
    
    if child:
        print('Recorded child mask at')
    else:
        print('Recorded parent mask at')
    def mouse_event(event):
        global click_x, click_y

        print('x: {} and y: {}'.format(event.xdata, event.ydata))
        click_x, click_y = event.xdata, event.ydata
        fig.canvas.mpl_disconnect(cid)

    fig = plt.figure()
    click_x, click_y = None, None
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

    plt.imshow(image)
    plt.show()
    if click_x is None or click_y is None:
        fig.canvas.mpl_disconnect(cid)
        return None
    return np.array([click_x,click_y])

def segment(pcd, top_n=1, eps=0.005, min_points=50, min_z=None):
    pcd_o3d = open3d.geometry.PointCloud()
    pcd_o3d.points = open3d.utility.Vector3dVector(pcd)
    # labels = np.array(pcd_o3d.cluster_dbscan(eps=0.005, min_points=50, print_progress=True))
    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    # labels = np.array(pcd_o3d.cluster_dbscan(eps=0.003, min_points=30, print_progress=True))

    clusters_detected = np.unique(labels)
    pcd_clusters = []
    cluster_sizes = []
    for seg_idx in clusters_detected:
        seg_inds = np.where(labels == seg_idx)[0]
        # above_table = np.where(proc_pcd[:, 2] > 0.01)
        # mask = np.logical_and(seg_inds, above_table)
        cluster = pcd[seg_inds]
        pcd_clusters.append(cluster)
        sz = cluster.shape[0]
        cluster_sizes.append(sz)
    topNsz = np.argsort(cluster_sizes)[-top_n:]

    clusters = [pcd_clusters[topNsz[i]] for i in range(top_n)]
    topNclusters = np.concatenate(clusters, axis=0)

    proc_pcd = copy.deepcopy(topNclusters)
    return proc_pcd
def main(args):
    #############################################################################
    # generic setup

    np.random.seed(args.seed)
    random.seed(args.seed)
    signal.signal(signal.SIGINT, util.signal_handler)

    cfg = util.AttrDict(dict())

    multi_object_class = f'parent_{args.parent_class}_child_{args.child_class}'
    parent_class = args.parent_class
    child_class = args.child_class

    demo_save_dir = osp.join(path_util.get_rndf_data(), args.demo_save_dir, 'multi_object', multi_object_class, args.exp)
    util.safe_makedirs(demo_save_dir)

    # TODO: decide on a good default save location for the demo files
    parent_save_dir = osp.join(path_util.get_rndf_data(), args.demo_save_dir, 'multi_object', parent_class, args.exp)
    util.safe_makedirs(parent_save_dir)
    child_save_dir = osp.join(path_util.get_rndf_data(), args.demo_save_dir, 'multi_object', child_class, args.exp)
    util.safe_makedirs(child_save_dir)
    single_save_dirs = dict(parent=parent_save_dir, child=child_save_dir)

    pcd_save_dir = osp.join(path_util.get_rndf_data(), args.demo_save_dir, 'multi_object', multi_object_class, args.exp, 'pcd_observations')
    util.safe_makedirs(pcd_save_dir)

    #############################################################################
    # Panda, IK, trajectory helper, planning + execution helper
    
    mc_vis = meshcat.Visualizer(zmq_url=f'tcp://127.0.0.1:{args.port_vis}')
    mc_vis['scene'].delete()

    # ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], robotiq=(args.gripper_type=='2f140'), mc_vis=mc_vis)
    ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], occnet=False, robotiq=(args.gripper_type=='2f140'), mc_vis=mc_vis)
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

    ik_helper.register_object(
        table_obs_fname,
        pos=[0.15 + 0.77/2.0, 0.0, 0.0015],
        ori=[0, 0, 0, 1],
        name='table')
    # ik_helper.register_object(
    #     cam1_obs_fname,
    #     pos=[0.135, 0.55, 0.1],
    #     ori=[0, 0, 0, 1],
    #     name='cam1')
    ik_helper.register_object(
        cam1_obs_fname,
        pos=[0.0, -0.525, 0.25],  # pos=[0.135, -0.525, 0.25],
        ori=[0, 0, 0, 1],
        name='cam2')

    franka_ip = "173.16.0.1" 
    panda = RobotInterface(ip_address=franka_ip)
    gripper = GripperInterface(ip_address=franka_ip)

    traj_helper = PolymetisTrajectoryUtil(robot=panda)
    planning = PlanningHelper(
        mc_vis=mc_vis,
        robot=panda,
        gripper=gripper,
        ik_helper=ik_helper,
        traj_helper=traj_helper,
        tmp_obstacle_dir=tmp_obstacle_dir
    )

    gripper_speed = planning.gripper_speed
    gripper_force = 40.0
    gripper_open_pos = 0.0 if args.gripper_type == 'panda' else gripper.get_state().max_width
    default_2f140_open_width = 0
    planning.set_gripper_speed(gripper_speed)
    planning.set_gripper_force(gripper_force)
    planning.set_gripper_open_pos(gripper_open_pos)
    planning.gripper_open()

    if args.gripper_type == '2f140':
        grasp_pose_viz = Robotiq2F140Hand(grasp_frame=False)
        place_pose_viz = Robotiq2F140Hand(grasp_frame=False)
        wrist2tip_tf = [0, 0, 0.24, 0, 0, 0, 1.0]
        tip2wrist_tf = [0, 0, -0.24, 0, 0, 0, 1.0]
    else:
        grasp_pose_viz = PandaHand(grasp_frame=True)
        place_pose_viz = PandaHand(grasp_frame=True)
        wrist2tip_tf = [0, 0, 0.1034, 0, 0, -0.3827, 0.9239]
        tip2wrist_tf = [0, 0, -0.1034, 0, 0, 0.3827, 0.9239]
    grasp_pose_viz.reset_pose()
    place_pose_viz.reset_pose()
    grasp_pose_viz.meshcat_show(mc_vis, name_prefix='grasp_pose')
    place_pose_viz.meshcat_show(mc_vis, name_prefix='place_pose')


    # setup camera interfaces 
    rs_cfg = get_default_multi_realsense_cfg()
    serials = rs_cfg.SERIAL_NUMBERS

    prefix = rs_cfg.CAMERA_NAME_PREFIX
    camera_names = [f'{prefix}{i}' for i in range(len(serials))]
    cam_list = [camera_names[int(idx)] for idx in args.cam_index]
    serials = [serials[int(idx)] for idx in args.cam_index]

    calib_dir = osp.join(path_util.get_rndf_src(), 'robot/camera_calibration_files')
    calib_filenames = [osp.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in args.cam_index]

    cams = MultiRealsenseLocal(cam_list, calib_filenames)

    realsense_interface = RealsenseInterface()
    ctx = rs.context() # Create librealsense context for managing devices

    # Define some constants 
    resolution_width = 640 # pixels
    resolution_height = 480 # pixels
    frame_rate = 30  # fps

    # pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)
    pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)

    #############################################################################
    # prepare for saving demos

    #############################################################################
    # prepare query points information

    # set up possible gripper query points that are used in optimization
    gripper_mesh_file_panda = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/panda_hand_full_with_offset.obj')
    # gripper_mesh_file_panda = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/panda_hand_full.obj')
    gripper_mesh_file_2f140 = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/robotiq_2f140/full_hand_2f140.obj')
    # gripper_mesh_file_2f140 = osp.join(path_util.get_rndf_descriptions(), 'franka_panda/meshes/robotiq_2f140/collision/robotiq_arg2f_base_link.obj')
    # rack_mesh_file = osp.join(path_util.get_rndf_descriptions(), cfg.PLACEMENT_OBJECTS.RACK_MESH_FILE)
    # shelf_mesh_file = osp.join(path_util.get_rndf_descriptions(), cfg.PLACEMENT_OBJECTS.SHELF_MESH_FILE)

    external_object_meshes = {
        'gripper_panda': gripper_mesh_file_panda,
        'gripper_2f140': gripper_mesh_file_2f140,
        # 'rack': rack_mesh_file,
        # 'shelf': shelf_mesh_file
    }

    gripper_panda_pose = None
    # gripper_panda_pose = np.eye(4); gripper_panda_pose[:-1, -1] = [0, 0, -0.105]
    gripper_2f140_pose = np.eye(4) #; gripper_2f140_pose[:-1, -1] = [0, 0, -0.23]
    external_object_poses = {
        'gripper_panda': gripper_panda_pose,
        'gripper_2f140': gripper_2f140_pose,
        'rack': None,
        'shelf': None
    }

    gripper_panda_finger_cutoff = {'x': None, 'y': None, 'z': 0.3}
    gripper_2f140_finger_cutoff = {'x': None, 'y': None, 'z': 0.3}
    external_object_cutoffs = {
        'gripper_panda': gripper_panda_finger_cutoff,
        'gripper_2f140': gripper_2f140_finger_cutoff,
        'rack': None,
        'shelf': None
    }

    gripper_2f140_scale = 0.7
    external_object_scales = {
        'gripper_panda': None,
        'gripper_2f140': gripper_2f140_scale,
        'rack': None,
        'shelf': None
    }

    query_point_info = DefaultQueryPoints(
        external_object_meshes=external_object_meshes, 
        external_object_poses=external_object_poses, 
        external_object_cutoffs=external_object_cutoffs,
        external_object_scales=external_object_scales,
        default_origin_scale=0.015)

    # check to see if there is already some custom query file to load from in the demo path
    if osp.exists(osp.join(demo_save_dir, 'custom_query_point_info.npz')):
        custom_qp_data = np.load(osp.join(demo_save_dir, 'custom_query_point_info.npz'))
        custom_query_points = custom_qp_data['custom_query_points']
        ee_query_pose_world = custom_qp_data['ee_query_pose_world']
        query_point_info.set_custom_query_points(custom_query_points)
        util.meshcat_pcd_show(mc_vis, custom_query_points, [255, 0, 128], name='scene/custom_query_points')
        util.meshcat_frame_show(mc_vis, 'scene/custom_query_points_pose_tip', util.matrix_from_pose(util.list2pose_stamped(ee_query_pose_world)))

    #############################################################################
    # constants and variables used for the robot

    current_grasp_pose = current_place_pose = poly_util.polypose2np(panda.get_ee_pose())
    # current_grasp_joints = current_place_joints = panda.joint_angles()
    current_grasp_joints = current_place_joints = panda.get_joint_positions().numpy().tolist()
    current_g2p_transform_list = util.pose_stamped2list(util.get_transform(
        pose_frame_target=util.list2pose_stamped(current_place_pose),
        pose_frame_source=util.list2pose_stamped(current_grasp_pose)
    ))
    current_g2p_transform_mat = util.matrix_from_pose(util.list2pose_stamped(current_g2p_transform_list))
    home_joints = np.array([-0.1329, -0.0262, -0.0448, -1.60,  0.0632,  1.9965, -0.8882]).tolist()
    # home_joints = cfg.HOME_JOINTS
    current_panda_plan = []

    #############################################################################
    # begin data collection and saving to demo files

    demo_iteration = args.start_iteration
    save_file_suffix = '%d.npz' % demo_iteration

    pcd_pts = None
    rgb_imgs = None
    depth_imgs = None
    full_scene_pcd = None
    proc_pcd = None
    proc_pcd_place = None
    cam_int_list = None
    cam_poses_list = None

    # constants for manually cropping the point cloud (simple way to segment the object)
    # cropx, cropy, cropz, crop_note = [0.375, 0.75], [-0.5, 0.5], [0.0075, 0.35], 'table'
    cropx, cropy, cropz, crop_note = [0.2, 0.7], [-0.45, 0.5], [0.003, 0.4], 'table'
    full_cropx, full_cropy, full_cropz, full_crop_note = [0.0, 0.8], [-0.65, 0.65], [-0.01, 1.0], 'full scene'

    print('\n\nBeginning demo iteration %d\n\n' % demo_iteration)
    got_observation = False
    default_2f140_open_width = 0
    offset_pose_set = None
    offset_pose_relative = [0, 0, 0, 0, 0, 0, 1]
    interaction_point = None

    pc_hand_viz_cls = PandaHand if args.gripper_type == 'panda' else Robotiq2F140Hand
    parent_child_manager = ParentChildObjectManager(
        mc_vis, 
        parent_class_name=parent_class, 
        child_class_name=child_class, 
        panda_hand_cls=pc_hand_viz_cls) 

    while True:
        current_g2p_transform_list = util.pose_stamped2list(util.get_transform(
            pose_frame_target=util.list2pose_stamped(current_place_pose),
            pose_frame_source=util.list2pose_stamped(current_grasp_pose)
        ))
        current_g2p_transform_mat = util.matrix_from_pose(util.list2pose_stamped(current_g2p_transform_list))

        # if offset_pose_set is not None:
        #     offset_pose_relative = util.pose_stamped2list(util.get_transform(
        #         pose_frame_target=util.list2pose_stamped(offset_pose_set), 
        #         pose_frame_source=util.list2pose_stamped(current_place_pose)
        #     ))
        #     util.meshcat_frame_show(mc_vis, 'scene/current_offset_pose', util.matrix_from_pose(util.list2pose_stamped(offset_pose_set)))

        grasp_pose_viz.reset_pose()
        place_pose_viz.reset_pose()
        grasp_ee_pose_mat = util.matrix_from_pose(util.list2pose_stamped(current_grasp_pose))
        grasp_pose_viz.transform_hand(grasp_ee_pose_mat)
        grasp_pose_viz.meshcat_show(mc_vis, name_prefix='grasp_pose')
        place_ee_pose_mat = util.matrix_from_pose(util.list2pose_stamped(current_place_pose))
        place_pose_viz.transform_hand(place_ee_pose_mat)
        place_pose_viz.meshcat_show(mc_vis, name_prefix='place_pose')

        gripper_qp_current = util.transform_pcd(query_point_info.external_object_qp_dict[f'gripper_{args.gripper_type}']['uniform'], grasp_ee_pose_mat)
        gripper_qp_rs_current = util.transform_pcd(query_point_info.external_object_qp_dict[f'gripper_{args.gripper_type}']['surface'], place_ee_pose_mat)

        util.meshcat_frame_show(mc_vis, 'scene/current_grasp_pose', grasp_ee_pose_mat)
        util.meshcat_frame_show(mc_vis, 'scene/current_place_pose', place_ee_pose_mat)
        util.meshcat_pcd_show(mc_vis, gripper_qp_current, color=[255, 0, 0], name='scene/gripper_qp_current')
        util.meshcat_pcd_show(mc_vis, gripper_qp_rs_current, color=[0, 0, 255], name='scene/gripper_qp_rs_current')

        if query_point_info.custom_query_points is not None:
            util.meshcat_pcd_show(mc_vis, query_point_info.custom_query_points, [255, 0, 128], name='scene/custom_query_points')
        if query_point_info.custom_ee_query_pose_world is not None:
            util.meshcat_frame_show(mc_vis, 'scene/custom_query_points_pose', util.matrix_from_pose(util.list2pose_stamped(query_point_info.custom_ee_query_pose_world)))

        if full_scene_pcd is not None:
            util.meshcat_pcd_show(mc_vis, full_scene_pcd, color=[128, 128, 128], name='scene/full_scene_pcd')
        # if proc_pcd is not None:
        #     util.meshcat_pcd_show(mc_vis, proc_pcd, color=[255, 0, 0], name='scene/observed_object_pcd')
        # if proc_pcd_place is not None:
        #     util.meshcat_pcd_show(mc_vis, proc_pcd_place, color=[255, 0, 255], name='scene/placed_object_pcd')

        parent_child_manager.visualize_current_state()
        user_val = input(
            '''
            Press:
                [o] to get observation
                [i] to set into low stiffness mode
                [l] to lock into current configuration with high stiffness
                [r] to grasp (if we have the gripper)
                [f] to open (if we have the gripper)
                [g] to record grasp pose
                [p] to record place pose
                [ff] to record offset from place pose
                [h] to go home
                [e] to execute currently stored plan
                [s] to save the currently stored grasps
                [s] to save the currently stored grasps
                [sm] to save the multi object pose
                [n] to enter a new save file suffix
                [em] to enter an interactive python terminal
                [q] to record a set of query points located at the current end effector pose
                [w] to set new default "open" grasp width
                [clear] to reset everything
                [b] to exit
            ''')
        if user_val == 'o':
            pcd_pts = []
            pcd_dict_list = []
            cam_int_list = []
            cam_poses_list = []
            rgb_imgs = []
            depth_imgs = []
            for idx, cam in enumerate(cams.cams):
                rgb, depth = realsense_interface.get_rgb_and_depth_image(pipelines[idx])
                rgb_imgs.append(rgb)                
                cam_intrinsics = realsense_interface.get_intrinsics_mat(pipelines[idx])
                cam.cam_int_mat = cam_intrinsics
                cam._init_pers_mat()
                cam_pose_world = cam.cam_ext_mat
                cam_int_list.append(cam_intrinsics)
                cam_poses_list.append(cam_pose_world)

                depth = depth * 0.001
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
                util.meshcat_pcd_show(mc_vis, pcd_world, name=f'scene/pcd_world_cam_{idx}')

            full_pcd = np.concatenate(pcd_pts, axis=0)
            full_scene_pcd = manually_segment_pcd(full_pcd, x=full_cropx, y=full_cropy, z=full_cropz, note=full_crop_note)
            
            # crop the point cloud to the table
            proc_pcd = manually_segment_pcd(full_pcd, x=cropx, y=cropy, z=cropz, note=crop_note)
            top2clusters = segment(proc_pcd, top_n=2, eps=0.008, min_points=20)
            util.meshcat_pcd_show(mc_vis, top2clusters, name='scene/top2clusters')

            proc_pcd = copy.deepcopy(top2clusters)

            if args.instance_seg_method == 'hand-label':
                log_warn('NOT IMPLEMENTED!')
            elif args.instance_seg_method == 'nn':
                child_pts = []
                parent_pts = []

                for idx, _ in enumerate(cams.cams):
                    rgb = rgb_imgs[idx]
                    pcd_world_img = pcd_dict_list[idx]['world_img']

                    child_pt = select_pt(rgb, child=True)
                    if child_pt is not None:
                        child_mask = get_mask_from_pt(child_pt, image=rgb)
                        child_partial_pcd = pcd_world_img[child_mask].reshape((-1,3))
                        child_pts.append(child_partial_pcd)

                    parent_pt = select_pt(rgb, child=False)
                    if parent_pt is not None:
                        parent_mask = get_mask_from_pt(parent_pt, image=rgb)
                        parent_partial_pcd = pcd_world_img[parent_mask].reshape((-1,3))
                        parent_pts.append(parent_partial_pcd)

                if len(child_pts) > 0:
                    child_pcd = np.concatenate(child_pts, axis=0)
                    child_pcd =  manually_segment_pcd(child_pcd, x=cropx, y=cropy, z=cropz, note=crop_note)
                    # child_pcd = segment(child_pcd, top_n=1, eps=0.008, min_points=20)
                    parent_child_manager.set_child_pointcloud(child_pcd)

                if len(parent_pts) > 0:
                    parent_pcd = np.concatenate(parent_pts, axis=0)
                    parent_pcd = manually_segment_pcd(parent_pcd, x=cropx, y=cropy, z=cropz, note=crop_note)
                    parent_child_manager.set_parent_pointcloud(parent_pcd)

            else:  # y-axis
                # get all clusters that are on the positive and negative side of the y-axis
                top2clusters = segment(proc_pcd, top_n=2, eps=0.008, min_points=20)

                pos_y_proc_pcd = proc_pcd[np.where(proc_pcd[:, 1] > 0.0)[0]]
                neg_y_proc_pcd = proc_pcd[np.where(proc_pcd[:, 1] < 0.0)[0]]

                parent_pcd = segment(pos_y_proc_pcd, eps=0.003, min_points=20)
                child_pcd = segment(neg_y_proc_pcd, eps=0.003, min_points=20)
                # crop each of these point clouds to get the single objects
                log_info('Segmenting the positive y-axis pcd to PARENT and negative y-axis pcd to CHILD')
                parent_child_manager.set_parent_pointcloud(parent_pcd)
                parent_child_manager.set_child_pointcloud(child_pcd)

            log_info('Setting the active object to be the CHILD')
            parent_child_manager.set_active_object('child')

            got_observation = True
            continue
        elif user_val == 'sa':
            # set the active object
            active_object = input('Please enter either "parent" or "child" ("p" or "c")')
            if active_object == 'p':
                active_object = 'parent'
            elif active_object == 'c':
                active_object = 'child'
            
            log_info(f'Setting the active object to: {active_object}')
            parent_child_manager.set_active_object(active_object)
            continue
        elif user_val == 'fa':
            # FLIP the active object
            parent_child_manager.flip_active_object()
            log_info(f'FLIPPING the active object to: {parent_child_manager.get_active_object()}')
            continue
        elif user_val == 'em':
            print('\n\nHere in interactive mode\n\n')
            embed()
            continue
        elif user_val == 'n':
            name_user_val = input('\n\nPlease enter a new suffix for the file name\n\n')
            save_file_suffix = name_user_val + '_%d.npz' % demo_iteration
            print('\n\nNew save file suffix: %s\n\n' % save_file_suffix)
            continue
        elif user_val == 's':
            if not got_observation:
                print('\n\nCannot save until we get an observation!\n\n')
                continue
            single_demo_save_dir = single_save_dirs[parent_child_manager.get_active_object()]
            save_file_path = osp.join(single_demo_save_dir, args.exp + '_' + save_file_suffix)
            print(f'\n\n[Active object: {parent_child_manager.get_active_object()}] Saving data to path: %s\n\n' % save_file_path)
            np.savez(
                save_file_path,
                pcd_pts=pcd_pts,
                processed_pcd=proc_pcd,
                rgb_imgs=rgb_imgs,
                depth_imgs=depth_imgs,
                cam_intrinsics=cam_int_list,
                cam_poses=cam_poses_list,
                grasp_pose_world=current_grasp_pose,
                place_pose_world=current_place_pose,
                grasp_joints=current_grasp_joints,
                place_joints=current_place_joints,
                ee_link=None # cfg.DEFAULT_EE_FRAME
            )

            ### save in the same format as our simulated demos
            grasp_save_path = '/'.join(save_file_path.split('/')[:-1] + ['grasp_demo_' + save_file_path.split('/')[-1]])

            # unused dummy variables for compatibility with simulated demos
            shapenet_id = None

            gripper_contact_pose = current_grasp_pose
            np.savez(
                grasp_save_path,
                shapenet_id=shapenet_id,
                ee_pose_world=np.asarray(gripper_contact_pose),  # TODO see if this works okay, or if we want to record where the grasp target was
                robot_joints=np.asarray(current_grasp_joints),
                obj_pose_world=np.asarray(util.pose_stamped2list(util.unit_pose())),
                obj_pose_camera=np.asarray(util.pose_stamped2list(util.unit_pose())),
                object_pointcloud=proc_pcd,
                rgb=rgb_imgs,
                depth_full=depth_imgs,
                depth=depth_imgs,
                seg=[np.arange(depth_imgs[0].flatten().shape[0])]*4,
                camera_poses=cam_poses_list,
                obj_model_file=None,
                obj_model_file_dec=None,
                gripper_pts=query_point_info.external_object_qp_dict[f'gripper_{args.gripper_type}']['surface'],
                gripper_pts_gaussian=query_point_info.external_object_qp_dict[f'gripper_{args.gripper_type}']['gaussian'],
                gripper_pts_uniform=query_point_info.external_object_qp_dict[f'gripper_{args.gripper_type}']['uniform'],
                gripper_contact_pose=gripper_contact_pose,
                table_urdf=None,
                pcd_raw=full_pcd,
                cam_intrinsics=cam_int_list
            )

            # place_save_path = 'place_demo_' + save_file_path
            place_save_path = '/'.join(save_file_path.split('/')[:-1] + ['place_demo_' + save_file_path.split('/')[-1]])
            place_ee_pose_world = current_place_pose

            # some unused dummy variables
            rack_pose_world = util.pose_stamped2list(util.unit_pose())
            shelf_pose_world = util.pose_stamped2list(util.unit_pose())
            rack_pcd_pts = np.random.random((500, 3))
            shelf_pcd_pts = np.random.random((500, 3))
            rack_contact_pose = copy.deepcopy(rack_pose_world)

            if query_point_info.custom_query_points is None:
                log_critical('!!! CUSTOM QUERY POINTS NOT SET! !!!')
            np.savez(
                place_save_path,
                shapenet_id=shapenet_id,
                ee_pose_world=np.asarray(place_ee_pose_world),  # TODO see if this works okay, or if we want to record where the grasp target was
                robot_joints=np.asarray(current_place_joints),
                obj_pose_world=np.asarray(current_g2p_transform_list),
                obj_pose_camera=np.asarray(util.pose_stamped2list(util.unit_pose())),
                object_pointcloud=proc_pcd,
                rgb=rgb_imgs,
                depth_full=depth_imgs,
                depth=depth_imgs,
                seg=[np.arange(depth_imgs[0].flatten().shape[0])]*4,
                camera_poses=cam_poses_list,
                obj_model_file=None,
                obj_model_file_dec=None,
                gripper_pts=query_point_info.external_object_qp_dict[f'gripper_{args.gripper_type}']['surface'],
                # rack_pointcloud_observed=rack_pcd_pts,
                # rack_pointcloud_gt=query_point_info.external_object_qp_dict['rack']['surface'],
                # rack_pointcloud_gaussian=query_point_info.external_object_qp_dict['rack']['gaussian'],
                # rack_pointcloud_uniform=query_point_info.external_object_qp_dict['rack']['uniform'],
                # rack_pose_world=rack_pose_world,
                # rack_contact_pose=rack_contact_pose,
                # shelf_pose_world=shelf_pose_world,
                # shelf_pointcloud_observed=shelf_pcd_pts,
                # shelf_pointcloud_uniform=query_point_info.external_object_qp_dict['shelf']['uniform'],
                # shelf_pointcloud_gt=query_point_info.external_object_qp_dict['shelf']['surface'],
                custom_query_points=query_point_info.custom_query_points,
                table_urdf=None,
                pcd_raw=full_pcd,
                cam_intrinsics=cam_int_list
            )

            img_dir = osp.join(demo_save_dir, args.exp + '_' + save_file_suffix.split('.npz')[0], 'imgs')
            depth_dir = osp.join(demo_save_dir, args.exp + '_' + save_file_suffix.split('.npz')[0], 'depth_imgs')
            pcd_dir = osp.join(demo_save_dir, args.exp + '_' + save_file_suffix.split('.npz')[0], 'pcds')
            util.safe_makedirs(img_dir)
            util.safe_makedirs(depth_dir)
            util.safe_makedirs(pcd_dir)

            plot3d(
                [proc_pcd],
                fname=osp.join(pcd_save_dir, 'pcd.html'),
                auto_scene=False,
                scene_dict=plotly_scene_dict,
                z_plane=False)
            for i in range(len(rgb_imgs)):
                cv2.imwrite(osp.join(img_dir, '%d.png' % i), cv2.cvtColor(rgb_imgs[i], cv2.COLOR_RGB2BGR))
                cv2.imwrite(osp.join(depth_dir, '%d.png' % i), depth_imgs[i].astype(np.uint16))

            save_file_suffix = save_file_suffix.replace('%d.npz' % demo_iteration, '%d.npz' % (demo_iteration + 1))

            demo_iteration += 1
            got_observation = False
            print('\n\nBeginning demo iteration %d\n\n' % demo_iteration)
            continue
        elif user_val == 'sm':
            # Save Multi
            log_info('Saving current info for multi object point clouds')

            if not got_observation:
                print('\n\nCannot save until we get an observation!\n\n')
                continue

            save_file_path = osp.join(demo_save_dir, args.exp + '_' + save_file_suffix)
            print('\n\nSaving data to path: %s\n\n' % save_file_path)

            ### save in the same format as our simulated demos
            grasp_save_path = '/'.join(save_file_path.split('/')[:-1] + ['grasp_demo_' + save_file_path.split('/')[-1]])

            # unused dummy variables for compatibility with simulated demos
            shapenet_id = None

            gripper_contact_pose = current_grasp_pose
            np.savez(
                grasp_save_path,
                shapenet_id=shapenet_id,
                ee_pose_world=np.asarray(gripper_contact_pose),  # TODO see if this works okay, or if we want to record where the grasp target was
                robot_joints=np.asarray(current_grasp_joints),
                obj_pose_world=np.asarray(util.pose_stamped2list(util.unit_pose())),
                obj_pose_camera=np.asarray(util.pose_stamped2list(util.unit_pose())),
                object_pointcloud=proc_pcd,
                rgb=rgb_imgs,
                depth_full=depth_imgs,
                depth=depth_imgs,
                seg=[np.arange(depth_imgs[0].flatten().shape[0])]*4,
                camera_poses=cam_poses_list,
                obj_model_file=None,
                obj_model_file_dec=None,
                gripper_pts=query_point_info.external_object_qp_dict[f'gripper_{args.gripper_type}']['surface'],
                gripper_pts_gaussian=query_point_info.external_object_qp_dict[f'gripper_{args.gripper_type}']['gaussian'],
                gripper_pts_uniform=query_point_info.external_object_qp_dict[f'gripper_{args.gripper_type}']['uniform'],
                gripper_contact_pose=gripper_contact_pose,
                table_urdf=None,
                pcd_raw=full_pcd,
                cam_intrinsics=cam_int_list
            )


            
            multi_obj_start_pcd_dict = dict(
                parent=parent_child_manager.get_parent_pointcloud(),
                child=parent_child_manager.get_child_pointcloud())

            multi_obj_final_pcd_dict = dict(
                parent=parent_child_manager.get_parent_tf_pointcloud(),
                child=parent_child_manager.get_child_tf_pointcloud())
            
            multi_obj_grasp_pose_dict = dict(
                parent=parent_child_manager.get_parent_grasp_pose(),
                child=parent_child_manager.get_child_grasp_pose())

            multi_obj_place_pose_dict = dict(
                parent=parent_child_manager.get_parent_place_pose(),
                child=parent_child_manager.get_child_place_pose())

            multi_obj_grasp_joints_dict = dict(
                parent=parent_child_manager.get_parent_grasp_joints(),
                child=parent_child_manager.get_child_grasp_joints())

            multi_obj_place_joints_dict = dict(
                parent=parent_child_manager.get_parent_place_joints(),
                child=parent_child_manager.get_child_place_joints())
            
            multi_obj_names_dict = dict(
                parent=parent_class,
                child=child_class)

            np.savez(
                save_file_path,
                pcd_pts=pcd_pts,
                processed_pcd=proc_pcd,
                rgb_imgs=rgb_imgs,
                depth_imgs=depth_imgs,
                cam_intrinsics=cam_int_list,
                cam_poses=cam_poses_list,
                multi_obj_names=multi_obj_names_dict,
                multi_obj_start_pcd=multi_obj_start_pcd_dict,
                multi_obj_final_pcd=multi_obj_final_pcd_dict,
                grasp_pose_world=multi_obj_grasp_pose_dict,
                place_pose_world=multi_obj_place_pose_dict,
                grasp_joints=multi_obj_grasp_joints_dict,
                place_joints=multi_obj_place_joints_dict,
                ee_link=None,
                gripper_type=args.gripper_type,
                offset_pose_relative=offset_pose_relative,
                interaction_point=interaction_point
            )

            img_dir = osp.join(demo_save_dir, args.exp + '_' + save_file_suffix.split('.npz')[0], 'imgs')
            depth_dir = osp.join(demo_save_dir, args.exp + '_' + save_file_suffix.split('.npz')[0], 'depth_imgs')
            pcd_dir = osp.join(demo_save_dir, args.exp + '_' + save_file_suffix.split('.npz')[0], 'pcds')
            util.safe_makedirs(img_dir)
            util.safe_makedirs(depth_dir)
            util.safe_makedirs(pcd_dir)

            plot3d(
                [parent_child_manager.get_parent_pointcloud(), parent_child_manager.get_child_pointcloud()], ['red', 'blue'],
                fname=osp.join(pcd_save_dir, 'pc_pcd.html'),
                auto_scene=False,
                scene_dict=plotly_scene_dict,
                z_plane=False)
            for i in range(len(rgb_imgs)):
                # np2img(rgb_imgs[i], osp.join(img_dir, '%d.png' % i))
                # np2img(depth_imgs[i].astype(np.uint16), osp.join(depth_dir, '%d.png' % i))
                cv2.imwrite(osp.join(img_dir, '%d.png' % i), cv2.cvtColor(rgb_imgs[i], cv2.COLOR_RGB2BGR))
                cv2.imwrite(osp.join(depth_dir, '%d.png' % i), depth_imgs[i].astype(np.uint16))
            save_file_suffix = save_file_suffix.replace('%d.npz' % demo_iteration, '%d.npz' % (demo_iteration + 1))
            demo_iteration += 1
            # sp.call('./sync_demo_files.sh run', shell=True)
            # sp.call(['./sync_demo_files.sh', 'run'])
            got_observation = False
            print('\n\nBeginning demo iteration %d\n\n' % demo_iteration)

            continue    
        elif user_val == 'i':
            print('\n\nSetting low stiffness in current pose, you can now move the robot')
            panda.start_cartesian_impedance(Kx=torch.zeros(6), Kxd=torch.zeros(6))
            continue
        elif user_val == 'l':
            print('\n\nSetting joint positions to current value\n\n')
            panda.start_joint_impedance()
            continue
        elif user_val == 'r':
            print('\n\nExecuting grasp\n\n')
            planning.gripper_grasp()
            continue
        elif user_val == 'f':
            print('\n\nOpening grasp\n\n')
            planning.gripper_open()
            continue
        elif user_val == 'g':
            current_grasp_pose = poly_util.polypose2np(panda.get_ee_pose())
            current_grasp_joints = panda.get_joint_positions().numpy().tolist()
            print('\n\nRecording grasp pose!\n\n')
            print(current_grasp_pose)

            parent_child_manager.set_grasp_pose(util.matrix_from_pose(util.list2pose_stamped(current_grasp_pose)))
            log_info(f'Setting current pose as grasp for {parent_child_manager.get_active_object()} object')
            continue
        elif user_val == 'p':
            current_place_pose = poly_util.polypose2np(panda.get_ee_pose())
            current_place_joints = panda.get_joint_positions().numpy().tolist()
            print('\n\nRecording place pose!\n\n')
            print(current_place_pose)

            current_g2p_transform_list = util.pose_stamped2list(util.get_transform(
                pose_frame_target=util.list2pose_stamped(current_place_pose),
                pose_frame_source=util.list2pose_stamped(current_grasp_pose)
            ))
            current_g2p_transform_mat = util.matrix_from_pose(util.list2pose_stamped(current_g2p_transform_list))

            parent_child_manager.set_place_pose(util.matrix_from_pose(util.list2pose_stamped(current_place_pose)))
            parent_child_manager.apply_transform_to_current(util.matrix_from_pose(util.list2pose_stamped(current_g2p_transform_list)))
            log_info(f'Setting current pose as place for {parent_child_manager.get_active_object()} object')
            continue
        elif user_val == 'ff':
            offset_pose_set = poly_util.polypose2np(panda.get_ee_pose())
            print('\n\nRecording offset pose!\n\n')
            print(offset_pose_set)
            offset_pose_relative = util.pose_stamped2list(util.get_transform(
                pose_frame_target=util.list2pose_stamped(offset_pose_set), 
                pose_frame_source=util.list2pose_stamped(current_place_pose)
            ))
            continue
        elif user_val == 'q':
            # this is the pose of the wrist (panda_link8)
            current_ee_query_pose = poly_util.polypose2np(panda.get_ee_pose())
            util.meshcat_frame_show(mc_vis, 'scene/custom_query_points_pose', util.matrix_from_pose(util.list2pose_stamped(current_ee_query_pose)))

            # this is the pose of the point in between the fingertips
            current_ee_query_pose = convert_wrist2tip(current_ee_query_pose, wrist2tip_tf=wrist2tip_tf) 
            util.meshcat_frame_show(mc_vis, 'scene/custom_query_points_pose_tip', util.matrix_from_pose(util.list2pose_stamped(current_ee_query_pose)))

            custom_query_points = copy.deepcopy(query_point_info.default_origin_pts)
            custom_query_points = util.transform_pcd(custom_query_points, util.matrix_from_pose(util.list2pose_stamped(current_ee_query_pose)))
            query_point_info.set_custom_query_points(custom_query_points)
            plot3d(
                [query_point_info.custom_query_points], 
                fname=osp.join(pcd_save_dir, 'custom_query_points.html'),
                auto_scene=False,
                scene_dict=plotly_scene_dict,
                z_plane=False)
            # util.meshcat_pcd_show(mc_vis, query_point_info.custom_query_points, color=[128, 128, 0], name='scene/custom_query_points')

            custom_query_save_path = osp.join(demo_save_dir, 'custom_query_point_info.npz')
            np.savez(
                custom_query_save_path,
                custom_query_points=custom_query_points,
                ee_query_pose_world=current_ee_query_pose)
            print('\n\nNew custom query points configuration recorded\n\n')

            interaction_point = np.asarray(current_ee_query_pose[:3])
            print('\n\nNew position of the interaction point recorded\n\n')
            continue
        elif user_val == 'h':
            # panda.move_to_joint_positions(home_joints)
            print('\n\nObtaining plan to go home!\n\n')
            current_panda_plan = planning.plan_home()
            continue
        elif user_val == 'e':
            print('\n\nExecuting current panda plan!\n\n')
            if len(current_panda_plan) > 0:
                planning.execute_loop(current_panda_plan)
            else:
                print('\n\nCurrent panda plan is empty!')
            continue
        elif user_val == 'w':
            new_gripper_open_value = input(f'Please enter an integer value 0-{gripper.get_state().max_width} for default open grasp width\n')
            # if not isinstance(new_gripper_open_value, int):
            #     print('Invalid value')
            #     continue

            if args.gripper_type != '2f140':
                print('Only valid for 2F140 gripper')
                continue
            default_2f140_open_width = int(np.clip(int(new_gripper_open_value), 0, gripper.get_state().max_width))
            planning.set_gripper_open_pos(default_2f140_open_width)
            planning.gripper.goto(planning.gripper_close_pos, gripper_speed, gripper_force, blocking=False)
            print(f'New default_2f140_open_width: {default_2f140_open_width}')
            continue
        elif user_val == 'clear':
            print('Resetting grasp pose and place pose to current, and "current_g2p_transform_list" to be identity transform')
            print('To create a new relative transformation to apply to the current object, save a grasp and a place pose')

            current_grasp_pose = current_place_pose = poly_util.polypose2np(panda.get_ee_pose())
            # current_grasp_pose = current_place_pose = [0, 0, 0, 0, 0, 0, 1]
            current_grasp_joints = current_place_joints = panda.get_joint_positions().numpy().tolist()
            current_g2p_transform_list = util.pose_stamped2list(util.get_transform(
                pose_frame_target=util.list2pose_stamped(current_place_pose),
                pose_frame_source=util.list2pose_stamped(current_grasp_pose)
            ))

            pcd_pts = None
            rgb_imgs = None
            depth_imgs = None
            full_scene_pcd = None
            proc_pcd = None
            proc_pcd_place = None
            cam_int_list = None
            cam_poses_list = None

            interaction_point = None
            offset_pose_relative = [0, 0, 0, 0, 0, 0, 1]

            parent_child_manager.clear()
            print('Clearing all info')
            mc_vis['scene'].delete()
            continue
        elif user_val == 'b':
            print('Exiting')
            break
        else:
            print('Command unused')
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='debug_real_demo')
    parser.add_argument('-p', '--port_vis', type=int, default=6000)
    parser.add_argument('--parent_class', type=str, required=True, help='Class name of parent object')
    parser.add_argument('--child_class', type=str, required=True, help='Class name of child object')
    parser.add_argument('--instance_seg_method', type=str, default='y-axis', help='Options: ["y-axis", "hand-label", "nn"]')
    parser.add_argument('--demo_save_dir', type=str, default='real_demo_data_multi_object')
    parser.add_argument('--config', type=str, default='base_demo_cfg.yaml')
    parser.add_argument('--n_cams', type=int, default=4)
    parser.add_argument('--start_iteration', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--color_seg', action='store_true')
    parser.add_argument('--demo_offset_z', type=float, default=0.0)
    parser.add_argument('--gripper_type', type=str, default='panda')
    parser.add_argument('--cam_index', nargs='+', help='set which cameras to get point cloud from', required=True)

    args = parser.parse_args()
    main(args)
