import os.path as osp
import time
import sys
import signal
import random
import cv2
import numpy as np
import argparse
import copy
import lcm
import threading
import meshcat
import open3d

import rospy
import rospkg
from franka_interface import ArmInterface
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as gripper_msg
rospack = rospkg.RosPack()

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

from ndf_robot.utils import util, path_util, lcm_util, trimesh_util
from ndf_robot.utils.real_util import RealImageLCMSubscriber, RealCamInfoLCMSubscriber
from ndf_robot.utils.plotly_save import plot3d, plotly_scene_dict
from ndf_robot.utils.visualize import PandaHand, Robotiq2F140Hand
from ndf_robot.utils.relational_utils import ParentChildObjectManager
from ndf_robot.utils.record_demo_utils import DefaultQueryPoints, manually_segment_pcd, convert_wrist2tip, convert_tip2wrist
# from ndf_robot.config.default_real_demo_cfg import get_real_demo_cfg_defaults

sys.path.append(osp.join(rospack.get_path('panda_ndf'), 'src'))
from panda_ndf_utils.panda_mg_wrapper import FrankaMoveIt
from panda_ndf_utils.simple_multicam import MultiRealsense
from panda_ndf_utils import util as real_robot_util
from panda_ndf_utils.config.default_real_demo_cfg import get_real_demo_cfg_defaults
from ndf_robot.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg


def handle_thread(lc):
    while True:
        lc.handle_timeout(1)
        time.sleep(0.001)


def gen_2f140_command(char, command):
    """Update the command according to the character entered by the user."""

    if char == 'a':
        command = gripper_msg.Robotiq2FGripper_robot_output()
        command.rACT = 1
        command.rGTO = 1
        command.rSP  = 255
        command.rFR  = 150

    if char == 'r':
        command = gripper_msg.Robotiq2FGripper_robot_output()
        command.rACT = 0

    if char == 'c':
        command.rPR = 255

    if char == 'o':
        command.rPR = 0

    #If the command entered is a int, assign this value to rPRA
    try:
        command.rPR = int(char)
        if command.rPR > 255:
            command.rPR = 255
        if command.rPR < 0:
            command.rPR = 0
    except ValueError:
        pass

    if char == 'f':
        command.rSP += 25
        if command.rSP > 255:
            command.rSP = 255

    if char == 'l':
        command.rSP -= 25
        if command.rSP < 0:
            command.rSP = 0


    if char == 'i':
        command.rFR += 25
        if command.rFR > 255:
            command.rFR = 255

    if char == 'd':
        command.rFR -= 25
        if command.rFR < 0:
            command.rFR = 0

    return command



def main(args):
    #############################################################################
    # generic setup

    np.random.seed(args.seed)
    random.seed(args.seed)
    signal.signal(signal.SIGINT, util.signal_handler)

    cfg = get_real_demo_cfg_defaults()
    config_fname = osp.join('config', 'real_demo_cfgs', args.config)
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        print('Config file %s does not exist, using defaults' % config_fname)
    cfg.freeze()

    multi_object_class = f'parent_{args.parent_class}_child_{args.child_class}'
    parent_class = args.parent_class
    child_class = args.child_class

    demo_save_dir = osp.join(path_util.get_ndf_data(), args.demo_save_dir, 'multi_object', multi_object_class, args.exp)
    util.safe_makedirs(demo_save_dir)

    # TODO: decide on a good default save location for the demo files
    parent_save_dir = osp.join(path_util.get_ndf_data(), args.demo_save_dir, 'multi_object', parent_class, args.exp)
    util.safe_makedirs(parent_save_dir)
    child_save_dir = osp.join(path_util.get_ndf_data(), args.demo_save_dir, 'multi_object', child_class, args.exp)
    util.safe_makedirs(child_save_dir)
    single_save_dirs = dict(parent=parent_save_dir, child=child_save_dir)

    pcd_save_dir = osp.join(path_util.get_ndf_data(), args.demo_save_dir, 'multi_object', multi_object_class, args.exp, 'pcd_observations')
    util.safe_makedirs(pcd_save_dir)

    #############################################################################
    # ROS (for panda), LCM (for cameras), meshcat for visualization

    rospy.init_node('PandaDemo')
    panda = ArmInterface()
    panda.set_EE_frame_to_link(cfg.DEFAULT_EE_FRAME)
    panda_mp = FrankaMoveIt(panda, cfg=cfg, gripper_type=args.gripper_type)
    panda.set_joint_position_speed(0.5)

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
    lc_th = threading.Thread(target=handle_thread, args=(lc,))
    lc_th.daemon = True
    lc_th.start()
    # lc_handler_manager = lcm_util.LCMHandleThread()
    # lc = lc_handler_manager.get_dispatch()

    mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    mc_vis['scene'].delete()

    gripper_pub = None 
    gripper_command = None
    if args.gripper_type == '2f140':
        grasp_pose_viz = Robotiq2F140Hand()
        place_pose_viz = Robotiq2F140Hand()

        wrist2tip_tf = cfg.WRIST2TIP.ROBOTIQ_2F140
        tip2wrist_tf = cfg.TIP2WRIST.ROBOTIQ_2F140

        gripper_pub = rospy.Publisher('Robotiq2FGripperRobotOutput', gripper_msg.Robotiq2FGripper_robot_output)
        gripper_command = gripper_msg.Robotiq2FGripper_robot_output()

        print('Resetting 2F140 gripper')
        gripper_command = gen_2f140_command('r', gripper_command)
        gripper_pub.publish(gripper_command)
        print('Activating 2F140 gripper')
        gripper_command = gen_2f140_command('a', gripper_command)
        gripper_pub.publish(gripper_command)
    else:
        grasp_pose_viz = PandaHand(grasp_frame=True)
        place_pose_viz = PandaHand(grasp_frame=True)

        wrist2tip_tf = cfg.WRIST2TIP.PANDA_HAND
        tip2wrist_tf = cfg.TIP2WRIST.PANDA_HAND
    grasp_pose_viz.reset_pose()
    place_pose_viz.reset_pose()
    grasp_pose_viz.meshcat_show(mc_vis, name_prefix='grasp_pose')
    place_pose_viz.meshcat_show(mc_vis, name_prefix='place_pose')


    # setup camera interfaces as LCM subscribers
    rs_cfg = get_default_multi_realsense_cfg()
    serials = rs_cfg.SERIAL_NUMBERS

    rgb_topic_name_suffix = rs_cfg.RGB_LCM_TOPIC_NAME_SUFFIX
    depth_topic_name_suffix = rs_cfg.DEPTH_LCM_TOPIC_NAME_SUFFIX
    info_topic_name_suffix = rs_cfg.INFO_LCM_TOPIC_NAME_SUFFIX
    pose_topic_name_suffix = rs_cfg.POSE_LCM_TOPIC_NAME_SUFFIX

    prefix = rs_cfg.CAMERA_NAME_PREFIX
    camera_names = [f'{prefix}{i}' for i in range(len(serials))]
    cam_list = [camera_names[int(idx)] for idx in args.cam_index]

    # update the topic names based on each individual camera
    rgb_sub_names = [f'{cam_name}_{rgb_topic_name_suffix}' for cam_name in camera_names]
    depth_sub_names = [f'{cam_name}_{depth_topic_name_suffix}' for cam_name in camera_names]
    info_sub_names = [f'{cam_name}_{info_topic_name_suffix}' for cam_name in camera_names]
    pose_sub_names = [f'{cam_name}_{pose_topic_name_suffix}' for cam_name in camera_names]

    img_subscribers = []
    for i, name in enumerate(cam_list):
        img_sub = RealImageLCMSubscriber(lc, rgb_sub_names[i], depth_sub_names[i])
        info_sub = RealCamInfoLCMSubscriber(lc, pose_sub_names[i], info_sub_names[i])
        img_subscribers.append((name, img_sub, info_sub))

    cams = MultiRealsense(cam_list, cfg=cfg)

    #############################################################################
    # prepare for saving demos

    #############################################################################
    # prepare query points information

    # set up possible gripper query points that are used in optimization
    gripper_mesh_file_panda = osp.join(path_util.get_ndf_descriptions(), 'franka_panda/meshes/panda_hand_full_with_offset.obj')
    # gripper_mesh_file_panda = osp.join(path_util.get_ndf_descriptions(), 'franka_panda/meshes/panda_hand_full.obj')
    gripper_mesh_file_2f140 = osp.join(path_util.get_ndf_descriptions(), 'franka_panda/meshes/robotiq_2f140/full_hand_2f140.obj')
    # gripper_mesh_file_2f140 = osp.join(path_util.get_ndf_descriptions(), 'franka_panda/meshes/robotiq_2f140/collision/robotiq_arg2f_base_link.obj')
    rack_mesh_file = osp.join(path_util.get_ndf_descriptions(), cfg.PLACEMENT_OBJECTS.RACK_MESH_FILE)
    shelf_mesh_file = osp.join(path_util.get_ndf_descriptions(), cfg.PLACEMENT_OBJECTS.SHELF_MESH_FILE)

    external_object_meshes = {
        'gripper_panda': gripper_mesh_file_panda,
        'gripper_2f140': gripper_mesh_file_2f140,
        'rack': rack_mesh_file,
        'shelf': shelf_mesh_file
    }

    gripper_panda_pose = None
    # gripper_panda_pose = np.eye(4); gripper_panda_pose[:-1, -1] = [0, 0, -0.105]
    gripper_2f140_pose = np.eye(4); gripper_2f140_pose[:-1, -1] = [0, 0, -0.23]
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

    current_grasp_pose = current_place_pose = real_robot_util.panda_full_pose(panda.endpoint_pose())
    current_grasp_joints = current_place_joints = panda.joint_angles()
    current_g2p_transform_list = util.pose_stamped2list(util.get_transform(
        pose_frame_target=util.list2pose_stamped(current_place_pose),
        pose_frame_source=util.list2pose_stamped(current_grasp_pose)
    ))
    current_g2p_transform_mat = util.matrix_from_pose(util.list2pose_stamped(current_g2p_transform_list))
    home_joints = cfg.OUT_OF_FRAME_JOINTS
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
    cropx, cropy, cropz, crop_note = [0.2, 0.75], [-0.5, 0.5], [0.008, 0.4], 'table'
    full_cropx, full_cropy, full_cropz, full_crop_note = [0.0, 0.8], [-0.65, 0.65], [-0.01, 1.0], 'full scene'

    print('\n\nBeginning demo iteration %d\n\n' % demo_iteration)
    got_observation = False
    default_2f140_open_width = 0
    offset_pose_set = None
    offset_pose_relative = [0, 0, 0, 0, 0, 0, 1]
    interaction_point = None

    parent_child_manager = ParentChildObjectManager(mc_vis, parent_class_name=parent_class, child_class_name=child_class, panda_hand_cls=PandaHand) 

    while not rospy.is_shutdown():
        current_g2p_transform_list = util.pose_stamped2list(util.get_transform(
            pose_frame_target=util.list2pose_stamped(current_place_pose),
            pose_frame_source=util.list2pose_stamped(current_grasp_pose)
        ))
        current_g2p_transform_mat = util.matrix_from_pose(util.list2pose_stamped(current_g2p_transform_list))

        if offset_pose_set is not None:
            offset_pose_relative = util.pose_stamped2list(util.get_transform(
                pose_frame_target=util.list2pose_stamped(offset_pose_set), 
                pose_frame_source=util.list2pose_stamped(current_place_pose)
            ))
            util.meshcat_frame_show(mc_vis, 'scene/current_offset_pose', util.matrix_from_pose(util.list2pose_stamped(offset_pose_set)))

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
        if proc_pcd is not None:
            util.meshcat_pcd_show(mc_vis, proc_pcd, color=[255, 0, 0], name='scene/observed_object_pcd')
        if proc_pcd_place is not None:
            util.meshcat_pcd_show(mc_vis, proc_pcd_place, color=[255, 0, 255], name='scene/placed_object_pcd')

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
                rgb, depth = img_subscribers[idx][1].get_rgb_and_depth(block=True)
                rgb_imgs.append(rgb)
                cam_intrinsics = img_subscribers[idx][2].get_cam_intrinsics(block=True)
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

            full_pcd = np.concatenate(pcd_pts, axis=0)
            full_scene_pcd = manually_segment_pcd(full_pcd, x=full_cropx, y=full_cropy, z=full_cropz, note=full_crop_note)
            
            # crop the point cloud to the table
            proc_pcd = manually_segment_pcd(full_pcd, x=cropx, y=cropy, z=cropz, note=crop_note)

            pcd_o3d = open3d.geometry.PointCloud()
            pcd_o3d.points = open3d.utility.Vector3dVector(proc_pcd)
            # labels = np.array(pcd_o3d.cluster_dbscan(eps=0.005, min_points=50, print_progress=True))
            labels = np.array(pcd_o3d.cluster_dbscan(eps=0.008, min_points=20, print_progress=True))
            
            clusters_detected = np.unique(labels)
            pcd_clusters = []
            cluster_sizes = []
            for seg_idx in clusters_detected:
                seg_inds = np.where(labels == seg_idx)[0]
                cluster = proc_pcd[seg_inds]
                pcd_clusters.append(cluster)
                sz = cluster.shape[0]
                cluster_sizes.append(sz)
            top2sz = np.argsort(cluster_sizes)[-2:]

            top2clusters = np.concatenate([pcd_clusters[top2sz[0]], pcd_clusters[top2sz[1]]], axis=0)
            # util.meshcat_multiple_pcd_show(mc_vis, pcd_clusters)
            util.meshcat_pcd_show(mc_vis, top2clusters, name='scene/top2clusters')
            proc_pcd = copy.deepcopy(top2clusters)

            if args.instance_seg_method == 'hand-label':
                log_warn('NOT IMPLEMENTED!')
            elif args.instance_seg_method == 'nn':
                log_warn('NOT IMPLEMENTED!')
            else:  # y-axis
                # get all clusters that are on the positive and negative side of the y-axis
                pos_y_proc_pcd = proc_pcd[np.where(proc_pcd[:, 1] > 0.0)[0]]
                neg_y_proc_pcd = proc_pcd[np.where(proc_pcd[:, 1] < 0.0)[0]]

                # crop each of these point clouds to get the single objects
                log_info('Segmenting the positive y-axis pcd to PARENT and negative y-axis pcd to CHILD')
                parent_child_manager.set_parent_pointcloud(pos_y_proc_pcd)
                parent_child_manager.set_child_pointcloud(neg_y_proc_pcd)

            log_info('Setting the active object to be the PARENT')
            parent_child_manager.set_active_object('parent')

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
            from IPython import embed; embed()
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
                ee_link=cfg.DEFAULT_EE_FRAME
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
                rack_pointcloud_observed=rack_pcd_pts,
                rack_pointcloud_gt=query_point_info.external_object_qp_dict['rack']['surface'],
                rack_pointcloud_gaussian=query_point_info.external_object_qp_dict['rack']['gaussian'],
                rack_pointcloud_uniform=query_point_info.external_object_qp_dict['rack']['uniform'],
                rack_pose_world=rack_pose_world,
                rack_contact_pose=rack_contact_pose,
                shelf_pose_world=shelf_pose_world,
                shelf_pointcloud_observed=shelf_pcd_pts,
                shelf_pointcloud_uniform=query_point_info.external_object_qp_dict['shelf']['uniform'],
                shelf_pointcloud_gt=query_point_info.external_object_qp_dict['shelf']['surface'],
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
                ee_link=cfg.DEFAULT_EE_FRAME,
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
            panda.set_cart_impedance_pose(panda.endpoint_pose(), stiffness=[0]*6)
            continue
        elif user_val == 'l':
            print('\n\nSetting joint positions to current value\n\n')
            panda.move_to_joint_positions(panda.joint_angles())
            continue
        elif user_val == 'r':
            print('\n\nExecuting grasp\n\n')
            if args.gripper_type == '2f140':
                gripper_command = gen_2f140_command('c', gripper_command)
                gripper_pub.publish(gripper_command)
            else:
                panda.hand.grasp(cfg.GRASP_CLOSE_WIDTH, cfg.GRASP_CLOSE_FORCE)
            continue
        elif user_val == 'f':
            print('\n\nOpening grasp\n\n')
            if args.gripper_type == '2f140':
                # gripper_command = gen_2f140_command('o', gripper_command)
                gripper_command = gen_2f140_command(default_2f140_open_width, gripper_command)
                gripper_pub.publish(gripper_command)
            else:
                panda.hand.open()
            continue
        elif user_val == 'g':
            current_grasp_pose = real_robot_util.panda_full_pose(panda.endpoint_pose())
            current_grasp_joints = panda.joint_angles()
            print('\n\nRecording grasp pose!\n\n')
            print(current_grasp_pose)

            parent_child_manager.set_grasp_pose(util.matrix_from_pose(util.list2pose_stamped(current_grasp_pose)))
            log_info(f'Setting current pose as grasp for {parent_child_manager.get_active_object()} object')
            continue
        elif user_val == 'p':
            current_place_pose = real_robot_util.panda_full_pose(panda.endpoint_pose())
            current_place_joints = panda.joint_angles()
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
            offset_pose_set = real_robot_util.panda_full_pose(panda.endpoint_pose())
            print('\n\nRecording offset pose!\n\n')
            print(offset_pose_set)
            offset_pose_relative = util.pose_stamped2list(util.get_transform(
                pose_frame_target=util.list2pose_stamped(offset_pose_set), 
                pose_frame_source=util.list2pose_stamped(current_place_pose)
            ))
            continue
        elif user_val == 'q':
            # this is the pose of the wrist (panda_link8)
            current_ee_query_pose = real_robot_util.panda_full_pose(panda.endpoint_pose())
            util.meshcat_frame_show(mc_vis, 'scene/custom_query_points_pose', util.matrix_from_pose(util.list2pose_stamped(current_ee_query_pose)))

            # this is the pose of the point in between the fingertips
            # current_ee_query_pose = convert_wrist2tip(current_ee_query_pose, wrist2tip_tf=wrist2tip_tf) 
            # util.meshcat_frame_show(mc_vis, 'scene/custom_query_points_pose_tip', util.matrix_from_pose(util.list2pose_stamped(current_ee_query_pose)))

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
            current_panda_plan = panda_mp.plan_joint_target_mp(home_joints, execute=False)
            continue
        elif user_val == 'e':
            print('\n\nExecuting current panda plan!\n\n')
            if len(current_panda_plan) > 0:
                panda.execute_position_path(current_panda_plan)
            else:
                print('\n\nCurrent panda plan is empty!')
            continue
        elif user_val == 'w':
            new_gripper_open_value = input('Please enter an integer value 0-255 for default open grasp width\n')
            # if not isinstance(new_gripper_open_value, int):
            #     print('Invalid value')
            #     continue

            if args.gripper_type != '2f140':
                print('Only valid for 2F140 gripper')
                continue
            default_2f140_open_width = int(np.clip(int(new_gripper_open_value), 0, 255))
            gripper_command = gen_2f140_command(default_2f140_open_width, gripper_command)
            gripper_pub.publish(gripper_command)
            print(f'New default_2f140_open_width: {default_2f140_open_width}')
            continue
        elif user_val == 'clear':
            print('Resetting grasp pose and place pose to current, and "current_g2p_transform_list" to be identity transform')
            print('To create a new relative transformation to apply to the current object, save a grasp and a place pose')

            current_grasp_pose = current_place_pose = real_robot_util.panda_full_pose(panda.endpoint_pose())
            # current_grasp_pose = current_place_pose = [0, 0, 0, 0, 0, 0, 1]
            current_grasp_joints = current_place_joints = panda.joint_angles()
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
    parser.add_argument('--parent_class', type=str, required=True, help='Class name of parent object')
    parser.add_argument('--child_class', type=str, required=True, help='Class name of child object')
    parser.add_argument('--instance_seg_method', type=str, default='y-axis', help='Options: ["y-axis", "hand-label"]')
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
