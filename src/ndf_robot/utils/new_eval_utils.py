import numpy as np
import trimesh
from scipy.spatial import KDTree
import time
import pybullet as p
import copy
from ndf_robot.utils import util, trimesh_util

def soft_grasp_close(robot, joint_id2, force=100):
    p.setJointMotorControl2(robot.arm.robot_id, joint_id2, p.VELOCITY_CONTROL, targetVelocity=-1, force=force)
    # p.setJointMotorControl2(robot.arm.robot_id, joint_id2, p.VELOCITY_CONTROL, targetVelocity=-1, force=100)
    time.sleep(0.2)        

def safeRemoveConstraint(cid):
    if cid is not None:
        p.removeConstraint(cid)

def safeCollisionFilterPair(bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB, enableCollision, *args, **kwargs):
    if bodyUniqueIdA is not None and bodyUniqueIdB is not None and linkIndexA is not None and linkIndexB is not None:
        p.setCollisionFilterPair(bodyUniqueIdA=bodyUniqueIdA, bodyUniqueIdB=bodyUniqueIdB, linkIndexA=linkIndexA, linkIndexB=linkIndexB, enableCollision=enableCollision)

def object_is_still_grasped(robot, obj_id, right_pad_id, left_pad_id):
    obj_finger_right_info = p.getClosestPoints(bodyA=obj_id, bodyB=robot.arm.robot_id, distance=0.002,
                                            linkIndexA=-1, linkIndexB=right_pad_id)
    obj_finger_left_info = p.getClosestPoints(bodyA=obj_id, bodyB=robot.arm.robot_id, distance=0.002,
                                            linkIndexA=-1, linkIndexB=left_pad_id)
    obj_still_in_grasp = len(obj_finger_left_info) > 0 or len(obj_finger_right_info) > 0
    return obj_still_in_grasp

def constraint_grasp_close(robot, obj_id):
    obj_pose_world = p.getBasePositionAndOrientation(obj_id)
    obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))

    ee_link_id = robot.arm.ee_link_id
    ee_pose_world = np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()
    ee_pose_world = util.list2pose_stamped(ee_pose_world)

    obj_pose_ee = util.convert_reference_frame(
        pose_source=obj_pose_world,
        pose_frame_target=ee_pose_world,
        pose_frame_source=util.unit_pose()
    )
    obj_pose_ee_list = util.pose_stamped2list(obj_pose_ee)

    cid = p.createConstraint(
        parentBodyUniqueId=robot.arm.robot_id,
        parentLinkIndex=ee_link_id,
        childBodyUniqueId=obj_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=obj_pose_ee_list[:3],
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=obj_pose_ee_list[3:])
    return cid

def process_xq_data(data, shelf=True):
    optimizer_gripper_pts = data['gripper_pts_uniform']
    return optimizer_gripper_pts

def process_xq_data_rs(data, shelf=True):
    optimizer_gripper_pts_rs = data['gripper_pts']
    return optimizer_gripper_pts_rs

def process_demo_data(data):
    demo_obj_pts = data['object_pointcloud']  # observed shape point cloud at start
    demo_pts_mean = np.mean(demo_obj_pts, axis=0)
    inliers = np.where(np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
    demo_obj_pts = demo_obj_pts[inliers]

    demo_gripper_pts_rs = data['gripper_pts']  # points we use to represent the gripper at their canonical pose position shown in the demonstration
    demo_gripper_pcd_rs = trimesh.PointCloud(demo_gripper_pts_rs)
    demo_ee_mat = util.matrix_from_pose(util.list2pose_stamped(data['ee_pose_world']))  # end-effector pose before grasping
    demo_gripper_pcd_rs.apply_transform(demo_ee_mat)
    demo_gripper_pts_rs = np.asarray(demo_gripper_pcd_rs.vertices) # points we use to represent the gripper at their canonical pose position shown in the demonstration

    # demo_gripper_pts = data['gripper_pts_gaussian'] * gaussian_scale # query points for the gripper (Gaussian distributed)
    demo_gripper_pts = data['gripper_pts_uniform'] # query points for the gripper (Uniform distributed)
    demo_gripper_pcd = trimesh.PointCloud(demo_gripper_pts)
    demo_ee_mat = util.matrix_from_pose(util.list2pose_stamped(data['ee_pose_world']))  # end-effector pose before grasping
    demo_gripper_pcd.apply_transform(demo_ee_mat)
    demo_gripper_pts = np.asarray(demo_gripper_pcd.vertices) # points we use to represent the gripper at their canonical pose position shown in the demonstration

    target_info = dict(
        demo_query_pts=demo_gripper_pts, 
        demo_query_pts_real_shape=demo_gripper_pts_rs,
        demo_obj_pts=demo_obj_pts, 
        demo_ee_pose_world=data['ee_pose_world'],
        demo_query_pt_pose=data['gripper_contact_pose'],
        demo_obj_rel_transform=np.eye(4))

    shapenet_id = data['shapenet_id'].item()
    return target_info, shapenet_id

def post_process_grasp(ee_pose, target_obj_pcd, thin_feature=True, grasp_viz=True, grasp_dist_thresh=0.0025):
    
    grasp_pt = ee_pose[:3]
    rix = np.random.permutation(target_obj_pcd.shape[0])
    target_obj_voxel_down = target_obj_pcd[rix[:int(target_obj_pcd.shape[0]/5)]]

    target_obj_tree = KDTree(target_obj_pcd)
    target_obj_down_tree = KDTree(target_obj_voxel_down)
    grasp_close_idxs = target_obj_tree.query(grasp_pt, k=100)[1]
    grasp_close_pts = target_obj_pcd[grasp_close_idxs].squeeze()

    n_pts_within_ball = 0
    k = 0
    while True:
        # get the mean of the nearest neighbors, and check how many points are inside the ball with size roughly the contact patch of the fingers
        new_grasp_pt = np.mean(grasp_close_pts, axis=0)
        new_idxs_within_ball = target_obj_down_tree.query_ball_point(new_grasp_pt, r=0.015)
        new_idxs_within_larger_ball = target_obj_down_tree.query_ball_point(new_grasp_pt, r=0.02)
        pts_within_ball = target_obj_voxel_down[new_idxs_within_ball].squeeze()
        pts_within_larger_ball = target_obj_voxel_down[new_idxs_within_larger_ball].squeeze()
        n_pts_within_ball = len(new_idxs_within_ball)

        if n_pts_within_ball > 75:
            break
        else:
            grasp_pt = np.mean(pts_within_larger_ball, axis=0) 
            grasp_close_idxs = target_obj_tree.query(grasp_pt, k=100)[1]
            grasp_close_pts = target_obj_pcd[grasp_close_idxs].squeeze()
        
        k += 1
        if k > 5:
            break

    # get antipodal point
    # local_grasp_normal = np.mean(np.asarray(close_o3d.normals), axis=0)
    local_grasp_normal = util.vec_from_pose(util.list2pose_stamped(ee_pose))[1]

    if thin_feature:
        # sample along this direction to find an antipodal point
        search_vec = -1.0 * local_grasp_normal
        search_final_pt = grasp_pt + search_vec
        search_pts = np.linspace(grasp_pt, search_final_pt, 250)
        min_dist = np.inf
        for pt in search_pts:
            a_close_idx = target_obj_tree.query(pt, k=1)[1]
            a_close_pt = target_obj_pcd[a_close_idx].squeeze()
            dist = np.linalg.norm(a_close_pt - pt)
            if dist < min_dist and (np.linalg.norm(a_close_pt - grasp_pt) > grasp_dist_thresh):
                antipodal_close_idx = a_close_idx
                antipodal_close_pt = a_close_pt
                min_dist = dist
        detected_pt = copy.deepcopy(grasp_pt)
        new_grasp_pt = (antipodal_close_pt + grasp_pt) / 2.0  

    if grasp_viz:
    # if True:
        # scene = trimesh_util.trimesh_show(
        #     [target_obj_pcd_obs[::5], grasp_close_pts, pts_within_ball], show=False)
        scene = trimesh_util.trimesh_show(
            [target_obj_voxel_down, grasp_close_pts, pts_within_ball], show=False)
        print('GRASP PT', grasp_pt)
        scale = 0.1
        grasp_sph = trimesh.creation.uv_sphere(0.005)
        grasp_sph.visual.face_colors = np.tile([40, 40, 40, 255], (grasp_sph.faces.shape[0], 1))
        grasp_sph.apply_translation(grasp_pt)

        new_grasp_sph = trimesh.creation.uv_sphere(0.005)
        new_grasp_sph.visual.face_colors = np.tile([40, 255, 40, 255], (new_grasp_sph.faces.shape[0], 1))
        new_grasp_sph.apply_translation(new_grasp_pt)
        scene.add_geometry([grasp_sph, new_grasp_sph])

        ee_pose[:3] = new_grasp_pt
        pregrasp_offset_tf = get_ee_offset(ee_pose=ee_pose)
        pre_ee_pose = util.pose_stamped2list(
            util.transform_pose(pose_source=util.list2pose_stamped(ee_pose), pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        ee_pose_mat = util.matrix_from_pose(util.list2pose_stamped(pre_ee_pose))
        ee_mesh = trimesh.load('../floating/panda_gripper.obj')
        scene.add_geometry([ee_mesh], transform=ee_pose_mat)
        print('END EFFECTOR POSE:', ee_pose_mat)

        scene.show()

    return new_grasp_pt

def get_ee_offset(ee_pose):
    """
    Gets the updated world frame normal direction of the palms
    """
    dist = 0.1
    normal_x = util.list2pose_stamped([dist, 0, 0, 0, 0, 0, 1])
    normal_y = util.list2pose_stamped([0, dist, 0, 0, 0, 0, 1])
    normal_z = util.list2pose_stamped([0, 0, dist, 0, 0, 0, 1])

    normal_x = util.transform_pose(normal_x, util.list2pose_stamped(ee_pose))
    normal_y = util.transform_pose(normal_y, util.list2pose_stamped(ee_pose))
    normal_z = util.transform_pose(normal_z, util.list2pose_stamped(ee_pose))
    
    dx_vec = np.asarray(ee_pose)[:3] - util.pose_stamped2np(normal_x)[:3]
    dy_vec = np.asarray(ee_pose)[:3] - util.pose_stamped2np(normal_y)[:3]
    dz_vec = np.asarray(ee_pose)[:3] - util.pose_stamped2np(normal_z)[:3]

    return dz_vec.tolist() + [0, 0, 0, 1]

def constraint_obj_world(obj_id, pos, ori):
    o_cid = p.createConstraint(
        obj_id,
        -1,
        -1,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        pos, childFrameOrientation=ori,
    )
    return o_cid
