import os, os.path as osp
import copy
import numpy as np
import pyrealsense2 as rs

from rndf_robot.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rndf_robot.utils import util, path_util
from rndf_robot.cameras.multicam import MultiCams
from rndf_robot.cameras.simple_multicam import MultiRealsenseLocal
from rndf_robot.cameras.realsense import RealsenseLocal, enable_devices
from rndf_robot.segmentation.pcd_utils import manually_segment_pcd

from airobot import Robot, log_warn
from IPython import embed
class CameraSys:

    def __init__(self, args, mc_vis, cfg, sim_robot=None) -> None:
        self.args = args
        self.mc_vis = mc_vis
        self.cfg = cfg
        if self.args.env_type == 'sim':
            if not isinstance(sim_robot, Robot):
                print("Please pass in a PyBullet Robot to setup cams")
                return
            else:
                self.sim_robot = sim_robot
        self.setup_cams()

    def setup_cams(self):
        if self.args.env_type == 'sim':
            if self.sim_robot is not None:
                self.cams = self.setup_sim_cams()
            else:
                print("Failed to initialize PyBullet cameras. Please pass in the PyBullet robot")
        elif self.args.env_type == 'real':
            self.cams, self.pipelines, self.cam_interface = self.setup_real_cams()
        else:
            raise NotImplementedError('CameraSys: Unfamilar env type')

    def setup_sim_cams(self):
        self.sim_robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, self.cfg.TABLE_Z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)
        cams = MultiCams(self.cfg.CAMERA, self.sim_robot.pb_client, n_cams=self.cfg.N_CAMERAS)
        cam_info = {}
        cam_info['pose_world'] = []
        for cam in cams.cams:
            cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))
        return cams 
    
    def setup_real_cams(self):
        rs_cfg = get_default_multi_realsense_cfg()
        serials = rs_cfg.SERIAL_NUMBERS

        prefix = rs_cfg.CAMERA_NAME_PREFIX
        camera_names = [f'{prefix}{i}' for i in range(len(serials))]
        cam_list = [camera_names[int(idx)] for idx in self.args.cam_index]       
        serials = [serials[int(idx)] for idx in self.args.cam_index]

        calib_dir = osp.join(path_util.get_rndf_cameras(), 'camera_calibration_files')
        calib_filenames = [osp.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in self.args.cam_index]

        cams = MultiRealsenseLocal(cam_names=cam_list, calib_filenames=calib_filenames)
        ctx = rs.context() # Create librealsense context for managing devices

        # Define some constants
        resolution_width = 640 # pixels
        resolution_height = 480 # pixels
        frame_rate = 30  # fps

        pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)
        cam_interface = RealsenseLocal()
        return cams, pipelines, cam_interface
    
    def get_pb_seg(self, obj_id_to_class, table_id=None, crop_show=True):
        '''
        obj_id_to_class (dic): {obj_id: obj_class}

        return: {obj_class: [(score, pcd, obj_id, clip embedding)]}
        '''

        pc_obs_info = {}
        for obj_id in obj_id_to_class:
            pc_obs_info[obj_id] = {'pcd': [], 'rbgs': []}

        all_table_pts = []
        for i, cam in enumerate(self.cams.cams): 
            # get image and raw point cloud
            rgb, depth, pyb_seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)

            seg = pyb_seg
            # flatten and find corresponding pixels in segmentation mask
            flat_seg = seg.flatten()
            for obj_id in obj_id_to_class:
                obj_inds = np.where(flat_seg == obj_id)                
                obj_pts = pts_raw[obj_inds[0], :]
                pc_obs_info[obj_id]['pcd'].append(obj_pts)
                pc_obs_info[obj_id]['rgbs'].append(rgb.flatten()[obj_inds])

            if table_id:
                table_inds = np.where(flat_seg == table_id)                
                table_pts = pts_raw[table_inds[0], :]
                all_table_pts.append(table_pts)

        if all_table_pts != []:
            table_pcd = np.concatenate(all_table_pts, axis=0)  # object shape point cloud
            util.meshcat_pcd_show(self.mc_vis, table_pcd, color=(0,0,255), name='scene/table_pcd')

        pcds_output = {}
        for obj_id, obj_info in pc_obs_info.items():
            if obj_info['pcd'] == []:
                log_warn(f'WARNING: COULD NOT FIND {obj_id} OBJ')
                continue

            target_obj_pcd_obs = np.concatenate(obj_info['pcd'], axis=0)  # object shape point cloud
            target_obj_pcd_obs = manually_segment_pcd(target_obj_pcd_obs, mean_inliers=True)

            if not target_obj_pcd_obs.all():
                log_warn(f'WARNING: COULD NOT FIND {obj_id} OBJ')
                continue

            clip_embeddings = None
            obj_class = obj_id_to_class[obj_id]
            util.meshcat_pcd_show(self.mc_vis, target_obj_pcd_obs, color=(255, 0, 0), name=f'scene/{obj_class}_{obj_id}')

            if obj_class not in pcds_output:
                pcds_output[obj_class] = []
            pcds_output[obj_class].append((1.0, target_obj_pcd_obs, obj_id, clip_embeddings))
        return pcds_output
    
    def get_all_real_views(self, crop_show=True):
        if self.args.env_type != 'real':
            print("This method is for the realsense camera system")
            return
        
        rgb_imgs = []
        pcd_pts = []
        pcd_2ds = []
        depth_imgs = []
        for idx, cam in enumerate(self.cams.cams):
            cam_intrinsics = self.cam_interface.get_intrinsics_mat(self.pipelines[idx])
            rgb, depth = self.cam_interface.get_rgb_and_depth_image(self.pipelines[idx])

            cam.cam_int_mat = cam_intrinsics
            cam._init_pers_mat()
            cam_pose_world = cam.cam_ext_mat

            valid = depth < cam.depth_max
            valid = np.logical_and(valid, depth > cam.depth_min)
            depth_valid = copy.deepcopy(depth)
            depth_valid[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth

            pcd_cam = cam.get_pcd(in_world=False, filter_depth=False, rgb_image=rgb, depth_image=depth_valid)[0]
            pcd_world = util.transform_pcd(pcd_cam, cam_pose_world)
            pcd_world_img = pcd_world.reshape(depth.shape[0], depth.shape[1], 3)

            util.meshcat_pcd_show(self.mc_vis, pcd_world, color=(0, 0, 0), name=f'scene/scene_{idx}')

            rgb_imgs.append(rgb)
            pcd_pts.append(pcd_world)
            pcd_2ds.append(pcd_world_img)
            depth_imgs.append(depth_valid)

        pcd_full = np.concatenate(pcd_pts, axis=0)

        if crop_show:
            cropx, cropy, cropz, crop_note = [0.2, 0.75], [-0.4, 0.4], [0.01, 0.7], 'table'
            bounds = (cropx, cropy, cropz)
            proc_pcd = manually_segment_pcd(pcd_full, bounds=bounds, mean_inliers=True)
        else:
            proc_pcd = pcd_full
        util.meshcat_pcd_show(self.mc_vis, proc_pcd, name=f'scene/cropped_scene')

        # util.meshcat_pcd_show(self.mc_vis, pcd_full, color=(0, 255, 0), name='scene/full_scene')
        return pcd_2ds, rgb_imgs, depth_imgs
