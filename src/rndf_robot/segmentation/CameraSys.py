
class CameraSys:

    def __init__(self, args, mc_vis, env_type, sim_robot=None) -> None:
        self.args = args
        self.mc_vis = mc_vis
        self.env_type = env_type

        if env_type == 'sim':
            self.sim_robot = sim_robot
            self.cams = self.setup_sim_cams()
        elif env_type == 'real':
            self.cams, self.pipelines, self.cam_interface = self.setup_real_cams()


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

        calib_dir = osp.join(path_util.get_rndf_src(), 'robot/camera_calibration_files')
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