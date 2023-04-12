import time
import numpy as np
import pyrealsense2 as rs


class RealsenseLocal:
    def __init__(self):
        self.depth_scale = 0.001

    def get_rgb_and_depth_image(self, pipeline):
        align_to = rs.stream.color
        align = rs.align(align_to)

        device, pipe = pipeline

        try:
            # Get frameset of color and depth
            frames = pipe.wait_for_frames(100)
        except RuntimeError as e:
            # print(e)
            print(f"Couldn't get frame for device: {device}")
            return -1

        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return -1

        depth_image = np.asanyarray(aligned_depth_frame.get_data()) * self.depth_scale
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def get_intrinsics_mat(self, pipeline):
        align_to = rs.stream.color
        align = rs.align(align_to)

        device, pipe = pipeline

        try:
            # Get frameset of color and depth
            frames = pipe.wait_for_frames(100)
        except RuntimeError as e:
            # print(e)
            print(f"Couldn't get frame for device: {device}")
            return np.eye(3)

        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return np.eye(3)

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        intrinsics_matrix = np.array(
            [[depth_intrin.fx, 0., depth_intrin.ppx],
            [0., depth_intrin.fy, depth_intrin.ppy],
            [0., 0., 1.]]
        )
        return intrinsics_matrix


def enable_devices(serials, ctx, resolution_width = 640,resolution_height = 480, frame_rate = 30):
    pipelines = []
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        cfg.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.rgb8, frame_rate)
        pipe.start(cfg)
        time.sleep(1.0)
        pipelines.append([serial,pipe])

    return pipelines


def pipeline_stop(pipelines):
    for (device,pipe) in pipelines:
        # Stop streaming
        pipe.stop()