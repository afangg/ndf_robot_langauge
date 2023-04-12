import numpy as np
import pyrealsense2 as rs

rs_pipe = rs.pipeline()
config = rs.config()
config.enable_device('013102060174')
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = rs_pipe.start()

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
# decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()

try:
    while True:
        # Stream frames
        frames = rs_pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        rgb_frame = frames.get_color_frame()
        if not depth_frame: continue
        
        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(rgb_frame.get_data())

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        mapped_frame, color_source = rgb_frame, color_image


        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        break
finally:
    rs_pipe.stop()
