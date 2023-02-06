from airobot import Robot, log_info, set_log_level, log_warn
from rndf_robot.utils.pb2mc.pybullet_meshcat import PyBulletMeshcat
import meshcat
import threading
import time

class VizServer():

    def __init__(self, pb_client) -> None:
        zmq_url = 'tcp://127.0.0.1:6000'
        log_warn(f'Starting meshcat at zmq_url: {zmq_url}')
        self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
        self.mc_vis['scene'].delete()
        self.recorder = PyBulletMeshcat(pb_client=pb_client)
        self.recorder.clear()

    # def setup_event(self):
        #check the scope of this. I think pause_mc_thread is referencing things out of scope
        rec_stop_event = threading.Event()
        rec_run_event = threading.Event()
        rec_th = threading.Thread(target=self.pb2mc_update, args=(self.recorder, self.mc_vis, rec_stop_event, rec_run_event))# , mc_vis))
        rec_th.daemon = True
        rec_th.start()
        self.pause_mc_thread = lambda pause_bool : rec_run_event.clear() if pause_bool else rec_run_event.set()

    def pb2mc_update(self, mc_vis, stop_event, run_event):
        iters = 0
        # while True:
        while not stop_event.is_set():
            run_event.wait()
            iters += 1
            self.recorder.add_keyframe()
            self.recorder.update_meshcat_current_state(mc_vis)
            time.sleep(1/230.0)