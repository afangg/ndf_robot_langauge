import argparse
import sys
import os
import meshcat

from rndf_robot.system.SysEnv import Environment
from rndf_robot.data.NDFLibrary import NDFLibrary
from rndf_robot.language.PromptModule import PromptModule
from rndf_robot.segmentation.VisionModule import VisionModule
from rndf_robot.cameras.CameraSys import CameraSys

from rndf_robot.descriptions import ObjectData
from rndf_robot.robot.SimRobot import SimRobot
from rndf_robot.robot.RealRobot import RealRobot

from rndf_robot.descriptions.ObjectData import OBJECT_CLASSES
from airobot import log_debug, log_warn, log_info

config = dict(
    # objects={'mug': {(1,0,0.1,1):1, (0,0.5,0.8,1):1}}
    objects = {'mug': {(0,1,0.1,1):1}, 'rack': {(0,0.5,0.8,1):1}}
)
class Pipeline:

    def __init__(self, args):
        self.mc_vis = meshcat.Visualizer(zmq_url=f'tcp://127.0.0.1:{args.port_vis}')
        self.args = args

        if args.env_type == 'sim':
            folder = 'sim_demos'
            self.robot = SimRobot(self.ndf_lib)
            sim_robot = self.robot
        elif args.env_type == 'real':
            folder = 'real_demos'
            self.robot = RealRobot(self.ndf_lib)
            sim_robot = None
        else:
            raise NotImplementedError('Not sure what env type this is')

        self.ndf_lib = NDFLibrary(args, folder, obj_classes=OBJECT_CLASSES)
        self.system_env = Environment(args, self.mc_vis, self.robot, config=config)

        camera_sys = CameraSys(args, self.mc_vis, sim_robot)
        self.vision_mod = VisionModule(args.seg_method, args, self.mc_vis)    
        self.vision_mod.initialize_cameras(camera_sys)

        self.prompter = PromptModule(self.ndf_lib.default_demos)

        primitive_template = self.robot.skill_library.get_primitive_templates()
        print(primitive_template)
    
    def ask(self):
        prompt = self.prompter.prompt_user()
        if not prompt:
            return
        
        action, geometry = self.prompter.decompose_demo(prompt) 

        if action.startswith('grasp'):
            state = 0
        elif action.startswith('place'):
            state = 1
        elif action.startswith('place') and state == -1:
            state = 2

        keywords, rank_to_class = self.prompter.get_keywords_and_classes(prompt, state)
        self.system_env.assign_nouns(rank_to_class)

        obj_desc = [keyword[1] if keyword[1] else keyword[0] for keyword in keywords]
        return action, self.system_env.get_relevant_classes(), geometry, obj_desc
    
    def find(self, obj_desc):
    
        if self.args.env_type == 'real':
            labels = obj_desc
        elif self.args.env_type == 'sim':
            labels = [obj_id for obj_id in self.system_env.obj_info]
            
        labels_to_pcds = self.vision_mod.segment_all_scenes(labels)
        if not labels_to_pcds:
            log_warn('WARNING: Target object not detected, try again')
            return
        
        i = input('Happy with segmentation (y/n)')
        if i == 'n':
            return False, None
    
        success = self.system_env.assign_pcds(labels_to_pcds)
        if not success:
            return False, None
        else:
            return True, self.system_env.get_relevant_pcds()

    def main(self):
        self.FUNCTIONS = {'ask': self.ask,
                          'find': self.find,
                          'grasp': self.robot.grasp, 
                          'place': self.robot.place, 
                          'place_relative': self.robot.place_relative,}
        
        self.system_env.next_iter()
        while True:
            run(self.system_env, self.FUNCTIONS)

def run(system_env: Environment, FUNCTIONS):
    action, obj_classes, geometry, obj_desc = FUNCTIONS['ask']()
    segmentation_success, pcds = FUNCTIONS['find'](obj_desc)
    if not segmentation_success:
        return

    skill_func = FUNCTIONS[action]

    pcd_class_pairs = zip(pcds, obj_classes)
    skill_func(*pcd_class_pairs, geometry)
    system_env.next_iter()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=5)

    parser.add_argument('--port_vis', type=int, default=6000)
    parser.add_argument('--pybullet_viz', action='store_true')
    parser.add_argument('--show_pcds', action='store_true')
    parser.add_argument('--opt_visualize', action='store_true')
    parser.add_argument('--grasp_viz', action='store_true')
    parser.add_argument('--env_type', type=str)

    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--opt_iterations', type=int, default=500)

    parser.add_argument('--seg_method', type=str, default='bboxes')

    # hardware things
    parser.add_argument('--gripper_type', type=str, default='panda')
    parser.add_argument('--cam_index', nargs='+', help='set which cameras to get point cloud from', required=True)

    args = parser.parse_args()

    pipline = Pipeline(args)
    pipline.main()