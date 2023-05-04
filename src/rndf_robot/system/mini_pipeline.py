import torch
import argparse
import meshcat
import random
import numpy as np

from rndf_robot.system.SysEnv import Environment
from rndf_robot.robot.Robot import Robot
from rndf_robot.data.NDFLibrary import NDFLibrary
from rndf_robot.language.PromptModule import PromptModule
from rndf_robot.segmentation.VisionModule import VisionModule
from rndf_robot.cameras.CameraSys import CameraSys

from rndf_robot.robot.SimRobot import SimRobot
from rndf_robot.robot.RealRobot import RealRobot

from rndf_robot.descriptions.ObjectData import OBJECT_CLASSES
from airobot import log_debug, log_warn, log_info

from IPython import embed

config = dict(
    # objects={'mug': {(1,0,0.1,1):1, (0,0.5,0.8,1):1}}
    objects = {'mug': {(0,1,0.1,1):1}, 'container': {(0,0.5,0.8,1):1}}
)
class Pipeline:

    def __init__(self, args):
        self.mc_vis = meshcat.Visualizer(zmq_url=f'tcp://127.0.0.1:{args.port_vis}')
        self.args = args
        obj_classes = list(OBJECT_CLASSES.keys())
        obj_models = {obj_class: OBJECT_CLASSES[obj_class]['model_weights'] for obj_class in obj_classes}

        folder = 'release_demos' if args.env_type == 'sim' else 'real_release_demos'
        self.system_env = Environment(args, self.mc_vis, scene_objs=config)
        self.ndf_lib = NDFLibrary(args, self.mc_vis, self.system_env.cfg, folder, obj_classes=obj_classes, obj_models=obj_models)

        if args.env_type == 'sim':
            self.robot = SimRobot(args, self.ndf_lib, self.mc_vis, self.system_env.cfg)
            sim_robot = self.robot.robot
        elif args.env_type == 'real':
            self.robot = RealRobot(args, self.ndf_lib, self.mc_vis, self.system_env.cfg)
            sim_robot = None
        else:
            raise NotImplementedError('Not sure what env type this is')

        self.system_env.set_robot(self.robot)
        camera_sys = CameraSys(args, self.mc_vis, self.system_env.cfg, sim_robot)
        self.vision_mod = VisionModule(args.seg_method, self.mc_vis)    
        self.vision_mod.initialize_cameras(camera_sys)

        skill_names = list(self.ndf_lib.default_demos.keys())
        self.prompter = PromptModule(skill_names, obj_classes)

        # primitive_template = self.robot.skill_library.get_primitive_templates()
        # print(primitive_template)
    
    def ask(self):
        prompt = self.prompter.prompt_user()
        if not prompt:
            return None
        
        action, geometry = self.prompter.decompose_demo(prompt) 

        if action.startswith('grasp'):
            state = 0
        elif action.startswith('place'):
            state = 1
        elif action.startswith('place') and state == -1:
            state = 2

        self.robot.state = state

        keywords, rank_to_class = self.prompter.get_keywords_and_classes(prompt, state)
        self.system_env.assign_nouns(rank_to_class)
        obj_desc = [keyword[1] if keyword[1] else keyword[0] for keyword in keywords]
        return action, self.system_env.get_relevant_classes(), geometry, obj_desc
    
    def find(self, obj_desc):
        if self.args.env_type == 'real':
            labels = obj_desc
        elif self.args.env_type == 'sim':
            labels = {}
            for obj_id, info in self.system_env.obj_info.items():
                labels[obj_id] = info['class']
            
        labels_to_pcds = self.vision_mod.segment_all_scenes(labels)
        if not labels_to_pcds:
            log_warn('WARNING: Target object not detected, try again')
            return False, None
        
        if self.args.seg_method != 'pb_seg':
            i = input('Happy with segmentation (y/n)')
            if i == 'n':
                return False, None

        labels_to_pcds_copy = labels_to_pcds.copy()
        success = self.system_env.assign_pcds(labels_to_pcds_copy)
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
            self.system_env.next_iter()

def run(system_env: Environment, FUNCTIONS):
    prompt_output = FUNCTIONS['ask']()
    torch.cuda.empty_cache()
    if prompt_output is None:
        return
    
    action, obj_classes, geometry, obj_desc = prompt_output
    segmentation_success, pcds = FUNCTIONS['find'](obj_desc)
    torch.cuda.empty_cache()
    if not segmentation_success:
        return
    
    if action not in FUNCTIONS:
        print('The skill outputted by the LLM does not exist')
        return
    
    skill_func = FUNCTIONS[action]

    pcd_class_pairs = zip(pcds, obj_classes)
    skill_func(*pcd_class_pairs, geometry)
    torch.cuda.empty_cache()

    
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
    parser.add_argument('--env_type', type=str, required=True)

    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--opt_iterations', type=int, default=500)

    parser.add_argument('--seg_method', type=str, help='[bboxes, point, pb_seg, owl]')

    # hardware things
    parser.add_argument('--gripper_type', type=str, default='panda')
    parser.add_argument('--cam_index', nargs='+', help='set which cameras to get point cloud from', required=False)

    #sim things
    parser.add_argument('--table_type', type=str, default='default')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    pipline = Pipeline(args)
    pipline.main()