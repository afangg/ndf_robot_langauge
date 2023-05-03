import os, os.path as osp
import torch
import random

from rndf_robot.robot.Robot import Robot
from rndf_robot.robot.RealRobot import RealRobot
from rndf_robot.robot.SimRobot import SimRobot

from rndf_robot.descriptions.ObjectData import ObjectData, OBJECT_CLASSES

from rndf_robot.utils import util, path_util
from rndf_robot.config.default_eval_cfg import get_eval_cfg_defaults

from IPython import embed
from airobot import log_debug, log_warn, log_info

class Environment:
    def __init__(self, args, mc_vis, scene_objs=None) -> None:
        '''
        ranked_objs (dict): Maps a rank (0 for child, 1 for parent) to its ID in obj_info
        obj_info (dict): Maps an object ID to a dictionary containing information about the obj
            i.e. class, pcd, rank, description
        desc_to_id (dict): Maps a language description to a set of object IDs that match the 
            description.
        '''

        self.mc_vis = mc_vis
        self.args = args
        self.scene_objs = scene_objs
        self.robot = None
        
        self.cfg = self.get_env_cfgs()
        self.object_data = ObjectData(self.cfg)
        self.ranked_objs = {}
        self.obj_info = {}
        self.desc_to_id = {}

    def set_robot(self, robot: Robot):
        self.robot = robot
        self.setup_random_scene()

    def next_iter(self):
        while True:
            i = input(
                '''What should we do
                    [h]: Move to home
                    [o]: Open gripper
                    [n]: Continue to next iteration
                    [g]: *SIM ONLY* - generate a new random scene from configs
                    [i]: *REAL ONLY* - to set into low stiffness mode
                    [l]: *REAL ONLY* - to lock into current configuration with high stiffness
                    [em]: Launch interactive mode
                    [clear]: Clears the env variables - will segment everything again
                ''')
            
            if i == 'h':
                self.robot.go_home()
                continue
            elif i == 'o':
                self.robot.gripper_state(open=True)
            elif i == 'n':
                self.mc_vis['scene'].delete()
                self.mc_vis['optimizer'].delete()
                self.mc_vis['ee'].delete()
                break
            elif i == 'g':
                self.robot.delete_scene(list(self.obj_info.keys()))
                self.obj_info = {}
                self.ranked_objs = {}     

                self.setup_random_scene()
                continue
            elif i == 'i':
                if isinstance(self.robot, RealRobot):
                    self.robot.gravity_comp(on=True)
                continue
            elif i == 'l':
                if isinstance(self.robot, RealRobot):
                    self.robot.gravity_comp(on=False)
                continue
            elif i =='em':
                embed()
            elif i == 'clear':
                self.ranked_objs = {}     
                self.robot.state = -1
                continue
            else:
                print('Unknown command')
                continue
        torch.cuda.empty_cache()

    def assign_nouns(self, rank_to_class):
        log_warn('Clearing ranked object dictionary')
        self.ranked_objs = rank_to_class

    def assign_pcds(self, labels_to_pcds, re_seg=True):
        '''
        labels_to_pcds (dic): {'label': [(score, pcd, obj_id/None)]}

        return: Boolean signaling if all objects have pcds 
        '''
        # pick the pcd with the highest score
        for label in labels_to_pcds:
            labels_to_pcds[label].sort(key=lambda x: x[0])

        for obj_rank in self.ranked_objs:
            # if we decide to not re-segment, must exclude alreadu assign pcds
            if not re_seg and 'pcd' in self.ranked_objs[obj_rank]: continue
            description = self.ranked_objs[obj_rank]['description']
            obj_class = self.ranked_objs[obj_rank]['potential_class']

            if description in labels_to_pcds:
                pcd_key = description
            elif obj_class in labels_to_pcds:
                pcd_key = obj_class
            else:
                log_warn(f'Could not find pcd for ranked obj {obj_rank}')
                return False

            score, pcd, obj_id = labels_to_pcds[pcd_key].pop(-1)
            self.ranked_objs[obj_rank]['pcd'] = pcd
            self.ranked_objs[obj_rank]['obj_id'] = obj_id

            log_debug(f'Best score for {pcd_key} was {score}')

        for rank, obj in self.ranked_objs.items():
            label = f'scene/initial_{rank}_pcd'
            color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
            util.meshcat_pcd_show(self.mc_vis, obj['pcd'], color=color, name=label)

        return True
    
    def get_relevant_pcds(self):
        ranks = sorted(self.ranked_objs.keys())
        return [self.ranked_objs[rank]['pcd'] for rank in ranks]

    def get_relevant_classes(self):
        ranks = sorted(self.ranked_objs.keys())
        return [self.ranked_objs[rank]['potential_class'] for rank in ranks]
    
    def get_env_cfgs(self):
        # general experiment + environment setup/scene generation configs
        cfg = get_eval_cfg_defaults()
        # if 0 not in self.ranked_objs: return

        # config = 'base_config.yaml' if 1 in self.ranked_objs else 'eval_'+self.obj_info[self.ranked_objs[0]]['class']+'_gen.yaml'
        config = 'base_config.yaml'
        config_fname = osp.join(path_util.get_rndf_config(), 'eval_cfgs', config)
        if osp.exists(config_fname):
            cfg.merge_from_file(config_fname)
            log_debug('Config file loaded')
        else:
            log_info('Config file %s does not exist, using defaults' % config_fname)
        cfg.freeze()
        return cfg

    def setup_random_scene(self):
        '''
        @config_dict: Key are ['objects': {'class': #}]
        '''
        if self.scene_objs is None or not isinstance(self.robot, SimRobot):
            print('Can not generate scene without PB or scene config')
            return

        for obj_class, colors in self.scene_objs['objects'].items():
            for color, n in colors.items():
                for _ in range(n):
                    
                    obj_file = self.object_data.choose_obj(obj_class)
                    obj_scale = OBJECT_CLASSES[obj_class]['scale_defaults']
                    obj_up_ori = OBJECT_CLASSES[obj_class]['upright_orientation']

                    sim_obj_data = self.robot.add_obj(obj_file, 
                                                      obj_scale, 
                                                      obj_up_ori, 
                                                      existing_objs=self.obj_info,
                                                      color=color)
                    obj_id, obj_pose_world = sim_obj_data
                    obj = {
                        'class': obj_class,
                        'pose': obj_pose_world,
                        'rank': -1
                    }
                    self.obj_info[obj_id] = obj

    def update_obj_info(self, obj_id, obj):
        '''
        Updates the info stored in self.obj_info with the key obj_id with values in obj
        Warning: May override existing information

        Args:
            obj_id (dic): 
            obj (dic): Dictionary containing information about the object to add
        '''
        assert obj_id in self.obj_info
        for key, val in obj:
            self.obj_info[obj_id][key] = val

    def add_obj(self, obj):
        '''
        Adds a new object to self.obj_info and the information contained in obj.

        Args:
            obj (dic): Dictionary containing information about the object to add
        '''
        new_key = len(self.obj_info)
        self.obj_info[new_key] = {}
        self.update_obj_info(new_key, obj)

    def add_attribute(self, attribute_key, attribute_val, obj_id=None):
        '''
        Adds an attribute with specified key and value to self.obj_info. If obj_id is specified,
            it will update the obj dictionary, otherwise it adds a new obj to the dic
        Warning: May override existing information

        Args:
            attribute_key (str): key in object dictionary
            attribute_val (any): value in object dictionary
            optional:
                obj_id (int): ID of the object's dictionary to update with the new attribute

        '''
        attribute_dic = {attribute_key: attribute_val}
        if not obj_id:
            self.add_obj(attribute_dic)
        else:
            self.update_obj_info(obj_id, attribute_dic)

    def add_pcd(self, pcd, obj_id=None):
        '''
        Adds a pointcloud to the object info dictionary with the key 'pcd'. 
        obj_id is optional if object already exists
        '''
        self.add_attribute(self, 'pcd', pcd, obj_id=obj_id)

    def add_segmentation_instance(self, rgb, obj_id=None):
        '''
        Adds a list of rgb images to the object info dictionary with the key 'rgb'. 
        obj_id is optional if object already exists
        '''
        self.add_attribute(self, 'rgb', rgb, obj_id=obj_id)

