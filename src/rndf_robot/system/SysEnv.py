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

    def intake_segmentation(self, labels_to_pcds):
        for label, infos in labels_to_pcds.items():
            for info in infos:
                score, pcd, obj_id = info
                if obj_id is None:
                    obj_id = len(self.obj_info)
                self.obj_info[obj_id] = {'class': label, 'score': score, 'pcd': pcd}

    def delete_sim_scene(self):
        self.robot.delete_scene(list(self.obj_info.keys()))
        self.obj_info = {}
        self.clear_obj_info()
        self.setup_random_scene()

    def clear_obj_info(self):
        self.ranked_objs = {}     
        self.robot.state = -1
