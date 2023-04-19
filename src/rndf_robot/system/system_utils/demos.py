import os, os.path as osp
from rndf_robot.utils import util, path_util

def load_demos_dict():
    all_demos = osp.join(path_util.get_rndf_data(), 'real_demos')
    demo_dic = {}
    for class_pair in os.listdir(all_demos):
        class_pair_path = osp.join(all_demos, class_pair)
        for fname in os.listdir(class_pair_path):
            if not fname.startswith('grasp') and not fname.startswith('place'): continue
            if not fname.endswith('npz'): continue
            verb = fname.split('_demo_')[0]
            demo_path = osp.join(class_pair_path, fname)
            concept = verb + ' ' + class_pair
            if concept not in demo_dic:
                demo_dic[concept] = []
            demo_dic[concept].append(demo_path)
    return demo_dic

all_demos = load_demos_dict()

def get_concept_demos(concept):
    return all_demos[concept] if concept in all_demos else []

def create_target_desc_subdir(demo_path, parent_model_path, child_model_path, create=False):
    parent_model_name_full = parent_model_path.split('ndf_vnn/')[-1]
    child_model_name_full = child_model_path.split('ndf_vnn/')[-1]

    parent_model_name_specific = parent_model_name_full.split('.pth')[0].replace('/', '--')
    child_model_name_specific = child_model_name_full.split('.pth')[0].replace('/', '--')
    
    subdir_name = f'parent_model--{parent_model_name_specific}_child--{child_model_name_specific}'
    dirname = osp.join(demo_path, subdir_name)
    if create:
        util.safe_makedirs(dirname)
    return dirname

def get_model_paths(descriptor_dirname):
    delim = '--rndf_weights--'
    _, parent_model_path, child_model_path = descriptor_dirname.split(delim)
    parent_model_path = parent_model_path.split('_child')[0]
    parent_model_path = 'ndf_vnn/rndf_weights/' + parent_model_path + '.pth'
    child_model_path = 'ndf_vnn/rndf_weights/' + child_model_path + '.pth'

    return parent_model_path, child_model_path