import os, os.path as osp
from rndf_robot.utils import util, path_util

def load_demos_dict():
    all_demos = osp.join(path_util.get_rndf_data(), 'release_real_demos')
    demo_dic = {}
    for demo_type in os.listdir(all_demos):
        demo_type_folder = osp.join(all_demos, demo_type)
        for demo_npz in os.listdir(demo_type_folder):
            if not demo_npz.startswith('grasp') and not demo_npz.startswith('place'): continue
            if not demo_npz.endswith('npz'): continue
            verb = demo_npz.split('_demo_')[0]
            demo_path = osp.join(demo_type_folder, demo_npz)
            concept = verb + ' ' + demo_type
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