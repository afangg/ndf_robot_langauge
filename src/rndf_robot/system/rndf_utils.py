import trimesh
import numpy as np
import os, os.path as osp

from airobot.utils import common
from rndf_robot.utils import util, path_util

from rndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list


mesh_data_dirs = {
    'mug': 'mug_centered_obj_normalized', 
    'bottle': 'bottle_centered_obj_normalized', 
    'bowl': 'bowl_centered_obj_normalized',
    'rack': 'syn_racks_easy_obj',
    'container': 'box_containers_unnormalized'
}
mesh_data_dirs = {k: osp.join(path_util.get_rndf_obj_descriptions(), v) for k, v in mesh_data_dirs.items()}

bad_ids = {
    'rack': [],
    'bowl': bad_shapenet_bowls_ids_list,
    'mug': bad_shapenet_mug_ids_list,
    'bottle': bad_shapenet_bottles_ids_list,
    'container': []
}

upright_orientation_dict = {
    'mug': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
    'bottle': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
    'bowl': common.euler2quat([np.pi/2, 0, 0]).tolist(),
    'rack': common.euler2quat([0, 0, 0]).tolist(),
    'container': common.euler2quat([0, 0, 0]).tolist(),
}

scale_default = {
    'mug': 0.3, 
    'bottle': 0.3, 
    'bowl': 0.3,
    'rack': 0.3,
    'container': 1.0,
}

moveable = {'mug', 'bowl', 'bottle'}
static = {'rack', 'mug', 'container'}

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

def get_parent_child_models(descriptor_dirname):
    delim = '--rndf_weights--'
    _, parent_model_path, child_model_path = descriptor_dirname.split(delim)
    parent_model_path = parent_model_path.split('_child')[0]
    parent_model_path = 'ndf_vnn/rndf_weights/' + parent_model_path + '.pth'
    child_model_path = 'ndf_vnn/rndf_weights/' + child_model_path + '.pth'

    return parent_model_path, child_model_path

def reshape_bottle(obj_file_dec, scale, obj_class):
    mesh = trimesh.load(obj_file_dec)
    mesh.apply_scale(scale)
    upright_orientation = upright_orientation_dict['bottle']
    upright_mat = np.eye(4); upright_mat[:-1, :-1] = common.quat2rot(upright_orientation)
    mesh.apply_transformation(upright_mat)
    obj_2d = np.asarray(mesh.vertices)[:, :-1]
    obj_flat = np.hstack([obj_2d, np.zeros(obj_2d.shape[0]).reshape(-1, 1)])
    bbox = trimesh.PointCloud(obj_flat).bounding_box
    extents = bbox.extents
    return bbox, extents
