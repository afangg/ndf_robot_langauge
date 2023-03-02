import trimesh
import numpy as np
import os, os.path as osp
import random

from airobot.utils import common
from rndf_robot.utils import util, path_util

from rndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
import pybullet as p

# should i make this a class? maybe

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
    'rack': 1.0,
    'container': 1.0,
}

moveable = {'mug', 'bowl', 'bottle'}
static = {'rack', 'mug', 'container'}

def choose_obj(meshes_dic, obj_class):
    shapenet_id = random.sample(meshes_dic[obj_class], 1)[0]
    # log_debug('Loading %s shape: %s' % (obj_class, shapenet_id))
    if obj_class in ['bottle', 'bowl', 'mug']:
        return osp.join(mesh_data_dirs[obj_class], shapenet_id, 'models/model_normalized.obj')
    # IF IT'S NOT SHAPENET NO NESTED FOLDERS
    else:
        return osp.join(mesh_data_dirs[obj_class], shapenet_id)


def load_meshes_dict(cfg):
    mesh_names = {}
    for k, v in mesh_data_dirs.items():
        # get train samples
        objects_raw = os.listdir(v)
        avoid_ids = bad_ids[k]
        if k == 'mug':
            avoid_ids += cfg.MUG.AVOID_SHAPENET_IDS
        elif k == 'bowl':
            avoid_ids += cfg.BOWL.AVOID_SHAPENET_IDS
        elif k == 'bottle':
            avoid_ids += cfg.BOTTLE.AVOID_SHAPENET_IDS
        else:
            pass
        objects_filtered = [fn for fn in objects_raw if (fn.split('/')[-1] not in avoid_ids and '_dec' not in fn)]
        # objects_filtered = objects_raw
        total_filtered = len(objects_filtered)
        train_n = int(total_filtered * 0.9); test_n = total_filtered - train_n

        train_objects = sorted(objects_filtered)[:train_n]
        test_objects = sorted(objects_filtered)[train_n:]

        mesh_names[k] = objects_filtered
    return mesh_names


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

def get_extents(obj_file, obj_class, scale_default,):
    obj_file_dec = obj_file.split('.obj')[0] + '_dec.obj'
    if not osp.exists(obj_file_dec):
        p.vhacd(
            obj_file,
            obj_file_dec,
            'log.txt',
            concavity=0.0025,
            alpha=0.04,
            beta=0.05,
            gamma=0.00125,
            minVolumePerCH=0.0001,
            resolution=1000000,
            depth=20,
            planeDownsampling=4,
            convexhullDownsampling=4,
            pca=0,
            mode=0,
            convexhullApproximation=1
        )
    
    mesh = trimesh.load(obj_file_dec)
    mesh.apply_scale(scale_default)

    # make upright
    upright_orientation = upright_orientation_dict[obj_class]
    upright_mat = np.eye(4); upright_mat[:-1, :-1] = common.quat2rot(upright_orientation)
    mesh.apply_transform(upright_mat)

    # get the 2D projection of the vertices
    obj_2d = np.asarray(mesh.vertices)[:, :-1]
    flat = np.hstack([obj_2d, np.zeros(obj_2d.shape[0]).reshape(-1, 1)])
    obj_bbox = trimesh.PointCloud(flat).bounding_box
    return obj_bbox.extents