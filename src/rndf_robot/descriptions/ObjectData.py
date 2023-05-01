import trimesh
import numpy as np
import os, os.path as osp
import random

from airobot.utils import common
from rndf_robot.utils import util, path_util

from rndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
import pybullet as p

OBJECT_CLASSES = {
    'mug': {'mesh_data': 'mug_centered_obj_normalized', 
            'bad_ids': bad_shapenet_mug_ids_list, 
            'upright_orientation': common.euler2quat([np.pi/2, 0, 0]).tolist(),
            'scale_defaults': 0.3},
    'bowl': {'mesh_data': 'bowl_centered_obj_normalized', 
             'bad_ids': bad_shapenet_bowls_ids_list, 
             'upright_orientation': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
             'scale_defaults': 0.3},
    'bottle': {'mesh_data': 'bottle_centered_obj_normalized', 
               'bad_ids': bad_shapenet_bottles_ids_list, 
               'upright_orientation': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
               'scale_defaults': 0.3},
    'rack': {'mesh_data': 'syn_racks_easy_obj', 
             'bad_ids': [], 
             'upright_orientation': common.euler2quat([0, 0, 0]).tolist(), 
             'scale_defaults': 1.0},
    'container': {'mesh_data': 'box_containers_unnormalized', 
                  'bad_ids': [], 
                  'upright_orientation': common.euler2quat([0, 0, 0]).tolist(), 
                  'scale_defaults': 1.0},
}

class ObjectData:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mesh_dirs = self.get_mesh_dirs()
        self.mesh_objs = self.load_meshes_dic()
    
    def get_mesh_dirs(self):
        mesh_dirs = {}
        for obj_class, object_dic in OBJECT_CLASSES.items():
            mesh_dirs[obj_class] = osp.join(path_util.get_rndf_obj_descriptions(), object_dic['mesh_data'])
        return mesh_dirs
    
    def load_meshes_dic(self):
        mesh_names = {}
        for k, v in self.mesh_dirs.items():
            objects_raw = os.listdir(v)
            avoid_ids = OBJECT_CLASSES[k]['bad_ids']
            if k == 'mug':
                avoid_ids += self.cfg.MUG.AVOID_SHAPENET_IDS
            elif k == 'bowl':
                avoid_ids += self.cfg.BOWL.AVOID_SHAPENET_IDS
            elif k == 'bottle':
                avoid_ids += self.cfg.BOTTLE.AVOID_SHAPENET_IDS
            else:
                pass
            objects_filtered = [fn for fn in objects_raw if (fn.split('/')[-1] not in avoid_ids and '_dec' not in fn)]
            mesh_names[k] = objects_filtered
        return mesh_names

    def choose_obj(self, obj_class):
        shapenet_id = random.sample(self.mesh_objs[obj_class], 1)[0]
        # log_debug('Loading %s shape: %s' % (obj_class, shapenet_id))
        if obj_class in ['bottle', 'bowl', 'mug']:
            return osp.join(self.mesh_dirs[obj_class], shapenet_id, 'models/model_normalized.obj')
        # IF IT'S NOT SHAPENET NO NESTED FOLDERS
        else:
            return osp.join(self.mesh_dirs[obj_class], shapenet_id)

def reshape_bottle(obj_file_dec, scale):
    mesh = trimesh.load(obj_file_dec)
    mesh.apply_scale(scale)
    upright_orientation = OBJECT_CLASSES['bottle']['upright_orientation']
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
    upright_orientation = OBJECT_CLASSES[obj_class]['upright_orientation']
    upright_mat = np.eye(4); upright_mat[:-1, :-1] = common.quat2rot(upright_orientation)
    mesh.apply_transform(upright_mat)

    # get the 2D projection of the vertices
    obj_2d = np.asarray(mesh.vertices)[:, :-1]
    flat = np.hstack([obj_2d, np.zeros(obj_2d.shape[0]).reshape(-1, 1)])
    obj_bbox = trimesh.PointCloud(flat).bounding_box
    return obj_bbox.extents
