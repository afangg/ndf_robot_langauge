import trimesh
import meshcat

import sys
import os
sys.path.append(os.environ['SOURCE_DIR'])

from rndf_robot.utils import util

box = trimesh.creation.box([0.15, 0.15, 0.03])
cyl = trimesh.creation.cylinder(radius=0.025, height=0.25)
cyl.apply_translation([0.0, 0.0, 0.15])
box.apply_translation([0.0, 0.0, 0.03/2])

full = trimesh.util.concatenate([box, cyl])

mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
util.meshcat_trimesh_show(mc_vis, 'scene/mesh', full)

full.export('simple_peg.obj')
