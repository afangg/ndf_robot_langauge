import trimesh
import meshcat

import sys
import os
sys.path.append(os.environ['SOURCE_DIR'])

from rndf_robot.utils import util

box = trimesh.creation.box([0.2, 0.2, 0.04])
cyl = trimesh.creation.cylinder(radius=0.025, height=0.3)
cyl.apply_translation([0.0, 0.0, 0.15])
box.apply_translation([0.0, 0.0, 0.002])

full = trimesh.util.concatenate([box, cyl])

mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
util.meshcat_trimesh_show(mc_vis, 'scene/mesh', full)

full.export('simple_peg.obj')
