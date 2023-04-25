import argparse
import os
import os.path as osp
import sys
import numpy as np
import meshcat

sys.path.append(os.environ['SOURCE_DIR'])
from rndf_robot.utils import util

parser = argparse.ArgumentParser()
parser.add_argument('--demo_name', type=str)
parser.add_argument('--demo_keys', nargs='+')

args = parser.parse_args()

path = os.getcwd()

demo_path = osp.join(path, 'release_demos', args.demo_name)
demo = np.load(demo_path, allow_pickle=True)
for key in demo:
    print(key)

if args.demo_keys:
    mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    print(f'MeshCat URL: {mc_vis.url()}')   
    
    for key in args.demo_keys:
        if not isinstance(demo[key], np.ndarray):
            print(f'{key} is of type {type(demo[key])}')
            continue
        color = list(np.random.choice(range(256), size=3))
        util.meshcat_pcd_show(mc_vis, demo[key][0], color=color, name=f'scene/{key}')

