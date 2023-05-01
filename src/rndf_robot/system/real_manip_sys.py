from pipeline_real import Pipeline 
from system_utils.vizServer import VizServer 

import argparse
from airobot import log_info, set_log_level, log_warn
from IPython import embed;
import torch

def main(pipeline):
    
    prompt = pipeline.prompt_user()
    if not prompt: 
        pipeline.next_iter()
        return
    corresponding_concept, query_text = prompt
    demos = get_concept_demos(corresponding_concept)
    log_debug('Number of Demos %s' % len(demos)) 
        
    self.skill_demos = demos
    if corresponding_concept.startswith('grasp'):
        self.state = 0
    elif corresponding_concept.startswith('place'):
        self.state = 1
    # elif corresponding_concept.startswith('place') and self.state == -1:
    #     self.state = 2
    log_debug('Current State is %s' %self.state)
    
    concept, keywords, rank_to_class = pipeline.identify_classes_from_query(query_text, corresponding_concept)
    pipeline.set_nouns(rank_to_class)
    torch.cuda.empty_cache()

    descriptions = [keyword[1] if keyword[1] else keyword[0] for keyword in keywords]
    labels_to_pcds = pipeline.segment_scene(descriptions)

    if not labels_to_pcds:
        log_warn('WARNING: Target object not detected, try again')
        return
    
    i = input('Happy with segmentation (y/n)')
    if i == 'y':
        pass
    else:
        return
    pipeline.assign_pcds(labels_to_pcds)
    torch.cuda.empty_cache()

    log_info(f'Demo name:{concept}')
    pipeline.get_initial_model_paths(concept)
    pipeline.load_demos(concept)
    pipeline.load_models()

    ee_poses = pipeline.find_correspondence()
    pipeline.execute(ee_poses)
    pipeline.next_iter()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--query_text', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=5)

    parser.add_argument('--pybullet_viz', action='store_true')
    parser.add_argument('--show_pcds', action='store_true')
    parser.add_argument('--opt_visualize', action='store_true')
    parser.add_argument('--grasp_viz', action='store_true')

    parser.add_argument('--parent_model_path', type=str, default='')
    parser.add_argument('--child_model_path', type=str, default='')

    # parser.add_argument('--random', action='store_true', help='utilize random weights')
    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--opt_iterations', type=int, default=500)

    parser.add_argument('--relation_method', type=str, default='intersection', help='either "intersection", "ebm"')

    parser.add_argument('--pc_reference', type=str, default='parent', help='either "parent" or "child"')
    parser.add_argument('--skip_alignment', action='store_true')
    parser.add_argument('--new_descriptors', action='store_true')
    parser.add_argument('--create_descriptors', action='store_true')
    parser.add_argument('--n_demos', type=int, default=15)
    parser.add_argument('--target_idx', type=int, default=-1)
    parser.add_argument('--query_scale', type=float, default=0.025)
    parser.add_argument('--target_rounds', type=int, default=3)

    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--noise_idx', type=int, default=0)
    parser.add_argument('--cam_index', nargs='+', help='set which cameras to get point cloud from', required=True)

    parser.add_argument('--pb_seg', action='store_true')
    parser.add_argument('--gripper_type', type=str, default='panda')
    parser.add_argument('--port_vis', type=int, default=6000)

    args = parser.parse_args()

    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    # all_objs_dirs = [path for path in path_util.get_ndf_obj_descriptions() if '_centered_obj_normalized' in path] 
    # all_demos_dirs = osp.join(path_util.get_ndf_data(), 'demos')

    pipeline = Pipeline(args)
    pipeline.setup_client()
    pipeline.setup_table()
    pipeline.setup_cams()

    while True:
        generate_new_scene = main(pipeline)
