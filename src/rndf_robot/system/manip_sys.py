from pipeline import Pipeline 
from system_utils.vizServer import VizServer 

import argparse
from airobot import log_info, set_log_level, log_warn
from IPython import embed;

random = True
use_privilege_info = False
generate_new_scene = True

config = dict(
    # objects={'mug': {(1,0,0.1,1):1, (0,0.5,0.8,1):1}}
    objects = {'container': {(1,0,0.1,1):1}, 'mug': {(0,1,0.1,1):1}}
)
def main(pipeline, generate_new_scene=True):
    # torch.manual_seed(args.seed)
    if random and generate_new_scene:
        pipeline.setup_random_scene(config)

    prompt = pipeline.prompt_user()
    if not prompt: 
        return pipeline.step()
    corresponding_concept, query_text = prompt
    concept, keywords = pipeline.identify_classes_from_query(query_text, corresponding_concept)
    if pipeline.args.pb_seg:
        labels_to_pcds = pipeline.segment_scene_pb()
    else:
        descriptions = [keyword[1] if keyword[1] else keyword[0] for keyword in keywords]
        labels_to_pcds = pipeline.segment_scene(descriptions)
    if not labels_to_pcds:
        log_warn('WARNING: Target object not detected, resetting the scene')
        pipeline.reset()
        return
    ranks = [0] if pipeline.state == 0 else [0,1]
    pipeline.assign_pcds(labels_to_pcds,ranks)
    
    pipeline.get_intial_model_paths(concept)
    pipeline.load_demos(concept)
    pipeline.load_models()

    ee_poses = pipeline.find_correspondence()
    pipeline.execute(0, ee_poses)


    if pipeline.state == 0:
        return pipeline.step(ee_poses[-1])
    else:
        return pipeline.step()

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

    parser.add_argument('--pb_seg', action='store_true')


    args = parser.parse_args()

    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    # all_objs_dirs = [path for path in path_util.get_ndf_obj_descriptions() if '_centered_obj_normalized' in path] 
    # all_demos_dirs = osp.join(path_util.get_ndf_data(), 'demos')

    pipeline = Pipeline(args)
    pipeline.setup_client()

    server = VizServer(pipeline.robot.pb_client)
    pipeline.register_vizServer(server)

    pipeline.setup_table()
    pipeline.reset_robot()
    log_info('Loaded new table')
    while True:
        generate_new_scene = main(pipeline, generate_new_scene)
