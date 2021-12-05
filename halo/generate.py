import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
import pickle
# from im2mesh import config
# from artihand import config, data
# from artihand.checkpoints import CheckpointIO

from models import config  # , data
from models.checkpoints import CheckpointIO

# from im2mesh.utils.io import export_pointcloud
# from im2mesh.utils.visualize import visualize_data
# from im2mesh.utils.voxels import VoxelGrid


parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--latest', action='store_true', help='Use latest model instead of best.')
parser.add_argument('--test_data', type=str,
                    help='Path to test data directory. Use test split in the'
                         'config file if not provided.')
parser.add_argument('--inference', action='store_true',
                    help='Inference on new data.')
parser.add_argument('--gen_denoiser_data', action='store_true',
                    help='Use validation set to generate keypoints for denoiser training.')
parser.add_argument('--random_rotate', action='store_true',
                    help='Randomly rotate input objects.')

# python generate.py /home/korrawe/halo_vae/configs/vae/bmc_loss_grab_z16.yaml --latest --test_data data/grab_object --inference --random_rotate
# python generate.py /home/korrawe/halo_vae/configs/vae/obman_baseline.yaml --test_data data/grab_object --inference
# python generate.py /home/korrawe/halo_vae/configs/vae/grab_refine_inter_number.ymal --test_data data/obman_test/ --inference

args = parser.parse_args()
cfg = config.load_config(args.config, '../configs/halo_vae/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
if args.latest:
    generation_dir = generation_dir + '_latest'
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
if args.test_data:
    cfg['data']['path'] = args.test_data
if args.inference:
    cfg['data']['dataset'] = 'inference'
# Random rotate
cfg['data']['random_rotate'] = args.random_rotate

# Use val instead of test to generate sampled keypoint dataset
if args.gen_denoiser_data:
    dataset = config.get_dataset('val', cfg, return_idx=True)
else:
    dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
if args.latest:
    checkpoint_io.load('model.pt')
else:
    load_dict = checkpoint_io.load(cfg['test']['model_file'])

epoch_it = load_dict.get('epoch_it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', 10000)
print('epoch_it:', epoch_it)
print('metric_val_best:', metric_val_best)

# Refinement
use_refine_net = cfg['model']['use_refine_net']

# Load HALO mesh model if needed
# if cfg['model']['use_mesh_model']:
model.initialize_halo(cfg['model']['halo_config_file'],
                      cfg['model']['denoiser_pth'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_keypoint = cfg['generation']['generate_keypoint']
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']

if generate_keypoint and not hasattr(generator, 'sample_keypoint'):
    generate_keypoint = False
    print('Warning: generator does not support keypoint generation.')

if generate_mesh and not hasattr(generator, 'generate_mesh'):
    generate_mesh = False
    print('Warning: generator does not support mesh generation.')

if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
    generate_pointcloud = False
    print('Warning: generator does not support pointcloud generation.')


# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Statistics
time_dicts = []

# Generate
model.eval()

# Count how many models already created
# model_counter = defaultdict(int)
model_counter = 0

keypoint_list = []

for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    kps_dir = os.path.join(generation_dir, 'kps')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis', )
    generation_vis_compare_dir = os.path.join(generation_dir, 'vis_compare')

    # Get index etc.
    idx = data['idx'].item()



    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)
        os.makedirs(generation_vis_compare_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
        os.makedirs(kps_dir)


    if generate_keypoint:
        t0 = time.time()
        # N = 20
        out = generator.sample_keypoint(data, idx, N=1, gen_mesh=generate_mesh, mesh_dir=mesh_dir,
                                        random_rotate=args.random_rotate, use_refine_net=use_refine_net)
        # print(out)
        keypoint_list.extend(out)


