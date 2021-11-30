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
# python generate.py /home/korrawe/halo_vae/configs/vae/grab_refine_inter_number.ymal --test_data data/obmean_test/ --inference

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
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

    # try:
    #     model_dict = dataset.get_model_dict(idx)
    # except AttributeError:
    #     model_dict = {'model': str(idx), 'category': 'n/a'}
    # # print(model_dict)
    # modelname = model_dict

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)
        os.makedirs(generation_vis_compare_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
        os.makedirs(kps_dir)

    # if generate_pointcloud and not os.path.exists(pointcloud_dir):
    #     os.makedirs(pointcloud_dir)

    # if not os.path.exists(in_dir):
    #     os.makedirs(in_dir)

    # # Timing dict
    # time_dict = {
    #     'idx': idx,
    #     # 'class id': category_id,
    #     # 'class name': category_name,
    #     'modelname': modelname,
    # }
    # time_dicts.append(time_dict)

    # # Generate outputs
    # out_file_dict = {}

    # # Also copy ground truth
    # if cfg['generation']['copy_groundtruth']:
    #     modelpath = os.path.join(
    #         # dataset.dataset_folder, category_id, modelname, 
    #         dataset.dataset_folder, cfg['data']['test_split'],
    #         cfg['data']['watertight_folder'],
    #         modelname + cfg['data']['watertight_file'])
    #     out_file_dict['gt'] = modelpath
    # import pdb;pdb.set_trace()
    if generate_keypoint:
        t0 = time.time()
        # N = 20
        out = generator.sample_keypoint(data, idx, N=1, gen_mesh=generate_mesh, mesh_dir=mesh_dir,
                                        random_rotate=args.random_rotate, use_refine_net=use_refine_net)
        # print(out)
        keypoint_list.extend(out)

    # if it > 2:
    #     print(len(keypoint_list))
    #     break

    # if generate_mesh:
    #     t0 = time.time()
    #     out = generator.generate_mesh(data)
    #     time_dict['mesh'] = time.time() - t0

    #     # Get statistics
    #     try:
    #         mesh, stats_dict = out
    #     except TypeError:
    #         mesh, stats_dict = out, {}
    #     time_dict.update(stats_dict)

    #     # Write output
    #     # mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
    #     mesh_out_file = os.path.join(mesh_dir, '%s.obj' % modelname)
    #     # print("mesh_out_file", mesh_out_file)
    #     os.makedirs(os.path.dirname(mesh_out_file), exist_ok=True)
    #     mesh.export(mesh_out_file)
    #     out_file_dict['mesh'] = mesh_out_file

    # if generate_pointcloud:
    #     t0 = time.time()
    #     pointcloud = generator.generate_pointcloud(data)
    #     time_dict['pcl'] = time.time() - t0
    #     pointcloud_out_file = os.path.join(
    #         pointcloud_dir, '%s.ply' % modelname)
    #     export_pointcloud(pointcloud, pointcloud_out_file)
    #     out_file_dict['pointcloud'] = pointcloud_out_file

    # if cfg['generation']['copy_input']:
    #     # Save inputs
    #     if input_type == 'trans_matrix':
    #         pass
    #         # TODO: create image/model of joints 
    #     elif input_type == 'img':
    #         inputs_path = os.path.join(in_dir, '%s.jpg' % modelname)
    #         inputs = data['inputs'].squeeze(0).cpu()
    #         visualize_data(inputs, 'img', inputs_path)
    #         out_file_dict['in'] = inputs_path
    #     elif input_type == 'voxels':
    #         inputs_path = os.path.join(in_dir, '%s.off' % modelname)
    #         inputs = data['inputs'].squeeze(0).cpu()
    #         voxel_mesh = VoxelGrid(inputs).to_mesh()
    #         voxel_mesh.export(inputs_path)
    #         out_file_dict['in'] = inputs_path
    #     elif input_type == 'pointcloud':
    #         inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
    #         inputs = data['inputs'].squeeze(0).cpu().numpy()
    #         export_pointcloud(inputs, inputs_path, False)
    #         out_file_dict['in'] = inputs_path

    # # Copy to visualization directory for first vis_n_output samples
    # # c_it = model_counter[category_id]
    # c_it = model_counter
    # if c_it < vis_n_outputs:
    #     # Save output files
    #     img_name = '%02d.off' % c_it
    #     for k, filepath in out_file_dict.items():
    #         ext = os.path.splitext(filepath)[1]
    #         out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
    #                                 % (c_it, k, ext))
    #         shutil.copyfile(filepath, out_file)

    # model_counter += 1

if args.gen_denoiser_data:
    # Dump generated keypoints
    outfile_name = "/media/korrawe/ssd/halo_vae/data/gen_val_kps/val_gen_20perObj.pkl"
    with open(outfile_name, 'wb') as p_f:
        pickle.dump(keypoint_list, p_f)


# with open(outfile_name, 'rb') as f:
#     mynewlist = pickle.load(f)

# import pdb; pdb.set_trace()

# # Create pandas dataframe and save
# time_df = pd.DataFrame(time_dicts)
# time_df.set_index(['idx'], inplace=True)
# time_df.to_pickle(out_time_file)

# # Create pickle files  with main statistics
# # time_df_class = time_df.groupby(by=['class name']).mean()
# time_df_class = time_df.groupby(by=['modelname']).mean()

# time_df_class.to_pickle(out_time_file_class)

# # Print results
# time_df_class.loc['mean'] = time_df_class.mean()
# print('Timings [s]:')
# print(time_df_class)