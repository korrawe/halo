import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
import numpy as np
# from im2mesh import config
from artihand import config, data
from artihand.checkpoints import CheckpointIO

from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
from im2mesh.utils.voxels import VoxelGrid


parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--latest', action='store_true', help='Use latest model instead of best.')
parser.add_argument('--sweep', action='store_true', help='Sweep threshold to visualize distance field.')
parser.add_argument('--test_data', type=str, 
                    help='Path to test data directory. Use test split in the'
                         'config file if not provided.')

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
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
if args.latest:
    checkpoint_io.load('model.pt')
else:
    checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']

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

for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis', )

    # Get index etc.
    # print(data.keys())
    # print(data['inputs'].shape)
    # print(data['bone_lengths'].shape)
    # break
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    
    # print(model_dict)
    modelname = model_dict
    # modelname = model_dict['model']
    # category_id = model_dict.get('category', 'n/a')

    # try:
    #     category_name = dataset.metadata[category_id].get('name', 'n/a')
    # except AttributeError:
    #     category_name = 'n/a'

    # print(category_name)

    # if category_id != 'n/a':
    #     mesh_dir = os.path.join(mesh_dir, str(category_id))
    #     pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
    #     in_dir = os.path.join(in_dir, str(category_id))

    #     folder_name = str(category_id)
    #     if category_name != 'n/a':
    #         folder_name = str(folder_name) + '_' + category_name.split(',')[0]

    #     generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    if generate_pointcloud and not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    
    # Timing dict
    time_dict = {
        'idx': idx,
        # 'class id': category_id,
        # 'class name': category_name,
        'modelname': modelname,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    # Also copy ground truth
    if cfg['generation']['copy_groundtruth']:
        modelpath = os.path.join(
            # dataset.dataset_folder, category_id, modelname, 
            dataset.dataset_folder, cfg['data']['test_split'],
            cfg['data']['watertight_folder'],
            modelname + cfg['data']['watertight_file'])
        out_file_dict['gt'] = modelpath

    if generate_mesh:
        t0 = time.time()
        if args.sweep:
            for cur_threshold in np.linspace(-0.05, 0.05, num=11):
                out = generator.generate_mesh(data, threshold=cur_threshold)
                time_dict['mesh'] = time.time() - t0
                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}
                time_dict.update(stats_dict)

                # Write output
                # mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
                print(cur_threshold)
                mesh_out_file = os.path.join(mesh_dir, '%s_%.2f.obj' % (modelname, cur_threshold))
                print("mesh_out_file", mesh_out_file)
                os.makedirs(os.path.dirname(mesh_out_file), exist_ok=True)
                mesh.export(mesh_out_file)
                out_file_dict['mesh'] = mesh_out_file

        else:
            out = generator.generate_mesh(data)
            time_dict['mesh'] = time.time() - t0

            # Get statistics
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}
            time_dict.update(stats_dict)

            # Write output
            # mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
            mesh_out_file = os.path.join(mesh_dir, '%s.obj' % modelname)
            # print("mesh_out_file", mesh_out_file)
            os.makedirs(os.path.dirname(mesh_out_file), exist_ok=True)
            mesh.export(mesh_out_file)
            out_file_dict['mesh'] = mesh_out_file

    if generate_pointcloud:
        t0 = time.time()
        pointcloud = generator.generate_pointcloud(data)
        time_dict['pcl'] = time.time() - t0
        pointcloud_out_file = os.path.join(
            pointcloud_dir, '%s.ply' % modelname)
        export_pointcloud(pointcloud, pointcloud_out_file)
        out_file_dict['pointcloud'] = pointcloud_out_file

    if cfg['generation']['copy_input']:
        # Save inputs
        if input_type == 'trans_matrix':
            pass
            # TODO: create image/model of joints 
        elif input_type == 'img':
            inputs_path = os.path.join(in_dir, '%s.jpg' % modelname)
            inputs = data['inputs'].squeeze(0).cpu()
            visualize_data(inputs, 'img', inputs_path)
            out_file_dict['in'] = inputs_path
        elif input_type == 'voxels':
            inputs_path = os.path.join(in_dir, '%s.off' % modelname)
            inputs = data['inputs'].squeeze(0).cpu()
            voxel_mesh = VoxelGrid(inputs).to_mesh()
            voxel_mesh.export(inputs_path)
            out_file_dict['in'] = inputs_path
        elif input_type == 'pointcloud':
            inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
            inputs = data['inputs'].squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in'] = inputs_path

    # Copy to visualization directory for first vis_n_output samples
    # c_it = model_counter[category_id]
    c_it = model_counter
    if c_it < vis_n_outputs:
        # Save output files
        img_name = '%02d.off' % c_it
        for k, filepath in out_file_dict.items():
            ext = os.path.splitext(filepath)[1]
            out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                    % (c_it, k, ext))
            shutil.copyfile(filepath, out_file)

    # model_counter[modelname] += 1
    model_counter += 1
    # model_counter[category_id] += 1

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
# time_df_class = time_df.groupby(by=['class name']).mean()
time_df_class = time_df.groupby(by=['modelname']).mean()

time_df_class.to_pickle(out_time_file_class)

# Print results
time_df_class.loc['mean'] = time_df_class.mean()
print('Timings [s]:')
print(time_df_class)