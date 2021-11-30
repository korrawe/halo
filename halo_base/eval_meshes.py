import argparse
# import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
# from im2mesh import config, data
from artihand import config, data
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.io import load_pointcloud


parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--eval_input', action='store_true',
                    help='Evaluate inputs instead.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
if not args.eval_input:
    # out_file = os.path.join('/home/korrawe/nasa_original/exp/models/shape_mesh_vert/', 'eval_meshes_full.pkl')
    # out_file_class = os.path.join('/home/korrawe/nasa_original/exp/models/shape_mesh_vert/', 'eval_meshes.csv')
    # out_file = os.path.join('/home/korrawe/iccv_halo_compare/Pose2Mesh_RELEASE/experiment/exp_06-16_23:40_gt_3d/', 'eval_meshes_full.pkl')
    # out_file_class = os.path.join('/home/korrawe/iccv_halo_compare/Pose2Mesh_RELEASE/experiment/exp_06-16_23:40_gt_3d/', 'eval_meshes.csv')
    out_file = os.path.join('/home/korrawe/iccv_halo_compare/minimal-hand/', 'eval_meshes_full.pkl')
    out_file_class = os.path.join('/home/korrawe/iccv_halo_compare/minimal-hand/', 'eval_meshes.csv')
    # out_file = os.path.join(generation_dir, 'eval_meshes_full.pkl')
    # out_file_class = os.path.join(generation_dir, 'eval_meshes.csv')
else:
    out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_input.csv')

# Dataset
# points_field = data.PointsField(
#     cfg['data']['points_iou_file'], 
#     unpackbits=cfg['data']['points_unpackbits'],
# )
points_helper = data.PointsHelper(
    cfg['data']['points_iou_file'], 
    unpackbits=cfg['data']['points_unpackbits'],
)

# pointcloud_field = data.PointCloudField(
#     cfg['data']['pointcloud_chamfer_file']
# )
pointcloud_helper = data.PointCloudHelper(
    cfg['data']['pointcloud_chamfer_file']
)
# fields = {
#     'points_iou': points_field,
#     'pointcloud_chamfer': pointcloud_field,
#     'idx': data.IndexField(),
# }
data_loader_helpers = {
    'points_iou': points_helper,
    'pointcloud_chamfer': pointcloud_helper,
}
# 

print('Test split: ', cfg['data']['test_split'])

dataset_folder = cfg['data']['path']
# dataset = data.Shapes3dDataset(
#     dataset_folder, fields,
#     cfg['data']['test_split'],
#     categories=cfg['data']['classes'])
dataset = data.SampleHandDataset(
    dataset_folder, data_loader_helpers,
    split=cfg['data']['test_split'],
    return_idx=True
)

# Evaluator
evaluator = MeshEvaluator(n_points=100000)

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

found = 0
# Evaluate all classes
eval_dicts = []
print('Evaluating meshes...')
for it, data in enumerate(tqdm(test_loader)):
        
    if data is None:
        print('Invalid data.')
        continue

    # # sample eval for debugging
    # if it > 50:
    #     break

    # Output folders
    if not args.eval_input:
        mesh_dir = os.path.join(generation_dir, 'meshes')
        # mesh_dir = '/home/korrawe/nasa_original/exp/models/shape_mesh_vert/meshes' ###############
        # pose2mesh output
        # mesh_dir = "/home/korrawe/iccv_halo_compare/Pose2Mesh_RELEASE/experiment/exp_06-16_07:22_final_compare/vis/"
        # mesh_dir = "/home/korrawe/iccv_halo_compare/Pose2Mesh_RELEASE/experiment/exp_06-16_23:40_gt_3d/vis/"
        mesh_dir = "/home/korrawe/iccv_halo_compare/minimal-hand/output/"
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    else:
        mesh_dir = os.path.join(generation_dir, 'input')
        pointcloud_dir = os.path.join(generation_dir, 'input')

    # Get index etc.
    idx = data['idx'].item()
    # print(idx)
    # print(data.keys())

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    
    modelname = model_dict
    # modelname = model_dict['model']
    # category_id = model_dict['category']

    # try:
    #     category_name = dataset.metadata[category_id].get('name', 'n/a')
    # except AttributeError:
    #     category_name = 'n/a'

    # if category_id != 'n/a':
    #     mesh_dir = os.path.join(mesh_dir, category_id)
    #     pointcloud_dir = os.path.join(pointcloud_dir, category_id)

    # # Evaluate
    pointcloud_tgt = data['pointcloud_chamfer.points'].squeeze(0).numpy()
    normals_tgt = data['pointcloud_chamfer.normals'].squeeze(0).numpy()
    points_tgt = data['points_iou.points'].squeeze(0).numpy()
    occ_tgt = data['points_iou.occ'].squeeze(0).numpy()

    # Evaluating mesh and pointcloud
    # Start row and put basic informatin inside
    eval_dict = {
        'idx': idx,
        # 'class id': category_id,
        # 'class name': category_name,
        'modelname': modelname,
    }
    eval_dicts.append(eval_dict)

    # Evaluate mesh
    if cfg['test']['eval_mesh']:
        mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)
        # print('looking for ', mesh_file)

        # try looking for .obj file
        if not os.path.exists(mesh_file):
            mesh_file = os.path.join(mesh_dir, '%s.obj' % modelname)
        
        # try looking for .ply file
        if not os.path.exists(mesh_file):
            mesh_file = os.path.join(mesh_dir, '%s.ply' % modelname)

        if os.path.exists(mesh_file):
            found += 1
            mesh = trimesh.load(mesh_file, process=False)
            eval_dict_mesh = evaluator.eval_mesh(
                mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)
            for k, v in eval_dict_mesh.items():
                eval_dict[k + ' (mesh)'] = v
                print(k, v)
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    # Evaluate point cloud
    if cfg['test']['eval_pointcloud']:
        pointcloud_file = os.path.join(
            pointcloud_dir, '%s.ply' % modelname)

        if os.path.exists(pointcloud_file):
            pointcloud = load_pointcloud(pointcloud_file)
            eval_dict_pcl = evaluator.eval_pointcloud(
                pointcloud, pointcloud_tgt)
            for k, v in eval_dict_pcl.items():
                eval_dict[k + ' (pcl)'] = v
        else:
            print('Warning: pointcloud does not exist: %s'
                    % pointcloud_file)

# print('found ', found)
# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
# add mean
eval_df.loc['mean'] = eval_df.mean()
eval_df.to_pickle(out_file)
eval_df.to_csv(out_file_class)

# Create CSV file  with main statistics
# eval_df_class = eval_df.groupby(by=['class name']).mean()
# print(eval_df)
# eval_df_class = eval_df.groupby(by=['modelname']).mean()
# eval_df_class.to_csv(out_file_class)

# Print results
# eval_df.loc['mean'] = eval_df.mean()
print(eval_df)
# print(eval_df['mean'])
# eval_df_class.loc['mean'] = eval_df_class.mean()
# print(eval_df_class)