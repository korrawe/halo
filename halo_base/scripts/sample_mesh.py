import argparse
import trimesh
import numpy as np
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial
# TODO: do this better
sys.path.append('..')
from im2mesh.utils import binvox_rw, voxels
from im2mesh.utils.libmesh import check_mesh_contains


parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')

parser.add_argument('--in_meta_folder', type=str,
                    help='Input path for meta (bone trans and surface vertices'
                         'with labels).')

parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')
parser.add_argument('--global_scale', type=float, default=0.,
                    help='If not zero, resize all mesh with the same scale.')

parser.add_argument('--rotate_xz', type=float, default=0.,
                    help='Angle to rotate around y axis.')

parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--bbox_in_folder', type=str,
                    help='Path to other input folder to extract'
                         'bounding boxes.')

parser.add_argument('--pointcloud_folder', type=str,
                    help='Output path for point cloud.')
parser.add_argument('--pointcloud_size', type=int, default=100000,
                    help='Size of point cloud.')

parser.add_argument('--points_folder', type=str,
                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=100000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=1.,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')

parser.add_argument('--mesh_folder', type=str,
                    help='Output path for mesh.')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                help='Whether to save truth values as bit array.')
parser.add_argument('--subfolder', action='store_true',
                help='Whether the data is in sequential subfolders (./0/xx, ./1/xx, ...).')
    
def main(args):
    if args.subfolder:
        input_files = glob.glob(os.path.join(args.in_folder, '*/*.off'))
        sub_dirs = [os.path.basename(n) for n in glob.glob(os.path.join(args.in_folder, '*'))]
        # Create sub directories
        if args.pointcloud_folder is not None:
            for sub_dir in sub_dirs:
                if not os.path.isdir(os.path.join(args.pointcloud_folder, sub_dir)):
                    os.makedirs(os.path.join(args.pointcloud_folder, sub_dir))
        if args.points_folder is not None:
            for sub_dir in sub_dirs:
                if not os.path.isdir(os.path.join(args.points_folder, sub_dir)):
                    os.makedirs(os.path.join(args.points_folder, sub_dir))
        if args.mesh_folder is not None:
            for sub_dir in sub_dirs:
                if not os.path.isdir(os.path.join(args.mesh_folder, sub_dir)):
                    os.makedirs(os.path.join(args.mesh_folder, sub_dir))
    else:
        datalist_file = os.path.join(args.in_folder, '..' ,'datalist.txt')
        with open(datalist_file) as f:
            data_list = f.readlines()
        data_list = [name.strip() for name in data_list]
        # input_files = [os.path.join(args.in_folder, name + '.off') for name in data_list]
        input_files = [name + '.off' for name in data_list]
        # import pdb; pdb.set_trace()
        # input_files = glob.glob(os.path.join(args.in_folder, '*.off'))

    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            process_path(p, args)


def process_path(mesh_file, args):
    modelname = os.path.splitext(mesh_file)[0]
    mesh_path = os.path.join(args.in_folder, mesh_file)
    mesh = trimesh.load(mesh_path, process=False)
    # if args.subfolder:
    #     sub_dir = os.path.split(os.path.split(modelname)[0])[1]
    #     modelname = os.path.join(sub_dir, modelname)
    
    # Load meta file ('joints_trans', 'verts', 'vert_labels')
    meta_data_file = os.path.join(args.in_meta_folder, modelname + ".npz")
    meta_data = np.load(meta_data_file)
    meta = {'verts': meta_data['verts'], 
            'joints_trans': meta_data['joints_trans'],
            'vert_labels': meta_data['vert_labels'],
            'shape': meta_data['shape'],
            'bone_lengths': meta_data['bone_lengths'],
            'hand_joints': meta_data['hand_joints']
    }

    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    elif args.global_scale != 0:
        loc = np.zeros(3)
        scale = args.global_scale

        # Transform input mesh
        mesh.apply_scale(1 / scale)
        meta['verts'] = meta['verts'] / scale
        meta['hand_joints'] = meta['hand_joints'] / scale
        # Transform meta data
        # Assume that the transformation matrices are already inverted
        scale_mat = np.identity(4) * scale 
        scale_mat[3,3] = 1.
        meta['joints_trans'] = np.matmul(meta['joints_trans'], scale_mat)
        # (optional) scale canonical pose by the same global scale to make learning occupancy function easier
        # canonical_scale_mat = np.identity(4) / scale 
        # canonical_scale_mat[3,3] = 1.
        # meta['joints_trans'] = np.matmul(canonical_scale_mat, meta['joints_trans'])
        # end optional scaling #

        # if the matrices are not pre-invert
        # scale_mat = np.identity(4) / scale
        # scale_mat[3,3] = 1.
        # meta['joints_trans'] = np.matmul(scale_mat, meta['joints_trans'])

        bbox = mesh.bounding_box.bounds
        # Compute location and scale
        # For uniform point sampling
        # loc = (bbox[0] + bbox[1]) / 2
        # scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)
        scale = 1.0
        # import pdb; pdb.set_trace()
        
    else:
        if args.bbox_in_folder is not None:
            in_path_tmp = os.path.join(args.bbox_in_folder, modelname + '.off')
            mesh_tmp = trimesh.load(in_path_tmp, process=False)
            bbox = mesh_tmp.bounding_box.bounds
        else:
            bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        if args.rotate_xz != 0:
            angle = args.rotate_xz / 180 * np.pi
            R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            mesh.apply_transform(R)

    # Export various modalities
    if args.pointcloud_folder is not None:
        export_pointcloud(mesh, modelname, loc, scale, args)

    if args.points_folder is not None:
        export_points(mesh, modelname, loc, scale, args, meta)

    if args.mesh_folder is not None:
        export_mesh(mesh, modelname, loc, scale, args)


def export_pointcloud(mesh, modelname, loc, scale, args):
    filename = os.path.join(args.pointcloud_folder,
                            modelname + '.npz')
    if not args.overwrite and os.path.exists(filename):
        print('Pointcloud already exist: %s' % filename)
        return

    points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    normals = mesh.face_normals[face_idx]

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)
    normals = normals.astype(dtype)

    print('Writing pointcloud: %s' % filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)


def export_points(mesh, modelname, loc, scale, args, meta_data):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return

    filename = os.path.join(args.points_folder, modelname + '.npz')

    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform

    boxsize = 1 + args.points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)
    points_uniform = (scale * points_uniform) + loc
    # print("uniform", points_uniform)
    # print("box size", boxsize)
    # print("loc", loc)
    # print("scale", scale)
    points_surface = mesh.sample(n_points_surface)
    points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)

    # import pdb; pdb.set_trace()

    occupancies = check_mesh_contains(mesh, points)

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)

    if args.packbits:
        occupancies = np.packbits(occupancies)

    print('Writing points: %s' % filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, points=points, occupancies=occupancies,
             loc=loc, scale=scale,
             shape=meta_data['shape'],
             joints_trans=meta_data['joints_trans'], 
             verts=meta_data['verts'], 
             vert_labels=meta_data['vert_labels'],
             bone_lengths=meta_data['bone_lengths'],
             hand_joints=meta_data['hand_joints']
    )


def export_mesh(mesh, modelname, loc, scale, args):
    filename = os.path.join(args.mesh_folder, modelname + '.off')    
    if not args.overwrite and os.path.exists(filename):
        print('Mesh already exist: %s' % filename)
        return
    print('Writing mesh: %s' % filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    mesh.export(filename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
