import torch
import numpy as np
import trimesh
import argparse
import os
import glob
import pickle
import sys

from manopth.manolayer import ManoLayer
from manopth import demo
from sample_utils import sample_surface_with_label

from visualize_utils import display_surface_points, display_vertices_with_joints, display_bones
from matplotlib import pyplot as plt
from manopth.tensutils import (th_posemap_axisang, th_with_zeros)

from visualize_utils import set_equal_xyz_scale, compare_joints

sys.path.insert(0, "/home/korrawe/halo_vae")
from models.halo_adapter.converter import PoseConverter, transform_to_canonical
from models.halo_adapter.interface import (convert_joints, change_axes)
#                                            get_bone_lengths, scale_halo_trans_mat)
from models.halo_adapter.transform_utils import xyz_to_xyz1

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('--root_folder', type=str,
                    default='../../data/test_data_code/test',
                    help='Output path root.')
parser.add_argument('--mano_folder', type=str,
                    default='../../data/youtubehand_raw',
                    help='Mano params path root.')
parser.add_argument('--mesh_folder', type=str,
                    default='../../data/test_data_code/test/mesh',
                    help='Output path for mesh.')
parser.add_argument('--meta_folder', type=str,
                    default='../../data/test_data_code/test/meta',
                    help='Output path for meta data.')

parser.add_argument('--subfolder', action='store_true',
                    help='Whether the data is stored in a sequential subfolder (./0/, ./1/, ./2/).')
parser.add_argument('--fixshape', action='store_true',
                    help='If ture, fix the hand shape to mean shape and ignore the shape parameters in the dataset.')
parser.add_argument('--local_coord', action='store_true',
                    help='If ture, store transformantion matrices to local coordinate frames(origin) instead of to canonical pose.')
parser.add_argument('--normalize', action='store_true',
                    help='If ture, hand rotation with global normalizer')


def get_verts_association(skinning_weight_npy):
    data = np.load(skinning_weight_npy)
    max_weight = data.argmax(1)
    return max_weight


def seal(mesh_to_seal):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    # pylint: disable=unsubscriptable-object # pylint/issues/3139
    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])
    return mesh_to_seal


def get_translation_mat(vec):
    # vec must be 3x1 matrix
    rot = torch.eye(3)
    mat_4_3 = torch.cat([rot, torch.zeros(1, 3)], 0).repeat(16, 1, 1)
    trans_4_1 = torch.cat([vec[:16], vec.new_ones(16, 1)], 1)
    translation = torch.cat([mat_4_3, trans_4_1.unsqueeze(2)], 2)
    return translation


def get_bone_lengths(joints):
    bones = np.array([
        (0,4), # use distance from root to middle finger as palm bone length
        (1,2),
        (2,3),
        (3,17),
        (4,5),
        (5,6),
        (6,18),
        (7,8),
        (8,9),
        (9,20),
        (10,11),
        (11,12),
        (12,19),
        (13,14),
        (14,15),
        (15,16)
    ])
    # display_bones(joints, bones)
    bone_length = joints[bones[:,0]] - joints[bones[:,1]]
    bone_length = np.linalg.norm(bone_length, axis=1)
    return bone_length


def load_data(filename):
    with open(filename, 'rb') as f:
        fit_params = pickle.load(f)

    # Convert mm to m
    mano_joints = fit_params['joints'] / 1000.0

    # Convert MANO joints to HALO joint
    mano_joints = convert_joints(mano_joints, source='mano', target='halo')

    data_size = len(fit_params['mano_joints'])
    print("data size: ", data_size)
    data_list = []
    for idx in range(data_size):
        dat = {
            'pose': fit_params['pose'][idx],
            'shape': fit_params['shape'][idx],
            'rot': fit_params['rot'][idx],
        }
        data_list.append(dat)

    return data_list


def main(args):
    print("Load input MANO parameters from", args.mano_folder)
    print("Recovering hand meshes from mano parameter to", args.mesh_folder)
    print("Save bone transformations and vertex labels to", args.meta_folder)

    # ### Initialization
    visualize = False  # True
    # Get bone association for each vertices
    # Use the same label order as MANO (without reordering to match visualization tool)
    skinning_weight_npy = 'resource/skinning_weight_r.npy'
    verts_joints_assoc = get_verts_association(skinning_weight_npy)
    # Add wrist label to the new vertex
    verts_joints_assoc = np.concatenate([verts_joints_assoc, [0]])
    # Select number of principal components for pose space
    ncomps = 45  # 6
    # Initialize MANO layer
    mano_layer = ManoLayer(
        mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False)
    faces = mano_layer.th_faces.detach().cpu().numpy()

    # Initialize PoseConverter
    pose_converter = PoseConverter()

    # Mesh output list
    mesh_name_list = []

    # Load data
    # Whether MANO parameters are stored in multiple small files or one big file
    # If data is stored in multiple files, defer loading until use
    multiple_file = True
    if multiple_file:
        # Load file names
        if args.subfolder:
            input_files = glob.glob(os.path.join(args.mano_folder, '*/*.npz'))
            sub_dirs = [os.path.basename(n) for n in glob.glob(os.path.join(args.mano_folder, '*'))]

            for sub_d in sub_dirs:
                if not os.path.isdir(os.path.join(args.mesh_folder, sub_d)):
                    os.makedirs(os.path.join(args.mesh_folder, sub_d))

                if not os.path.isdir(os.path.join(args.meta_folder, sub_d)):
                    os.makedirs(os.path.join(args.meta_folder, sub_d))
        else:
            input_files = glob.glob(os.path.join(args.mano_folder, '*.npz'))

            if not os.path.isdir(args.mesh_folder):
                os.makedirs(args.mesh_folder)

            if not os.path.isdir(args.meta_folder):
                os.makedirs(args.meta_folder)
    else:
        # Load the whole data file
        data_file = "datafile.pkl"
        input_files = load_data(data_file)

        if not os.path.isdir(args.mesh_folder):
            os.makedirs(args.mesh_folder)
        if not os.path.isdir(args.meta_folder):
            os.makedirs(args.meta_folder)

    for idx, mano_in_file in enumerate(input_files):
        if multiple_file:
            print(mano_in_file)
            file_idx = os.path.splitext(os.path.basename(mano_in_file))[0]
            file_idx = int(file_idx)
            if args.subfolder:
                sub_dir = os.path.split(os.path.split(mano_in_file)[0])[1]
                # print("sub_dir", sub_dir)
            mano_params = np.load(mano_in_file)
            pose = torch.from_numpy(mano_params["pose"]).unsqueeze(0)
            shape = torch.from_numpy(mano_params["shape"]).unsqueeze(0)
            rot = torch.from_numpy(mano_params["rot"]).unsqueeze(0)
        else:
            # if idx % 100 == 0:
            print(idx)
            file_idx = idx
            mano_params = mano_in_file
            pose = torch.from_numpy(mano_params["pose"]).unsqueeze(0)
            shape = torch.from_numpy(mano_params["shape"]).unsqueeze(0)
            rot = torch.from_numpy(mano_params["rot"]).unsqueeze(0)

        if args.fixshape:
            shape = torch.zeros(1, 10)

        pose_para = torch.cat([rot, pose], 1)

        # Forward pass through MANO layer. All info is in meter scale
        hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = mano_layer(pose_para, shape, no_root_rot=False) # fixed_shape)

        hand_verts = hand_verts[0,:]
        hand_joints = hand_joints[0,:]
        joints_trans = joints_trans[0,:]
        rest_pose_verts = rest_pose_verts[0,:]
        rest_pose_joints = rest_pose_joints[0, :16]
        rest_pose_joints_original = rest_pose_joints + 0

        root_xyz = hand_joints[0]

        # Move root joint to origin
        hand_verts = hand_verts - root_xyz
        hand_joints = hand_joints - root_xyz
        rest_pose_verts = rest_pose_verts - root_xyz
        rest_pose_joints = rest_pose_joints - root_xyz

        # display check rot
        th_pose_map, th_rot_map = th_posemap_axisang(rot)
        root_rot = th_rot_map[:, :9].view(1, 3, 3)
        root_rot_mat = th_with_zeros(torch.cat([root_rot, torch.zeros([1, 3, 1])], 2))
        root_rot_mat = root_rot_mat[0]

        neg_rest_pose_joints = -1. * rest_pose_joints
        translation = get_translation_mat(neg_rest_pose_joints)
        new_trans_mat = torch.matmul(joints_trans, translation)
        rest_pose_joints = torch.cat([rest_pose_joints[:16], rest_pose_joints.new_ones(16, 1)], 1)
        new_trans_mat[:, :3, 3] = new_trans_mat[:, :3, 3] - root_xyz
        # final_joints = torch.matmul(new_trans_mat, rest_pose_joints.unsqueeze(2))

        rest_pose_joints_original = torch.cat([rest_pose_joints_original, rest_pose_joints_original.new_ones(16, 1)], 1)
        # final_joints_from_local = torch.matmul(joints_trans_local[0], rest_pose_joints_original.unsqueeze(2))

        # make posed joints 16 x 4 x 1
        posed_joints = torch.cat([hand_joints[:16], torch.ones([16, 1])], 1)
        # print("posed_joints", posed_joints.shape)

        # check back project to origin
        tmp_joints = joints_trans + 0
        tmp_joints[:, :3, 3] = tmp_joints[:, :3, 3] - root_xyz
        inv_global_trans_mat = np.linalg.inv(tmp_joints)
        # try adding global scale after inverse
        # Transform meta data
        scale_mat = np.identity(4) / 0.4
        scale_mat[3,3] = 1.
        inv_global_trans_mat = np.matmul(scale_mat, inv_global_trans_mat)

        if visualize:
            demo.display_hand({
                'verts': hand_verts,
                'joints': tmp_joints[:, :3, 3],
                'rest_joints': rest_pose_joints,
                'verts_assoc': verts_joints_assoc
            },
                mano_faces=mano_layer.th_faces)

        # Save mesh
        num_str = f'{file_idx:03}'
        if args.subfolder:
            mesh_name_list.append(sub_dir + "/" + num_str)
            mesh_out = os.path.join(args.mesh_folder, sub_dir, num_str + ".off")
            meta_out = os.path.join(args.meta_folder, sub_dir, num_str + ".npz")
        else:
            mesh_name_list.append(num_str)
            mesh_out = os.path.join(args.mesh_folder, num_str + ".off")
            meta_out = os.path.join(args.meta_folder, num_str + ".npz")

        # ##### Compute local coord trans mat
        kps_local_cs = convert_joints(hand_joints.unsqueeze(0), source='halo', target='biomech').cuda()
        is_right_one = torch.ones(1, device=kps_local_cs.device)

        palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
        palm_align_kps_local_cs_nasa_axes, swap_axes_mat = change_axes(palm_align_kps_local_cs)

        swap_axes_mat = swap_axes_mat.unsqueeze(0).cuda()

        rot_then_swap_mat = torch.matmul(swap_axes_mat, glo_rot_right).unsqueeze(0)
        root_rot_mat = rot_then_swap_mat.squeeze().detach().cpu().numpy()

        # Pose_converter
        trans_mat_pc, _ = pose_converter(palm_align_kps_local_cs_nasa_axes, is_right_one)
        trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='halo')

        # Ignore transformation of the root bones
        joints_for_nasa_input = torch.tensor([0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16])
        trans_mat_pc = trans_mat_pc[:, joints_for_nasa_input]
        # trans_mat_pc = trans_mat_pc.squeeze(0).cpu()

        if args.normalize:
            # Apply root_rot_mat to hand joints
            hand_joints = torch.matmul(rot_then_swap_mat.squeeze().cpu(), xyz_to_xyz1(hand_joints).unsqueeze(-1))[:, :3, 0]
            # Apply normalization to mesh vertices
            hand_verts = torch.matmul(rot_then_swap_mat.squeeze().cpu(), xyz_to_xyz1(hand_verts).unsqueeze(-1))[:, :3, 0]
            hand_verts = hand_verts.detach().cpu().numpy()
            # Bone transformations without normalization matrix
            trans_mat_pc_all = trans_mat_pc
        else:
            # If not normalizaed, include normalization matrix in the bone transformations
            trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)

        trans_mat_pc = trans_mat_pc.squeeze(0).cpu()

        # Save mesh
        mesh = trimesh.Trimesh(hand_verts, faces, process=False)
        mesh = seal(mesh)
        mesh.export(mesh_out)

        # Save bone transformation and surface vertices with label
        # # Inverse the translation matrix for future use
        # # joints_trans_inv = np.linalg.inv(tmp_joints) 
        # joints_trans_inv = np.linalg.inv(new_trans_mat)
        joints_trans_inv = trans_mat_pc_all.cpu().numpy().squeeze(0)
        # print('hand joints', hand_joints)

        # Resample surface points
        vertices, vert_labels = sample_surface_with_label(mesh, verts_joints_assoc)  # , viz=True, joints=hand_joints)
        bone_lengths = get_bone_lengths(hand_joints)
        np.savez(meta_out, joints_trans=joints_trans_inv, verts=vertices, vert_labels=vert_labels,
                 shape=mano_params["shape"], bone_lengths=bone_lengths, hand_joints=hand_joints, root_rot_mat=root_rot_mat)

        # display_vertices_with_joints(hand_joints, vertices, vert_labels)

    # Save files list
    with open(os.path.join(args.root_folder, 'datalist.txt'), 'w') as listfile:
        for m in mesh_name_list:
            listfile.write('%s\n' % m)


nasa_joint_parent = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9])
biomech_joint_parent = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

def vis_joints(target_joints_right, ax=None, vis=True, set_axes=True, color='#1f77b4', parent='nasa', sixteen=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    jj = target_joints_right[0]

    ax.scatter(jj[:, 0], jj[:, 1], jj[:, 2], color='b', s=10)
    if set_axes:
        set_equal_xyz_scale(ax, jj[:, 0], jj[:, 1], jj[:, 2])

    # bone
    joints = target_joints_right
    if parent == 'nasa':
        b_start_loc = joints[0, nasa_joint_parent]
    elif parent == 'mano':
        print("mano parent")
        b_start_loc = joints[0, mano_joint_parent]
    elif parent == 'adrian':
        b_start_loc = joints[0, biomech_joint_parent]

    b_end_loc = joints[0]

    n_joint = 16 if sixteen else 21
    for b in range(n_joint):
        ax.plot([b_start_loc[b, 0], b_end_loc[b, 0]],
                [b_start_loc[b, 1], b_end_loc[b, 1]],
                [b_start_loc[b, 2], b_end_loc[b, 2]], color=color)

    if vis:
        plt.show()



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
