import torch
import numpy as np
import trimesh
import argparse
import os
import glob

from manopth.manolayer import ManoLayer
from manopth import demo
from sample_utils import sample_surface_with_label

from visualize_utils import display_surface_points, display_vertices_with_joints, display_bones
from matplotlib import pyplot as plt
from manopth.tensutils import (th_posemap_axisang, th_with_zeros)


parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('--root_folder', type=str,
                    default='../../data/test_data_code/test',
                    help='Output path root.')
parser.add_argument('--mano_folder', type=str,
                    default='../../data/youtubehand_raw', # train_sub_folder
                    help='Output path root.')
parser.add_argument('--mesh_folder', type=str,
                    default='../../data/test_data_code/test/mesh',
                    help='Output path for mesh.')
parser.add_argument('--meta_folder', type=str,
                    default='../../data/test_data_code/test/meta',
                    help='Output path for points.')

parser.add_argument('--subfolder', action='store_true',
                help='Whether the data is stored in a sequential subfolder (./0/, ./1/, ./2/).')
parser.add_argument('--fixshape', action='store_true',
                help='If ture, fix the hand shape to mean shape and ignore the shape parameters in the dataset.')
parser.add_argument('--local_coord', action='store_true',
                help='If ture, store transformantion matices to local coordinate frames instead of to canonical pose..')

def get_verts_association(skinning_weight_npy):
    # skinning_weight_npy = 'resource/skinning_weight_r.npy'
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
    # print(mat_4_3.shape)
    trans_4_1 = torch.cat([vec, vec.new_ones(16, 1)], 1)
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


def main(args):
    print("Load input MANO parameters from", args.mano_folder)
    print("Recovering hand meshes from mano parameter to", args.mesh_folder)
    print("Save bone transformations and vertex labels to", args.meta_folder)

    visualize = False # True

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

    # Total number of hand to generate
    sample_size = 16

    # Get bone association for each vertices
    # Use the same label order as MANO (without reordering to match visualization tool)
    skinning_weight_npy = 'resource/skinning_weight_r.npy'
    verts_joints_assoc = get_verts_association(skinning_weight_npy)
    # Add wrist label to the new vertex
    verts_joints_assoc = np.concatenate([verts_joints_assoc, [0]])
    # Select number of principal components for pose space
    ncomps = 45 # 6
    # Initialize MANO layer
    mano_layer = ManoLayer(
        mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False)
    faces = mano_layer.th_faces.detach().cpu().numpy()
    
    # Mesh list
    mesh_name_list = []
    # Sample one at a time
    batch_size = 1

    fixed_shape = torch.zeros(batch_size, 10) 

    for mano_in_file in input_files:
        print(mano_in_file)
        i = os.path.splitext(os.path.basename(mano_in_file))[0]
        i = int(i)
        if args.subfolder:
            sub_dir = os.path.split(os.path.split(mano_in_file)[0])[1]
            # print("sub_dir", sub_dir)
        mano_params = np.load(mano_in_file)
        pose = torch.from_numpy(mano_params["pose"]).unsqueeze(0)
        shape = torch.from_numpy(mano_params["shape"]).unsqueeze(0)
        rot = torch.from_numpy(mano_params["rot"]).unsqueeze(0)

        if args.fixshape:
            shape = torch.zeros(batch_size, 10)

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
        new_trans_mat = torch.matmul(joints_trans , translation)
        # print("new_trans_mat", new_trans_mat[:3])
        rest_pose_joints = torch.cat([rest_pose_joints, rest_pose_joints.new_ones(16, 1)], 1)
        # print("root_xyz", torch.cat([root_xyz, torch.ones(1)]).shape)
        new_trans_mat[:, :3, 3] = new_trans_mat[:, :3, 3] - root_xyz
        final_joints = torch.matmul(new_trans_mat, rest_pose_joints.unsqueeze(2))
        
        rest_pose_joints_original = torch.cat([rest_pose_joints_original, rest_pose_joints_original.new_ones(16, 1)], 1)
        # print(rest_pose_joints_original.shape)
        final_joints_from_local = torch.matmul(joints_trans_local[0], rest_pose_joints_original.unsqueeze(2))

        # make pose joints 16 x 4 x 1
        posed_joints = torch.cat([hand_joints[:16], torch.ones([16, 1])], 1)
        # print("posed_joints", posed_joints.shape)

        # check back project to origin
        # print("joints_trans", joints_trans)
        tmp_joints = joints_trans + 0
        tmp_joints[:, :3, 3] = tmp_joints[:, :3, 3] - root_xyz
        # print("tmp_joints",tmp_joints)
        inv_global_trans_mat = np.linalg.inv(tmp_joints)
        # try adding global scale after inverse
        # Transform meta data
        scale_mat = np.identity(4) / 0.4
        scale_mat[3,3] = 1.
        inv_global_trans_mat = np.matmul(scale_mat, inv_global_trans_mat)

        global_back_project = torch.matmul(torch.from_numpy(inv_global_trans_mat).float(), posed_joints.unsqueeze(2))
        
        # inverse
        inv_new_trans_mat = np.linalg.inv(new_trans_mat)
        inv_new_trans_mat = np.matmul(scale_mat, inv_new_trans_mat)
        back_projected_joints = torch.matmul(torch.from_numpy(inv_new_trans_mat).float(), posed_joints.unsqueeze(2))
        
        if visualize:
            demo.display_hand({
                'verts': hand_verts,
                'joints': tmp_joints[:, :3, 3],
                'rest_joints':  rest_pose_joints,
                'verts_assoc': verts_joints_assoc
            },
                            mano_faces=mano_layer.th_faces)
        
        num_str = f'{i:03}'
        if args.subfolder:
            mesh_name_list.append(sub_dir + "/" + num_str)
            mesh_out = os.path.join(args.mesh_folder, sub_dir, num_str + ".off")
            meta_out = os.path.join(args.meta_folder, sub_dir, num_str + ".npz")
        else:
            mesh_name_list.append(num_str)
            mesh_out = os.path.join(args.mesh_folder, num_str + ".off")
            meta_out = os.path.join(args.meta_folder, num_str + ".npz")

        # Save mesh
        mesh = trimesh.Trimesh(hand_verts, faces, process=False)
        mesh = seal(mesh)
        mesh.export(mesh_out)

        # Save bone transformation and surface vertices with label
        # Inverse the translation matrix for future use
        # joints_trans_inv = np.linalg.inv(tmp_joints) # tmp_joints # 
        joints_trans_inv = np.linalg.inv(new_trans_mat)
        # Resample surface points 
        # vertices, vert_labels = sample_surface_with_label(mesh, verts_joints_assoc) # , viz=True, joints=hand_joints) ########
        bone_lengths = get_bone_lengths(hand_joints)
        # with point resampling
        # np.savez(meta_out, joints_trans=joints_trans_inv, verts=vertices, vert_labels=vert_labels, 
        #          shape=mano_params["shape"], bone_lengths=bone_lengths, hand_joints=hand_joints, root_rot_mat=root_rot_mat)
        # without point resampling
        np.savez(meta_out, joints_trans=joints_trans_inv, verts=hand_verts, vert_labels=verts_joints_assoc, 
                 shape=mano_params["shape"], bone_lengths=bone_lengths, hand_joints=hand_joints, root_rot_mat=root_rot_mat)
        
        # display_vertices_with_joints(hand_joints, vertices, vert_labels)
    
    # Save files list
    with open(os.path.join(args.root_folder, 'datalist.txt'), 'w') as listfile:
        for m in mesh_name_list:
            listfile.write('%s\n' % m)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)