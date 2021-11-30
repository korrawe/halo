import torch
import numpy as np
import trimesh
import argparse
import os
import glob
import pickle

from manopth.manolayer import ManoLayer
from manopth import demo
from sample_utils import sample_surface_with_label

from visualize_utils import display_surface_points, display_vertices_with_joints, display_bones
from matplotlib import pyplot as plt
from manopth.tensutils import (th_posemap_axisang, th_with_zeros)

from visualize_utils import set_equal_xyz_scale, compare_joints

import sys
# sys.path.insert(0, "/home/korrawe/interhand/InterHand2.6M")
# from common.nets.converter import transform_to_canonical  # PoseConverter
# from common.utils.transforms_torch import swap_axes_for_nasa
# from common.utils.transforms_torch import convert_joints

sys.path.insert(0, "/home/korrawe/halo_vae")
from models.halo_adapter.converter import PoseConverter, transform_to_canonical
from models.halo_adapter.interface import (convert_joints, change_axes)
#                                            get_bone_lengths, scale_halo_trans_mat)
from models.halo_adapter.transform_utils import xyz_to_xyz1

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('--root_folder', type=str,
                    default='/home/korrawe/nasa/data/youtube_hand/youtube_keypoint_no_scale/train',
                    # default='/home/korrawe/nasa/data/overfit/Subject_2_squeeze_paper_1_resample/test',
                    help='Output path root.')
parser.add_argument('--mano_folder', type=str,
                    default='/home/korrawe/nasa/data/youtube_hand/raw/train_sub_folder',  # train_sub_folder
                    # default='/home/korrawe/nasa/data/FHB/test/Subject_2_squeeze_paper_1/test/mano',
                    help='Output path root.')
parser.add_argument('--mesh_folder', type=str,
                    default='/home/korrawe/nasa/data/youtube_hand/youtube_keypoint_no_scale/train/mesh',
                    # default='/home/korrawe/nasa/data/overfit/Subject_2_squeeze_paper_1_resample/test/mesh',
                    help='Output path for mesh.')
parser.add_argument('--meta_folder', type=str,
                    default='/home/korrawe/nasa/data/youtube_hand/youtube_keypoint_no_scale/train/meta',
                    # default='/home/korrawe/nasa/data/overfit/Subject_2_squeeze_paper_1_resample/test/meta',
                    help='Output path for points.')

parser.add_argument('--subfolder', action='store_true',
                    help='Whether the data is stored in a sequential subfolder (./0/, ./1/, ./2/).')
parser.add_argument('--fixshape', action='store_true',
                    help='If ture, fix the hand shape to mean shape and ignore the shape parameters in the dataset.')
parser.add_argument('--local_coord', action='store_true',
                    help='If ture, store transformantion matrices to local coordinate frames(origin) instead of to canonical pose.')
parser.add_argument('--normalize', action='store_true',
                    help='If ture, hand rotation with global normalizer')
parser.add_argument('--noisy', action='store_true',
                    help='If ture, use the noisy keypoints instead of MANO keypoints')

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
    # print("bone_length_shape", bone_length)
    # print("bone_length", bone_length.shape)
    return bone_length


def load_data(filename):
    # mano_fit_filename = "/media/korrawe/ssd/halo_vae/data/gen_val_kps/val_gen_6perObj_mano_fit.pkl"
    with open(filename, 'rb') as f:
        fit_params = pickle.load(f)

    # Convert mm to m
    gen_joints = fit_params['gen_joints'] / 1000.0
    mano_joints = fit_params['joints'] / 1000.0

    # Convert MANO joints to HALO joint
    gen_joints = convert_joints(gen_joints, source='mano', target='halo')
    mano_joints = convert_joints(mano_joints, source='mano', target='halo')

    data_size = len(fit_params['gen_joints'])
    print("data size: ", data_size)
    data_list = []
    for idx in range(data_size):
        dat = {
            'pose': fit_params['pose'][idx],
            'shape': fit_params['shape'][idx],
            'rot': fit_params['rot'][idx],
            'gen_joints': gen_joints[idx]
        }
        data_list.append(dat)

    # import pdb; pdb.set_trace()
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
    if args.noisy:
        multiple_file = False
    else:
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
        # print(input_files)
    else:
        # Load the whole data file
        data_file = "/media/korrawe/ssd/halo_vae/data/gen_val_kps/val_gen_6perObj_mano_fit.pkl"
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
            gen_joints = mano_params["gen_joints"]

        if args.fixshape:
            shape = torch.zeros(1, 10)

        # print("pose", pose)
        # print("shape", shape)
        # print("rot", rot)

        pose_para = torch.cat([rot, pose], 1)
        # print(pose_para)
        # assert False

        # Forward pass through MANO layer. All info is in meter scale
        hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = mano_layer(pose_para, shape, no_root_rot=False) # fixed_shape)

        # hand_verts_rot, hand_joints_rot, joints_trans_rot, _, _, _ = mano_layer(pose_para, shape) # fixed_shape)
        # hand_verts_rot = hand_verts_rot[0,:] - hand_joints_rot[0,0]

        hand_verts = hand_verts[0,:]
        hand_joints = hand_joints[0,:]
        joints_trans = joints_trans[0,:]
        rest_pose_verts = rest_pose_verts[0,:]
        rest_pose_joints = rest_pose_joints[0, :]
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
        # print("root rot ---", root_rot)
        # print("zeros", torch.zeros([1, 1, 3]))
        root_rot_mat = th_with_zeros(torch.cat([root_rot, torch.zeros([1, 3, 1])], 2))
        root_rot_mat = root_rot_mat[0]
        # print("root rot ---", root_rot_mat)
        # check_rot = True
        # if check_rot:
        #     fig = plt.figure(figsize=plt.figaspect(0.5))
        #     ax = fig.add_subplot(121, projection='3d')
        #     hand_verts = torch.matmul(root_rot_mat[0], torch.cat([hand_verts, torch.ones(778,1)], 1).unsqueeze(-1))
        #     hand_verts = hand_verts[:, :3, 0]
        #     print("hand_verts shape", hand_verts.shape)

        #     display_surface_points(hand_verts, verts_joints_assoc[:-1], ax=ax, show=False)
        #     ax = fig.add_subplot(122, projection='3d')
        #     display_surface_points(hand_verts_rot, verts_joints_assoc[:-1], ax=ax, show=True)

        neg_rest_pose_joints = -1. * rest_pose_joints
        translation = get_translation_mat(neg_rest_pose_joints)
        # print("translation", translation)
        # print("translation", translation.shape)
        new_trans_mat = torch.matmul(joints_trans, translation)
        # print("new_trans_mat", new_trans_mat[:3])
        rest_pose_joints = torch.cat([rest_pose_joints, rest_pose_joints.new_ones(16, 1)], 1)
        # print("root_xyz", torch.cat([root_xyz, torch.ones(1)]).shape)
        new_trans_mat[:, :3, 3] = new_trans_mat[:, :3, 3] - root_xyz
        final_joints = torch.matmul(new_trans_mat, rest_pose_joints.unsqueeze(2))

        rest_pose_joints_original = torch.cat([rest_pose_joints_original, rest_pose_joints_original.new_ones(16, 1)], 1)
        # print(rest_pose_joints_original.shape)
        final_joints_from_local = torch.matmul(joints_trans_local[0], rest_pose_joints_original.unsqueeze(2))

        # print("final_joints", final_joints.shape)

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
        # print("global_back_project", global_back_project)
        # print(tmp_joints.shape)
        # tmp_joints = tmp_joints[:, :3, 3]

        # check inv trans mat
        # print("new tran_mat", new_trans_mat.shape)
        # print("hand_joints", hand_joints.shape)

        # inverse
        inv_new_trans_mat = np.linalg.inv(new_trans_mat)
        # inv_new_trans_mat = np.matmul(scale_mat, inv_new_trans_mat)
        back_projected_joints = torch.matmul(torch.from_numpy(inv_new_trans_mat).float(), posed_joints.unsqueeze(2))
        # print("back_projected_joints", back_projected_joints.shape)
        # back_projected_joints, #
        # ----

        if visualize:
            demo.display_hand({
                'verts': hand_verts,
                'joints': tmp_joints[:, :3, 3],  # back_projected_joints, # global_back_project, # posed_joints,  # # final_joints[:, :3, 0],  # final_joints_from_local, #
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
        if args.noisy:
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(gen_joints[:, 0], gen_joints[:, 1], gen_joints[:, 2], color='b')
            # ax.scatter(hand_joints[:, 0], hand_joints[:, 1], hand_joints[:, 2], color='r')
            # plt.show()

            # compare_joints(gen_joints, hand_joints, nasa_joint_parent)
            # import pdb; pdb.set_trace()
            hand_joints = torch.Tensor(gen_joints)

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
            # import pdb; pdb.set_trace()
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

        # visualization test
        # cs_joints = hand_joints[:16].unsqueeze(0).cuda()
        # cs_joints_4 = torch.cat([cs_joints, torch.ones([1, 16, 1], device=cs_joints.device)], dim=2)
        # cs_joint_after_transform = torch.matmul(trans_mat_pc_all, cs_joints_4.unsqueeze(-1))

        # import pdb; pdb.set_trace()
        # not use
        # pred_joints = convert_joints(palm_align_kps_local_cs_nasa_axes, source='local_cs', target='nasa')
        # pred_joints = pred_joints[:, :16]
        # pred_joints_4 = torch.cat([pred_joints, torch.ones([1, 16, 1], device=pred_joints.device)], dim=2)
        # cs_joint_after_transform = torch.matmul(trans_mat_pc, pred_joints_4.unsqueeze(-1))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # vis_joints(cs_joint_after_transform.squeeze(-1).cpu(), parent='nasa', sixteen=True, ax=ax, vis=False)
        # vis_joints(hand_joints[:16].unsqueeze(0).cpu(), parent='nasa', sixteen=True, ax=ax, vis=False, color='black')
        # # vis_joints(rest_pose_joints.unsqueeze(0).cpu(), parent='nasa', sixteen=True, ax=ax, color='orange')
        # vis_joints(back_projected_joints.unsqueeze(0).squeeze(-1).cpu(), parent='nasa', sixteen=True, ax=ax, color='orange')

        trans_mat_pc = trans_mat_pc.squeeze(0).cpu()

        # Save mesh
        # hand_shape["faces"]
        mesh = trimesh.Trimesh(hand_verts, faces, process=False)
        mesh = seal(mesh)
        mesh.export(mesh_out)

        # Save bone transformation and surface vertices with label
        # Inverse the translation matrix for future use
        # # joints_trans_inv = np.linalg.inv(tmp_joints) # tmp_joints # 
        # joints_trans_inv = np.linalg.inv(new_trans_mat)
        # import pdb; pdb.set_trace()
        joints_trans_inv = trans_mat_pc_all.cpu().numpy().squeeze(0)
        # print('hand joints', hand_joints)

        # import pdb; pdb.set_trace()
        # Resample surface points
        vertices, vert_labels = sample_surface_with_label(mesh, verts_joints_assoc)  # , viz=True, joints=hand_joints)
        bone_lengths = get_bone_lengths(hand_joints)
        np.savez(meta_out, joints_trans=joints_trans_inv, verts=vertices, vert_labels=vert_labels,
                 shape=mano_params["shape"], bone_lengths=bone_lengths, hand_joints=hand_joints, root_rot_mat=root_rot_mat)

        # display_vertices_with_joints(hand_joints, vertices, vert_labels)

        # loaded_meta = np.load(meta_out)
        # print(loaded_meta['joints_trans'].shape)
        # print(loaded_meta['verts'].shape)
        # print(loaded_meta['vert_labels'].shape)

    # Save files list
    with open(os.path.join(args.root_folder, 'datalist.txt'), 'w') as listfile:
        for m in mesh_name_list:
            listfile.write('%s\n' % m)


nasa_joint_parent = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9])
adrian_joint_parent = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

def vis_joints(target_joints_right, ax=None, vis=True, set_axes=True, color='#1f77b4', parent='nasa', sixteen=False):

    print(target_joints_right.shape)
    # for i in range(21):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    jj = target_joints_right[0]

    ax.scatter(jj[:, 0], jj[:, 1], jj[:, 2], color='b', s=10)

    # ax.scatter(jj[i, 0], jj[i, 1], jj[i, 2], color='black', s=50)
    if set_axes:
        set_equal_xyz_scale(ax, jj[:, 0], jj[:, 1], jj[:, 2])
    # plt.show()
    # plt.close()

    # bone
    joints = target_joints_right
    # b_start = target_joints_right[0, nasa_joint_parent]
    # b_end = target_joints_right[0]
    # print('b_start', b_start)
    # print('b_end', b_end)
    if parent == 'nasa':
        b_start_loc = joints[0, nasa_joint_parent]
    elif parent == 'mano':
        print("mano parent")
        b_start_loc = joints[0, mano_joint_parent]
    elif parent == 'adrian':
        b_start_loc = joints[0, adrian_joint_parent]

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
