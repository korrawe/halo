from os import makedirs
from numpy.lib.utils import source
import torch
import sys
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "/home/korrawe/halo_vae")
from models.halo_adapter.projection import JointProjectionLayer, get_projection_layer
from models.halo_adapter.adapter import HaloAdapter
from models.halo_adapter.converter import PoseConverter, transform_to_canonical
from models.halo_adapter.interface import convert_joints, change_axes
from models.halo_adapter.transform_utils import xyz_to_xyz1
from models.utils import visualize as vis

sys.path.insert(0, "/home/korrawe/nasa/scripts")
from manopth.manolayer import ManoLayer
from manopth.manolayer_mod import ManoLayerMod
from manopth import demo

import pdb
import json


def t2np(tensor):
    return tensor.detach().cpu().numpy()


def seal(mesh_to_seal):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype=np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    # pylint: disable=unsubscriptable-object # pylint/issues/3139
    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i - 1], circle_v_id[i], center_v_id]
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])
    return mesh_to_seal


def scale_halo_trans_mat(trans_mat, scale=0.4):
    ''' Scale the transformation matrices to match the scale of HALO.
        Maybe this should be in the HALO package.
    Args:
        trans_mat: Transformation matrices that are already inverted (from pose to unpose)
    '''
    # Transform meta data
    # Assume that the transformation matrices are already inverted
    scale_mat = torch.eye(4, device=trans_mat.device).reshape(1, 1, 4, 4).repeat(trans_mat.shape[0], 1, 1, 1) * scale
    scale_mat[:, :, 3, 3] = 1.

    nasa_input = torch.matmul(trans_mat, scale_mat)
    # (optional) scale canonical pose by the same global scale to make learning occupancy function easier
    canonical_scale_mat = torch.eye(4, device=trans_mat.device).reshape(1, 1, 4, 4).repeat(trans_mat.shape[0], 1, 1, 1) / scale
    canonical_scale_mat[:, :, 3, 3] = 1.
    nasa_input = torch.matmul(canonical_scale_mat, nasa_input)
    return nasa_input


def get_halo_matrices(trans_mat):
    # Use 16 out of 21 joints for nasa inputs
    joints_for_nasa_input = torch.tensor([0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16])
    trans_mat = trans_mat[:, joints_for_nasa_input]
    return trans_mat


def cano2cano_mat(mano_cano_joints, device):
    kps = mano_cano_joints
    is_right_vec = torch.ones(kps.shape[0], device=device) * True
    # Scale from cm (VAE) to m (HALO)
    # scale = 100.0
    # kps = kps / scale

    # Global normalization
    normalized_kps, normalization_mat = transform_to_canonical(kps, is_right=is_right_vec)

    normalized_kps, change_axes_mat = change_axes(normalized_kps, target='halo')
    normalization_mat = torch.matmul(change_axes_mat, normalization_mat)

    # unpose_mat, _ = self.pose_normalizer(normalized_kps, is_right_vec)
    # # Change to HALO joint order
    # unpose_mat = convert_joints(unpose_mat, source='biomech', target='halo')
    # unpose_mat = self.get_halo_matrices(unpose_mat)
    # unpose_mat_scaled = scale_halo_trans_mat(unpose_mat)
    return normalization_mat


def root_center(hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local):
    root_xyz = hand_joints[:, 0]

    # Move root joint to origin
    hand_verts = hand_verts - root_xyz
    hand_joints = hand_joints - root_xyz
    rest_pose_verts = rest_pose_verts - root_xyz
    rest_pose_joints = rest_pose_joints - root_xyz

    return hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local


def get_mano_hand(idx, mano_param, mano_layer, halo_adapter, pose_converter, device):
    faces = mano_layer.th_faces.detach().cpu().numpy()

    # shape = torch.zeros(1, 10).to(device)
    # rot = torch.zeros(1, 10).to(device)
    # pose = torch.rand(1, 45).to(device) - 0.5
    # pose_para = torch.cat([rot, pose], 1)
    # print(pose_para)
    # assert False
    # pdb.set_trace()
    shape = torch.tensor(mano_param['shape']).to(device).unsqueeze(0)
    pose_para = torch.tensor(mano_param['pose']).to(device).unsqueeze(0)

    # Forward pass through MANO layer. All info is in meter scale
    hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = mano_layer(pose_para, shape, no_root_rot=False)
    # Root centered
    (hand_verts, hand_joints, joints_trans,
     rest_pose_verts, rest_pose_joints,
     joints_trans_local) = root_center(hand_verts, hand_joints, joints_trans,
                                       rest_pose_verts, rest_pose_joints, joints_trans_local)
    # pdb.set_trace()
    # Canonical pose correction
    # cano_mat = cano2cano_mat(rest_pose_joints, device)
    # undo_norm_kps = torch.matmul(torch.inverse(cano_mat), xyz_to_xyz1(hand_joints).unsqueeze(-1)).squeeze(-1)
    # undo_norm_kps = undo_norm_kps[:, :, :3]
    # hand_joints = undo_norm_kps

    # Get HALO mesh
    halo_mesh = halo_adapter(hand_joints * 100.0, joint_order='halo', original_position=True)
    halo_mesh.vertices = halo_mesh.vertices / (100.)
    halo_verts = halo_mesh.vertices  # / 2.5
    halo_out = '/home/korrawe/halo_vae/exp/interhand_seq/'
    halo_mesh.export(halo_out + str(idx) + '.obj')
    # hand_joints = convert_joints(hand_joints, source='halo', target=)
    # vis.visualise_skeleton(hand_joints)
    # pdb.set_trace()

    # Visualize
    hand_info = {
        'verts': t2np(hand_verts.squeeze(0)),
        'joints': t2np(hand_joints.squeeze(0)),
        'rest_joints': t2np(rest_pose_joints.squeeze(0))
    }
    mano_mesh = trimesh.Trimesh(t2np(hand_verts.squeeze(0)), faces, process=False)
    mano_mesh = seal(mano_mesh)
    # mano_mesh.export(halo_out + str(idx) + '.obj')  # 'mano_hand.obj')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # vis.display_mano_hand(hand_info, faces, ax=ax, alpha=0.2, show=False)
    # # ax.scatter(halo_verts[:, 0], halo_verts[:, 1], halo_verts[:, 2])
    # plt.show()


def read_interhand():
    interhand_path = '/home/korrawe/nasa/data/interhand/InterHand2.6M_train_MANO_NeuralAnnot.json'
    with open(interhand_path) as f:
        data = json.load(f)

    data_list = []
    max_hand = 200
    count = 0
    for seq in data.keys():
        # print(seq)
        for frame_idx in data[seq].keys():
            if data[seq][frame_idx]['right'] is not None:
                # print(data[seq][frame_idx]['right'])
                data_list.append(data[seq][frame_idx]['right'])
                count += 1
                if count >= max_hand:
                    break

            # print(frame_idx)
        if count >= max_hand:
            break
        # pdb.set_trace()

    return data_list

def main():
    data_list = read_interhand()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Select number of principal components for pose space
    ncomps = 45  # 6
    # Initialize MANO layer
    mano_layer = ManoLayer(
        mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=False).to(device)
    faces = mano_layer.th_faces.detach().cpu().numpy()

    halo_config_file = "/home/korrawe/nasa/configs/iccv/yt3d_b16_keypoint_normalized_fix.yaml"
    halo_adapter = HaloAdapter(halo_config_file, device=device, denoiser_pth=None)

    # Initialize PoseConverter
    pose_converter = PoseConverter()

    for idx, mano_param in enumerate(data_list):
        # pdb.set_trace()
        get_mano_hand(idx, mano_param, mano_layer, halo_adapter, pose_converter, device)


if __name__ == '__main__':
    main()
