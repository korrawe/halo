# from os import makedirs
import os
# from numpy.lib.utils import source
import torch
import sys
import pickle
import numpy as np
import trimesh
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "/home/korrawe/halo_vae")
from models.halo_adapter.adapter import HaloAdapter
from models.halo_adapter.converter import PoseConverter, transform_to_canonical
from models.halo_adapter.interface import convert_joints, change_axes
from models.halo_adapter.transform_utils import xyz_to_xyz1
from models.utils import visualize as vis

from models.mano_converter.mano_converter import ManoConverter

sys.path.insert(0, "/home/korrawe/nasa/scripts")
from manopth.manolayer import ManoLayer
# from manopth.manolayer_mod import ManoLayerMod
# from manopth import demo

import pdb


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


def rot_mat_to_axis_angle(R):
    """
    Taken from 
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/derivation/index.htm
    """
    # val_1 = (R[:,0,0] + R[:,1,1] + R[:,2,2] - 1) / 2
    # angles = torch.acos(val_1)
    # denom = 2 * torch.sqrt((val_1 ** 2 - 1).abs())
    # x = (R[:,2,1] - R[:,1,2]) / denom
    # y = (R[:,0,2] - R[:,2,0]) / denom
    # z = (R[:,1,0] - R[:,0,1]) / denom
    angles = torch.acos((R[...,0,0] + R[...,1,1] + R[...,2,2] - 1) / 2)
    denom = torch.sqrt(
            (R[...,2,1] - R[...,1,2]) ** 2 + 
            (R[...,0,2] - R[...,2,0]) ** 2 + 
            (R[...,1,0] - R[...,0,1]) ** 2
            )
    x = (R[...,2,1] - R[...,1,2]) / denom
    y = (R[...,0,2] - R[...,2,0]) / denom
    z = (R[...,1,0] - R[...,0,1]) / denom
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)
    axis = torch.cat((x,y,z), dim=-1)
    return axis, angles


def trans2bmc_cano_mat(mano_cano_joints, pose_normalizer, device):
    # if joint_order != 'biomech':
    kps = convert_joints(mano_cano_joints, source='halo', target='biomech')

    is_right_vec = torch.ones(kps.shape[0], device=device) * True
    # Scale from cm (VAE) to m (HALO)
    # scale = 100.0
    # kps = kps / scale

    # pdb.set_trace()

    # Global normalization
    normalized_kps, normalization_mat = transform_to_canonical(kps, is_right=is_right_vec)

    normalized_kps, change_axes_mat = change_axes(normalized_kps, target='halo')
    normalization_mat = torch.matmul(change_axes_mat, normalization_mat)

    unpose_mat, _ = pose_normalizer(normalized_kps, is_right_vec)
    # # Change to HALO joint order
    unpose_mat = convert_joints(unpose_mat, source='biomech', target='halo')
    unpose_mat = get_halo_matrices(unpose_mat)
    # unpose_mat_scaled = scale_halo_trans_mat(unpose_mat)

    full_trans_mat = torch.matmul(unpose_mat, normalization_mat)

    return full_trans_mat  # normalization_mat


def get_translation_mat(vec):
    # vec must be 3x1 matrix
    rot = torch.eye(3).to(vec.device)
    mat_4_3 = torch.cat([rot, torch.zeros(1, 3).to(vec.device)], 0).repeat(16, 1, 1)
    # print(mat_4_3.shape)
    trans_4_1 = torch.cat([vec, vec.new_ones(16, 1)], 1)
    translation = torch.cat([mat_4_3, trans_4_1.unsqueeze(2)], 2)
    return translation


def root_center(hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local):
    root_xyz = hand_joints[:, 0]

    rest_pose_joints_before = rest_pose_joints
    hand_joints_before = hand_joints
    # Move root joint to origin
    hand_verts = hand_verts - root_xyz
    hand_joints = hand_joints - root_xyz
    rest_pose_verts = rest_pose_verts - root_xyz
    rest_pose_joints = rest_pose_joints - root_xyz

    pdb.set_trace()
    # Modify transformation matrices for root centered hand
    # neg_rest_pose_joints = -1. * rest_pose_joints.squeeze(0)
    # translation = get_translation_mat(neg_rest_pose_joints)
    # print("translation", translation)
    # print("translation", translation.shape)
    # new_trans_mat = torch.matmul(joints_trans, translation)
    # print("new_trans_mat", new_trans_mat[:3])
    # rest_pose_joints = torch.cat([rest_pose_joints, rest_pose_joints.new_ones(16, 1)], 1)
    # print("root_xyz", torch.cat([root_xyz, torch.ones(1)]).shape)
    # joints_trans[:, :, :3, 3] = joints_trans[:, :, :3, 3] - root_xyz
    final_joints = torch.matmul(joints_trans_local, xyz_to_xyz1(rest_pose_joints_before).unsqueeze(-1))

    inv_new_trans_mat = torch.inverse(joints_trans_local)  # np.linalg.inv(joints_trans)
    # inv_new_trans_mat = np.matmul(scale_mat, inv_new_trans_mat)
    back_projected_joints = torch.matmul(inv_new_trans_mat, xyz_to_xyz1(hand_joints_before[:, :16]).unsqueeze(-1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    joint_parent = vis.get_joint_parent('halo')
    vis.plot_skeleton(ax, t2np(rest_pose_joints.squeeze(0)), joint_parent, 'r')
    vis.plot_skeleton(ax, t2np(hand_joints.squeeze(0)), joint_parent, 'g')
    vis.plot_skeleton(ax, t2np(final_joints.squeeze(0).squeeze(-1)), joint_parent, 'orange')
    # vis.plot_skeleton(ax, t2np(back_projected_joints.squeeze(0).squeeze(-1)), joint_parent, 'orange')
    # vis.display_mano_hand(hand_info, faces, ax=ax, alpha=0.2, show=False)
    # ax.scatter(halo_verts[:, 0], halo_verts[:, 1], halo_verts[:, 2])
    vis.cam_equal_aspect_3d(ax, t2np(hand_joints.squeeze(0)))
    plt.show()

    return hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local


def global_mat2mano_input(cano2pose_mat):

    global_rots = cano2pose_mat[:, :, :3, :3]
    R = global_rots

    # Convert to local rotation
    # TODO Once this works, take out global rotation too
    # lvl1_idx = torch.tensor([5,6,7,8,9])
    # lvl2_idx = torch.tensor([10,11,12,13,14])
    # lvl3_idx = torch.tensor([15,16,17,18,19])
    lvl1_idx = torch.tensor([1, 4, 7, 10, 13])
    lvl2_idx = torch.tensor([2, 5, 8, 11, 14])
    lvl3_idx = torch.tensor([3, 6, 9, 12, 15])
    R_lvl1 = R[:, lvl1_idx]
    R_lvl2 = R[:, lvl2_idx]
    R_lvl3 = R[:, lvl3_idx]
    R_lvl2_l = R_lvl1.transpose(-1, -2) @ R_lvl2
    R_lvl3_l = R_lvl2.transpose(-1, -2) @ R_lvl3
    # pdb.set_trace()
    R_local_no_order = torch.cat([R[:, None, 0], R_lvl1, R_lvl2_l, R_lvl3_l], dim=1)
    reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
    R_local = R_local_no_order[:, reorder_idxs]
    # th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]

    axis, angles = rot_mat_to_axis_angle(R_local)  # global_rots
    # axis[0, 0] = torch.tensor([0., 0., 0.])
    axis_angle_mano = axis * angles.unsqueeze(-1)
    return axis_angle_mano.reshape(axis_angle_mano.shape[0], -1)


def load_ho3d(mano_layer, mano_layer_axisang, halo_adapter, pose_converter, device):
    ho3d_path = "/media/korrawe/ssd/ho3d/data/HO3D_V2/"  # "/is/cluster/work/kkarunratanakul/HO3D/"
    split = "train"
    seq_list_split = os.path.join(ho3d_path, split + ".txt")

    # get sequence dir list
    sequence_dir_list = []
    with open(seq_list_split, 'r') as sequence_file:
        i = 0
        for line in sequence_file:
            # if 'MC2' in line:
            # if 'SB10' in line:
            # print(line)
            # if 'SMu42/1345' in line:
            sequence_dir_list.append(line.strip())
            # print(line.strip())
            i += 1

    i = 0
    for seq in sequence_dir_list:
        # meta files
        mata_file_path = os.path.join(ho3d_path, split, os.path.dirname(seq), "meta", os.path.basename(seq) + ".pkl")
        print("pkl list", mata_file_path)
        process_ho3d(mata_file_path, i, mano_layer, mano_layer_axisang, halo_adapter, pose_converter, device)
        i += 1


def process_ho3d(pkl_file, idx, mano_layer, mano_layer_axisang, halo_adapter, pose_converter, device):
    data = pickle.load(open(pkl_file, 'rb'))
    beta = data['handBeta']
    beta = (torch.from_numpy(beta)).unsqueeze(0).float().to(device)
    pose = data['handPose']
    pose = (torch.from_numpy(pose)).unsqueeze(0).float().to(device)
    # pdb.set_trace()
    get_mano_hand(pose, beta, idx, mano_layer, mano_layer_axisang, halo_adapter, pose_converter, device)


def get_mano_hand(pose_para, shape, idx, mano_layer, mano_layer_axisang, halo_adapter, pose_converter, device):
    outdir = '/home/korrawe/halo_vae/exp/mano_conversion/'

    faces = mano_layer.th_faces.detach().cpu().numpy()

    # shape = torch.rand(1, 10).to(device) * 3.0 - 1.5
    # # rot = torch.rand(1, 3).to(device) * 2.0 - 1.0
    # rot = torch.zeros(1, 3).to(device)
    # pose = torch.rand(1, 45).to(device) * 2 - 1.0
    # pose_para = torch.cat([rot, pose], 1)
    # pdb.set_trace()
    # print(pose_para)
    # assert False
    print("shape: ", shape)

    # pose_para[:, 0] = 0.2
    # pose_para[:, 1] = 0.123
    # pose_para[:, 2] = 1.4
    # Forward pass through MANO layer. All info is in meter scale
    hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_global = mano_layer(pose_para, shape, no_root_rot=False)

    # GT rest pose mesh
    # mano_mesh = trimesh.Trimesh(t2np(rest_pose_verts.squeeze(0)), faces, process=False)
    # mano_mesh = seal(mano_mesh)
    # mano_mesh.export(outdir + 'mano_hand_rest.obj')

    # GT mesh
    mano_mesh = trimesh.Trimesh(t2np(hand_verts.squeeze(0)), faces, process=False)
    mano_mesh = seal(mano_mesh)
    mano_mesh.export(outdir + str(idx) + '_' + 'mano_hand.obj')

    # Add Gaussian noise to GT key points up to +-2 mm
    # print('before:', hand_joints)
    # noise = (torch.rand(hand_joints.shape).to(device) - 0.5) * 2. * 2. / 1000.
    # hand_joints = hand_joints + noise
    # print('after:', hand_joints)
    # pdb.set_trace()

    # Derive mano param from keypoints
    mano_converter = ManoConverter(mano_layer, pose_converter, device=device)
    mano_shape, mano_pose = mano_converter.to_mano(hand_joints)
    
    print("MANO pose", mano_pose)
    pdb.set_trace()
    mano_pose[:, 0] = 0
    mano_pose[:, 1] = 0
    mano_pose[:, 2] = 0
    # Regularize shape with 1-step update of L2 loss
    # print("shape est before reg: ", mano_shape)
    # mano_shape = mano_shape - 0.5 * mano_shape
    print("shape est: ", mano_shape)

    hand_verts_test, hand_joints_test, joints_trans_test, _, _, _ = mano_layer_axisang(mano_pose, mano_shape)

    mano_mesh_from_kps = trimesh.Trimesh(t2np(hand_verts_test.squeeze(0)), faces, process=False)
    mano_mesh_from_kps = seal(mano_mesh_from_kps)
    mano_mesh_from_kps.export(outdir + str(idx) + '_' + 'mano_from_kps.obj')

    # Calculate error
    joint_err = hand_joints_test - hand_joints
    joint_err = torch.norm(joint_err, dim=2)
    print("joint error (m)", joint_err)

    surface_err = hand_verts_test - hand_verts
    surface_err = torch.norm(surface_err, dim=2).mean()
    print("surface error (m): ", surface_err.item())

    # Get noisy HALO mesh
    halo_mesh = halo_adapter(hand_joints * 100.0, joint_order='halo', original_position=True)
    halo_mesh.vertices = halo_mesh.vertices / (100.)
    halo_mesh.export(outdir + str(idx) + '_' + 'halo_noisy.obj')

    # Get HALO mesh - shape from Adrain's formulation
    halo_mesh = halo_adapter(hand_joints_test * 100.0, joint_order='halo', original_position=True)
    halo_mesh.vertices = halo_mesh.vertices / (100.)
    halo_mesh.export(outdir + str(idx) + '_' + 'halo_new_shape.obj')
    return



    ##########

    # Canonical pose correction
    mano_rest2bmc_cano_mat = trans2bmc_cano_mat(rest_pose_joints, pose_converter, device)
    posed_hand2bmc_cano_mat = trans2bmc_cano_mat(hand_joints, pose_converter, device)

    hand2mano_mat = torch.matmul(torch.inverse(mano_rest2bmc_cano_mat), posed_hand2bmc_cano_mat)
    # bmc_joints = torch.matmul(mano_rest2bmc_cano_mat, xyz_to_xyz1(rest_pose_joints[:, :16]).unsqueeze(-1))
    # bmc_joints = torch.matmul(hand2mano_mat, xyz_to_xyz1(hand_joints[:, :16]).unsqueeze(-1))
    bmc_joints = torch.matmul(torch.inverse(hand2mano_mat), xyz_to_xyz1(rest_pose_joints[:, :16]).unsqueeze(-1))

    diff = bmc_joints.squeeze(-1)[..., :3] - hand_joints[:, :16]
    diff = torch.norm(diff, dim=2)
    print("joint error", diff)

    posed_mano_mat = torch.inverse(hand2mano_mat)

    # test mano input
    mano_input = global_mat2mano_input(posed_mano_mat)
    # mano_input = global_mat2mano_input(joints_trans_global)
    # remove mean pose angle
    mano_input = torch.cat([
        mano_input[:, :3],
        mano_input[:, 3:] - mano_layer.th_hands_mean
    ], 1)
    hand_verts_test, hand_joints_test, joints_trans_test, _, _, joints_trans_global_test = mano_layer_axisang(mano_input, shape)
    # hand_verts_test, hand_joints_test, joints_trans_test, _, _, joints_trans_global_test = mano_layer_axisang(th_full_pose, shape)
    mano_bmc_mesh_local = trimesh.Trimesh(t2np(hand_verts_test.squeeze(0)), faces, process=False)
    mano_bmc_mesh_local = seal(mano_bmc_mesh_local)
    mano_bmc_mesh_local.export(outdir + 'mano_bmc_hand_local.obj')

    joint_err = hand_joints_test - hand_joints
    joint_err = torch.norm(joint_err, dim=2)
    print("joint error", joint_err)

    pdb.set_trace()
    surface_err = hand_verts_test - hand_verts
    surface_err = torch.norm(surface_err, dim=2).mean()
    print("surface error: ", surface_err)
    # end test mano input
    # posed_mano_bmc = mano_layer_mod(posed_mano_mat, shape)

    # surface_err_no_poseblend = posed_mano_bmc - hand_verts
    # surface_err_no_poseblend = torch.norm(surface_err_no_poseblend, dim=2).mean()
    # print("surface error with out pose blend shape: ", surface_err_no_poseblend)

    # # Root centered
    # (hand_verts, hand_joints, joints_trans,
    #  rest_pose_verts, rest_pose_joints,
    #  joints_trans_local) = root_center(hand_verts, hand_joints, joints_trans,
    #                                    rest_pose_verts, rest_pose_joints, joints_trans_local)

    # pdb.set_trace()

    # Get HALO mesh
    halo_mesh = halo_adapter(hand_joints * 100.0, joint_order='halo', original_position=True)
    halo_mesh.vertices = halo_mesh.vertices / (100.)
    halo_verts = halo_mesh.vertices  # / 2.5

    halo_mesh.export(outdir + 'halo_hand.obj')
    # hand_joints = convert_joints(hand_joints, source='halo', target=)
    # vis.visualise_skeleton(hand_joints)
    # pdb.set_trace()

    # Visualize
    hand_info = {
        'verts': t2np(hand_verts.squeeze(0)),
        # 'verts': t2np(rest_pose_verts.squeeze(0)),
        'joints': t2np(hand_joints.squeeze(0)),
        # 'joints': t2np(rest_pose_joints.squeeze(0)),
        # 'rest_joints': t2np(rest_pose_joints.squeeze(0))
        'rest_joints': t2np(bmc_joints.squeeze(0))
    }
    # hand_info = {
    #     'verts': t2np(posed_mano_bmc.squeeze(0)),
    #     'joints': t2np(hand_joints.squeeze(0)),
    #     'rest_joints': t2np(rest_pose_joints.squeeze(0))
    # }
    

    # mano_bmc_mesh = trimesh.Trimesh(t2np(posed_mano_bmc.squeeze(0)), faces, process=False)
    # mano_bmc_mesh = seal(mano_bmc_mesh)
    # mano_bmc_mesh.export(outdir + 'mano_bmc_hand.obj')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vis.display_mano_hand(hand_info, faces, ax=ax, alpha=0.2, show=False)
    # ax.scatter(halo_verts[:, 0], halo_verts[:, 1], halo_verts[:, 2])

    hand_verts_vis = t2np(hand_verts.squeeze(0))
    # ax.scatter(hand_verts_vis[:, 0], hand_verts_vis[:, 1], hand_verts_vis[:, 2])
    # rest_verts = t2np(rest_pose_verts.squeeze(0))
    # ax.scatter(rest_verts[:, 0], rest_verts[:, 1], rest_verts[:, 2])
    plt.show()
    # pdb.set_trace()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Select number of principal components for pose space
    ncomps = 45  # 6
    # Initialize MANO layer
    mano_layer = ManoLayer(
        # mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False).to(device)
        mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True).to(device)
    # MANO axis-angle
    mano_layer_axisang = ManoLayer(
        # mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=False).to(device)
        mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True).to(device)
    # Initialize modified MANO layer that take trans_mat as inputs
    # mano_layer_mod = ManoLayerMod(
    #     # mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False).to(device)
    #     mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True).to(device)
    faces = mano_layer.th_faces.detach().cpu().numpy()

    halo_config_file = "/home/korrawe/nasa/configs/iccv/yt3d_b16_keypoint_normalized_fix.yaml"
    halo_adapter = HaloAdapter(halo_config_file, device=device, denoiser_pth=None)

    # Initialize PoseConverter
    pose_converter = PoseConverter()

    load_ho3d(mano_layer, mano_layer_axisang, halo_adapter, pose_converter, device)

    # get_mano_hand(mano_layer, mano_layer_mod, mano_layer_axisang, halo_adapter, pose_converter, device)


if __name__ == '__main__':
    main()
