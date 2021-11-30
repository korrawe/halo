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

from visualize_utils import set_equal_xyz_scale

import sys
import json
import pickle
from tqdm import tqdm
from pycocotools.coco import COCO
sys.path.insert(0, "/home/korrawe/interhand/InterHand2.6M")
from common.nets.converter import PoseConverter, transform_to_canonical
from common.utils.transforms_torch import convert_interhand_to_nasa_joint, swap_axes_for_nasa
from common.utils.transforms_torch import convert_joints


# from common.nets.converter import transform_to_canonical
from common.utils.transforms import world2cam

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('--root_folder', type=str,
                    default='/home/korrawe/nasa/data/interhand/test',
                    # default='/home/korrawe/nasa/data/overfit/Subject_2_squeeze_paper_1_resample/test',
                    help='Output path root.')
parser.add_argument('--root_path', type=str,
                    default='/home/korrawe/interhand/data_with_mano/', # train_sub_folder
                    # default='/home/korrawe/nasa/data/FHB/test/Subject_2_squeeze_paper_1/test/mano',
                    help='Output path root.')
parser.add_argument('--mesh_folder', type=str,
                    default='/home/korrawe/nasa/data/interhand/test/mesh',
                    # default='/home/korrawe/nasa/data/overfit/Subject_2_squeeze_paper_1_resample/test/mesh',
                    help='Output path for mesh.')
parser.add_argument('--meta_folder', type=str,
                    default='/home/korrawe/nasa/data/interhand/test/meta',
                    # default='/home/korrawe/nasa/data/overfit/Subject_2_squeeze_paper_1_resample/test/meta',
                    help='Output path for points.')

parser.add_argument('--subfolder', action='store_true',
                help='Whether the data is stored in a sequential subfolder (./0/, ./1/, ./2/).')
parser.add_argument('--fixshape', action='store_true',
                help='If ture, fix the hand shape to mean shape and ignore the shape parameters in the dataset.')
parser.add_argument('--local_coord', action='store_true',
                help='If ture, store transformantion matices to local coordinate frames(origin) instead of to canonical pose..')

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


def handtype_str2array(hand_type):
    if hand_type == 'right':
        return np.array([1,0], dtype=np.float32)
    elif hand_type == 'left':
        return np.array([0,1], dtype=np.float32)
    elif hand_type == 'interacting':
        return np.array([1,1], dtype=np.float32)
    else:
        assert 0, print('Not supported hand type: ' + hand_type)


def process_interhand_data(split='train'):
    ## Initialize MANO layer
    ncomps = 45 # 6
    mano_layer_right = ManoLayer(
        mano_root='/home/korrawe/nasa/scripts/mano/models', use_pca=False, ncomps=ncomps, side='right', flat_hand_mean=False)
    mano_layer_left = ManoLayer(
        mano_root='/home/korrawe/nasa/scripts/mano/models', use_pca=False, ncomps=ncomps, side='left', flat_hand_mean=False)

    mano_layer = {'right': mano_layer_right, 'left': mano_layer_left}
    # smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True)
    # flat_hand_mean=False

    root_path = '/home/korrawe/interhand/data_with_mano/'
    img_root_path = os.path.join(root_path, 'images')
    annot_root_path = os.path.join(root_path, 'annotations')
    subset = 'all'
    # split = 'train'
    # capture_idx = '13'
    # seq_name = '0266_dh_pray'
    # cam_idx = '400030'

    # save_path = os.path.join(subset, split, capture_idx, seq_name, cam_idx)
    # os.makedirs(save_path, exist_ok=True)

    joint_num = 21 # single hand
    root_joint_idx = {'right': 20, 'left': 41}
    joint_type = {'right': np.arange(0, joint_num), 'left': np.arange(joint_num, joint_num*2)}


    db = COCO(os.path.join(annot_root_path, subset, 'InterHand2.6M_' + split + '_data.json'))

    with open(os.path.join(annot_root_path, subset, 'InterHand2.6M_' + split + '_MANO.json')) as f:
        mano_params = json.load(f)
    with open(os.path.join(annot_root_path, subset, 'InterHand2.6M_' + split + '_camera.json')) as f:
        cam_params = json.load(f)
    with open(os.path.join(annot_root_path, subset, 'InterHand2.6M_' + split + '_joint_3d.json')) as f:
        joints = json.load(f)

    left_mano_list = []
    right_mano_list = []
    left_interhand_list = []
    right_interhand_list = []
    joint_error_left = []
    joint_error_right = []

    mano_canonical = []
    interhand_canonical = []

    mano_list_right = []
    mano_list_left = []
    interhand_joints_right_list = []
    interhand_joints_left_list = []
    hand_name_right_list = []
    hand_name_left_list = []

    print("load successfully")
    # return 

    i = 0
    for aid in tqdm(db.anns.keys()):
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]

        capture_id = img['capture']
        seq_name = img['seq_name']
        cam = img['camera']
        frame_idx = img['frame_idx']

        has_mano = False
        mano_joint_right = None
        mano_joint_left = None

        mano_param_right = None
        mano_param_left = None
        # 2 4 6 chars
        hand_name = "/".join([str(capture_id).zfill(2), str(frame_idx).zfill(4), cam.zfill(6)])
        # print("hand name", hand_name)
        for hand_type in ('right', 'left'):
            # get mano joints
            try:
                # import pdb; pdb.set_trace()
                mano_param = mano_params[str(capture_id)][str(frame_idx)][cam][hand_type]
                if mano_param is None:
                    continue
                has_mano = True
                # print("mano_param", mano_param)
            except KeyError:
                print("KeyError", capture_id, frame_idx, cam, hand_type)
                continue

            mano_pose = torch.FloatTensor(mano_param['pose']).view(1,-1)    
            # mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
            # root_pose = mano_pose[0].view(1,3)
            # hand_pose = mano_pose[1:,:].view(1,-1)
            shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
            trans = torch.FloatTensor(mano_param['trans']).view(1,-1)
            output = mano_layer[hand_type](mano_pose, th_betas=shape, th_trans=trans)
            verts, nasa_joints, _, _, _, _ = output
            # mesh = output.vertices[0].numpy() # meter unit
            # joint = output.joints[0].numpy() # meter unit
            # print("mano joint", joint)
            # import pdb; pdb.set_trace()
            # vis_joints(nasa_joints, parent='nasa')
            nasa_joints_local_cs = convert_joints(nasa_joints, source='nasa', target='local_cs')
            if hand_type == 'right':
                mano_joint_right = nasa_joints_local_cs
                mano_param_right = mano_param
            else:
                mano_joint_left = nasa_joints_local_cs
                mano_param_left = mano_param
        
        campos, camrot = np.array(cam_params[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cam_params[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
        focal, princpt = np.array(cam_params[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cam_params[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
        joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
        joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
        # joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]

        joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(joint_num*2)
        # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
        joint_valid[joint_type['right']] *= joint_valid[root_joint_idx['right']]
        joint_valid[joint_type['left']] *= joint_valid[root_joint_idx['left']]
        hand_type = ann['hand_type']
        hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)

        hand_type = handtype_str2array(hand_type)

        # print("joint_cam", joint_cam, joint_cam.shape)
        cam_param = {'focal': focal, 'princpt': princpt}
        joint = {'cam_coord': joint_cam, 'valid': joint_valid}

        
        root_valid = joint_valid[root_joint_idx['right']] * joint_valid[root_joint_idx['left']]  if hand_type[0]*hand_type[1] == 1 else np.zeros((1),dtype=np.float32)
        
        # if root_valid == 0:
        #     continue

        right_hand_kps_nasa, left_hand_kps_nasa = convert_interhand_to_nasa_joint(np.expand_dims(joint_cam, axis=0))
        # print("right_hand_kps_nasa", right_hand_kps_nasa)

        right_hand_kps_nasa = (right_hand_kps_nasa / 1000.0)
        left_hand_kps_nasa = (left_hand_kps_nasa / 1000.0)

        # if joint_valid[root_joint_idx['right']]:
        if np.all(joint_valid[joint_type['right']]):
            interhand_right = right_hand_kps_nasa
            # print("All right hand joints valid")
        else:
            interhand_right = None
        
        if np.all(joint_valid[joint_type['left']]):
            interhand_left = left_hand_kps_nasa
            # print("All left hand joints valid")
        else:
            interhand_left = None

        # right hand
        if interhand_right is not None and mano_joint_right is not None:
            
            # import pdb;pdb.set_trace()
            interhand_right_local_cs = convert_joints(interhand_right, source='nasa', target='local_cs')
            error = np.sqrt(np.sum((interhand_right_local_cs - mano_joint_right.cpu().detach().numpy())**2, 2)).mean()
            # print("fitting error", error)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # vis_joints(interhand_right_local_cs, parent='adrian', ax=ax, vis=False)
            # vis_joints(mano_joint_right, parent='adrian', ax=ax, color='orange')

            if error < 0.02:
                joint_error_right.append(error)
                
                right_mano_list.append(mano_joint_right.cpu().detach().numpy())
                right_interhand_list.append(interhand_right_local_cs)

                mano_list_right.append(mano_param_right)
                hand_name_right_list.append(hand_name + '_right')
                # print(hand_name_right_list[-1])
            # i += 1
        
        # left hand
        if interhand_left is not None and mano_joint_left is not None:
            # import pdb;pdb.set_trace()
            interhand_left_local_cs = convert_joints(interhand_left, source='nasa', target='local_cs')
            error = np.sqrt(np.sum((interhand_left_local_cs - mano_joint_left.cpu().detach().numpy())**2, 2)).mean()
            # print("fitting error", error)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # vis_joints(interhand_left_local_cs, parent='adrian', ax=ax, vis=False)
            # vis_joints(mano_joint_left, parent='adrian', ax=ax, color='orange')

            if error < 0.02:
                joint_error_left.append(error)

                left_mano_list.append(mano_joint_left.cpu().detach().numpy())
                left_interhand_list.append(interhand_left_local_cs)

                mano_list_left.append(mano_param_left)
                hand_name_left_list.append(hand_name + '_left')
                # print(hand_name_left_list[-1])

            # i += 1
        
        # print("frame_idx", frame_idx)
        # i += 1
        # if i > 50:
        #     break
    
     
    # import pdb;pdb.set_trace()
    out_dir = '/home/korrawe/nasa/data/interhand'

    left_mano_list = np.array(left_mano_list).squeeze(1)
    left_interhand_list = np.array(left_interhand_list).squeeze(1)
    # list - not numpy
    # mano_list_left
    # hand_name_left_list

    right_mano_list = np.array(right_mano_list).squeeze(1)
    right_interhand_list = np.array(right_interhand_list).squeeze(1)
    # list - not numpy
    # mano_list_right
    # hand_name_right_list

    joint_error_left = np.array(joint_error_left)
    joint_error_right = np.array(joint_error_right)

    print("Joint Error right:", joint_error_right.mean())
    print("JOint Error left:", joint_error_left.mean())

    print("left hand total:", left_mano_list.shape[0])
    print("right hand total:", right_mano_list.shape[0])

    data_dict = {
        'mano_joint_left': left_mano_list,
        'interhand_joint_left': left_interhand_list,
        'mano_param_left': mano_list_left,
        'hand_name_left': hand_name_left_list,

        'mano_joint_right': right_mano_list,
        'interhand_joint_right': right_interhand_list,
        'mano_param_right': mano_list_right,
        'hand_name_right': hand_name_right_list,
    }
    pickle.dump( data_dict, open( os.path.join(out_dir, split + ".pkl"), "wb" ) )
    
    # pickle.load(open(os.path.join(out_dir, split + ".pkl"), 'rb'))
    # import pdb;pdb.set_trace()
    
    #  ----------------

    # interhand = []
    # mano_GT = []
    return 

# def load_data_pickle(split='train'):
#     out_dir = '/home/korrawe/interhand/InterHand2.6M/data/mano_projection'
#     loaded_data = pickle.load(open(os.path.join(out_dir, split + "_align.pkl"), 'rb'))
#     interhand_joints_data = loaded_data['interhand']
#     mano_joints_data = loaded_data['mano']

#     return interhand_joints_data, mano_joints_data

def main(args):
    print("Load input MANO parameters from", args.root_path)
    print("Recovering hand meshes from mano parameter to", args.mesh_folder)
    print("Save bone transformations and vertex labels to", args.meta_folder)

    out_dir = '/home/korrawe/nasa/data/interhand'
    split = 'test' # 'train'
    target_size = 3000 # 50000
    # process_interhand_data(split="test")
    # return 
 
    visualize = False # True

    # load data

    data_pickle = pickle.load(open(os.path.join(out_dir, split + ".pkl"), 'rb'))
    data_size = len(data_pickle['mano_joint_right'])
    print("data_size", data_size)

    keep_ids = np.random.choice(len(data_pickle['mano_joint_right']), target_size)
    print(keep_ids.shape)
    print(keep_ids[:20])

    # return 

    # if args.subfolder:
    #     input_files = glob.glob(os.path.join(args.mano_folder, '*/*.npz'))
    #     sub_dirs = [os.path.basename(n) for n in glob.glob(os.path.join(args.mano_folder, '*'))]

    #     for sub_d in sub_dirs:
    #         if not os.path.isdir(os.path.join(args.mesh_folder, sub_d)):
    #             os.makedirs(os.path.join(args.mesh_folder, sub_d))
    
    #         if not os.path.isdir(os.path.join(args.meta_folder, sub_d)):
    #             os.makedirs(os.path.join(args.meta_folder, sub_d))
    # else:
    #     input_files = glob.glob(os.path.join(args.mano_folder, '*.npz'))
        
    if not os.path.isdir(args.mesh_folder):
        os.makedirs(args.mesh_folder)

    if not os.path.isdir(args.meta_folder):
        os.makedirs(args.meta_folder)

    # print(input_files)


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
        mano_root='mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=False)
        # mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False)
    faces = mano_layer.th_faces.detach().cpu().numpy()
    
    # Initialize PoseConverter
    pose_converter = PoseConverter()

    # Mesh list
    mesh_name_list = []
    # Sample one at a time
    batch_size = 1

    # start_pose = rand_pose(1, ncomps, pose_std, rot_std)

    # for mano_in_file in input_files:
    i = 0
    for idx in tqdm(keep_ids):
        i += 1
        # if i > 5:
        #     break 
        # 'mano_joint_right': right_interhand_list,
        # 'interhand_joint_right': right_interhand_list,
        # 'mano_param_right': mano_list_right,
        # 'hand_name_right': hand_name_right_list,
        interhand_joint = data_pickle['interhand_joint_right'][idx]
        mano_joint = data_pickle['mano_joint_right'][idx]
        hand_name = data_pickle['hand_name_right'][idx]

        mano_param = data_pickle['mano_param_right'][idx]
        # print(mano_param)
        print("hand name", hand_name)

        pose = torch.FloatTensor(mano_param['pose']).view(1,-1)    
        rot = pose[:, :3]
        # mano_pose = torch.FloatTensor(mano_param['pose']).view(-1,3)
        # root_pose = mano_pose[0].view(1,3)
        # hand_pose = mano_pose[1:,:].view(1,-1)
        shape = torch.FloatTensor(mano_param['shape']).view(1,-1)
        trans = torch.FloatTensor(mano_param['trans']).view(1,-1)
        
        # output = mano_layer[hand_type](mano_pose, th_betas=shape, th_trans=trans)

        # print(mano_in_file)
        # i = os.path.splitext(os.path.basename(mano_in_file))[0]
        # i = int(i)
        # if args.subfolder:
        #     sub_dir = os.path.split(os.path.split(mano_in_file)[0])[1]
        #     # print("sub_dir", sub_dir)
        # mano_params = np.load(mano_in_file)
        # pose = torch.from_numpy(mano_params["pose"]).unsqueeze(0)
        # shape = torch.from_numpy(mano_params["shape"]).unsqueeze(0)
        # rot = torch.from_numpy(mano_params["rot"]).unsqueeze(0)

        # print("pose", pose)
        # print("shape", shape)
        # print("rot", rot)

        # pose_para = torch.cat([rot, pose], 1)
        # print(pose_para)

        # Forward pass through MANO layer. All info is in meter scale
        hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = mano_layer(pose, shape, no_root_rot=False) # fixed_shape)
        # continue
        # hand_verts_rot, hand_joints_rot, joints_trans_rot, _, _, _ = mano_layer(pose_para, shape) # fixed_shape)
        # hand_verts_rot = hand_verts_rot[0,:] - hand_joints_rot[0,0]
        
        interhand_joint = interhand_joint - interhand_joint[0]
        mano_joint = mano_joint - mano_joint[0]

        # import pdb;pdb.set_trace()
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
        new_trans_mat = torch.matmul(joints_trans , translation)
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
        ## ----
        
        if visualize:
            demo.display_hand({
                'verts': hand_verts,
                'joints': tmp_joints[:, :3, 3], # back_projected_joints, # global_back_project, # posed_joints,  # # final_joints[:, :3, 0],  # final_joints_from_local, #
                'rest_joints':  interhand_joint, # rest_pose_joints,
                'verts_assoc': verts_joints_assoc
            },
                            mano_faces=mano_layer.th_faces)
        
        num_str = f'{i:03}'
        # if args.subfolder:
        #     mesh_name_list.append(sub_dir + "/" + num_str)
        #     mesh_out = os.path.join(args.mesh_folder, sub_dir, num_str + ".off")
        #     meta_out = os.path.join(args.meta_folder, sub_dir, num_str + ".npz")
        # else:
        mesh_name_list.append(hand_name)
        mesh_out = os.path.join(args.mesh_folder, hand_name + ".off")
        meta_out = os.path.join(args.meta_folder, hand_name + ".npz")

        # Create dir if not exists
        if not os.path.isdir(os.path.dirname(mesh_out)):
            os.makedirs(os.path.dirname(mesh_out))
        
        if not os.path.isdir(os.path.dirname(meta_out)):
            os.makedirs(os.path.dirname(meta_out))


        ###### Compute local coord trans mat
        joints_for_nasa_input = torch.tensor([0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16])

        # kps_local_cs = convert_joints(hand_joints.unsqueeze(0), source='nasa', target='local_cs').cuda()
        interhand_joint_nasa = convert_joints(interhand_joint[None, :], source='local_cs', target='nasa')[0]

        # interhand_joint already in local_cs
        kps_local_cs = torch.from_numpy(interhand_joint).unsqueeze(0).cuda()
        is_right_one = torch.ones(1, device=kps_local_cs.device)
        # import pdb; pdb.set_trace()
        
        palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(kps_local_cs, is_right=is_right_one)
        palm_align_kps_local_cs_nasa_axes = swap_axes_for_nasa(palm_align_kps_local_cs)

        glo_rot_right = torch.cat([glo_rot_right, torch.tensor([[[0, 0, 0, 1]]], device=glo_rot_right.device)], dim=1)
        swap_axes_mat = torch.zeros([4,4])
        swap_axes_mat[0,1] = 1
        swap_axes_mat[1,2] = 1
        swap_axes_mat[2,0] = 1
        swap_axes_mat[3,3] = 1
        swap_axes_mat = swap_axes_mat.unsqueeze(0).cuda()
        rot_then_swap_mat = torch.matmul(swap_axes_mat, glo_rot_right).unsqueeze(0)

        trans_mat_pc, _ = pose_converter(palm_align_kps_local_cs_nasa_axes, is_right_one)
        trans_mat_pc = convert_joints(trans_mat_pc, source='local_cs', target='nasa')
        trans_mat_pc = trans_mat_pc[:, joints_for_nasa_input]
        # trans_mat_pc = trans_mat_pc.squeeze(0).cpu()

        trans_mat_pc_all = torch.matmul(trans_mat_pc, rot_then_swap_mat)
        
        # visualization test
        cs_joints = convert_joints(kps_local_cs, source='local_cs', target='nasa')[:, :16]
        cs_joints_4 = torch.cat([cs_joints, torch.ones([1, 16, 1], device=cs_joints.device)], dim=2)
        cs_joint_after_transform = torch.matmul(trans_mat_pc_all, cs_joints_4.unsqueeze(-1))

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
        # vis_joints(interhand_joint_nasa[None, :], parent='nasa', sixteen=True, ax=ax, color='red')
        # vis_joints(back_projected_joints.unsqueeze(0).squeeze(-1).cpu(), parent='nasa', sixteen=True, ax=ax, color='orange')
        
        trans_mat_pc = trans_mat_pc.squeeze(0).cpu()

        # Save mesh
        # hand_shape["faces"]
        mesh = trimesh.Trimesh(hand_verts, faces, process=False)
        mesh = seal(mesh)
        mesh.export(mesh_out)

        # Save bone transformation and surface vertices with label
        # Inverse the translation matrix for future use
        ## joints_trans_inv = np.linalg.inv(tmp_joints) # tmp_joints # 
        # joints_trans_inv = np.linalg.inv(new_trans_mat)
        # import pdb; pdb.set_trace()
        joints_trans_inv = trans_mat_pc_all.cpu().numpy().squeeze(0)

        # Resample surface points 
        vertices, vert_labels = sample_surface_with_label(mesh, verts_joints_assoc) # , viz=True, joints=hand_joints)
        bone_lengths = get_bone_lengths(interhand_joint_nasa)
        # import pdb;pdb.set_trace()
        np.savez(meta_out, joints_trans=joints_trans_inv, verts=vertices, vert_labels=vert_labels, 
                 shape=shape.cpu().detach().numpy(), bone_lengths=bone_lengths, hand_joints=interhand_joint_nasa, root_rot_mat=root_rot_mat)
        
        # display_vertices_with_joints(hand_joints, vertices, vert_labels)

        # loaded_meta = np.load(meta_out)
        # print(loaded_meta['joints_trans'].shape)
        # print(loaded_meta['verts'].shape)
        # print(loaded_meta['vert_labels'].shape)
    
    # import pdb;pdb.set_trace()
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