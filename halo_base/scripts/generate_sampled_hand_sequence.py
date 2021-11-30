import torch
import numpy as np
import trimesh
import argparse
import os

from manopth.manolayer import ManoLayer
from manopth import demo

from sample_utils import sample_surface_with_label
from manopth.tensutils import (th_posemap_axisang, th_with_zeros)

from prepare_data_from_mano_param import get_bone_lengths


parser = argparse.ArgumentParser('Prepare data by morphing pose s to pose t')
parser.add_argument('--root_folder', type=str,
                    default='/home/korrawe/nasa/data/eval_fist/test',
                    help='Output path root.')
parser.add_argument('--mesh_folder', type=str,
                    default='/home/korrawe/nasa/data/eval_fist/test/mesh',
                    help='Output path for mesh.')
parser.add_argument('--meta_folder', type=str,
                    default='/home/korrawe/nasa/data/eval_fist/test/meta',
                    help='Output path for points.')

parser.add_argument('-s', type=str,
                    default='/home/korrawe/nasa/data/youtube_hand/raw/test/0617.npz',
                    help='Start pose.')
parser.add_argument('-t', type=str,
                    # default='/home/korrawe/nasa/data/youtube_hand/raw/test/0238.npz',
                    default='/home/korrawe/interhand/InterHand2.6M/output/result/GT_custom/0/00000_right.npz',
                    help='End pose.')

parser.add_argument('--subfolder', action='store_true',
                help='Whether the data is stored in a sequential subfolder (./0/, ./1/, ./2/).')

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


def rand_pose(batch_size, ncomps, pose_std, rot_std, rot_param=None):
    random_pose = torch.rand(batch_size, ncomps) * 2.0 - 1.0
    random_pose = random_pose * pose_std
    if rot_param is not None:
        random_pose = torch.cat([rot_param, random_pose], 1)
    else:
        random_rot = torch.rand(batch_size, 3) * 2.0 - 1.0 
        random_rot = random_rot * rot_std
        random_pose = torch.cat([random_rot, random_pose], 1)
    return random_pose


def main(args):
    print("Generating sampled hand meshes to", args.mesh_folder)
    print("Save bone transformations and vertex labels to", args.meta_folder)

    visualize = False # True

    # Total number of hand to generate
    sample_size = 20

    # How close the sampled hands are to the mean pose
    rot_std = 3.0
    pose_std = 2.0 # 1.0

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
    
    if not os.path.isdir(args.mesh_folder):
        os.makedirs(args.mesh_folder)
    
    if not os.path.isdir(args.meta_folder):
        os.makedirs(args.meta_folder)
    
    # Mesh list
    mesh_name_list = []
    # Sample one at a time
    batch_size = 1

    fixed_shape = torch.zeros(batch_size, 10)
    
    if args.s is not None and args.t is not None:
        meta_s = np.load(args.s)
        meta_t = np.load(args.t)

        start_pose = torch.from_numpy(meta_s["pose"]).unsqueeze(0)
        start_shape = torch.from_numpy(meta_s["shape"]).unsqueeze(0)
        start_rot = torch.from_numpy(meta_s["rot"]).unsqueeze(0)

        end_pose = torch.from_numpy(meta_t["pose"]).unsqueeze(0)
        end_shape = torch.from_numpy(meta_t["shape"]).unsqueeze(0)
        end_rot = torch.from_numpy(meta_t["rot"]).unsqueeze(0)

        # no rot
        print("start shape", start_shape)
        print("end shape", end_shape)
        print("start shape norm", np.linalg.norm(start_shape))
        print("end shape norm", np.linalg.norm(end_shape))
        
        start_rot = end_rot
        # start_pose = end_pose

        # print("pose", pose)
        # print("shape", shape)
        # print("rot", rot)

        start_pose = torch.cat([start_rot, start_pose], 1)
        end_pose = torch.cat([end_rot, end_pose], 1)

    else:
        random_seq = True
        start_pose = rand_pose(1, ncomps, pose_std, rot_std)
        end_pose = rand_pose(1, ncomps, pose_std, rot_std, rot_param=start_pose[:,:3])

        start_shape = torch.zeros(batch_size, 10)
        end_shape = torch.zeros(batch_size, 10)

    # Forward pass through MANO layer. All info is in meter scale
    # Cancel here if the pose is not good.
    hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = mano_layer(start_pose, start_shape)
    demo.display_hand({
            'verts': hand_verts[0],
            'joints': joints_trans[0, :, :3, 3], # back_projected_joints, # global_back_project, # posed_joints,  # # final_joints[:, :3, 0],  # final_joints_from_local, #
            'rest_joints':  rest_pose_joints[0],
            'verts_assoc': verts_joints_assoc
        },
        mano_faces=mano_layer.th_faces)

    hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = mano_layer(end_pose, end_shape)
    demo.display_hand({
            'verts': hand_verts[0],
            'joints': joints_trans[0, :, :3, 3], # back_projected_joints, # global_back_project, # posed_joints,  # # final_joints[:, :3, 0],  # final_joints_from_local, #
            'rest_joints':  rest_pose_joints[0],
            'verts_assoc': verts_joints_assoc
        },
        mano_faces=mano_layer.th_faces)


    j = 0
    for j in range(4):
        for ii in range(sample_size):
            # Generate random pose parameters, including 3 values for global axis-angle rotation
            # torch.rand() returns uniform samples in [0,1)
            # random_pose = torch.rand(batch_size, ncomps) * 2.0 - 1.0
            # random_pose = random_pose * pose_std
            # random_rot = torch.rand(batch_size, 3) * 2.0 - 1.0 
            # random_rot = random_rot * rot_std
            # random_pose = torch.cat([random_rot, random_pose], 1)
            # print("random pose", random_pose.shape)

            # more than 2 poses
            i = j * sample_size + ii
            print(i)

            interpolated_pose = start_pose + ((end_pose - start_pose) * ii) / sample_size
            interpolated_shape = start_shape + ((end_shape - start_shape) * ii) / sample_size
            interpolated_rot = start_rot + ((end_rot - start_rot) * ii) / sample_size

            # Forward pass through MANO layer. All info is in meter scale
            hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = mano_layer(interpolated_pose, interpolated_shape) # fixed_shape)
            
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
            th_pose_map, th_rot_map = th_posemap_axisang(interpolated_rot)
            root_rot = th_rot_map[:, :9].view(1, 3, 3)
            # print("root rot ---", root_rot)
            # print("zeros", torch.zeros([1, 1, 3]))
            root_rot_mat = th_with_zeros(torch.cat([root_rot, torch.zeros([1, 3, 1])], 2))
            root_rot_mat = root_rot_mat[0]
        
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
            inv_new_trans_mat = np.matmul(scale_mat, inv_new_trans_mat)
            back_projected_joints = torch.matmul(torch.from_numpy(inv_new_trans_mat).float(), posed_joints.unsqueeze(2))
            # print("back_projected_joints", back_projected_joints.shape)
            # back_projected_joints, #
            ## ----
            
            if visualize:
                demo.display_hand({
                    'verts': hand_verts,
                    'joints': tmp_joints[:, :3, 3], # back_projected_joints, # global_back_project, # posed_joints,  # # final_joints[:, :3, 0],  # final_joints_from_local, #
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

            # Data list
            
            # Save mesh
            # hand_shape["faces"]
            mesh = trimesh.Trimesh(hand_verts, faces, process=False)
            mesh = seal(mesh)
            mesh.export(mesh_out)

            # Save bone transformation and surface vertices with label
            # Inverse the translation matrix for future use
            # joints_trans_inv = np.linalg.inv(tmp_joints) # tmp_joints # 
            joints_trans_inv = np.linalg.inv(new_trans_mat)
            # Resample surface points 
            vertices, vert_labels = sample_surface_with_label(mesh, verts_joints_assoc) # , viz=True, joints=hand_joints)
            bone_lengths = get_bone_lengths(hand_joints)
            np.savez(meta_out, joints_trans=joints_trans_inv, verts=vertices, vert_labels=vert_labels, 
                    shape=interpolated_shape, bone_lengths=bone_lengths, hand_joints=hand_joints, root_rot_mat=root_rot_mat)
            
            # display_vertices_with_joints(hand_joints, vertices, vert_labels)

            # loaded_meta = np.load(meta_out)
            # print(loaded_meta['joints_trans'].shape)
            # print(loaded_meta['verts'].shape)
            # print(loaded_meta['vert_labels'].shape)

        start_pose = end_pose
        end_pose = rand_pose(1, ncomps, pose_std, rot_std, rot_param=start_pose[:,:3])
    
    # Save files list
    with open(os.path.join(args.root_folder, 'datalist.txt'), 'w') as listfile:
        for m in mesh_name_list:
            listfile.write('%s\n' % m)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)