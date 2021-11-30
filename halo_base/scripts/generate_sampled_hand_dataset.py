import torch
import numpy as np
import trimesh
import argparse
import os

from manopth.manolayer import ManoLayer
from manopth import demo


parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('--root_folder', type=str,
                    default='/home/korrawe/nasa/data/check_scale_bug/test',
                    help='Output path root.')
parser.add_argument('--mesh_folder', type=str,
                    default='/home/korrawe/nasa/data/check_scale_bug/test/mesh',
                    help='Output path for mesh.')
parser.add_argument('--meta_folder', type=str,
                    default='/home/korrawe/nasa/data/check_scale_bug/test/meta',
                    help='Output path for points.')


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


def main(args):
    print("Generating sampled hand meshes to", args.mesh_folder)
    print("Save bone transformations and vertex labels to", args.meta_folder)

    visualize = False # True

    # Total number of hand to generate
    sample_size = 5

    # How close the sampled hands are to the mean pose
    rot_std = 3.0
    pose_std = 2.0

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
    for i in range(sample_size):
        # Generate random shape parameters
        fixed_shape = torch.zeros(batch_size, 10) # torch.rand(batch_size, 10)
        # Generate random pose parameters, including 3 values for global axis-angle rotation
        # torch.rand() returns uniform samples in [0,1)
        random_pose = torch.rand(batch_size, ncomps) * 2.0 - 1.0
        random_pose = random_pose * pose_std
        random_rot = torch.rand(batch_size, 3) * 2.0 - 1.0 
        random_rot = random_rot * rot_std
        random_pose = torch.cat([random_rot, random_pose], 1)
        # print("random pose", random_pose.shape)

        # Forward pass through MANO layer. All info is in meter scale
        hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = mano_layer(random_pose, fixed_shape)
        
        hand_verts = hand_verts[0,:]
        hand_joints = hand_joints[0,:]
        joints_trans = joints_trans[0,:]
        rest_pose_verts = rest_pose_verts[0,:]
        rest_pose_joints = rest_pose_joints[0, :]
        rest_pose_joints_original = rest_pose_joints + 0

        root_xyz = hand_joints[0]
        # print("posed root joints", root_xyz)
        # print("rest pose root joints", rest_pose_verts[0])
        # print("rest pose joints", rest_pose_joints)
        # print("root", root_xyz)
        # print(hand_verts.shape)
        # print(hand_joints.shape)
        # print(joints_trans.shape)

        # Move root joint to origin
        hand_verts = hand_verts - root_xyz
        hand_joints = hand_joints - root_xyz
        rest_pose_verts = rest_pose_verts - root_xyz
        rest_pose_joints = rest_pose_joints - root_xyz
        # joints_trans[:, :3, 3] = joints_trans[:, :3, 3] - root_xyz
        # print(joints_trans.shape)

        # rest_pose_joints = rest_pose_joints[0,:] - rest_pose_joints[0,0]
        # print("rest_pose_joints", rest_pose_joints.shape)
        # rest_pose_joints = rest_pose_joints[:, :3] - root_xyz
        # print("joint trans", joints_trans.shape)

        # print("joint_js", rest_pose_joints.shape)
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
        # Data list
        mesh_name_list.append(num_str)
        # Save mesh
        mesh_out = os.path.join(args.mesh_folder, num_str + ".off")
        # hand_shape["faces"]
        mesh = trimesh.Trimesh(hand_verts, faces, process=False)
        mesh = seal(mesh)
        mesh.export(mesh_out)

        # Save bone transformation and surface vertices with label
        meta_out = os.path.join(args.meta_folder, num_str + ".npz")
        # Inverse the translation matrix for future use
        # joints_trans_inv = np.linalg.inv(tmp_joints) # tmp_joints # 
        joints_trans_inv = np.linalg.inv(new_trans_mat)
        np.savez(meta_out, joints_trans=joints_trans_inv, verts=mesh.vertices, vert_labels=verts_joints_assoc)
        # loaded_meta = np.load(meta_out)
        # print(loaded_meta['joints_trans'].shape)
        # print(loaded_meta['verts'].shape)
        # print(loaded_meta['vert_labels'].shape)
    
    # Save files list
    with open(os.path.join(args.root_folder, 'datalist.txt'), 'w') as listfile:
        for m in mesh_name_list:
            listfile.write('%s\n' % m)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)