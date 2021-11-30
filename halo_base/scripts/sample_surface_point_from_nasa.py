import argparse
import trimesh
import numpy as np
import os
import torch
import sys
sys.path.insert(0, "/home/korrawe/nasa")
from matplotlib import pyplot as plt

import visualize_utils
from visualize_utils import display_surface_points
from sample_utils import sample_surface_with_label
from mesh_utils import (get_bone_lengths, get_verts_association, seal)

from manopth.manolayer import ManoLayer
from manopth import demo

from artihand import config, data
from artihand.checkpoints import CheckpointIO


parser = argparse.ArgumentParser('Prepare data by morphing pose s to pose t')
parser.add_argument('--nasa_folder', type=str,
                    default='/home/korrawe/nasa/data/eval_fist/test',
                    help='Path to a trained NASA model.')
parser.add_argument('--out_folder', type=str,
                    default='/home/korrawe/nasa/tmp_out',
                    help='Output path.')

parser.add_argument('--config', type=str, 
                     default='/home/korrawe/nasa/configs/sample_hands/yt3d_all_bone_no_wn.yaml',
                     help='Path to config file.')

parser.add_argument('-n', type=int,
                    default=10000,
                    help='number of points to sample')

# parser.add_argument('--subfolder', action='store_true',
#                 help='Whether the data is stored in a sequential subfolder (./0/, ./1/, ./2/).')


def adjust_joint_locations(hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints):
    # Squeeze dimension
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

    return hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, root_xyz


def _get_translation_mat(vec):
    # vec must be 3x1 matrix
    rot = torch.eye(3)
    mat_4_3 = torch.cat([rot, torch.zeros(1, 3)], 0).repeat(16, 1, 1)
    # print(mat_4_3.shape)
    trans_4_1 = torch.cat([vec, vec.new_ones(16, 1)], 1)
    translation = torch.cat([mat_4_3, trans_4_1.unsqueeze(2)], 2)
    return translation


def calculate_new_trans_mat(rest_pose_joints, joints_trans, root_xyz):
    neg_rest_pose_joints = -1. * rest_pose_joints        
    translation = _get_translation_mat(neg_rest_pose_joints)
    # print("translation", translation)
    # print("translation", translation.shape)
    new_trans_mat = torch.matmul(joints_trans , translation)
    # print("new_trans_mat", new_trans_mat[:3])
    rest_pose_joints = torch.cat([rest_pose_joints, rest_pose_joints.new_ones(16, 1)], 1)
    # print("root_xyz", torch.cat([root_xyz, torch.ones(1)]).shape)
    new_trans_mat[:, :3, 3] = new_trans_mat[:, :3, 3] - root_xyz

    return new_trans_mat


def color_to_label(vert_colors):
    print(vert_colors)
    print(vert_colors.shape)
    bone_colors = np.array([
        (119, 41, 191, 255), (75, 170, 46, 255), (116, 61, 134, 255), (44, 121, 216, 255), (250, 191, 216, 255), (129, 64, 130, 255),
        (71, 242, 184, 255), (145, 60, 43, 255), (51, 68, 187, 255), (208, 250, 72, 255), (104, 155, 87, 255), (189, 8, 224, 255),
        (193, 172, 145, 255), (72, 93, 70, 255), (28, 203, 124, 255), (131, 207, 80, 255)
        ], dtype=np.uint8
    )

    vert_label = []
    for v_c in vert_colors:
        found = False
        for bone_id, color in enumerate(bone_colors):
            # print("v_c", v_c)
            # print("color", color)
            if (v_c == color).all():
                found = True
                vert_label.append(bone_id)
                break

        if not found:
            print(v_c)
            assert False
    
    return np.asarray(vert_label)


def get_colors(vert_labels):
    bone_colors = np.array([
        (119, 41, 191, 255), (75, 170, 46, 255), (116, 61, 134, 255), (44, 121, 216, 255), (250, 191, 216, 255), (129, 64, 130, 255),
        (71, 242, 184, 255), (145, 60, 43, 255), (51, 68, 187, 255), (208, 250, 72, 255), (104, 155, 87, 255), (189, 8, 224, 255),
        (193, 172, 145, 255), (72, 93, 70, 255), (28, 203, 124, 255), (131, 207, 80, 255)
        ], dtype=np.uint8
    )
    return bone_colors[vert_labels]


def get_hand_mano(mano_layer, shape, pose_para, rest_pose=True):
    ## Load MANO
    # ncomps = 45 # 6
    # ## Initialize MANO layer
    # mano_layer = ManoLayer(
    #     mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False)
    faces = mano_layer.th_faces.detach().cpu().numpy()
    batch_size = 1
    
    ## Get MANO canonical pose with mean shape
    # Forward pass through MANO layer. All info are in meter scale
    hand_verts, hand_joints, joints_trans, rest_pose_verts, rest_pose_joints, joints_trans_local = mano_layer(pose_para, shape, no_root_rot=False)

    (hand_verts, hand_joints, joints_trans,
     rest_pose_verts, rest_pose_joints, root_xyz) = adjust_joint_locations(hand_verts, hand_joints, joints_trans, 
                                                                           rest_pose_verts, rest_pose_joints)
    
    # print(hand_verts)
    print(hand_verts.shape)
    # Transformation matrics
    new_trans_mat = calculate_new_trans_mat(rest_pose_joints, joints_trans, root_xyz)

    if rest_pose:
        mesh = trimesh.Trimesh(rest_pose_verts, faces, process=False)
    else:
        mesh = trimesh.Trimesh(hand_verts, faces, process=False)
    mesh = seal(mesh)

    bone_lengths = get_bone_lengths(hand_joints)
    return mesh, hand_joints, new_trans_mat, bone_lengths, rest_pose_verts, rest_pose_joints


def trans_mat2nasa_input(mesh, joints_trans_inv, scale = 0.4):
    '''Note that the scale of the mesh also changed in-place
    '''
    # Transform input mesh
    mesh.apply_scale(1 / scale)
    # meta['verts'] = meta['verts'] / scale
    # meta['hand_joints'] = meta['hand_joints'] / scale

    # Transform meta data
    # Assume that the transformation matrices are already inverted
    scale_mat = np.identity(4) * scale 
    scale_mat[3,3] = 1.

    nasa_input = np.matmul(joints_trans_inv, scale_mat)
    # (optional) scale canonical pose by the same global scale to make learning occupancy function easier
    canonical_scale_mat = np.identity(4) / scale 
    canonical_scale_mat[3,3] = 1.
    nasa_input = np.matmul(canonical_scale_mat, nasa_input)
    
    return nasa_input


def create_template_nasa(mano_layer, args):
    
    ## Save bone transformation matrices and bone length
    mesh_out = os.path.join(args.out_folder, 'mesh.obj')
    surface_points_out = os.path.join(args.out_folder, 'points.obj')
    meta_out = os.path.join(args.out_folder, 'meta.npz')
    
    shape = torch.zeros(1, 10)
    pose_para = torch.zeros(1, 45 + 3)

    # shape = torch.rand(1, 10) * 3.0 - 1.5
    # pose_para = torch.rand(1, 45 + 3)  * 3.0 - 1.5

    mesh, hand_joints, new_trans_mat, bone_lengths, rest_pose_verts, rest_pose_joints = get_hand_mano(mano_layer, shape, pose_para)
    mesh.export(mesh_out)

    # Inverse the translation matrix for future use
    joints_trans_inv = np.linalg.inv(new_trans_mat)

    # Resample surface points 
    verts_joints_assoc = get_verts_association('/home/korrawe/nasa/scripts/resource/skinning_weight_r.npy', add_wrist=True)
    vertices, vert_labels = sample_surface_with_label(mesh, verts_joints_assoc, viz=False) # , joints=hand_joints)    

    # print(meta_out)
    np.savez(meta_out, joints_trans=joints_trans_inv, verts=vertices, vert_labels=vert_labels, 
             shape=shape, bone_lengths=bone_lengths, hand_joints=hand_joints) #  root_rot_mat=root_rot_mat)

    # 
    # num_surface_points = 10000
    # surface_points, face_index = trimesh.sample.sample_surface(mesh, 10000)
    # print(surface_points)
    # For visualization in MeshLab
    surface_points_mesh = trimesh.Trimesh(vertices, process=False)
    surface_points_mesh.export(surface_points_out)

    return mesh, hand_joints, joints_trans_inv, bone_lengths, vertices, vert_labels, rest_pose_verts, rest_pose_joints


def canonical_to_posed(vertices, trans_mat, vert_labels, inv=False):
    ''' Converts the mesh vertices from canonical pose to posed mesh using the input 
    transformation matrices and the labels.
    Args:
        vertices (numpy array?): vertices of the mesh
        trans_mat ()?: latent conditioned code c. Must be a transformation matices without projection.
        vert_labels ()?: labels indicating which sub-model each vertex belongs to.
        inv (bool): whether trans_mat is inverse. If true, inverse the matrices back  ((B)(x)). 
            Else, use it as-is.
    '''
    if inv:
        trans_mat = np.linalg.inv(trans_mat)
    # print(trans_mat.shape)
    # print(vertices.shape)
    # print(type(vertices))
    # print(vert_labels.shape)

    pointsf = torch.FloatTensor(vertices) # .to(self.device)
    # print("pointssf before", pointsf.shape)
    # [V, 3] -> [V, 4, 1]
    pointsf = torch.cat([pointsf, pointsf.new_ones(pointsf.shape[0], 1)], dim=1)
    print("pointsf", pointsf.shape)
    pointsf = pointsf.unsqueeze(2)
    print("pointsf", pointsf.shape)

    trans_mat = torch.from_numpy(trans_mat).float()
    print("trans mat", trans_mat.shape)
    print("vert_labels", vert_labels.shape)
    vert_trans_mat = trans_mat[vert_labels]

    # print(vert_trans_mat.shape)
    new_vertices = torch.matmul(vert_trans_mat, pointsf)

    vertices = new_vertices[:,:3].squeeze(2).detach().cpu().numpy()
    # print("return", vertices.shape)

    return vertices # new_vertices


def posed_to_canonical(vertices, trans_mat, vert_labels, inv=False):
    ''' Converts the mesh vertices back to canonical pose using the input transformation matrices
    and the labels.
    Args:
        vertices (numpy array?): vertices of the mesh
        c (tensor): latent conditioned code c. Must be a transformation matices without projection.
        vert_labels (tensor): labels indicating which sub-model each vertex belongs to.
        inv (bool): whether trans_mat is inverse. If true, use the matrices as-is. (B^(-1)(x)). 
            Else, inverse it first.
    '''
    if not inv:
        trans_mat = np.linalg.inv(trans_mat)
    # print(trans_mat.shape)
    # print(vertices.shape)
    # print(type(vertices))
    # print(vert_labels.shape)

    pointsf = torch.FloatTensor(vertices) # .to(self.device)
    # print("pointssf before", pointsf.shape)
    # [V, 3] -> [V, 4, 1]
    pointsf = torch.cat([pointsf, pointsf.new_ones(pointsf.shape[0], 1)], dim=1)
    pointsf = pointsf.unsqueeze(2)
    # print("pointsf", pointsf.shape)

    trans_mat = torch.from_numpy(trans_mat).float()
    vert_trans_mat = trans_mat[vert_labels]
    print('trans_mat', trans_mat.shape)
    # print(vert_trans_mat.shape)
    new_vertices = torch.matmul(vert_trans_mat, pointsf)

    vertices = new_vertices[:,:3].squeeze(2).detach().cpu().numpy()
    # print("return", vertices.shape)

    return vertices # new_vertices


def get_nasa_model(config_file):
    no_cuda = False
    print("config_file", config_file)
    cfg = config.load_config(config_file, '/home/korrawe/nasa/configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    # Model
    model = config.get_model(cfg, device=device)

    out_dir = cfg['training']['out_dir']
    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])

    # print(checkpoint_io.module_dict['model'])

    # print(model.state_dict().keys())

    # Generator
    generator = config.get_generator(model, cfg, device=device)
    # print("upsampling", generator.upsampling_steps)
    return model, generator


def main(args):

    ## Initialize MANO layer
    ncomps = 45 # 6
    mano_layer = ManoLayer(
        mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False)

    ## Load NASA
    nasa_model, nasa_generator = get_nasa_model(args.config)

    # Create base template
    t_mesh, t_hand_joints, t_trans_mat_inv, t_bone_lengths, t_surface_vertices, t_vert_labels = create_template_nasa(mano_layer, args)

    print('template bone lengths', t_bone_lengths)


    # get nasa input from MANO
    print(t_mesh.vertices.shape)
    print(t_trans_mat_inv.shape)
    nasa_input = trans_mat2nasa_input(t_mesh, t_trans_mat_inv)
    t_mesh_scale_out = os.path.join(args.out_folder, 't_mesh_scale.obj')
    t_mesh.export(t_mesh_scale_out)

    tmp_vert = t_surface_vertices.copy()

    # try posing
    new_vertices = canonical_to_posed(tmp_vert, t_trans_mat_inv, t_vert_labels, inv=True)
    t_mesh.vertices = new_vertices
    t_mesh.faces = None
    posed_points_out = os.path.join(args.out_folder, 'points_posed.obj')
    t_mesh.export(posed_points_out)

    # try converting back to canonical
    new_vertices = posed_to_canonical(new_vertices, t_trans_mat_inv, t_vert_labels, inv=True)
    t_mesh.vertices = new_vertices
    posed_points_out = os.path.join(args.out_folder, 'points_posed2cano.obj')
    t_mesh.export(posed_points_out)

    ## Obtain the NASA shape in canonical pose from the input from MANO
    # print('nasa input', nasa_input.shape)
    data_in = {}
    data_in['inputs'] = torch.FloatTensor(nasa_input).unsqueeze(0)
    data_in['bone_lengths'] = torch.from_numpy(t_bone_lengths).unsqueeze(0)
    print(data_in['inputs'].shape)
    print('nasa input', type(data_in['inputs']))
    print(data_in['bone_lengths'].shape)
    nasa_mesh_out, stat = nasa_generator.generate_mesh(data_in)
    # nasa_mesh_out.vertices = nasa_mesh_out.vertices * 0.4
    # print(nasa_mesh_out)

    nasa_out_file = os.path.join(args.out_folder, 'nasa_out.obj')
    nasa_mesh_out.export(nasa_out_file)

    nasa_cano_vertices = nasa_mesh_out.vertices.copy()

    ## Modify the generation code to also get the bone association back.
    nasa_vert_label = color_to_label(nasa_mesh_out.visual.vertex_colors)
    print(nasa_vert_label)
    print(nasa_vert_label.shape)
    print(nasa_mesh_out.vertices.shape)

    ## Load NASA output mesh

    ## Sample ~20k points on the surface on NASA mesh to be use as template
    n_points_surface = 6000
    sample_points_nasa, sample_points_nasa_labels = sample_surface_with_label(nasa_mesh_out, nasa_vert_label, sample_n=n_points_surface, viz=False, joints=None)
    print("sample_points_nasa", sample_points_nasa.shape)
    

    ## Add noise to surface point
    points_sigma = 0.01
    sample_points_nasa += points_sigma * np.random.randn(n_points_surface, 3)

    # Save sampled points
    vertex_colors = get_colors(sample_points_nasa_labels)
    sample_points_nasa_mesh = trimesh.Trimesh(sample_points_nasa, faces=None, vertex_colors=vertex_colors, process=False)
    sample_points_nasa_out = os.path.join(args.out_folder, 'sample_points_nasa.obj')
    sample_points_nasa_mesh.export(sample_points_nasa_out)

    # Transform to canonical pose
    t_cano_nasa_vertices = posed_to_canonical(sample_points_nasa, nasa_input, sample_points_nasa_labels, inv=True)
    sample_points_nasa_mesh.vertices = t_cano_nasa_vertices
    sample_points_nasa_cano_out = os.path.join(args.out_folder, 'sample_points_nasa_cano.obj')
    sample_points_nasa_mesh.export(sample_points_nasa_cano_out)

    ## Save nasa template
    nasa_template_out = os.path.join(args.out_folder, 'nasa_cano_template.npz')
    print(type(t_trans_mat_inv))
    print(type(t_bone_lengths))
    print(type(t_hand_joints))
    shape=np.zeros([1, 10])
    np.savez(nasa_template_out, joints_trans=t_trans_mat_inv, verts=t_cano_nasa_vertices, vert_labels=sample_points_nasa_labels, 
             shape=np.zeros([1, 10]), bone_lengths=t_bone_lengths, hand_joints=t_hand_joints)

    # try scaling (should be according to bone lengths)
    scaled_cano_nasa_vertices = t_cano_nasa_vertices * 1.1

    # transform back to pose
    posed_nasa_vertices = canonical_to_posed(scaled_cano_nasa_vertices, nasa_input, sample_points_nasa_labels, inv=True)
    sample_points_nasa_mesh.vertices = posed_nasa_vertices
    sample_points_nasa_posed_out = os.path.join(args.out_folder, 'sample_points_nasa_posed.obj')
    sample_points_nasa_mesh.export(sample_points_nasa_posed_out)

    ## try loading the template points
    template_nasa_npz = "/home/korrawe/nasa/tmp_out/nasa_cano_template.npz"
    template_nasa = np.load(template_nasa_npz)

    t_bone_lengths = template_nasa['bone_lengths']
    print("template bone_lengths", t_bone_lengths)
    sample_points_nasa_labels = template_nasa['vert_labels']
    t_cano_nasa_vertices = template_nasa['verts']

    for i in range(10):
        # change shape
        shape = torch.rand(1, 10)
        pose_para = torch.rand(1, 45 + 3)
        # get MANO shape - 1
        mesh, hand_joints, new_trans_mat, bone_lengths = get_hand_mano(mano_layer, shape, pose_para, rest_pose=False)
        mano_out = os.path.join(args.out_folder, str(i) + '_mano.obj')
        mesh.vertices = mesh.vertices * 2.5
        mesh.export(mano_out)
        mesh.vertices = mesh.vertices * 0.4

        # Inverse the translation matrix for future use
        new_trans_mat_inv = np.linalg.inv(new_trans_mat)
        # get nasa input from MANO
        nasa_input = trans_mat2nasa_input(mesh, new_trans_mat_inv)
        data_in = {
            'inputs': torch.FloatTensor(nasa_input).unsqueeze(0),
            'bone_lengths': torch.from_numpy(bone_lengths).unsqueeze(0)
        }
        # get nasa shape - 2
        nasa_mesh_out, stat = nasa_generator.generate_mesh(data_in)
        nasa_out = os.path.join(args.out_folder, str(i) + '_nasa.obj')
        nasa_mesh_out.export(nasa_out)

        # get a base canonical nasa shape scaled according to bone lengths
        scale = bone_lengths[0] / t_bone_lengths[0]
        print("bone_length current", bone_lengths[0])
        print("bone_length template", t_bone_lengths[0])
        print("scale: %.4f" % scale)
        scaled_cano_nasa_vertices = t_cano_nasa_vertices * scale

        # transform the scaled canonical shape to posed shape with nasa input -3
        posed_nasa_vertices = canonical_to_posed(scaled_cano_nasa_vertices, nasa_input, sample_points_nasa_labels, inv=True)
        scaled_nasa_mesh = trimesh.Trimesh(posed_nasa_vertices, faces=None, vertex_colors=vertex_colors, process=False)
        scaled_nasa_out = os.path.join(args.out_folder, str(i) + '_scaled_nasa.obj')
        scaled_nasa_mesh.export(scaled_nasa_out)
        # compare 1 2 3

        # maybe compute chamfer dist?
        pass



    ## *** Verify if the mesh scaled using the bone lengths match the NASA output from changing MANO shape parameters or not (how close)
    ## *** Also, compare them to the ground truth.
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)