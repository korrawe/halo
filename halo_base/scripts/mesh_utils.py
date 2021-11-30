import torch
import numpy as np
import trimesh

# from manopth import demo
# from visualize_utils import display_surface_points

# from matplotlib import pyplot as plt


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


def get_verts_association(skinning_weight_npy, add_wrist=False):
    # skinning_weight_npy = 'resource/skinning_weight_r.npy'
    data = np.load(skinning_weight_npy)
    verts_joints_assoc = data.argmax(1)

    if add_wrist:
        # Add wrist label to the new vertex
        verts_joints_assoc = np.concatenate([verts_joints_assoc, [0]])
    return verts_joints_assoc


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