# import torch
import numpy as np
import trimesh

from manopth import demo
from visualize_utils import display_surface_points

from matplotlib import pyplot as plt


# def get_verts_association(skinning_weight_npy):
#     # skinning_weight_npy = 'resource/skinning_weight_r.npy'
#     data = np.load(skinning_weight_npy)
#     max_weight = data.argmax(1)
#     return max_weight

def get_face_labels(faces, vert_labels):
    face_labels = []
    for row in vert_labels[faces]:
        (values, counts)= np.unique(row, return_counts=True)
        ind = np.argmax(counts)
        face_labels.append(values[ind])

    face_labels = np.array(face_labels)
    return face_labels

def sample_surface_with_label(mesh, vert_labels, sample_n=6000, viz=False, joints=None):
    '''Sample surface points from the given mesh.
    The bone assosiation is choosen by majority vote between the three face vertices.
    In case of a tie, the lower number bone will be used.
    Args:
        mesh (trimesh.Trimesh): MANO mesh with sealed wrist
            vertices (779, 3): locations of vertices from MANO template.
            faces (1554, 3): MANO faces
        vert_labels (779): vertex labels according to the highest skinning weight.
        sample_n (int): a number of points to sample.
    '''

    face_labels = get_face_labels(mesh.faces, vert_labels)
    points, face_idx = trimesh.sample.sample_surface(mesh, sample_n)

    if viz:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(121, projection='3d')
        display_surface_points(mesh.vertices, vert_labels, ax=ax, show=False)
        if joints is not None:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black')
        ax = fig.add_subplot(122, projection='3d')
        # if joints is not None:
        #     ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black', s=200)
        display_surface_points(points, face_labels[face_idx], ax=ax, show=True)
        

    return points, face_labels[face_idx]
