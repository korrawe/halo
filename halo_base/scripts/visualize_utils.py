import argparse
import trimesh
import numpy as np
import os
import glob
import sys
from matplotlib import pyplot as plt


def set_equal_xyz_scale(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    return ax


def display_surface_points(points, points_labels=None, bones=None, faces=None, ax=None, show=True):
    """
    Displays surface points with bone labels
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Surface vertices
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b', alpha=0.5, s=0.5)

    if points_labels is not None:
        for bone_idx in range(16):
            selected_points = points_labels == bone_idx
            ax.scatter(points[selected_points, 0], points[selected_points, 1], points[selected_points, 2])
                        # color='g', s=5.0)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # ax.scatter(joints_trans[bone_idx, 0, 3], joints_trans[bone_idx, 1, 3], 
    #                 joints_trans[bone_idx, 2, 3], color='g', s=70.0)

    set_equal_xyz_scale(ax, points[:, 0], points[:, 1], points[:, 2])

    if show:
        plt.show()


def display_vertices_with_joints(joints, vertices, points_labels=None, ax=None, show=True):
    """
    Displays surface points with bone labels
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Surface vertices
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='b', s=20)

    if points_labels is not None:
        for bone_idx in range(16):
            selected_points = points_labels == bone_idx
            ax.scatter(vertices[selected_points, 0], vertices[selected_points, 1], vertices[selected_points, 2])
                        # color='g', s=5.0)
    # ax.scatter(joints_trans[bone_idx, 0, 3], joints_trans[bone_idx, 1, 3], 
    #                 joints_trans[bone_idx, 2, 3], color='g', s=70.0)
    
    if show:
        plt.show()


def compare_joints(joints_1, joints_2, bone_parent):
    """
    Compare two skeleton. First - blue, second - red
    """
    # if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Surface vertices
    ax.scatter(joints_1[:, 0], joints_1[:, 1], joints_1[:, 2], color='b', s=20)
    ax.scatter(joints_2[:, 0], joints_2[:, 1], joints_2[:, 2], color='r', s=20)

    b_start_loc = joints_1[bone_parent]
    b_end_loc = joints_1
    for b in range(21):
        ax.plot([b_start_loc[b, 0], b_end_loc[b, 0]],
                [b_start_loc[b, 1], b_end_loc[b, 1]],
                [b_start_loc[b, 2], b_end_loc[b, 2]], color='b')

    b_start_loc = joints_2[bone_parent]
    b_end_loc = joints_2
    for b in range(21):
        ax.plot([b_start_loc[b, 0], b_end_loc[b, 0]],
                [b_start_loc[b, 1], b_end_loc[b, 1]],
                [b_start_loc[b, 2], b_end_loc[b, 2]], color='r')

    plt.show()


def display_bones(joints, bones=None, ax=None, show=True):
    """
    Displays surface points with bone labels
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Surface vertices
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='b', alpha=0.5, s=5)
    set_equal_xyz_scale(ax, joints[:, 0], joints[:, 1], joints[:, 2])

    idxx = 16
    ax.scatter(joints[idxx, 0], joints[idxx, 1], joints[idxx, 2], color='red', alpha=1, s=20)

    # bone
    b_start = bones[:,0]
    b_end = bones[:,1]
    print('b_start', b_start)
    print('b_end', b_end)
    b_start_loc = joints[b_start]
    b_end_loc = joints[b_end]
    for b in range(16):
        ax.plot([b_start_loc[b, 0], b_end_loc[b, 0]], 
                [b_start_loc[b, 1], b_end_loc[b, 1]], 
                [b_start_loc[b, 2], b_end_loc[b, 2]], color='#1f77b4')

    if show:
        plt.show()


def display_iou(points, intersect, union, ax=None, show=True):
    """
    Displays intercestion over union
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Surface vertices
    intersect_points = points[intersect == 1]
    union_points = points[union == 1]
    error_points = points[np.logical_and(union == 1, intersect == 0)]

    # ax.scatter(intersect_points[:, 0], intersect_points[:, 1], intersect_points[:, 2], color='b', s=2)
    ax.scatter(error_points[:, 0], error_points[:, 1], error_points[:, 2], color='r', alpha=0.5, s=2)

    set_equal_xyz_scale(ax, union_points[:, 0], union_points[:, 1], union_points[:, 2])

    if show:
        plt.show()
