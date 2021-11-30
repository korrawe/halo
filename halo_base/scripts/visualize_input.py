import argparse
import trimesh
import numpy as np
import os
import glob
import sys
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser('Visualize mesh.')
parser.add_argument('input_path', type=str,
                    help='Path to input watertight meshes. (./points)')


bones = np.array([
    (0,1),
    (1,2),
    (2,3),
    (0,4),
    (4,5),
    (5,6),
    (0,7),
    (7,8),
    (8,9),
    (0,10),
    (10,11),
    (11,12),
    (0,13),
    (13,14),
    (14,15),
])

def main(args):
    # args.input = 
    if args.input_path[-4:] == '.npz':
        print(args.input_path)
        process_path(args.input_path, args)
        return

    input_files = glob.glob(os.path.join(args.input_path, '*.npz'))
    max_display = 10
    i = 0
    for p in input_files:
        print(p)
        process_path(p, args)
        i += 1
        if i >= max_display:
            break
        

def set_equal_xyz_scale(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    return ax


def append_1_expand_dims(verts):
    verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    verts = np.expand_dims(verts, axis=2)
    return verts


def display_points(points, occupancies, joints_trans=None, verts=None, vert_labels=None, joints=None, loc=None,
                   inv_trans=False, bone_idx=0, viz_inside=True, viz_rot=True, ax=None, show=True):
    """
    Displays points with occupancies
    """
    if ax is None:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(121, projection='3d')

    sample_idx = np.random.choice(len(points), 5000)
    points = points[sample_idx]
    occupancies = occupancies[sample_idx]

    canonical_pose = True
    if canonical_pose:
        bone_mat_idx = 13
        # print('point shape', points.shape)
        # points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
        # points = np.expand_dims(points, axis=2)
        # points = np.matmul(joints_trans[bone_mat_idx], points)

    inside_points = points[occupancies == 1]
    outside_points = points[occupancies == 0]
    viz_side = 'both'
    if viz_side == 'in':
        ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], color='g', s=1)
        ax = set_equal_xyz_scale(ax, inside_points[:, 0], inside_points[:, 1], inside_points[:, 2])
    elif viz_side == 'out':
        ax.scatter(outside_points[:, 0], outside_points[:, 1], outside_points[:, 2], color='r', s=0.1)
        ax = set_equal_xyz_scale(ax, outside_points[:, 0], outside_points[:, 1], outside_points[:, 2])
    else:
        ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], color='g', s=1)
        ax.scatter(outside_points[:, 0], outside_points[:, 1], outside_points[:, 2], color='r', s=0.1)
        ax = set_equal_xyz_scale(ax, points[:, 0], points[:, 1], points[:, 2])
    
    if loc is not None:
        ax.scatter(loc[0], loc[1], loc[2], color='black', s=20)

    if joints is not None:
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black', s=50)

    inv_trans = False # True
    if inv_trans:
        print("before", joints_trans[:3])
        joints_trans = np.linalg.inv(joints_trans)
        # print("after", joints_trans)

    # Bones
    if joints_trans is not None:
        # print("joints_trans", joints_trans[:3])
        # print("joints_trans shape", joints_trans.shape)
        # scale_mat = np.identity(4) / 2.5 # 2.5 
        # scale_mat[3,3] = 1.
        # joints_trans = np.matmul(joints_trans, scale_mat)

        print("joints_trans", joints_trans[:3])
        # ax.scatter(joints_trans[:, 0, 3], joints_trans[:, 1, 3], joints_trans[:, 2, 3], color='black')
        # if viz_rot:
        #     dx = np.array([0.005, 0., 0., 1])
        #     dy = np.array([0., 0.005, 0., 1])
        #     dz = np.array([0., 0., 0.005, 1])
        #     joints_dx = np.matmul(joints_trans, dx)
        #     joints_dy = np.matmul(joints_trans, dy)
        #     joints_dz = np.matmul(joints_trans, dz)
        #     color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        #     for idx, joints_d in enumerate((joints_dx, joints_dy, joints_dz)):
        #         for b in range(len(joints_trans)):
        #             ax.plot([joints_trans[b, 0, 3], joints_d[b, 0]], 
        #                     [joints_trans[b, 1, 3], joints_d[b, 1]], 
        #                     [joints_trans[b, 2, 3], joints_d[b, 2]], color=color[idx])
    
    # # Back-prejected bones
    # if joints_trans is not None:
    #     print("joints_trans", joints_trans.shape)
    #     pass

    # Surface vertices
    if verts is not None:
        ax = fig.add_subplot(122, projection='3d')

        if canonical_pose:
            if joints is not None:
                # print("joints", joints)
                # print("joints", joints.shape)
                # print('joints_trans', joints_trans.shape)
                cano_joints = np.concatenate([joints[:16], np.ones([16, 1])], axis=1)
                cano_joints = np.expand_dims(cano_joints, axis=2)
                cano_joints = np.matmul(joints_trans, cano_joints)
                # print('cano_joints', cano_joints.shape)

                # bone
                b_start = bones[:,0]
                b_end = bones[:,1]
                print('b_start', b_start)
                print('b_end', b_end)
                b_start_loc = joints[b_start]
                b_end_loc = joints[b_end]
                print('b_start_loc', b_start_loc.shape)
                b_start_loc = append_1_expand_dims(b_start_loc)
                b_end_loc = append_1_expand_dims(b_end_loc)
                print('b_start_loc', b_start_loc.shape)
                b_start_loc = np.matmul(joints_trans[b_start], b_start_loc)
                b_end_loc = np.matmul(joints_trans[b_end], b_end_loc)
                print('b_start_loc', b_start_loc.shape)
                print('b_end_loc', b_end_loc.shape)
                for b in range(15):
                    ax.plot([b_start_loc[b, 0, 0], b_end_loc[b, 0, 0]], 
                            [b_start_loc[b, 1, 0], b_end_loc[b, 1, 0]], 
                            [b_start_loc[b, 2, 0], b_end_loc[b, 2, 0]], color='#1f77b4')
        
                
                ax.scatter(b_start_loc[:, 0], b_start_loc[:, 1], b_start_loc[:, 2], color='r', s=10)
                ax.scatter(b_end_loc[:, 0], b_end_loc[:, 1], b_end_loc[:, 2], color='g', s=10)

            vert_trans = joints_trans[vert_labels]
            verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
            verts = np.expand_dims(verts, axis=2)
            verts = np.matmul(vert_trans, verts)
            print('vert', verts.shape)
            # ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='b', alpha=0.5, s=1)

        else:
            pass
            # ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='b', alpha=0.5, s=0.5)

        # set equal xyz scale
        X, Y, Z = verts[:, 0], verts[:, 1], verts[:, 2]
        ax = set_equal_xyz_scale(ax, X, Y, Z)

        if vert_labels is not None:
            for bone_idx in range(16):
                selected_verts = vert_labels == bone_idx
                if bone_idx == 14:
                    ax.scatter(verts[selected_verts, 0], verts[selected_verts, 1], verts[selected_verts, 2],
                                s=5.0, alpha=0.2, color='black')
                else:
                    ax.scatter(verts[selected_verts, 0], verts[selected_verts, 1], verts[selected_verts, 2],
                                s=5.0, alpha=0.2) # color='g'
                
                if joints is not None:
                    ax.scatter(cano_joints[bone_idx, 0], cano_joints[bone_idx, 1], 
                    cano_joints[bone_idx, 2], color='black', s=30.0)


    # cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()

def process_path(in_path, args):

    data = np.load(in_path)
    # print(data.files)
    # ['points', 'occupancies', 'loc', 'scale']
    points, occupancies, loc, scale = data['points'], data['occupancies'], data['loc'], data['scale']
    joints_trans, verts, vert_labels = data['joints_trans'], data['verts'], data['vert_labels']

    if 'hand_joints' in data.keys():
        joints = data['hand_joints']
        print("joints in data keys")
    else:
        joints = None

    # print(points.shape)
    # print(occupancies.shape)
    # print(loc)
    # print(scale)
    # print("joints_trans", joints_trans.shape)
    # print("verts", verts.shape)
    # print("vert_labels", vert_labels.shape)

    occupancies = np.unpackbits(occupancies)
    # print(occupancies.shape)
    # print(occupancies)

    display_points(points, occupancies, joints_trans, verts, vert_labels, joints, loc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)