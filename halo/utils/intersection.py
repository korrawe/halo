''' compute mesh to mesh intersection volume '''

import numpy as np
import argparse
import igl as igl
import trimesh
import torch
import json
import os


def uniform_box_sampling(min_corner, max_corner, res = 0.005):
    x_min = min_corner[0] - res
    x_max = max_corner[0] + res
    y_min = min_corner[1] - res
    y_max = max_corner[1] + res
    z_min = min_corner[2] - res
    z_max = max_corner[2] + res

    h = int((x_max-x_min)/res)+1
    l = int((y_max-y_min)/res)+1
    w = int((z_max-z_min)/res)+1

    # print('Sampling size: %d x %d x %d'%(h, l, w))

    with torch.no_grad():
        xyz = x = torch.zeros(h, l, w, 3, dtype=torch.float32) + torch.tensor([x_min, y_min, z_min], dtype=torch.float32)
        for i in range(1,h):
            xyz[i,0,0] = xyz[i-1,0,0] + torch.tensor([res,0,0])
        for i in range(1,l):
            xyz[:,i,0] = xyz[:,i-1,0] + torch.tensor([0,res,0])
        for i in range(1,w):
            xyz[:,:,i] = xyz[:,:,i-1] + torch.tensor([0,0,res])
    return res, xyz


def bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1):
    min_x = max(min_corner0[0], min_corner1[0])
    min_y = max(min_corner0[1], min_corner1[1])
    min_z = max(min_corner0[2], min_corner1[2])

    max_x = min(max_corner0[0], max_corner1[0])
    max_y = min(max_corner0[1], max_corner1[1])
    max_z = min(max_corner0[2], max_corner1[2])

    if max_x > min_x and max_y > min_y and max_z > min_z:
        # print('Intersected bounding box size: %f x %f x %f'%(max_x - min_x, max_y - min_y, max_z - min_z))
        return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])
    else:
        return np.zeros((1,3), dtype = np.float32), np.zeros((1,3), dtype = np.float32)


def writeOff(output, vertex, face):
    with open(output, 'w') as f:
        f.write("COFF\n")
        f.write("%d %d 0\n" %(vertex.shape[0], face.shape[0]))
        for row in range(0, vertex.shape[0]):
            f.write("%f %f %f\n" %(vertex[row, 0], vertex[row, 1], vertex[row, 2]))
        for row in range(0, face.shape[0]):
            f.write("3 %d %d %d\n" %(face[row, 0], face[row, 1], face[row, 2]))


def intersection_eval(mesh0, mesh1, res=0.005, scale=1., trans=None, visualize_flag=False, visualize_file='output.off'):
    '''Calculate intersection depth and volumn of the two inputs meshes.
    args:
        mesh1, mesh2 (Trimesh.trimesh): input meshes
        res (float): voxel resolustion in meter(m)
        scale (float): scaling factor
        trans (float) (1, 3): translation
    returns:
        volume (float): intersection volume in cm^3
        mesh_mesh_dist (float): maximum depth from the center of voxel to the surface of another mesh
    '''
    # mesh0 = trimesh.load(mesh_file_0, process=False)
    # mesh1 = trimesh.load(mesh_file_1, process=False)

    # scale = 1 # 10
    # res = 0.5
    mesh0.vertices = mesh0.vertices * scale
    mesh1.vertices = mesh1.vertices * scale

    S, I, C = igl.signed_distance(mesh0.vertices + 1e-10, mesh1.vertices, mesh1.faces, return_normals=False)

    mesh_mesh_distance = S.min()
    # print("dist", S)
    # print("Mesh to mesh distance: %f cm" % mesh_mesh_distance)

    #### print("Mesh to mesh distance: %f" % (max(S.min(), 0)))

    if mesh_mesh_distance > 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance

    # Get bounding box for each mesh:
    min_corner0 = np.array([mesh0.vertices[:,0].min(), mesh0.vertices[:,1].min(), mesh0.vertices[:,2].min()])
    max_corner0 = np.array([mesh0.vertices[:,0].max(), mesh0.vertices[:,1].max(), mesh0.vertices[:,2].max()])

    min_corner1 = np.array([mesh1.vertices[:,0].min(), mesh1.vertices[:,1].min(), mesh1.vertices[:,2].min()])
    max_corner1 = np.array([mesh1.vertices[:,0].max(), mesh1.vertices[:,1].max(), mesh1.vertices[:,2].max()])

    # Compute the intersection of two bounding boxes:
    min_corner_i, max_corner_i = bounding_box_intersection(min_corner0, max_corner0, min_corner1, max_corner1)
    if ((min_corner_i - max_corner_i)**2).sum() == 0:
        # print('No intersection!')
        return 0, mesh_mesh_distance

    # Uniformly sample the intersection bounding box:
    _, xyz = uniform_box_sampling(min_corner_i, max_corner_i, res)
    xyz = xyz.view(-1, 3)
    xyz = xyz.detach().cpu().numpy()

    S, I, C = igl.signed_distance(xyz, mesh0.vertices, mesh0.faces, return_normals=False)

    inside_sample_index = np.argwhere(S < 0.0)
    # print("inside sample index", inside_sample_index, len(inside_sample_index))

    # Compute the signed distance for inside_samples to mesh 1:
    inside_samples = xyz[inside_sample_index[:,0], :]

    S, I, C = igl.signed_distance(inside_samples, mesh1.vertices, mesh1.faces, return_normals=False)

    inside_both_sample_index = np.argwhere(S < 0)

    # Compute intersection volume:
    i_v = inside_both_sample_index.shape[0] * (res**3)
    # print("Intersected volume: %f cm^3" % (i_v))

    # Visualize intersection volume:
    if visualize_flag:
        writeOff(visualize_file, inside_samples[inside_both_sample_index[:,0], :], np.zeros((0,3)))

    # From (m) to (cm)
    return i_v * 1e6, mesh_mesh_distance * 1e2

def evaluate():
    pass


if __name__ == "__main__":
    i_v, mesh_mesh_distance = intersection_eval("mesh0.off", "mesh1.off", res=0.001, visualize_flag=True)
    print("volume", i_v)
    print("mesh_mesh_distance", mesh_mesh_distance)
