from os.path import join, isfile, isdir
import os
import numpy as np

from tqdm import tqdm
import pyrender
import trimesh
from rendering import GraspRender

from scipy.spatial.transform import Rotation as R
import cv2

# import sys
# sys.path.insert(0, "/home/korrawe/halo_vae/scripts")

SKIN_COLOR = np.array([[187, 109, 74]])
OBJ_COLOR = np.array([[20, 120, 240]])

# ho3d_dir = '/ps/scratch/ps_shared/kkarunratanakul/neurips/baseline_obman_ho3d/ho3d_train_GT'
# ho3d_dir = '/ps/scratch/ps_shared/kkarunratanakul/fix_ho3d/gt'
# save_dir = '/ps/scratch/ps_shared/jyang/ho3d_render'
# save_dir = '/ps/scratch/ps_shared/jyang/ho3d_fixed_render'
# os.makedirs(save_dir, exist_ok=True)

# data_dir = ho3d_dir
# hand_files = sorted([s for s in os.listdir(data_dir) if "_hand" in s])

# save_gif_folder = join(save_dir, 'gif')
# save_obj_folder = join(save_dir, 'obj')

grasp_render = GraspRender()

# HALO
# mesh_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation/mano/'
# save_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation/render/'
mesh_dir = '/home/korrawe/halo_vae/exp/grab_baseline_3/generation/mano/'
save_dir = '/home/korrawe/halo_vae/exp/grab_baseline_3/generation/render/'
# GrabNet
# mesh_dir = '/home/korrawe/halo_vae/dataset/GrabNet/tests/grab_new_objects/'
# save_dir = '/home/korrawe/halo_vae/dataset/GrabNet/tests/grab_new_objects_render/'

data_dir = mesh_dir

object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
for obj_idx, object_type in enumerate(object_list):

    # HALO
    obj_mesh_filename = os.path.join(mesh_dir, object_type + '.ply')
    obj_mesh = trimesh.load(join(data_dir, obj_mesh_filename), process=False)
    object_vertices = obj_mesh.vertices
    # GrabNet
    # obj_dir = os.path.join(mesh_dir, object_type)

    n_sample = 20
    for idx in range(n_sample):
        # HALO
        hand_mesh_filename = os.path.join(mesh_dir, '%s_%03d.ply' % (object_type, idx))
        # hand_mesh_filename = os.path.join(mesh_dir, '%s_%03d_refine.ply' % (object_type, idx))
        hand_mesh = trimesh.load(join(data_dir, hand_mesh_filename), process=False)

        # GrabNet
        # hand_mesh_filename = os.path.join(obj_dir, 'rh_mesh_gen_coarse_%s.ply' % idx)
        # hand_mesh_filename = os.path.join(obj_dir, 'rh_mesh_gen_%s.ply' % idx)
        # hand_mesh = trimesh.load(join(data_dir, hand_mesh_filename), process=False)
        # obj_mesh_filename = os.path.join(obj_dir, 'obj_mesh_%s.ply' % idx)
        # obj_mesh = trimesh.load(join(data_dir, obj_mesh_filename), process=False)

        center = 0.5 * (hand_mesh.vertices.mean(0) + obj_mesh.vertices.mean(0))
        hand_mesh.vertices -= center
        # HALO
        obj_mesh.vertices = object_vertices - center
        # GrabNet
        # obj_mesh.vertices -= center

        hand_mesh.visual.vertex_colors = np.zeros_like(hand_mesh.vertices) + SKIN_COLOR
        obj_mesh.visual.vertex_colors = np.zeros_like(obj_mesh.vertices) + OBJ_COLOR

        steps = 4  # 6
        for i in range(steps):
            fix_r = R.from_rotvec(np.array([0, 0, np.pi / 4.]))
            fix_rmat = fix_r.as_dcm()
            hand_mesh.vertices = hand_mesh.vertices.dot(fix_rmat)
            obj_mesh.vertices = obj_mesh.vertices.dot(fix_rmat)

            # r = R.from_rotvec(np.pi*2/steps * np.array([0, 1, 0]))
            r = R.from_rotvec(np.pi*2/steps * np.array([0, 1, 0]))
            rmat = r.as_dcm()

            hand_mesh.vertices = hand_mesh.vertices.dot(rmat)
            obj_mesh.vertices = obj_mesh.vertices.dot(rmat)

            grasp_render.add_geometry([hand_mesh, obj_mesh])
            image = grasp_render.render()
            grasp_render.clear_geometry()

            # sample_name = hand_file.replace('_hand.obj', '')
            # os.makedirs(join(save_dir, sample_name), exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)
            sample_name = '%s_%03d' % (object_type, idx)
            # sample_name = '%s_%03d_refine' % (object_type, idx)
            os.makedirs(join(save_dir, sample_name), exist_ok=True)
            output_image_file = join(save_dir, sample_name, str(i) + '.png')
            cv2.imwrite(output_image_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # import pdb; pdb.set_trace()


# for index, hand_file in enumerate(tqdm(hand_files)):
#     # if index < 64:
#     #     continue

#     hand_mesh = trimesh.load(join(data_dir, hand_file), process=False)
#     obj_file = join(data_dir, hand_file.replace('_hand', '_obj'))
#     if not isfile(obj_file):
#         continue
#     obj_mesh = trimesh.load(join(data_dir, obj_file), process=False)

#     center = 0.5*(hand_mesh.vertices.mean(0) + obj_mesh.vertices.mean(0))
#     # center = obj_mesh.vertices.mean(0)
#     # grasp_volume = np.vstack((hand_mesh.vertices, obj_mesh.vertices))
#     # center = grasp_volume.mean(0)
#     # center = 0.5*(grasp_volume.max(0)+grasp_volume.min(0))

#     hand_mesh.vertices -= center
#     obj_mesh.vertices -= center

#     hand_mesh.visual.vertex_colors = np.zeros_like(hand_mesh.vertices) + SKIN_COLOR
#     obj_mesh.visual.vertex_colors = np.zeros_like(obj_mesh.vertices) + OBJ_COLOR

#     steps = 6
#     for i in range(steps):
#         r = R.from_rotvec(np.pi*2/steps * np.array([0, 1, 0]))
#         rmat = r.as_dcm()

#         hand_mesh.vertices = hand_mesh.vertices.dot(rmat)
#         obj_mesh.vertices = obj_mesh.vertices.dot(rmat)

#         grasp_render.add_geometry([hand_mesh, obj_mesh])
#         image = grasp_render.render()
#         grasp_render.clear_geometry()

#         sample_name = hand_file.replace('_hand.obj', '')
#         os.makedirs(join(save_dir, sample_name), exist_ok=True)
#         output_image_file = join(save_dir, sample_name, str(i) + '.png')
#         cv2.imwrite(output_image_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
