import os
import numpy as np
import torch
import trimesh
import scipy.cluster
from scipy.stats import entropy

from utils.intersection import intersection_eval
import matplotlib.pyplot as plt

from models.utils import visualize as vis

from models.halo_adapter.converter import PoseConverter, transform_to_canonical
from models.halo_adapter.interface import convert_joints


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


def eval_grabnet():
    # For GrabNet
    # object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
    datalist_file = "../data/obman_test/datalist.txt"
    object_list = []
    with open(datalist_file, 'r') as f:
        for line in f:
            object_list.append(line.strip()[:-4])

    kps_all_list = []

    grab_to_mano = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

    # GrabNet model results
    mesh_dir = '..vae_out/tests/grab_new_objects/'
    for object_type in object_list:
        print()
        obj_dir = os.path.join(mesh_dir, object_type)
        kps_list = []
        n_sample = 5  # for Obman # 10  for ho3d  # 20 for GRAB
        for idx in range(n_sample):
            # Refine
            # hand_mesh_filename = os.path.join(obj_dir, 'rh_mesh_gen_%s.ply' % idx)
            # hand_kps_filename = os.path.join(obj_dir, 'j_rh_mesh_gen_%s.npy' % idx)
            # Coarse
            # hand_mesh_filename = os.path.join(obj_dir, 'rh_mesh_gen_coarse_%s.ply' % idx)
            hand_kps_filename = os.path.join(obj_dir, 'j_rh_mesh_gen_coarse_%s.npy' % idx)

            print(hand_kps_filename)
            hand_kps = np.load(hand_kps_filename)
            hand_kps = hand_kps[grab_to_mano]
            hand_kps = hand_kps * 100.0
            hand_kps_before = hand_kps

            hand_kps = torch.from_numpy(hand_kps).unsqueeze(0)
            is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)

            hand_kps = convert_joints(hand_kps, source='mano', target='biomech')

            hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
            hand_kps_after = convert_joints(hand_kps_after, source='biomech', target='mano')
            hand_kps = hand_kps_after.squeeze(0).numpy()


            kps_flat = hand_kps.reshape(-1)
            kps_list.append(hand_kps)
            kps_all_list.append(kps_flat)

        print(" -- ", object_type, " -- ")

    # import pdb; pdb.set_trace()
    kps_all = np.stack(kps_all_list, axis=0)  # [120, 63]
    cls_num = 20
    entropy, dist = diversity(kps_all, cls_num=cls_num)
    print(" entropy :", entropy)
    print(" dist :", dist)


def eval_halo():
    # For HALO
    # object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
    datalist_file = "../data/ho3d/datalist.txt"
    object_list = []
    with open(datalist_file, 'r') as f:
        for line in f:
            object_list.append(line.strip()[:-4])

    kps_all_list = []

    # GrabNet model results
    kps_dir = '../exp/grab_refine_inter_2/generation_ho3d/kps/'
    for obj_idx, object_type in enumerate(object_list):

        kps_list = []
        mesh_dist_list = []
        n_sample = 10  # 5  # 20
        for idx in range(n_sample):
            # hand_mesh_filename = os.path.join(obj_dir, 'obj%03d_h%03d.obj' % (obj_idx, idx))
            # hand_kps_filename = os.path.join(kps_dir, '%s_%03d_refine.npy' % (object_type, idx))
            hand_kps_filename = os.path.join(kps_dir, '%s_%03d.npy' % (object_type, idx))

            print(hand_kps_filename)

            hand_kps = np.load(hand_kps_filename)
            hand_kps_before = hand_kps
            hand_kps = torch.from_numpy(hand_kps).unsqueeze(0)
            is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)

            hand_kps = convert_joints(hand_kps, source='mano', target='biomech')

            hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
            hand_kps_after = convert_joints(hand_kps_after, source='biomech', target='mano')
            hand_kps = hand_kps_after.squeeze(0).numpy()

            kps_flat = hand_kps.reshape(-1)
            kps_list.append(hand_kps)
            kps_all_list.append(kps_flat)

        print(" -- ", object_type, " -- ")

    # import pdb; pdb.set_trace()
    kps_all = np.stack(kps_all_list, axis=0)  # [120, 63]
    cls_num = 20
    entropy, dist = diversity(kps_all, cls_num=cls_num)
    print(" entropy :", entropy)
    print(" dist :", dist)


def eval_halo_mano():
    # For HALO
    object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
    kps_all_list = []

    kps_dir = '../exp/grab_baseline_mano/generation/kps/'
    for obj_idx, object_type in enumerate(object_list):

        kps_list = []
        mesh_dist_list = []
        n_sample = 20
        for idx in range(n_sample):
            # hand_mesh_filename = os.path.join(obj_dir, 'obj%03d_h%03d.obj' % (obj_idx, idx))
            hand_kps_filename = os.path.join(kps_dir, '%s_%03d.npy' % (object_type, idx))
            # hand_kps_filename = os.path.join(kps_dir, '%s_%03d_refine.npy' % (object_type, idx))

            print(hand_kps_filename)

            hand_kps = np.load(hand_kps_filename)
            hand_kps_before = hand_kps
            hand_kps = torch.from_numpy(hand_kps).unsqueeze(0)
            is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)

            hand_kps = convert_joints(hand_kps, source='mano', target='biomech')

            hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
            hand_kps_after = convert_joints(hand_kps_after, source='biomech', target='mano')
            hand_kps = hand_kps_after.squeeze(0).numpy()

            kps_flat = hand_kps.reshape(-1)
            kps_list.append(hand_kps)
            kps_all_list.append(kps_flat)

        print(" -- ", object_type, " -- ")

    kps_all = np.stack(kps_all_list, axis=0)  # [120, 63]
    cls_num = 20
    entropy, dist = diversity(kps_all, cls_num=cls_num)
    print(" entropy :", entropy)
    print(" dist :", dist)


def eval_gf():
    # For HALO
    # object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']
    datalist_file = "../data/ho3d/datalist.txt"
    object_list = []
    with open(datalist_file, 'r') as f:
        for line in f:
            object_list.append(line.strip()[:-4])

    kps_all_list = []

    kps_dir = '../grasping_field/output/kps/'
    for obj_idx, object_type in enumerate(object_list):

        kps_list = []
        mesh_dist_list = []
        n_sample = 10  # 5
        for idx in range(n_sample):
            # import pdb; pdb.set_trace()
            # hand_mesh_filename = os.path.join(obj_dir, 'obj%03d_h%03d.obj' % (obj_idx, idx))
            hand_kps_filename = os.path.join(kps_dir, '%s_%d_hand_mano.npy' % (object_type, idx))
            # hand_kps_filename = os.path.join(kps_dir, '%s_%03d_refine.npy' % (object_type, idx))

            print(hand_kps_filename)

            hand_kps = np.load(hand_kps_filename)
            hand_kps = hand_kps * 100.0
            hand_kps_before = hand_kps
            hand_kps = torch.from_numpy(hand_kps).unsqueeze(0)
            is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)

            hand_kps = convert_joints(hand_kps, source='mano', target='biomech')

            hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
            hand_kps_after = convert_joints(hand_kps_after, source='biomech', target='mano')
            hand_kps = hand_kps_after.squeeze(0).numpy()

            kps_flat = hand_kps.reshape(-1)
            kps_list.append(hand_kps)
            kps_all_list.append(kps_flat)

        print(" -- ", object_type, " -- ")

    kps_all = np.stack(kps_all_list, axis=0)  # [120, 63]
    cls_num = 20
    entropy, dist = diversity(kps_all, cls_num=cls_num)
    print(" entropy :", entropy)
    print(" dist :", dist)


def diversity(params_list, cls_num=20):
    # params_list = scipy.cluster.vq.whiten(params_list)
    #  # k-means
    codes, dist = scipy.cluster.vq.kmeans(params_list, cls_num)  # codes: [20, 72], dist: scalar
    vecs, dist = scipy.cluster.vq.vq(params_list, codes)  # assign codes, vecs/dist: [1200]
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences  count: [20]
    ee = entropy(counts)
    return ee, np.mean(dist)


def main():
    eval_grabnet()
    # eval_halo()
    # eval_halo_mano()
    # eval_gf()


if __name__ == "__main__":
    main()
