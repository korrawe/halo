import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import trimesh

from manopth.manolayer import ManoLayer
from manopth import demo

sys.path.insert(0, "/home/korrawe/halo_vae")
from models.halo_adapter.converter import PoseConverter, transform_to_canonical
from models.halo_adapter.interface import convert_joints

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit_mano(gen_keypoint, kps_scale=1.0):
    batch_size = 1024  # 512  # 32
    w_pose = 5.0  # 10.0
    w_shape = 5.0  #10.0
    ncomps = 45
    epoch_coarse = 500
    epoch_fine = 2500 # 2500  # 1000

    out_params = {
        'gen_joints': [],
        'joints': [],
        'verts': [],
        'rot': [],
        'pose': [],
        'shape': [],
        'trans': []
    }

    data_size = len(gen_keypoint)
    # gen_keypoint = np.stack(gen_keypoint, 0)

    # from cm to mm
    gen_keypoint = gen_keypoint * kps_scale

    target_skeleton = np.array([[0, 1, 2, 3, 4],
                                [0, 5, 6, 7, 8],
                                [0, 9,10,11,12],
                                [0,13,14,15,16],
                                [0,17,18,19,20]])

    # Initialize MANO layer
    mano_layer = ManoLayer(
        mano_root='mano/models', center_idx=0, use_pca=True, ncomps=ncomps, flat_hand_mean=False)
    mano_layer = mano_layer.to(device)

    for frameId in tqdm(range(0, data_size, batch_size)):
        current_batch_size = min(batch_size, data_size - frameId)

        # target_js = torch.from_numpy(gen_keypoint[frameId: frameId + current_batch_size, :]).float().to(device)
        target_js = gen_keypoint[frameId: frameId + current_batch_size, :].float().to(device)

        # Root center
        # target_js = target_js[:, :] - target_js[:, None, 0]
        # # Use HALO adapter to normalize middle root bone
        # target_js = convert_joints(target_js, source='mano', target='biomech')
        # target_js, unused_mat = transform_to_canonical(target_js, torch.ones(target_js.shape[0], device=device))
        # target_js = convert_joints(target_js, source='biomech', target='mano')

        # Model para initialization:
        shape = torch.randn(current_batch_size, 10).to(device)
        shape.requires_grad_()
        rot = torch.rand(current_batch_size, 3).to(device)
        rot.requires_grad_()
        pose = torch.randn(current_batch_size, ncomps).to(device)
        pose.requires_grad_()
        trans = (target_js[:, 0]).to(device) / 1000.0
        trans.requires_grad_()
        # print(trans)

        # import pdb; pdb.set_trace()

        visualize_interm_result = False  # True
        if visualize_interm_result:
            print('Before pose shape optimization.')
            hand_verts, hand_joints = mano_layer(torch.cat((rot, pose), 1), shape, trans)
            demo.display_mosh(target_js.detach().cpu(),
                                target_skeleton,
                                {'verts': hand_verts.detach().cpu(),
                                'joints': hand_joints.detach().cpu()}, \
                                mano_faces=mano_layer.th_faces.detach().cpu(), \
                                alpha = 0.3)

        criteria_loss = nn.MSELoss().to(device)

        # Optimize for global rotation and translation
        optimizer = torch.optim.Adam([rot, trans], lr=1e-2) 
        for i in range(0, epoch_coarse):
            _, hand_joints = mano_layer(torch.cat((rot, pose), 1), shape, trans)
            loss = criteria_loss(hand_joints, target_js)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('After coarse alignment: %6f' % (loss.data))

        if visualize_interm_result:
            print('After optimizing global translation and orientation.')
            hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)
            demo.display_mosh(target_js.detach().cpu(),
                                target_skeleton,
                                {'verts': hand_verts.detach().cpu(),
                                'joints': hand_joints.detach().cpu()}, \
                                mano_faces=mano_layer.th_faces.detach().cpu(), \
                                alpha = 0.3)

        # Local optimization
        optimizer = torch.optim.Adam([rot, trans, pose, shape], lr=1e-2)
        for i in range(0, epoch_fine):
            hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)
            loss = criteria_loss(hand_joints, target_js) + w_shape*(shape*shape).mean() + w_pose*(pose*pose).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
        print('After fine alignment: %6f'%(loss.data))

        if visualize_interm_result:
            print('After optimizing pose, shape, global translation and orientation.')
            hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)
            demo.display_mosh(target_js.detach().cpu(),\
                                target_skeleton,
                                {'verts': hand_verts.detach().cpu(),
                                'joints': hand_joints.detach().cpu()}, \
                                mano_faces=mano_layer.th_faces.detach().cpu(), \
                                alpha = 0.1)

        # import pdb; pdb.set_trace()
        hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)

        out_params['gen_joints'].append(target_js.detach().cpu().numpy())
        out_params['joints'].append(hand_joints.detach().cpu().numpy())
        out_params['verts'].append(hand_verts.detach().cpu().numpy())
        out_params['rot'].append(rot.detach().cpu().numpy())
        out_params['pose'].append(pose.detach().cpu().numpy())
        out_params['shape'].append(shape.detach().cpu().numpy())
        out_params['trans'].append(trans.detach().cpu().numpy())

        # if frameId > 3000: break

    for k, v in out_params.items():
        out_params[k] = np.concatenate(out_params[k], 0)

    # import pdb; pdb.set_trace()

    mano_faces = mano_layer.th_faces.detach().cpu()

    return out_params, mano_faces


def get_kps_halo():
    # For HALO
    object_list = ['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass']

    kps_list_all = []

    # kps_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation/kps/'
    # mesh_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation/meshes/'

    kps_dir = '/home/korrawe/halo_vae/exp/grab_baseline_3/generation/kps/'
    mesh_dir = '/home/korrawe/halo_vae/exp/grab_baseline_3/generation/meshes/'

    for obj_idx, object_type in enumerate(object_list):
        print()

        obj_mesh_filename = os.path.join(mesh_dir, '%s_gt_obj_mesh.obj' % object_type)

        kps_list = []
        mesh_dist_list = []
        n_sample = 20
        for idx in range(n_sample):
            # import pdb; pdb.set_trace()
            # hand_mesh_filename = os.path.join(obj_dir, 'obj%03d_h%03d.obj' % (obj_idx, idx))
            hand_kps_filename = os.path.join(kps_dir, '%s_%03d.npy' % (object_type, idx))
            # hand_kps_filename = os.path.join(kps_dir, '%s_%03d_refine.npy' % (object_type, idx))

            print(hand_kps_filename)

            hand_kps = np.load(hand_kps_filename)
            # import pdb; pdb.set_trace()
            # hand_kps = hand_kps / 100.0
            hand_kps_before = hand_kps
            hand_kps = torch.from_numpy(hand_kps)  # .unsqueeze(0)

            # is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)
            # hand_kps = convert_joints(hand_kps, source='mano', target='biomech')

            # hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
            # hand_kps_after = convert_joints(hand_kps_after, source='biomech', target='mano')
            # hand_kps = hand_kps_after.squeeze(0).numpy()

            # vis.visualise_skeleton(hand_kps_before, joint_order='mano', show=False, color='green')
            # vis.visualise_skeleton(hand_kps, joint_order='mano', show=True)

            # kps_flat = hand_kps.reshape(-1)
            kps_list_all.append(hand_kps)
            # kps_all_list.append(kps_flat)

        print(" -- ", object_type, " -- ")

    # import pdb; pdb.set_trace()
    # kps_list_all
    kps_array = torch.stack(kps_list_all, 0)

    # cm to mm
    kps_scale = 10.0
    mano_outputs, mano_faces = fit_mano(kps_array, kps_scale=kps_scale)
    # import pdb; pdb.set_trace()

    # mano_out_dir = '/home/korrawe/halo_vae/exp/grab_refine_inter_2/generation/mano/'
    mano_out_dir = '/home/korrawe/halo_vae/exp/grab_baseline_3/generation/mano/'
    all_idx = 0
    for obj_idx, object_type in enumerate(object_list):
        print()

        # obj_mesh_filename = os.path.join(mesh_dir, '%s_gt_obj_mesh.obj' % object_type)
        n_sample = 20
        for idx in range(n_sample):
            hand_out_filename = os.path.join(mano_out_dir, '%s_%03d.ply' % (object_type, idx))
            # hand_out_filename = os.path.join(mano_out_dir, '%s_%03d_refine.ply' % (object_type, idx))
            hand_mesh = trimesh.Trimesh(vertices=mano_outputs['verts'][all_idx] / 1000.0, faces=mano_faces)
            # import pdb; pdb.set_trace()
            hand_mesh.export(hand_out_filename)
            all_idx += 1
        # mesh_out_path = os.path.join(mano_out_dir, )    


def get_kps_grab():
    pass


def main():

    # gen_keypoint_filename = "/media/korrawe/ssd/halo_vae/data/gen_val_kps/val_gen_20perObj.pkl"
    # # with open(gen_keypoint_filename, 'wb') as p_f:
    # #     pickle.dump(keypoint_list, p_f)

    # with open(gen_keypoint_filename, 'rb') as f:
    #     gen_keypoint = pickle.load(f)

    # # import pdb; pdb.set_trace()
    # out_params = fit_mano(gen_keypoint)
    # out_params_filename = "/media/korrawe/ssd/halo_vae/data/gen_val_kps/val_gen_20perObj_mano_fit.pkl"
    # with open(out_params_filename, 'wb') as p_f:
    #     pickle.dump(out_params, p_f)

    # test read files
    # with open(out_params_filename, 'rb') as f:
    #     fit_params = pickle.load(f)

    # import pdb; pdb.set_trace()

    get_kps_halo()

    # eval_grabnet()
    # eval_halo()


if __name__ == "__main__":
    main()
