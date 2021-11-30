import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm

from manopth.manolayer import ManoLayer
from manopth import demo

sys.path.insert(0, "/home/korrawe/halo_vae")
from models.halo_adapter.converter import PoseConverter, transform_to_canonical
from models.halo_adapter.interface import convert_joints

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit_mano(gen_keypoint):
    batch_size = 1024  # 512  # 32
    w_pose = 10.0
    w_shape = 10.0
    ncomps = 45
    epoch_coarse = 500
    epoch_fine = 500  # 1000

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
    gen_keypoint = np.stack(gen_keypoint, 0)

    # from cm to mm
    gen_keypoint = gen_keypoint * 10.0

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

        target_js = torch.from_numpy(gen_keypoint[frameId: frameId + current_batch_size, :]).float().to(device)
        target_js = target_js[:, :] - target_js[:, None, 0]

        # Use HALO adapter to normalize middle root bone
        target_js = convert_joints(target_js, source='mano', target='biomech')
        target_js, unused_mat = transform_to_canonical(target_js, torch.ones(target_js.shape[0], device=device))
        target_js = convert_joints(target_js, source='biomech', target='mano')

        # Model para initialization:
        shape = torch.randn(current_batch_size, 10).to(device)
        shape.requires_grad_()
        rot = torch.rand(current_batch_size, 3).to(device)
        rot.requires_grad_()
        pose = torch.randn(current_batch_size, ncomps).to(device)
        pose.requires_grad_()
        trans = (target_js[:, 0]).to(device) / 1000.0
        trans.requires_grad_()

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

        # Optimize for global rotation (no trans)
        optimizer = torch.optim.Adam([rot], lr=1e-2)  # trans
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

        # Local optimization - not update translation
        optimizer = torch.optim.Adam([rot, pose, shape], lr=1e-2)  # trans
        for i in range(0, epoch_fine):
            hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)
            loss = criteria_loss(hand_joints, target_js) + w_shape*(shape*shape).mean() + w_pose*(pose*pose).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

    return out_params


if __name__ == "__main__":

    gen_keypoint_filename = "/media/korrawe/ssd/halo_vae/data/gen_val_kps/val_gen_20perObj.pkl"
    # with open(gen_keypoint_filename, 'wb') as p_f:
    #     pickle.dump(keypoint_list, p_f)

    with open(gen_keypoint_filename, 'rb') as f:
        gen_keypoint = pickle.load(f)

    # import pdb; pdb.set_trace()
    out_params = fit_mano(gen_keypoint)
    out_params_filename = "/media/korrawe/ssd/halo_vae/data/gen_val_kps/val_gen_20perObj_mano_fit.pkl"
    with open(out_params_filename, 'wb') as p_f:
        pickle.dump(out_params, p_f)

    # test read files
    # with open(out_params_filename, 'rb') as f:
    #     fit_params = pickle.load(f)

    # import pdb; pdb.set_trace()
