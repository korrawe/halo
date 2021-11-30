import torch
import torch.nn as nn
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np
import trimesh

import argparse

import json
import os
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default='/media/korrawe/Elements/PS_scratch/FHB/Hand_pose_annotation_v1/Subject_1/charge_cell_phone/1/skeleton.txt', help='Path to file skeleton.txt')
parser.add_argument('--output_dir', type=str, default='./test_out/', help='Output directory for meshes')
parser.add_argument('--w_pose', type=float, default=10.0, help='Weight of pose regularization, default is 10.0')
parser.add_argument('--w_shape', type=float, default=10.0, help='Weight of shape regularization, default is 10.0')
parser.add_argument('--visualize_interm_result', type=bool, default=False, help='If visualize the result, default is False')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size, default is 8')
parser.add_argument('--ncomps', type=int, default=45, help='Number of hand pose PCAs, default is 45')
parser.add_argument('--epoch_coarse', type=int, default=500, help='Max optimization iterations for coarse alignment, default is 500')
parser.add_argument('--epoch_fine', type=int, default=1000, help='Max optimization iterations for fine alignment, default is 1000')


def marker2mano(arg_data_file = '/ps/scratch/kkarunratanakul/FHB/Hand_pose_annotation_v1/Subject_1/charge_cell_phone/1/skeleton.txt',\
                arg_output_dir = './test_out/',\
                arg_w_pose = 100.0,\
                arg_w_shape = 100.0,\
                arg_batch_size = 8,\
                arg_ncomps = 45,\
                arg_epoch_coarse = 500,\
                arg_epoch_fine = 1000,\
                arg_visualize_interm_result = False):

    data_file = arg_data_file
    output_dir = arg_output_dir

    paths = data_file.split('/')
    # print(paths)
    instance_out = "_".join(paths[-4:-1])
    # print(instance_out)

    w_pose = arg_w_pose
    w_shape = arg_w_shape

    visualize_interm_result = arg_visualize_interm_result

    batch_size = arg_batch_size
    # Select number of principal components for pose space
    ncomps = arg_ncomps

    epoch_coarse = arg_epoch_coarse
    epoch_fine = arg_epoch_fine


    with open(data_file) as f:
        data_array = [[float(x) for x in line.split()] for line in f]
    data_array_np = np.array(data_array)
    del data_array

    target_mano_joint = np.zeros([data_array_np.shape[0], data_array_np.shape[1] - 1])
    # FHB -> MANO
    MANO_from_FHB = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20],
     [0, 1, 6, 7, 8, 2, 9, 10,11,3,12,13,14, 4,15,16,17, 5,18,19,20]])

    # FHB skeleton (For visualization only)
    # target_skeleton = np.array([[0, 1, 6, 7, 8],
    #                             [0, 2, 9,10,11],
    #                             [0, 3,12,13,14],
    #                             [0, 4,15,16,17],
    #                             [0, 5,18,19,20]])
    # MANO skeleton (For visualization only)
    target_skeleton = np.array([[0, 1, 2, 3, 4],
                                [0, 5, 6, 7, 8],
                                [0, 9,10,11,12],
                                [0,13,14,15,16],
                                [0,17,18,19,20]])

    for i in range(0, MANO_from_FHB.shape[1]):
        j_id_mano = MANO_from_FHB[0, i]
        j_id_FHB = MANO_from_FHB[1, i]
        target_mano_joint[:,j_id_mano*3:j_id_mano*3+3] = data_array_np[:,j_id_FHB*3+1:j_id_FHB*3+3+1]
    frameId_arrar = data_array_np[:,0].copy()
    del data_array_np

    data_size = target_mano_joint.shape[0]
    print(str(data_size) + ' frames to be processed.')

    # Initialize MANO layer
    mano_layer = ManoLayer(
        mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False)
    mano_layer = mano_layer.to(device)


    # Batch optimization
    for frameId in range(0, data_size, batch_size):
        current_batch_size = min(batch_size, data_size - frameId)
        print("Processing " + str(frameId) + ' to ' + str(frameId + current_batch_size))

        # Targeted joint positions
        target_js = torch.from_numpy(target_mano_joint[frameId:frameId+current_batch_size, :]).float().to(device)
        target_js = target_js.view(current_batch_size, -1, 3)

        # Model para initialization:
        shape = torch.randn(current_batch_size, 10).to(device)
        shape.requires_grad_()
        rot = torch.rand(current_batch_size, 3).to(device)
        rot.requires_grad_()
        pose = torch.randn(current_batch_size, ncomps).to(device)
        pose.requires_grad_()
        trans = (target_js[:,0]).to(device)/1000.0
        trans.requires_grad_()

        if visualize_interm_result:
            print('Before pose shape optimization.')
            hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)
            demo.display_mosh(target_js.detach().cpu(),
                                target_skeleton,
                                {'verts': hand_verts.detach().cpu(),
                                'joints': hand_joints.detach().cpu()}, \
                                mano_faces=mano_layer.th_faces.detach().cpu(), \
                                alpha = 0.3)

        criteria_loss = nn.MSELoss().to(device)

        # Optimize for global translation and rotation
        optimizer = torch.optim.Adam([trans, rot], lr=1e-2)
        for i in range(0, 500):
            _, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)
            loss = criteria_loss(hand_joints, target_js)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('After coarse alignment: %6f'%(loss.data))

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
        optimizer = torch.optim.Adam([trans, rot, pose, shape], lr=1e-2)
        for i in range(0, 1000):
            _, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)
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
                                alpha = 0.3)

        # Output meshes:
        with torch.no_grad(): 
            hand_verts, hand_joints = mano_layer(torch.cat((rot, pose),1), shape, trans)

        for i in range(0, current_batch_size):
            frame_num = str(int(frameId_arrar[frameId+i])).zfill(4)
            # output_file = output_dir + '/' + frame_num + '.obj'

            # output_new_path = "_".join([instance_out, frame_num])
            # output_new_path = os.path.join(output_dir, instance_out, 'hand_' + frame_num, 'models', 'model_mm.obj')
            # for meta data
            # output_meta_path = os.path.join(output_dir, instance_out, frame_num + '.npy')
            output_meta_path = os.path.join(output_dir, instance_out, frame_num + '.npz')

            # print(output_new_path)
            hand_model_output_dir = os.path.dirname(output_meta_path)
            # print(hand_model_output_dir)

            if not os.path.isdir(hand_model_output_dir):
                os.makedirs(hand_model_output_dir)

            mesh = trimesh.base.Trimesh(vertices = hand_verts[i,:,:].detach().cpu().data,\
                                        faces = mano_layer.th_faces.detach().cpu().data,\
                                        process=False)
            
            ### Save joints
            joints_posi = hand_joints[i,:,:].detach().cpu().numpy()
            # print(joints_posi)
            # print(joints_posi.shape)
            # print(output_new_path)
            # np.save(output_new_path, joints_posi)
            # print("==================================")
            # mesh.export(output_new_path)

            # save MANO params
            np.savez(output_meta_path, shape=shape[i].detach().cpu().numpy(), 
                pose=pose[i].detach().cpu().numpy(),
                rot=rot[i].detach().cpu().numpy(),
                trans=trans[i].detach().cpu().numpy())

        del hand_verts, hand_joints, loss


if __name__ == '__main__':
    args = parser.parse_args()

    train_split = False
    fhb_path = '/media/korrawe/Elements/PS_scratch/FHB/'
    fhb_model_outdir = '/home/korrawe/nasa/data/FHB/'
    if train_split:
        split = 'train'
    else:
        split = 'test'
    
    # i = 0
    st_time = time.time()
    for split in ['test']: # ['train', 'test']:
        split_file = fhb_path + split + '.json'

        # load_test
        with open(split_file, "r") as f:
            read_file = json.load(f)
        
        split_out_dir = os.path.join(fhb_model_outdir, split)
        if not os.path.isdir(split_out_dir):
            os.makedirs(split_out_dir)
        
        print("Output to ", split_out_dir)
        
        
        for instance in read_file:
            instance = "/media/korrawe/Elements/PS_scratch/FHB/Hand_pose_annotation_v1/Subject_2/toast_wine/1/skeleton.txt"
            instance = "/media/korrawe/Elements/PS_scratch/FHB/Hand_pose_annotation_v1/Subject_2/squeeze_sponge/2/skeleton.txt"
            instance = "/media/korrawe/Elements/PS_scratch/FHB/Hand_pose_annotation_v1/Subject_1/read_letter/1/skeleton.txt"
            instance = "/media/korrawe/Elements/PS_scratch/FHB/Hand_pose_annotation_v1/Subject_2/squeeze_paper/1/skeleton.txt"
            print(instance)



            # paths = instance.split('/')
            # print(paths)

            # instance_out = "_".join(paths[-4:-1])
            # print(instance_out)
            

            marker2mano(arg_data_file = instance, # = args.data_file,\
                        arg_output_dir = split_out_dir, # args.output_dir,\
                        arg_w_pose = args.w_pose,\
                        arg_w_shape = args.w_shape,\
                        arg_batch_size = args.batch_size,\
                        arg_ncomps = args.ncomps,\
                        arg_epoch_coarse = args.epoch_coarse,\
                        arg_epoch_fine = args.epoch_fine,\
                        arg_visualize_interm_result = args.visualize_interm_result)
            
            # i += 1
            # if i > 2:
            #     break            
            break
        break
    
    print("time used: ", time.time() - st_time)
