import os
import logging
from matplotlib.pyplot import axis
from torch.utils import data
import numpy as np
import yaml
import pickle
import torch
from scipy.spatial import distance

from models.data.input_helpers import random_rotate


from models.utils import visualize as vis
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


logger = logging.getLogger(__name__)

class ObmanDataset(data.Dataset):
    ''' Obman dataset class.
    '''

    def __init__(self, dataset_folder, input_helpers=None, split=None,
                 no_except=True, transforms=None, return_idx=False, use_bps=False, random_rotate=False):
        ''' Initialization of the the 3D articulated hand dataset.
        Args:
            dataset_folder (str): dataset folder
            input_helpers dict[(callable)]: helpers for data loading
            split (str): which split is used
            no_except (bool): no exception
            transform dict{(callable)}: transformation applied to data points
            return_idx (bool): wether to return index
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.input_helpers = input_helpers
        self.split = split
        self.no_except = no_except
        self.transforms = transforms
        self.return_idx = return_idx
        self.use_bps = use_bps

        self.obman_data = False

        # ## Get all models
        # split_file = os.path.join(dataset_folder, split, 'datalist.txt')
        # with open(split_file, 'r') as f:
        #     self.models = f.read().strip().split('\n')

        # split_file = os.path.join(dataset_folder, split + '.pkl')
        if self.use_bps:
            split_file = os.path.join(dataset_folder, split + '_bps.npz')
            all_data = np.load(split_file)
            self.object_bps = all_data["object_bps"]
        elif self.obman_data:
            split_file = os.path.join(dataset_folder, split + '.pkl')
            # with (open('/home/korrawe/halo_vae/dataset/obman/processed/test.pkl', "rb")) as data_file:
            with open(split_file, "rb") as data_file:
                all_data = pickle.load(data_file)
            self.hand_joints = all_data["hand_joints3d"]
            self.object_points = all_data["object_points3d"]

        else:
            # split_file = os.path.join(dataset_folder, split + '.pkl')
            # split_file = os.path.join(dataset_folder, split + '.npz')
            split_file = os.path.join(dataset_folder, split + '_hand.npz')
            all_data = np.load(split_file)

            self.hand_joints = all_data["hand_joints3d"]
            self.object_points = all_data["object_points3d"]
            self.closest_point_idx = all_data["closest_obj_point_idx"]
            self.closest_point_dist = all_data["closest_obj_point_dist"]
            self.obj_names = all_data["obj_name"]
            self.rot_mats = all_data["rot_mat"]

            # load sample point inside
            sample_point_file = os.path.join(dataset_folder, split + '_sample_vol.npz')
            sample_points = np.load(sample_point_file)
            # import pdb; pdb.set_trace()
            points_dict = {}
            for k in sample_points.files:
                points_dict[k] = sample_points[k]

            self.sample_points = points_dict
            # import pdb; pdb.set_trace()
            self.hand_verts = all_data["hand_verts"]

        #####
        #####
        #####

        if split == 'val':
            val_keep = 1500
            if len(self.hand_joints) > val_keep:
                keep_idx = np.random.choice(len(self.hand_joints), val_keep, replace=False)
                # import pdb; pdb.set_trace()
                # if not self.use_bps:
                #     keep_hand_joints = []
                #     keep_object_points = []
                #     keep_closest_point_idx = []
                #     keep_closest_point_dist = []
                #     for idx in keep_idx:
                #         keep_hand_joints.append(self.hand_joints[idx])
                #         keep_object_points.append(self.object_points[idx])
                #         keep_closest_point_dist.append(self.closest_point_idx[idx])
                #         keep_closest_point_dist.append(self.closest_point_dist[idx])
                #     self.closest_point_idx = keep_closest_point_dist
                #     self.closest_point_dist = keep_closest_point_dist
                #     self.hand_joints = keep_hand_joints
                #     self.object_points = keep_object_points
                # else:
                if self.obman_data:
                    keep_hand_joints = []
                    keep_object_points = []
                    for idx in keep_idx:
                        keep_hand_joints.append(self.hand_joints[idx])
                        keep_object_points.append(self.object_points[idx])
                    self.hand_joints = keep_hand_joints
                    self.object_points = keep_object_points
                else:
                    self.hand_joints = self.hand_joints[keep_idx]
                    self.object_points = self.object_points[keep_idx]
                    self.closest_point_idx = self.closest_point_idx[keep_idx]
                    self.closest_point_dist = self.closest_point_dist[keep_idx]
                    self.hand_verts = self.hand_verts[keep_idx]
                    if self.use_bps:
                        self.object_bps = self.object_bps[keep_idx]                

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.hand_joints)  # len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''
        # model = self.models[idx]

        # split_path = os.path.join(self.dataset_folder, self.split)

        data = {}
        data['mesh_path'] = ''

        # for field_name, input_helper in self.input_helpers.items():
        #     try:
        #         field_data = input_helper.load(split_path, model, idx)
        #     except Exception:
        #         if self.no_except:
        #             logger.warn(
        #                 'Error occured when loading field %s of model %s'
        #                 % (field_name.__class__.__name__, model)
        #                 # % (self.input_helper.__class__.__name__, model)
        #             )
        #             return None
        #         else:
        #             raise

        #     if isinstance(field_data, dict):
        #         for k, v in field_data.items():
        #             if k is None:
        #                 data[field_name] = v
        #             elif field_name == 'inputs':
        #                 data[k] = v
        #             else:
        #                 data['%s.%s' % (field_name, k)] = v
        #     else:
        #         data[field_name] = field_data

        # if self.transforms is not None:
        #     for tran_name, tran in self.transforms.items():
        #         data = tran(data)

        # idx = 0
        # idx = idx % 1000
        data['object_points'] = self.object_points[idx]
        data['hand_joints'] = self.hand_joints[idx]
        data['object_points'], data['hand_joints'], data['obj_center'] = self.obj_to_origin(data['object_points'], data['hand_joints'])

        if not self.obman_data:
            data['closest_point_idx'] = self.closest_point_idx[idx]
            data['closest_point_dist'] = self.closest_point_dist[idx]

            # hand verts
            data['hand_verts'] = self.hand_verts[idx] - data['obj_center']

            # Get points sampled inside the mesh for occupancy query
            # import pdb; pdb.set_trace()
            inside_points = self.sample_points[self.obj_names[idx]]
            keep_idx = np.random.choice(len(inside_points), 600, replace=False)
            inside_points = inside_points[keep_idx]
            # inside_points = np.inside_points * self.rot_mats[idx].T
            inside_points = np.matmul(self.rot_mats[idx], np.expand_dims(inside_points, -1)).squeeze(-1)
            data['inside_points'] = inside_points - data['obj_center']

        # def rotate_vertices(self, rotation_matrix):
        # import cv2
        # rotation_matrix = np.matrix(cv2.Rodrigues(np.array(rotation_matrix))[0] if (np.array(rotation_matrix).shape != (3, 3)) else rotation_matrix)
        # self.v = np.array(self.v * rotation_matrix.T)
        # return self

        # fig = plt.figure()
        # ax = fig.gca(projection=Axes3D.name)
        # # vis.plot_skeleton_single_view(data['hand_joints'], joint_order='mano', object_points=data['object_points'], ax=ax, color='r', show=False)
        # vis.plot_skeleton_single_view(data['hand_joints'], joint_order='mano', object_points=inside_points, ax=ax, color='b', show=False)
        # fig.show()
        # import pdb; pdb.set_trace()
        # plt.close()

        if self.use_bps:
            data['object_bps'] = self.object_bps[idx]
        else:
            data = self.scale_to_cm(data)

        data = self.gen_refine_training_data(data)

        # import pdb; pdb.set_trace()
        # Randomly rotate to augment data
        # if self.split == 'train':
        #     data['object_points'], data['hand_joints'], rot_mat = random_rotate(data['object_points'], data['hand_joints'])

        # print("hand_joints", data['hand_joints'].shape)
        # print("object_points", data['object_points'].shape)

        if self.return_idx:
            data['idx'] = idx

        return data

    def gen_refine_training_data(self, data):
        hand_joints, obj_points = data['hand_joints'], data['object_points']
        mu = 0.0
        scale = 0.5  # 2mm
        noise = np.random.normal(mu, scale, 15 * 3)
        noise = noise.reshape((15, 3))
        noisy_joints = hand_joints.copy()
        noisy_joints[6:] = noisy_joints[6:] + noise

        trans_noise = np.random.rand() * 2.0  # 0.5
        noisy_joints = noisy_joints + trans_noise

        # fig = plt.figure()
        # ax = fig.gca(projection=Axes3D.name)
        # vis.plot_skeleton_single_view(hand_joints, joint_order='mano', object_points=obj_points, ax=ax, color='r', show=False)
        # vis.plot_skeleton_single_view(noisy_joints, joint_order='mano', ax=ax, color='b', show=False)
        # fig.show()
        # import pdb; pdb.set_trace()
        # fig.close()

        data['noisy_joints'] = noisy_joints
        tip_idx = np.array([4, 8, 12, 16, 20])
        # tip_loc = noisy_joints[tip_idx]
        p2p_dist = distance.cdist(noisy_joints, obj_points)
        p2p_dist = p2p_dist.min(axis=1)
        data['tip_dists'] = p2p_dist
        return data

    def obj_to_origin(self, object_points, hand_joints):
        # obj_center = object_points.mean(axis=0)
        min_p = object_points.min(0)
        max_p = object_points.max(0)
        obj_center = (max_p + min_p) / 2.0
        # print("obj center", obj_center)
        return object_points - obj_center, hand_joints - obj_center, obj_center

    def scale_to_cm(self, data_dict):
        scale = 100.0
        data_dict['object_points'] = data_dict['object_points'] * scale
        data_dict['hand_joints'] = data_dict['hand_joints'] * scale
        # data_dict['closest_point_idx'] = data_dict['closest_point_idx']
        # data_dict['closest_point_dist'] = data_dict['closest_point_dist'] * scale
        # data_dict['object_bps'] = data_dict['object_bps'] * scale
        # data_dict['inside_points'] = data_dict['inside_points'] * scale

        # Hand verts
        data_dict['hand_verts'] = data_dict['hand_verts'] * scale
        return data_dict

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.
        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True
