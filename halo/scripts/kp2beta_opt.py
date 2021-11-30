"""
Optimize for beta given the QP root finding formulation
"""

import json
import numpy as np
import torch
import os
from manopth import demo
from manopth.manolayer import ManoLayer

import sys
sys.path.append(".")
from pose.dataset_reader.hand.freihand_dataset import convert_order
from pose.utils.visualization_2 import plot_fingers
import matplotlib.pyplot as plt

def json_load(p):
    with open(p, "r") as f:
        d = json.load(f)
    return d


def ch2pyt(ch_arr):
    return torch.from_numpy(np.array(ch_arr)).float()


def split_theta(theta):
    poses = theta[:, :48]
    shapes = theta[:, 48:58]
    uv_root = theta[:, 58:60]
    scale = theta[:, 60:]
    return poses, shapes, uv_root, scale


def get_pc_idx():
    # Construct child/parent indices
    idx_p = torch.cat((torch.tensor([0]*5) , torch.arange(1,11)))
    idx_c = torch.arange(1,16)

    return idx_p, idx_c


def get_range(idx, len_range):
    n_idx = len(idx)
    idx = idx.repeat_interleave(len_range) * 3
    idx += torch.arange(len_range).repeat(n_idx)

    return idx


def initialize_QP(mano_layer, J):
    """
    For the QP problem:
    bT Q b + cT b + a = bl2

    returns Q, c, a, bl2
    """
    # Get T,S,M
    n_v = 778
    n_j = 16
    # Extract template mesh T and reshape from V x 3 to 3V
    T = mano_layer.th_v_template
    T = T.view(3 * n_v)
    # Extract Shape blend shapes and reshape from V x 3 x B to 3V x B
    S = mano_layer.th_shapedirs
    S = S.view(3 * n_v, 10)
    # Extract M and re-order to Zimmermann joint ordering.
    M = mano_layer.th_J_regressor
    # Add entries for the tips. TODO Add actual vertex positions
    M = torch.cat((M, torch.zeros((5, n_v))), dim=0)
    # Convert to our joint ordering
    M = M[mano_2_zimm][zimm_2_ours]
    # Remove entries for tips. TODO Once using actual tip position, remove this step
    M = M[:16]
    # Construct the 3J x 3V band matrix
    M_band = torch.zeros(3 * n_j, 3 * n_v)
    fr = -(n_j - 1)
    to = n_v
    for i in range(fr, to):
        # Extract diagonal from M
        d = M.diag(i)
        # Expand it
        d = d.repeat_interleave(3)
        # Add it to the final band matrix
        M_band.diagonal(3 * i)[:] = d
    # Construct Q, c, a and bl for the quadratic equation: bT Q b + cT b + (a - bl2) = 0
    # Joint idx in Zimmermann ordering
    # idx_p = 10
    # idx_c = 11
    # Construct child/parent indices
    idx_p = torch.cat((torch.tensor([0]*5) , torch.arange(1,11)))
    idx_c = torch.arange(1,16)
    # Compute bl squared
    bl2 = torch.norm(J[idx_c] - J[idx_p], dim=-1).pow(2)

    idx_p_range = get_range(idx_p, 3)
    idx_c_range = get_range(idx_c, 3)

    M_c = M_band[idx_c_range]
    M_p = M_band[idx_p_range]
    # Exploit additional dimension to make it work across all bones
    M_c = M_c.view(15, 3, 3*n_v)
    M_p = M_p.view(15, 3, 3*n_v)
    T = T.view(1,3*n_v, 1)
    S = S.view(1,2334,10)
    bl2 = bl2.view(15,1,1)
    # Construct M_cp
    M_cp = (M_c - M_p).transpose(-1,-2) @ (M_c - M_p)
    # DEBUG
    # bone_idx = 0  # Should be root/thumb_mcp
    # M_c = M_c[bone_idx]
    # M_p = M_p[bone_idx]
    # M_cp = M_cp[bone_idx]
    # bl2 = bl2[bone_idx]
    # Compute a
    a = T.transpose(-1,-2) @ M_cp @ T
    # Compute c
    c = (
            S.transpose(-1,-2) @ (M_cp.transpose(-1,-2) @ T) + 
            S.transpose(-1,-2) @ (M_cp @ T)
        )
    # Compute Q
    Q = S.transpose(-1,-2) @ M_cp @ S

    return Q, c, a, bl2


def eval_QP(Q,c,a, bl2, b):
    b = b.view(1,10,1)
    val = ((b.transpose(-1,-2) @ Q @ b) + (c.transpose(-1,-2) @ b) + a)

    r = (val - bl2)

    return r


def get_J(Q,c, b):
    # Jacobian of QP problem
    J = Q @ b + c

    return J


def newtons_method(Q,c,a, bl2, beta_init, tol=1e-4):
    F = eval_QP(Q, c, a, bl2, beta_init)
    beta = beta_init.view(1,10,1)
    i = 0
    while F.abs().max() > tol:
        J = get_J(Q,c, beta)
        F = eval_QP(Q,c, a, bl2, beta)
        # Reshape matrices
        J = J.squeeze(-1)
        F = F.squeeze(-1)

        J_inv = (J.transpose(-1,-2) @ J).inverse() @ J.transpose(-1,-2)
        beta = beta - (J_inv @ F).unsqueeze(0)

        print(f'(Gauss-Newton) Max. residual: {F.abs().max()} mm')

        i += 1

    return beta, i



ds_path = "/hdd/Datasets/freihand_dataset"
idx = 10000
mano_list = json_load(os.path.join(ds_path, "training_mano.json"))
K_list = json_load(os.path.join(ds_path, "training_K.json"))
use_mean_pose = True
use_pca = False

mano = np.array(mano_list[idx])
K = np.array(K_list[idx])
poses, shapes, uv_root, scale = split_theta(mano)
poses = torch.from_numpy(poses).float()
shapes = torch.from_numpy(shapes).float()
ncomps = poses.shape[1]
# Get mean hand
# poses *= 0
# shapes *= 0
mano_layer = ManoLayer(
    mano_root="mano_data",
    use_pca=use_pca,
    ncomps=ncomps,
    flat_hand_mean=not use_mean_pose,
)

V, J = mano_layer(poses, shapes)
b = shapes[0]
# demo.display_hand({"verts": V, "joints": J}, mano_faces=mano_layer.th_faces)
# Undo changes from Yana's MANO layer to get raw MANO joints
# Convert to m
J = J[0] / 1000 
V = V[0] / 1000
# Undo joint reordering
mano_2_zimm = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
zimm_2_ours = np.array([0, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20])
zimm_2_mano = np.argsort(mano_2_zimm)

# J = J[zimm_2_mano]
J = J[zimm_2_ours]
# Remove the tips
J = J[:16]


Q,c,a,bl2 = initialize_QP(mano_layer, J)
# r = eval_QP(Q,c,a,bl2, b)
beta_init = torch.zeros_like(b)
b_est, n_iter = newtons_method(Q,c,a,bl2, beta_init)
# Construct MANO hand with estimated betas
shapes_est = b_est.view(1,10)
V_est, J_est = mano_layer(poses, shapes_est)
# Convert to m
J_est = J_est[0] / 1000 
V_est = V_est[0] / 1000
J_est = J_est[zimm_2_ours]
J_est = J_est[:16]
# Compute euclidean distance for J,V and beta
err_V = (V - V_est).pow(2).sum(-1).sqrt().mean() * 1000
err_J = (J - J_est).pow(2).sum(-1).sqrt().mean() * 1000
err_b = (b.squeeze() - b_est.squeeze()).pow(2).sum(-1).sqrt().mean()

print(
        f"""
Num. iter: {n_iter}
Err. Vertices: {err_V} mm
Err. joints: {err_J} mm
Err. betas: {err_b}
Beta: {b.squeeze()}
Beta_est: {b_est.squeeze()}
        """
        )
