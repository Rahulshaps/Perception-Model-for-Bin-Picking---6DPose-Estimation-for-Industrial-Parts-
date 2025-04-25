#!/usr/bin/env python3
"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code gives differentiable rotation representations and pose losses for 6D estimation
Includes geodesic, Euler, quaternion, and 6D formulations with PyTorch.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

# Rotation Conversion Utilities

def geodesic_distance_from_matrix(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Calculates angular distance between rotation matrices (batchwise).
    Args are R1, R2: Tensors of shape (B,3,3)
    Returns tensor of angles in radians, shape (B,)
    """

    R_rel = torch.bmm(R1.transpose(1,2), R2)
    trace = R_rel[...,0,0] + R_rel[...,1,1] + R_rel[...,2,2]
    cos_val = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_val)


def rotmat_from_euler(euler: torch.Tensor) -> torch.Tensor:
    """
    Converting Euler angles to rotation matrices.
    Args is euler: Tensor (B,3) of Rx, Ry, Rz.
    Returns tensor (B,3,3).
    """
    x, y, z = euler.unbind(dim=1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    R11 = cy*cz;       R12 = -cy*sz;      R13 = sy
    R21 = sx*sy*cz + cx*sz; R22 = -sx*sy*sz + cx*cz; R23 = -sx*cy
    R31 = -cx*sy*cz + sx*sz; R32 = cx*sy*sz + sx*cz; R33 = cx*cy

    row1 = torch.stack([R11, R12, R13], dim=1)
    row2 = torch.stack([R21, R22, R23], dim=1)
    row3 = torch.stack([R31, R32, R33], dim=1)
    return torch.stack([row1, row2, row3], dim=1)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Converting quaternion to rotation matrix.
    Args is quat: Tensor (B,4) as [x, y, z, w].
    Returns tensor (B,3,3).
    """
    q = quat / quat.norm(dim=1, keepdim=True)
    x, y, z, w = q.unbind(dim=1)
    B = q.shape[0]
    mat = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
    ], dim=1).view(B, 3, 3)
    return mat


def rotmat_from_6d(rep6d: torch.Tensor) -> torch.Tensor:
    """
    Converting 6D representation to rotation matrix.
    Args is rep6d: Tensor (B,6) representing two 3D vectors.
    Returns tensor (B,3,3).
    """
    a1 = rep6d[:, :3]
    a2 = rep6d[:, 3:]
    b1 = F.normalize(a1, dim=1)
    proj = (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(a2 - proj, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


def euler_to_quat_torch(euler: torch.Tensor) -> torch.Tensor:
    """
    Converting Euler angles to quaternion [x, y, z, w]
    Args is euler: Tensor (B,3).
    Returns tensor (B,4).
    """
    half = 0.5 * euler
    cx, sx = torch.cos(half[:, 0]), torch.sin(half[:, 0])
    cy, sy = torch.cos(half[:, 1]), torch.sin(half[:, 1])
    cz, sz = torch.cos(half[:, 2]), torch.sin(half[:, 2])
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return torch.stack([x, y, z, w], dim=1)


# Metric Evaluations
def compute_rot_deg_mean_matrix(R_gt: torch.Tensor, R_pred: torch.Tensor) -> torch.Tensor:
    """
    This function gives Mean angular error (degrees) between two rotation batches
    Args is R_gt, R_pred: Tensors (B,3,3)
    Returns a scalar tensor (deg)
    """
    R_gt_np = R_gt.cpu().numpy()
    R_pred_np = R_pred.cpu().numpy()
    r1 = R.from_matrix(R_gt_np)
    r2 = R.from_matrix(R_pred_np)
    rel = r1.inv() * r2
    angles = rel.magnitude()
    return torch.tensor(np.mean(np.degrees(angles)), dtype=torch.float32)

# Loss Classes
class EulerAnglePoseLoss(nn.Module):
    """Loss based on Euler angles."""
    def __init__(self, w_rot: float = 1.0):
        super().__init__()
        self.w_rot = w_rot

    def forward(self,labels: dict,preds: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        R_gt = labels.get('R') if 'R' in labels else rotmat_from_euler(labels['euler'])
        pred_euler = preds[:, :3]
        R_pred = rotmat_from_euler(torch.remainder(pred_euler + math.pi, 2*math.pi) - math.pi)
        angles = geodesic_distance_from_matrix(R_pred, R_gt)
        rot_loss = angles.mean() * self.w_rot
        rot_deg = compute_rot_deg_mean_matrix(R_gt, R_pred)
        return rot_loss, {'rot_loss': rot_loss, 'rot_deg_mean': rot_deg}

class QuaternionPoseLoss(nn.Module):
    """Loss using quaternion representation"""
    def __init__(self, w_rot: float = 1.0):
        super().__init__()
        self.w_rot = w_rot

    def forward(self,labels: dict,preds: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        R_gt = labels.get('R') if 'R' in labels else quat_to_rotmat(euler_to_quat_torch(labels['euler']))
        pred_quat = preds[:, :4]
        pred_quat = pred_quat / (pred_quat.norm(dim=1, keepdim=True) + 1e-8)
        # Making quaternion continuity
        dot = (pred_quat * pred_quat).sum(dim=1, keepdim=True)
        pred_quat = torch.where(dot < 0, -pred_quat, pred_quat)
        R_pred = quat_to_rotmat(pred_quat)
        angles = geodesic_distance_from_matrix(R_pred, R_gt)
        rot_loss = angles.mean() * self.w_rot
        rot_deg = compute_rot_deg_mean_matrix(R_gt, R_pred)
        return rot_loss, {'rot_loss': rot_loss, 'rot_deg_mean': rot_deg}

class SixDPoseLoss(nn.Module):
    """Loss using 6D rotation representation"""
    def __init__(self, w_rot: float = 1.0):
        super().__init__()
        self.w_rot = w_rot

    def forward(self,labels: dict,preds: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        R_gt = labels.get('R') if 'R' in labels else rotmat_from_euler(labels['euler'])
        rep6d = preds[:, :6]
        R_pred = rotmat_from_6d(rep6d)
        angles = geodesic_distance_from_matrix(R_pred, R_gt)
        rot_loss = angles.mean() * self.w_rot
        rot_deg = compute_rot_deg_mean_matrix(R_gt, R_pred)
        return rot_loss, {'rot_loss': rot_loss, 'rot_deg_mean': rot_deg}
