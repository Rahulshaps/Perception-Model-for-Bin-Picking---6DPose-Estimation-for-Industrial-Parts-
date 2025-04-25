#!/usr/bin/env python3
"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code give triangulation utilities for multi-view 6D pose estimation on the Industrial Plentopic Dataset (IPD).
Provides Direct Linear Transform (DLT) triangulation, reprojection error computation, and assembly of final 6D pose vectors.
"""

import numpy as np
import logging
from typing import Sequence, Tuple

# Configuring logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')


def triangulate_multi_view(proj_mats: Sequence[np.ndarray],points: Sequence[Tuple[float, float]]) -> np.ndarray:
    """
    This functoion triangulate a 3D point from multiple camera views using Direct Linear Transform (DLT).
    Args are proj_mats: Sequence of 3x4 projection matrices (P) and 
    points: Sequence of corresponding 2D image coordinates (x, y)
    Returns a 3D point as numpy array of shape (3,) in homogeneous coordinates
    """
    # Building linear system A * X = 0
    A = []
    for P, (u, v) in zip(proj_mats, points):
        # Equation: u * P[2] - P[0] and v * P[2] - P[1]
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
    A = np.vstack(A)

    # Solving via SVD: smallest singular vector of A
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]

    # Converting  from homogeneous to Euclidean
    if abs(X_h[3]) < 1e-8:
        logging.warning("Triangulated point at infinity (w ~ 0)")
        return X_h[:3]
    return X_h[:3] / X_h[3]


def compute_reprojection_error(proj_mat: np.ndarray,point_3d: np.ndarray,point_2d: Tuple[float, float]) -> float:
    """
    This function computes the reprojection error of a 3D point onto a single view.
    Args are proj_mat: 3x4 projection matrix, point_3d: 3D point (x, y, z) and point_2d: Observed 2D image coordinate (u, v).
    Returns a euclidean pixel error between projected and observed points
    """
    if point_3d.shape[0] != 3:
        raise ValueError("point_3d must have shape (3,)")

    # Homogeneous 3D point
    X_h = np.append(point_3d, 1.0)
    proj = proj_mat @ X_h

    if abs(proj[2]) < 1e-8:
        logging.warning("Point projects to infinity (z ~ 0)")
        return float('inf')

    proj /= proj[2]
    u_proj, v_proj = proj[0], proj[1]
    u_obs, v_obs = point_2d

    return float(np.linalg.norm([u_proj - u_obs, v_proj - v_obs]))


def assemble_final_pose(rotation_params: np.ndarray,translations: np.ndarray) -> np.ndarray:
    """
    This function assemble final 6D poses by averaging rotation angles and combining with translations.
    Args are rotation_params: Array of shape (N, M, 3) containing per-view Euler angles [Rx, Ry, Rz] and 
    translations: Array of shape (N, 3) of triangulated 3D points [X, Y, Z].
    Returns a final_poses: Array of shape (N, 6) with [Rx_avg, Ry_avg, Rz_avg, X, Y, Z]
    """
    if rotation_params.ndim != 3 or rotation_params.shape[2] != 3:
        raise ValueError("rotation_params must be of shape (N, M, 3)")
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must be of shape (N, 3)")

    N = rotation_params.shape[0]
    final_poses = np.zeros((N, 6), dtype=np.float32)

    # Average rotations per object
    avg_rotations = rotation_params.mean(axis=1)
    final_poses[:, :3] = avg_rotations
    final_poses[:, 3:] = translations

    return final_poses
