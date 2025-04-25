#!/usr/bin/env python3
"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code give camera utilities for the Industrial Plentopic Dataset (IPD).
Provides loading of camera intrinsics/extrinsics and computation of epipolar geometry metrics.
"""

import logging
from pathlib import Path
import json
from typing import List, Dict
import numpy as np

# Configuring logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')


def load_camera_params(scene_dir: Path,cam_ids: List[str]) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    """
    This function load camera intrinsics and extrinsics from JSON files.
    Args are scene_dir: Path to the dataset scene directory and
    cam_ids: List of camera identifiers
    Returns a dictionary with camera parameters for each camera ID
    """
    params: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
    for cid in cam_ids:
        file_path = scene_dir / f'scene_camera_{cid}.json'
        if not file_path.is_file():
            logging.error(f"Camera file not found: {file_path}")
            raise FileNotFoundError(file_path)

        logging.info(f"Loading camera parameters from: {file_path}")
        with file_path.open('r') as f:
            data = json.load(f)

        cam_dict = {'K': {}, 'R': {}, 't': {}}
        for im_id_str, vals in data.items():
            im_id = int(im_id_str)
            cam_dict['K'][im_id] = np.array(vals['cam_K'], dtype=np.float32).reshape(3, 3)
            cam_dict['R'][im_id] = np.array(vals['cam_R_w2c'], dtype=np.float32).reshape(3, 3)
            cam_dict['t'][im_id] = np.array(vals['cam_t_w2c'], dtype=np.float32).flatten()

        params[cid] = cam_dict

    return params


def compute_fundamental_matrix(
    K1: np.ndarray,R1: np.ndarray,
    t1: np.ndarray,K2: np.ndarray,
    R2: np.ndarray,t2: np.ndarray) -> np.ndarray:
    """
    This function Computes the fundamental matrix between two calibrated cameras
    Args are K1, K2: Intrinsic matrices (3x3), R1, R2: Rotation matrices (3x3) mapping world to camera and 
    t1, t2: Translation vectors (3,) of camera centers.
    Returns a fundamental matrix F (3x3).
    """
    # Ensuring vectors
    t1_vec = t1.flatten()
    t2_vec = t2.flatten()
    # Relative pose from camera1 to camera2
    R_rel = R2 @ R1.T
    t_rel = t2_vec - R_rel @ t1_vec

    # Skew-symmetric matrix for t_rel
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ], dtype=np.float32)

    # Essential matrix E = tx * R_rel
    E = tx @ R_rel
    # Fundamental F = K2^-T * E * K1^-1
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    # Normalizing so that F[2,2] == 1
    if abs(F[2, 2]) > 1e-8:
        F /= F[2, 2]

    return F
