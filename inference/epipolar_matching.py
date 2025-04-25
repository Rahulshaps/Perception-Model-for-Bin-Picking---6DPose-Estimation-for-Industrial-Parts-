#!/usr/bin/env python3
"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This codge gives epipolar geometry and matching utilities for 6D pose estimation on the IPD dataset.
Includes symmetric epipolar error, multi-view assignment via Hungarian, and DLT triangulation.
"""

import logging
from typing import Sequence, Tuple, List, Dict, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Configure logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')


def epipolar_error(pt1: Tuple[float, float],pt2: Tuple[float, float],F: np.ndarray,
    img1: Optional[np.ndarray] = None,img2: Optional[np.ndarray] = None) -> float:
    """
    This functions computes symmetric epipolar distance between a point in two views.
    Args are pt1: (u1, v1) in image, pt2: (u2, v2) in image2, 
    F: Fundamental matrix relating image1 and image2 (3x3), img1: Optional BGR image for visualization and 
    img2: Optional BGR image for visualization.
    Returns a Symmetric epipolar error.
    """
    # Homogeneous coordinates
    p1 = np.array([pt1[0], pt1[1], 1.0], dtype=np.float32)
    p2 = np.array([pt2[0], pt2[1], 1.0], dtype=np.float32)

    # Epipolar lines
    l2 = F @ p1       # line in image2
    l1 = F.T @ p2     # line in image1

    # Normalize lines
    n1 = np.linalg.norm(l1[:2])
    n2 = np.linalg.norm(l2[:2])
    if n1 > 1e-8:
        l1 /= n1
    if n2 > 1e-8:
        l2 /= n2

    # Distance from points to corresponding epipolar lines
    d1 = abs(np.dot(l1, p1)) if n1 > 1e-8 else float('inf')
    d2 = abs(np.dot(l2, p2)) if n2 > 1e-8 else float('inf')
    error = 0.5 * (d1 + d2)

    # Visualization if images provided
    if img1 is not None and img2 is not None:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        def _line_endpoints(line: np.ndarray, h: int, w: int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
            # Compute two points on the line for plotting
            a, b, c = line
            if abs(a) > abs(b):
                y0, y1 = 0, h
                x0 = int(-c/a)
                x1 = int((-c - b*h)/a)
            else:
                x0, x1 = 0, w
                y0 = int(-c/b)
                y1 = int((-c - a*w)/b)
            return (x0, y0), (x1, y1)

        (x11,y11),(x12,y12) = _line_endpoints(l1, h1, w1)
        (x21,y21),(x22,y22) = _line_endpoints(l2, h2, w2)

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.plot(pt1[0], pt1[1], 'ro')
        plt.plot([x11,x12], [y11,y12], 'g-')
        plt.title('View 1')

        plt.subplot(1,2,2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.plot(pt2[0], pt2[1], 'bo')
        plt.plot([x21,x22], [y21,y22], 'm-')
        plt.title('View 2')
        plt.show()

    return float(error)


def epipolar_error_full(pt1: Tuple[float, float],pt2: Tuple[float, float],
    pt3: Tuple[float, float],F12: np.ndarray,F13: np.ndarray,F23: np.ndarray) -> float:
    """
    This functions give average symmetric epipolar error across three cameras.
    Args are pt1, pt2, pt3: Image points in three views and 
    F12, F13, F23: Fundamental matrices between pairs.
    Returns a Mean of epipolar errors
    """
    e12 = epipolar_error(pt1, pt2, F12)
    e13 = epipolar_error(pt1, pt3, F13)
    e23 = epipolar_error(pt2, pt3, F23)
    return (e12 + e13 + e23) / 3.0


def compute_cost_matrix(dets1: Sequence[Dict],dets2: Sequence[Dict],
    dets3: Sequence[Dict],F12: np.ndarray,F13: np.ndarray,F23: np.ndarray) -> np.ndarray:
    """
    This function constructs a 3D cost tensor using epipolar errors of bbox centers.
    Args are dets1, dets2, dets3: Lists of detections, each with 'bb_center' and 
    F12, F13, F23: Fundamental matrices.
    Returns a Cost array of shape (N1, N2, N3)
    """
    N1, N2, N3 = len(dets1), len(dets2), len(dets3)
    cost = np.zeros((N1, N2, N3), dtype=np.float32)
    for i, d1 in enumerate(dets1):
        for j, d2 in enumerate(dets2):
            for k, d3 in enumerate(dets3):
                cost[i,j,k] = epipolar_error_full(
                    d1['bb_center'], d2['bb_center'], d3['bb_center'],
                    F12, F13, F23
                )
    return cost


def match_objects(cost_matrix: np.ndarray,threshold: float) -> List[Tuple[int, int, int]]:
    """
    This function solve assignment over three views via Hungarian and thresholding.
    Args are  cost_matrix: (N1,N2,N3) cost tensor and threshold: Maximum allowable cost.
    Returns a list of matched index triplets.
    """
    N1, N2, N3 = cost_matrix.shape
    # Flatten to 2D: (N1*N2) x N3
    flat = cost_matrix.reshape(N1*N2, N3)
    rows, cols = linear_sum_assignment(flat)
    matches: List[Tuple[int,int,int]] = []
    for r, c in zip(rows, cols):
        if flat[r,c] < threshold:
            i = r // N2
            j = r % N2
            k = c
            matches.append((i,j,k))
    return matches


def triangulate_multi_view(proj_mats: Sequence[np.ndarray],
    points_2d: Sequence[Tuple[float, float]]) -> np.ndarray:
    """
    This function gives direct Linear Transform (DLT) triangulation from multiple views
    Args are proj_mats: List of 3x4 projection matrices and points_2d: Corresponding 2D points (u,v).
    Returns a Reconstructed 3D point (x,y,z).
    """
    A: List[np.ndarray] = []
    for P, (u, v) in zip(proj_mats, points_2d):
        A.append(u * P[2,:] - P[0,:])
        A.append(v * P[2,:] - P[1,:])
    A = np.vstack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-8:
        logging.warning("Triangulated point at infinity.")
        return X[:3]
    return X[:3] / X[3]