#!/usr/bin/env python3
"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code is for multi-view 6D pose estimation using YOLO detection and a fine-tuned pose network.
Includes detection, matching, triangulation, and rotation estimation steps.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from pose.models.losses import rotmat_from_euler, quat_to_rotmat, rotmat_from_6d
from pose.models.simple_pose_net import SimplePoseNet
from utils.data_utils import letterbox_preserving_aspect_ratio, calc_pose_matrix
from inference.epipolar_matching import compute_cost_matrix, match_objects, triangulate_multi_view
from inference.utils.camera_utils import load_camera_params, compute_fundamental_matrix

# Configuring logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')


@dataclass
class PoseEstimatorParams:
    """
    This class is for configuration parameters for pose estimation.
    """
    yolo_model_path: Path = Path("yolo11-detection-obj11.pt")
    pose_model_path: Path = Path("best_model.pth")
    matching_threshold: float = 30.0
    yolo_conf_thresh: float = 0.1
    rotation_mode: Optional[str] = None


def load_pose_model(
    pose_model_path: Path,device: torch.device = torch.device('cuda:0'),
    rotation_mode: Optional[str] = None) -> Tuple[SimplePoseNet, str]:
    """
    This function load a pose network checkpoint and infer rotation output mode.
    Returns a pose_model: Loaded SimplePoseNet in eval mode and 
    rotation_mode: One of 'euler', 'quat', '6d'.
    """
    ckpt = torch.load(pose_model_path, map_location=device)
    if "fc.weight" not in ckpt:
        raise KeyError("Checkpoint missing 'fc.weight'.")
    out_dim = ckpt['fc.weight'].shape[0]

    if rotation_mode is None:
        if out_dim == 3:
            rotation_mode = 'euler'
        elif out_dim == 4:
            rotation_mode = 'quat'
        elif out_dim == 6:
            rotation_mode = '6d'
        else:
            raise ValueError(f"Unknown output_dim={out_dim}")
        logging.info(f"Auto-detected rotation mode: {rotation_mode}")
    else:
        logging.info(f"Using provided rotation mode: {rotation_mode}")

    model = SimplePoseNet(loss_type=rotation_mode, pretrained=False)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, rotation_mode


class PosePrediction:
    """
    This class stores a set of detections and computes triangulated translation.
    """
    def __init__(self,detections: List[Dict],capture: 'Capture'):
        self.detections = detections
        self.capture = capture
        self.boxes = np.array([d['bbox'] for d in detections])
        self.centroids = np.array([d['bb_center'] for d in detections])
        self.translation = self._triangulate()

    def _triangulate(self) -> np.ndarray:
        proj_mats = []
        for idx, box in enumerate(self.boxes):
            K = self.capture.Ks[idx]
            RT = self.capture.RTs[idx]
            P = K @ RT[:3, :]
            proj_mats.append(P)
        return triangulate_multi_view(proj_mats, [tuple(c) for c in self.centroids])


class PoseEstimator:
    """
    This class handles detection, matching, and rotation estimation.
    """
    def __init__(self, params: PoseEstimatorParams):
        self.params = params
        from ultralytics import YOLO
        self.yolo = YOLO(str(params.yolo_model_path)).cuda()
        self.pose_model, self.rotation_mode = load_pose_model(Path(params.pose_model_path),
            device=torch.device('cuda:0'),rotation_mode=params.rotation_mode)

    def _detect(self, capture: 'Capture') -> Dict[int, List[Dict]]:
        preds = {}
        for idx, img in enumerate(capture.images):
            res = self.yolo(img, imgsz=1280)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss  = res.boxes.cls.cpu().numpy()
            valid = (clss == 0) & (confs >= self.params.yolo_conf_thresh)
            dets = []
            for box in boxes[valid]:
                x1,y1,x2,y2 = map(int, box)
                cx, cy = (x1+x2)/2, (y1+y2)/2
                dets.append({'bbox': (x1,y1,x2,y2), 'bb_center': (cx,cy)})
            preds[idx] = dets
        return preds

    def _match(self, capture: 'Capture', detections: Dict[int, List[Dict]]) -> List[PosePrediction]:
        cam_ids = sorted(detections.keys())
        pts = [detections[i] for i in cam_ids]
        K_list = capture.Ks
        RT_list = capture.RTs
        R_list = [RT[:3,:3] for RT in RT_list]
        t_list = [RT[:3,3] for RT in RT_list]
        F12 = compute_fundamental_matrix(K_list[0], R_list[0], t_list[0], K_list[1], R_list[1], t_list[1])
        F13 = compute_fundamental_matrix(K_list[0], R_list[0], t_list[0], K_list[2], R_list[2], t_list[2])
        F23 = compute_fundamental_matrix(K_list[1], R_list[1], t_list[1], K_list[2], R_list[2], t_list[2])

        if any(len(p)==0 for p in pts):
            logging.info("One view has no detections, skipping matching.")
            return []

        cost = compute_cost_matrix(pts[0], pts[1], pts[2], F12, F13, F23)
        matches = match_objects(cost, self.params.matching_threshold)
        preds = []
        for i,j,k in sorted(matches, key=lambda x: cost[x]):
            preds.append(PosePrediction([pts[0][i], pts[1][j], pts[2][k]], capture))
        return preds

    def _estimate_rotation(self, preds: List[PosePrediction], capture: 'Capture') -> None:
        for p in preds:
            rot_list = []
            for idx, det in enumerate(p.detections):
                x1,y1,x2,y2 = det['bbox']
                crop = capture.images[idx][y1:y2, x1:x2]
                img_lb, _, _, _ = letterbox_preserving_aspect_ratio(crop, 256)
                img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
                t = TF.to_tensor(img_rgb)
                t = TF.normalize(t, [0.485,0.456,0.406], [0.229,0.224,0.225]).unsqueeze(0).cuda()
                with torch.no_grad():
                    raw = self.pose_model(t)[0].cpu().numpy()
                if self.rotation_mode == 'euler':
                    wrapped = ((raw + np.pi)%(2*np.pi)) - np.pi
                    Rm = rotmat_from_euler(torch.tensor(wrapped).unsqueeze(0)).squeeze(0).numpy()
                elif self.rotation_mode == 'quat':
                    qt = torch.tensor(raw).unsqueeze(0)
                    qt = qt/ (qt.norm(dim=1,keepdim=True)+1e-8)
                    Rm = quat_to_rotmat(qt).squeeze(0).numpy()
                else:
                    rep6d = torch.tensor(raw).unsqueeze(0)
                    Rm = rotmat_from_6d(rep6d).squeeze(0).numpy()
                final = capture.RTs[idx][:3,:3].T @ Rm
                rot_list.append(final)
            p.rotation_preds = rot_list
            p.final_rotation = rot_list[2]
            p.pose = calc_pose_matrix(p.final_rotation, p.translation)

    def estimate(self, scene_dir: Path, cam_ids: List[str], image_ids: List[int], obj_id: int) -> List[PosePrediction]:
        """
        This function is the Full pipeline: load capture, detect, match, and estimate rotations.
        """
        # Load multi-view capture
        KRT = load_camera_params(scene_dir, cam_ids)
        # Build Capture object externally (not shown)
        from inference.utils.camera_utils import Capture
        capture = Capture.from_dir(scene_dir, cam_ids, image_ids[0], obj_id)

        detections = self._detect(capture)
        preds = self._match(capture, detections)
        self._estimate_rotation(preds, capture)
        return preds


if __name__ == '__main__':
    logging.info("Pose estimation module loaded. Use PoseEstimator. ")