"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This module provides data loading, augmentation, and preprocessing utilities
for 6D pose estimation on the Industrial Plentopic Dataset (IPD)
"""

import logging
from pathlib import Path
import math
import json
import glob
import random
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from scipy.spatial.transform import Rotation as R
from inference.utils.camera_utils import load_camera_params
import pyrender

# Making the proper OpenGL backend
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Configuring the logging for debugging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

# Setting the random seed for reproducibility for torch and numpy
def compute_2d_center(K: np.ndarray,R_mat: np.ndarray,t: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Projects a 3D translation vector into 2D image coordinates and Returns (u, v) if in front of camera, else None
    """
    if t[2, 0] <= 0:
        return None
    uv = K @ t
    if uv[2, 0] == 0:
        return None
    uv /= uv[2, 0]
    return float(uv[0, 0]), float(uv[1, 0])

def letterbox_preserving_aspect_ratio(img: np.ndarray,target_size: int = 256,fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[np.ndarray, float, int, int]:
    """
    Resizing and padding an image to a square of size target_size, preserving aspect ratio and returns padded image, scale, dx, dy offsets
    """
    h, w = img.shape[:2]
    scale = float(target_size) / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), fill_color, dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy

def matrix_to_euler_xyz(R_mat: np.ndarray) -> Tuple[float, float, float]:
    """
    Converting a rotation matrix to Euler angles (Rx, Ry, Rz) in radians and Assumesing rotation order Rx * Ry * Rz
    """
    sy = R_mat[0, 2]
    eps = 1e-7
    if abs(abs(sy) - 1.0) > eps:
        y = math.asin(sy)
        x = math.atan2(-R_mat[1, 2], R_mat[2, 2])
        z = math.atan2(-R_mat[0, 1], R_mat[0, 0])
    else:
        y = math.copysign(math.pi / 2, sy)
        x = math.atan2(R_mat[2, 1], R_mat[1, 1])
        z = 0.0
    return float(x), float(y), float(z)

def euler_to_quat(euler_angles: Tuple[float, float, float]) -> np.ndarray:
    """
    Converting Euler angles to quaternion [x, y, z, w]
    """
    return R.from_euler('xyz', euler_angles, degrees=False).as_quat()


def euler_to_6d(euler_angles: Tuple[float, float, float]) -> np.ndarray:
    """
    Converting Euler angles to a 6D rotation representation.
    """
    R_mat = R.from_euler('xyz', euler_angles, degrees=False).as_matrix()
    return np.concatenate([R_mat[:, 0], R_mat[:, 1]])


class BOPSingleObjDataset(Dataset):
    """
    Dataset for single-object 6D pose estimation on BOP data and Returning tuple: (orig_img, aug_img or None, label_dict, meta).
    """
    def __init__(self,root_dir: Path,scene_ids: List[int],cam_ids: List[str],
        target_obj_id: int,target_size: int = 256,augment: bool = False,split: str = "train",
        max_per_scene: Optional[int] = None,train_ratio: float = 0.8,seed: int = 42,use_real_val: bool = False):

        super().__init__()
        self.root_dir = root_dir
        self.scene_ids = scene_ids
        self.cam_ids = cam_ids
        self.obj_id = target_obj_id
        self.target_size = target_size
        self.augment = augment
        self.split = split.lower()
        self.max_per_scene = max_per_scene
        self.train_ratio = train_ratio
        random.seed(seed)

        # Choosing dataset path
        base_dir = self._select_dataset_dir(use_real_val)
        self.samples = self._collect_samples(base_dir)
        self.samples = self._split_samples(self.samples)
        logging.info(f"{self.__class__.__name__}(split={self.split}): {len(self.samples)} samples")

    def _select_dataset_dir(self, use_real_val: bool) -> Path:
        if self.split == "val" and use_real_val:
            real_dir = self.root_dir / "val"
            if real_dir.exists():
                return real_dir
            logging.warning(f"Real val dir not found: {real_dir}, using train_pbr.")
        train_pbr = self.root_dir / "train_pbr"

        if train_pbr.exists():
            return train_pbr
        raise FileNotFoundError(f"Dataset not found in {self.root_dir}")

    def _collect_samples(self, base_dir: Path) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for sid in self.scene_ids:
            scene_dir = base_dir / str(sid)
            if not scene_dir.is_dir():
                continue
            count = 0

            for cam in self.cam_ids:
                info_path = scene_dir / f"scene_gt_info_{cam}.json"
                pose_path = scene_dir / f"scene_gt_{cam}.json"
                cam_path  = scene_dir / f"scene_camera_{cam}.json"
                rgb_dir   = scene_dir / f"rgb_{cam}"

                if not all(p.exists() for p in [info_path, pose_path, cam_path, rgb_dir]):
                    continue
                info = json.load(info_path.open())
                poses = json.load(pose_path.open())
                cams  = json.load(cam_path.open())

                for im_id_s, inf in info.items():
                    if im_id_s not in cams:
                        continue
                    im_id = int(im_id_s)
                    K = np.array(cams[im_id_s]['cam_K'], dtype=np.float32).reshape(3, 3)
                    img_file = next((rgb_dir / f"{im_id:06d}.{ext}").with_suffix(ext) for ext in ['jpg','png'] if (rgb_dir / f"{im_id:06d}.{ext}").exists())

                    for inf_obj, pose_obj in zip(inf, poses[im_id_s]):
                        if pose_obj['obj_id'] != self.obj_id:
                            continue
                        x,y,w_,h_ = inf_obj['bbox_visib']
                        if w_ <= 0 or h_ <= 0:
                            continue
                        if inf_obj.get('visib_fract',1.0) < 0.1 or inf_obj.get('px_count_valid',w_*h_) < 1000:
                            continue
                        R_mat = np.array(pose_obj['cam_R_m2c'], dtype=np.float32).reshape(3,3)
                        t     = np.array(pose_obj['cam_t_m2c'], dtype=np.float32).reshape(3,1)
                        samples.append({
                            'img_path': str(img_file), 'K': K, 'R': R_mat, 't': t,
                            'bbox': (int(x),int(y),int(w_),int(h_)), 'scene': sid, 'cam': cam, 'im_id': im_id
                        })
                count += 1
                if self.max_per_scene and count >= self.max_per_scene:
                    break
        return samples

    def _split_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        from collections import defaultdict
        groups = defaultdict(list)

        for s in samples:
            groups[(s['scene'], s['cam'])].append(s)
        final = []

        for grp in groups.values():
            random.shuffle(grp)
            n = int(round(self.train_ratio * len(grp)))
            if self.split == 'train':
                final.extend(grp[:n])
            else:
                final.extend(grp[n:])
        return final

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        entry = self.samples[idx]
        img = cv2.imread(entry['img_path'])

        if img is None:
            raise IOError(f"Failed to load {entry['img_path']}")
        
        x,y,w,h = entry['bbox']
        crop = img[y:y+h, x:x+w]
        orig_img, _, dx, dy = letterbox_preserving_aspect_ratio(crop, self.target_size)
        orig = torch.from_numpy(orig_img).permute(2,0,1).float() / 255.0
        orig = TF.normalize(orig, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        aug = None

        if self.split=='train' and self.augment:
            aug = orig.clone()
        Rx,Ry,Rz = matrix_to_euler_xyz(entry['R'])
        euler = np.array([Rx,Ry,Rz], dtype=np.float32)
        labels = {'euler': torch.from_numpy(euler),
            'quat' : torch.from_numpy(euler_to_quat(tuple(euler))),
            '6d'   : torch.from_numpy(euler_to_6d(tuple(euler)))}
        meta = {'scene':entry['scene'], 'cam':entry['cam'], 'im_id': entry['im_id']}
        return orig, aug, labels, meta


def bop_collate_fn(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]]):
    """
    Collate for pose estimation and stack originals and augmentations if any
    """
    origs, augs, lbls, metas = zip(*batch)
    imgs = torch.stack(origs)
    if any(aug is not None for aug in augs):
        aug_tensors = [aug for aug in augs if aug is not None]
        imgs = torch.cat([imgs, torch.stack(aug_tensors)], dim=0)
        lbls = list(lbls) + list(lbls)
        metas = list(metas) + list(metas)
    return imgs, lbls, metas


def render_mask(mesh, K: np.ndarray, camera_pose: np.ndarray, imsize: Tuple[int,int], mesh_poses: List[np.ndarray]):
    """
    Rendering object masks via offscreen pyrender.
    """
    scene = pyrender.Scene(bg_color=[0,0,0])
    for mp in mesh_poses:
        scene.add(pyrender.Mesh.from_trimesh(mesh), pose=mp)
    cam = pyrender.IntrinsicsCamera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
    cam_pose = np.linalg.inv(camera_pose)
    scene.add(cam, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=5.0), pose=cam_pose)
    r = pyrender.OffscreenRenderer(imsize[0], imsize[1])
    color, depth = r.render(scene)
    return color, depth


def load_gt_poses(scene_dir: Path, cam_ids: List[str], image_id: int, obj_id: int) -> List[np.ndarray]:
    """
    Loading ground-truth poses as 4x4 matrices for a given image.
    """
    poses = []
    for cam in cam_ids:
        gt = json.load((scene_dir / f"scene_gt_{cam}.json").open())
        info = json.load((scene_dir / f"scene_gt_info_{cam}.json").open())
        key = str(image_id)
        if key not in gt or key not in info:
            continue
        for obj, bbox in zip(gt[key], info[key]):
            if obj['obj_id']!=obj_id:
                continue
            Rm = np.array(obj['cam_R_m2c'],dtype=np.float32).reshape(3,3)
            tm = np.array(obj['cam_t_m2c'],dtype=np.float32)
            pose = np.eye(4, dtype=np.float32)
            pose[:3,:3] = Rm; pose[:3,3]=tm
            poses.append(pose)
    return poses

class Capture:
    """
    Container for multi-view captures and poses
    """
    def __init__(self, images: List[np.ndarray], Ks: List[np.ndarray], RTs: List[np.ndarray], obj_id: int, gt_poses: Optional[np.ndarray]=None):
        self.images = images
        self.Ks = Ks
        self.RTs = RTs
        self.obj_id = obj_id
        self.gt_poses = None
        if gt_poses is not None:
            self.gt_poses = np.linalg.inv(RTs[0]) @ gt_poses

    @classmethod
    def from_dir(cls,scene_dir: Path,cam_ids: List[str],image_id: int,obj_id: int) -> 'Capture':
        
        cam_params = load_camera_params(str(scene_dir), cam_ids)
        Ks = [cam_params[c]['K'][image_id] for c in cam_ids]
        Rs = [cam_params[c]['R'][image_id] for c in cam_ids]
        Ts = [cam_params[c]['t'][image_id] for c in cam_ids]
        RTs = [np.vstack((np.hstack((r, t.reshape(3,1))),np.array([0,0,0,1],dtype=np.float32))) for r,t in zip(Rs,Ts)]
        images = [cv2.imread(str(p)) for p in sum([list(scene_dir.glob(f"rgb_{c}/{image_id:06d}.*")) for c in cam_ids], [])]
        gt = load_gt_poses(scene_dir, cam_ids, image_id, obj_id)
        return cls(images, Ks, RTs, obj_id, gt)