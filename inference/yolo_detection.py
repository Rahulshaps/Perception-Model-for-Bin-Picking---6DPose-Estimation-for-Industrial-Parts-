#!/usr/bin/env python3
"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code  is for YOLO-based detection utility for multi-view images in the Industrial Plentopic Dataset (IPD).
Supports JPG and PNG formats, returns bounding boxes and centers.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import cv2
from ultralytics import YOLO

# Configuring logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')


def detect_with_yolo(scene_dir: Path,cam_ids: List[str],
    image_id: int,yolo_model_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    This function run YOLO inference on each camera view for a given image ID.

    Args are scene_dir: Root directory of a scene containing rgb_<cam_id> subfolders, 
    cam_ids: List of camera identifiers (e.g., ['cam1','cam2','cam3']) and 
    image_id: Image index to load (zero-padded to 6 digits) and 
    yolo_model_path: Path to the YOLO weights file.

    Returns a dictionary mapping each cam_id to a list of detections.
    """
    # Loading YOLO model once
    model = YOLO(str(yolo_model_path))
    detections: Dict[str, List[Dict[str, Any]]] = {}

    for cid in cam_ids:
        cam_folder = scene_dir / f"rgb_{cid}"
        img_path_jpg = cam_folder / f"{image_id:06d}.jpg"
        img_path_png = cam_folder / f"{image_id:06d}.png"

        # finding existing image file
        if img_path_jpg.is_file():
            img_path = img_path_jpg
        elif img_path_png.is_file():
            img_path = img_path_png
        else:
            logging.warning(f"No image found for cam '{cid}' at ID {image_id}")
            detections[cid] = []
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logging.error(f"Failed to load image: {img_path}")
            detections[cid] = []
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = model(img_rgb)[0]
        cam_dets: List[Dict[str, Any]] = []

        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            cx = float((x1 + x2) * 0.5)
            cy = float((y1 + y2) * 0.5)
            cam_dets.append({
                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                'bb_center': (cx, cy)
            })

        logging.info(f"cam '{cid}': {len(cam_dets)} detections")
        detections[cid] = cam_dets

    return detections
