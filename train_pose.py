"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code is part of the pose estimation training for the Industrial Plentopic Dataset (IPD).
This code is designed to train a pose estimation model for single object detection using the IPD dataset.
"""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils.data_utils import BOPSingleObjDataset, bop_collate_fn
from pose.models.simple_pose_net import SimplePoseNet
from pose.models.losses import EulerAnglePoseLoss, QuaternionPoseLoss, SixDPoseLoss
from pose.trainers.trainer import train_pose_estimation


def parse_arguments() -> argparse.Namespace:
    """
    Parsing and returning CLI arguments for pose training
    """
    parser = argparse.ArgumentParser(description="Training a pose estimation model on a Industrial Plentopic Dataset single-object dataset")
    parser.add_argument('--root_dir', type=Path, required=True,help='Dataset root containing train_pbr and optional val folders')
    parser.add_argument('--use_real_val', action='store_true',help='Use provided validation data instead of train split')
    parser.add_argument('--target_obj_id', type=int, default=11,help='IPD object ID to train on')
    parser.add_argument('--batch_size', type=int, default=32,help='Number of samples per batch')
    parser.add_argument('--epochs', type=int, default=100,help='Total number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,help='Workers for data loading')
    parser.add_argument('--checkpoints_dir', type=Path, default=Path('checkpoints'),help='Directory to save and load checkpoints')
    parser.add_argument('--resume', action='store_true',help='Resume training from last checkpoint if available')
    parser.add_argument('--loss_type', type=str, choices=['euler', 'quat', '6d'], default='euler',help='Rotation loss: Euler, quaternion, or 6D')
    return parser.parse_args()


def find_scene_ids(dir_path: Path) -> list[int]:
    """
    Returning sorted list of numeric scene IDs in the directory.
    """
    return sorted([int(p.name) for p in dir_path.iterdir() if p.name.isdigit()])


def setup_dataset(root: Path, scene_ids: list[int], obj_id: int, split: str, use_real_val: bool, ratio: float=0.8) -> BOPSingleObjDataset:
    """
    Construct a IPDSingleObjDataset for train or val.
    """
    params = dict(root_dir=root,scene_ids=scene_ids,cam_ids=['cam1','cam2','cam3'],target_obj_id=obj_id,
        target_size=256,augment=False,split=split,use_real_val=use_real_val)
    
    if split == 'train':
        params['train_ratio'] = ratio
    return BOPSingleObjDataset(**params)


def main():
    # Initialization logging
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Starting pose estimation training")
    args = parse_arguments()

    # verifying files structure
    train_dir = args.root_dir / 'train_pbr'
    # checking if the training directory exists and contains scene IDs 
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing training directory at {train_dir}")
    train_ids = find_scene_ids(train_dir)
    logging.info(f"Training scenes: {train_ids}")

    # validating split determination
    use_real = args.use_real_val and (args.root_dir / 'val').exists()
    # checking if the validation directory exists and contains scene IDs
    if use_real:
        val_dir = args.root_dir / 'val'
        val_ids = find_scene_ids(val_dir)
        logging.info(f"Validation scenes (real): {val_ids}")
    else:
        val_ids = train_ids
        logging.info("Using train split for validation")

    # setting up dataset and dataloaders for training and validation 
    train_ds = setup_dataset(args.root_dir, train_ids, args.target_obj_id, 'train', use_real, ratio=(1.0 if use_real else 0.8))
    val_ds   = setup_dataset(args.root_dir, val_ids,   args.target_obj_id, 'val',   use_real)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, collate_fn=bop_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers, collate_fn=bop_collate_fn)

    # model initialization and loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplePoseNet(loss_type=args.loss_type, pretrained=not args.resume).to(device)
    logging.info(f"Model initialized with {args.loss_type} loss")

    # loading pretrained weights if available
    ckpt_dir = args.checkpoints_dir / f"obj_{args.target_obj_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / 'last_checkpoint.pth'
    # loading checkpoint if resuming training
    if args.resume and ckpt_file.exists():
        logging.info(f"Resuming from {ckpt_file}")
        ckpt = torch.load(ckpt_file)
        model.load_state_dict(ckpt['model_state_dict'])

    # loss function selection and initialization 
    if args.loss_type == 'euler':
        criterion = EulerAnglePoseLoss()
    elif args.loss_type == 'quat':
        criterion = QuaternionPoseLoss()
    else:
        criterion = SixDPoseLoss()

    # setting up the optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=max(1, args.epochs//3), gamma=0.8)
    if args.resume and ckpt_file.exists():
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # training the model
    logging.info("Starting training")
    train_pose_estimation(
        model=model,train_loader=train_loader,
        val_loader=val_loader,criterion=criterion,
        optimizer=optimizer,scheduler=scheduler,
        epochs=args.epochs,out_dir=str(ckpt_dir),
        device=device,resume=args.resume)


if __name__ == '__main__':
    main()
