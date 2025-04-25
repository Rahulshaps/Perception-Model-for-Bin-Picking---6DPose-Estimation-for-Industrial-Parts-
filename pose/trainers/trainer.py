#!/usr/bin/env python3
"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code implements the training and validation loops for a 6D pose
estimation model using PyTorch. It also handles checkpointing, TensorBoard logging, and performance tracking.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def batch_labels(label_list: list[Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Collate individual label dictionaries into a batched dict of tensors.

    Args are label_list: List of dictionaries mapping label names to tensors and
    device: Compute device to move tensors onto.

    Returns a dictionary of batched tensors on the specified device.
    """
    batched: Dict[str, torch.Tensor] = {}
    for key in label_list[0]:
        batched[key] = torch.stack([lab[key] for lab in label_list], dim=0).to(device)
    return batched


def train_pose_estimation(model: nn.Module,train_loader: DataLoader,val_loader: DataLoader,
    criterion: nn.Module,optimizer: optim.Optimizer,scheduler: optim.lr_scheduler._LRScheduler,epochs: int = 10,
    out_dir: Path = Path('checkpoints'),device: Optional[torch.device] = None,resume: bool = False) -> nn.Module:
    """
    This executes training and validation loops, log metrics, and manage checkpoints.

    Args are model: PyTorch model to train, train_loader: DataLoader for training data,
    val_loader: DataLoader for validation data, criterion: Loss module returning (loss, metrics dict),
    optimizer: Optimizer instance, scheduler: Learning rate scheduler, epochs: Total number of epochs.
    out_dir: Directory for saving checkpoints and logs, device: Compute device; defaults to CUDA,
    resume: resume from last checkpoint

    Returns the trained model (with best weights saved separately).
    """
    # Setting up device, output directory, and logging
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / 'logs'))
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    best_val_metric = float('inf')
    start_epoch = 1
    best_model_file = out_dir / 'best_model.pth'
    last_checkpoint = out_dir / 'last_checkpoint.pth'

    # Resuming checkpoint
    if resume and last_checkpoint.exists():
        logging.info(f"Resuming training from {last_checkpoint}")
        ckpt = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_metric = ckpt.get('best_val_metric', best_val_metric)

    model.to(device)

    # Epoch looping
    for epoch in range(start_epoch, epochs + 1):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_metric_sum = 0.0
        train_steps = 0
        total_images = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for imgs, labels, _ in train_bar:
            imgs = imgs.to(device)
            batched = batch_labels(labels, device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss, metrics = criterion(batched, outputs)
            loss.backward()
            optimizer.step()

            train_loss_sum += metrics['rot_loss'].item()
            train_metric_sum += metrics['rot_deg_mean'].item()
            train_steps += 1
            total_images += imgs.size(0)

            train_bar.set_postfix({'loss': f"{metrics['rot_loss'].item():.3f}",'deg' : f"{metrics['rot_deg_mean'].item():.2f}"})

        avg_train_loss = train_loss_sum / train_steps
        avg_train_metric = train_metric_sum / train_steps
        writer.add_scalar('train/rot_loss', avg_train_loss, epoch)
        writer.add_scalar('train/rot_deg_mean', avg_train_metric, epoch)
        logging.info(f"Epoch {epoch}: Train loss={avg_train_loss:.3f}, deg={avg_train_metric:.2f}, imgs={total_images}")

        # Validation of the model 
        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                batched = batch_labels(labels, device)
                loss, metrics = criterion(batched, model(imgs))
                val_loss_sum += metrics['rot_loss'].item()
                val_metric_sum += metrics['rot_deg_mean'].item()
                val_steps += 1

        avg_val_loss = val_loss_sum / val_steps
        avg_val_metric = val_metric_sum / val_steps
        writer.add_scalar('val/rot_loss', avg_val_loss, epoch)
        writer.add_scalar('val/rot_deg_mean', avg_val_metric, epoch)
        logging.info(f"Epoch {epoch}: Val   loss={avg_val_loss:.3f}, deg={avg_val_metric:.2f}")

        # Scheduler and checkpointing
        scheduler.step()
        ckpt_data = {'epoch': epoch,'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),'best_val_metric': best_val_metric}
        torch.save(ckpt_data, last_checkpoint)

        if avg_val_metric < best_val_metric:
            best_val_metric = avg_val_metric
            torch.save(model.state_dict(), best_model_file)
            logging.info(f"New best model saved at epoch {epoch} with deg={best_val_metric:.2f}")

    #Last Final model
    final_model = out_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model)
    logging.info(f"Training complete. Final model saved to {final_model}")

    writer.close()
    return model
