# pose/models/simple_pose_net.py
"""
Author: Rahul Sha Pathepur Shankar, Vaishnav Raja & Reza Farrokhi Saray
CS 5330 Pattern Recognition and Computer Vision - Spring 2025 - Final Project

This code implements a simple pose estimation network using a ResNet50 backbone
for feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

class SimplePoseNet(nn.Module):
    def __init__(self, loss_type="euler", pretrained=True):
        """
        Args are  loss_type: one of "euler", "quat", or "6d". Determines the number of output neurons and 
        pretrained: use pretrained ResNet50 weight for inference.
        """
        super(SimplePoseNet, self).__init__()
        # Loading a ResNet50 backbone.
        backbone = tv_models.resnet50(weights=(tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None))
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        
        # Determining the output dimension based on the loss type
        if loss_type == "euler":
            out_dim = 3
        elif loss_type == "quat":
            out_dim = 4
        elif loss_type == "6d":
            out_dim = 6
        else:
            raise ValueError("loss_type must be one of 'euler', 'quat', or '6d'")
        
        self.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        preds = self.fc(feats)
        return preds