import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

class DiceLoss(nn.Module):
    def __init__(self, num_classes, dims):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.dims = dims

    def forward(self, pred, target):
        epsilon = 1e-6
        pred = F.softmax(pred, dim=1)

        if self.dims == 2:
            target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        else:
            target_one_hot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()

        dice_loss = 0
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = target_one_hot[:, i]

            intersection = torch.sum(pred_i * target_i)
            union = torch.sum(pred_i) + torch.sum(target_i)

            dice_score = (2 * intersection + epsilon) / (union + epsilon)
            dice_loss = dice_loss + 1 - dice_score
        dice_loss = dice_loss / self.num_classes

        return dice_loss

class DiceCELoss(nn.Module):
    def __init__(self, num_classes, dims):
        super(DiceCELoss, self).__init__()
        self.dice_loss = DiceLoss(num_classes, dims)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        ce_loss = self.ce_loss(pred, target.long())
        return 0.5 * dice_loss + 0.5 * ce_loss