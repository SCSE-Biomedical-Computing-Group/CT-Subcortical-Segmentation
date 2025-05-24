import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Dataset3D(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.long)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

class Dataset2D(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, volume_sampling = 1.0):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        self.transform = transform
        self.volume_sampling = volume_sampling
        self.slice_indices = self._get_slice_indices()

    def _get_slice_indices(self):
        slice_indices = []
        volume_sample_size = round(len(self.img_files) * self.volume_sampling)
        sample_indices = set(random.sample([i for i in range(len(self.img_files))], volume_sample_size))
        img_files = [self.img_files[i] for i in range(len(self.img_files)) if i in sample_indices]
        mask_files = [self.mask_files[i] for i in range(len(self.mask_files)) if i in sample_indices]
        for img_file, mask_file in zip(img_files, mask_files):
            img_path = os.path.join(self.img_dir, img_file)
            img = nib.load(img_path).get_fdata()
            num_slices = img.shape[2]
            slice_indices.extend([(img_file, mask_file, i) for i in range(num_slices)])
        return slice_indices

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        img_file, mask_file, slice_idx = self.slice_indices[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        img = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        img_slice = img[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

        img_slice = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0)
        mask_slice = torch.tensor(mask_slice, dtype=torch.long)

        if self.transform:
            img_slice = self.transform(img_slice)

        return img_slice, mask_slice