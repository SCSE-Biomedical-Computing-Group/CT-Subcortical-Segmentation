import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import MODEL_CONFIG, DATASET_CONFIG
from loss import DiceLoss, DiceCELoss

def run_inference2D(model, test_dir, output_dir, device):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.nii.gz')])
    for img_file in img_files:
        img_path = os.path.join(test_dir, img_file)
        img = nib.load(img_path)
        img_data = img.get_fdata()

        pred_slices = []
        with torch.no_grad():
            for i in range(img_data.shape[2]):
                img_slice = torch.tensor(img_data[:, :, i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                output = model(img_slice)
                predicted_slice = torch.argmax(output, dim=1).cpu().numpy()[0]
                pred_slices.append(predicted_slice)

        pred_vol = np.stack(pred_slices, axis=-1).astype(np.int32)
        pred_img = nib.Nifti1Image(pred_vol, img.affine)
        nib.save(pred_img, os.path.join(output_dir, img_file))

def run_inference3D(model, test_dir, output_dir, device):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.nii.gz')])
    with torch.no_grad():
        for img_file in img_files:
            img_path = os.path.join(test_dir, img_file)
            img = nib.load(img_path)
            img_data = img.get_fdata()

            img_data = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(img_data)
            pred = torch.argmax(output, dim=1).long()

            pred = pred[0].cpu().numpy().astype(np.int16)
            pred_vol = nib.Nifti1Image(pred, affine=np.eye(4))
            output_path = os.path.join(output_dir, img_file)
            nib.save(pred_vol, output_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_model = MODEL_CONFIG["UNet2D"]
    num_classes = 8
    test_dir = "data/CT/test"

    model = selected_model(in_channels=1, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load("weights/UNet2D_best.pth"))
    torch.use_deterministic_algorithms(True)

    run_inference2D(model, test_dir, 'pred_results', device)