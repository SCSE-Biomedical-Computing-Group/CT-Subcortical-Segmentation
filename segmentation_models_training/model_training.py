import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import MODEL_CONFIG, DATASET_CONFIG
from loss import DiceLoss, DiceCELoss

def train_model(model, criterion, train_loader, val_loader, num_classes, device):
    # Configuration
    num_epochs = 1000
    patience = 5
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    # Variable initialisation
    lowest_loss = float('inf')
    patience_count = 0

    # Training Loop
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        
        # Model training on training dataset
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            torch.cuda.empty_cache()
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss/len(train_loader)

        # Evaluation on validation dataset
        model.eval()
        val_loss = 0
        torch.cuda.empty_cache()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Print train and validation loss
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}")

        # Early stopping mechanism
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            # Reset patience count
            patience_count = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_count += 1

        if patience_count >= patience:
            print("Patience count exceed, early stopping enabled")
            break

if __name__ == "__main__":
    train_images = 'data/CT/train'
    train_masks = 'data/multiclass_classification_data/train'
    val_images = 'data/CT/val'
    val_masks = 'data/multiclass_classification_data/val'

    num_classes = 8
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crtierion = nn.CrossEntropyLoss() # CE
    weights = torch.tensor([0.000711, 0.373, 0.808, 1.73, 1.31, 3.56, 1.51, 0.567]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights) # Weighted CE
    criterion = DiceLoss(num_classes, 2) # 2D Dice
    criterion = DiceCELoss(num_classes, 2) # 2D DiceCE
    criterion = DiceLoss(num_classes, 3) # 3D Dice
    criterion = DiceCELoss(num_classes, 3) # 3D DiceCE

    selected_model = MODEL_CONFIG["UNet2D"]
    dataset = DATASET_CONFIG["2D"]

    train_dataset = dataset(train_images, train_masks)
    val_dataset = dataset(val_images, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = selected_model(in_channels=1, out_channels=num_classes).to(device)
    train_model(model, criterion, train_loader, val_loader, num_classes, device)