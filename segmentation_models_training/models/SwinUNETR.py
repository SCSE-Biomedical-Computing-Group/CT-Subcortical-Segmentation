import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETR_model(nn.Module):
    def __init__(self, in_channels, out_channels, resolution):
        super(SwinUNETR_model, self).__init__()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=resolution,
            feature_size=48,
            use_checkpoint=True,
            pretrained=True
        )
    
    def forward(self, x):
        return self.model(x)