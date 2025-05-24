import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.enc1 = self.double_conv(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self.double_conv(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self.double_conv(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = self.double_conv(128, 256)
        self.pool4 = nn.MaxPool3d(2)
        
        self.bottleneck = self.double_conv(256, 512)

        self.up4 = self.upsampling(512, 256)
        self.dec4 = self.double_conv(512, 256)
        self.up3 = self.upsampling(256, 128)
        self.dec3 = self.double_conv(256, 128)
        self.up2 = self.upsampling(128, 64)
        self.dec2 = self.double_conv(128, 64)
        self.up1 = self.upsampling(64, 32)
        self.dec1 = self.double_conv(64, 32)

        self.final = nn.Conv3d(32, out_channels, kernel_size=1)
    

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upsampling(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))

        up4 = self.up4(bottleneck)
        up4 = F.interpolate(up4, size=enc4.shape[2:], mode="trilinear", align_corners=False)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.up3(dec4)
        up3 = F.interpolate(up3, size=enc3.shape[2:], mode="trilinear", align_corners=False)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        up2 = F.interpolate(up2, size=enc2.shape[2:], mode="trilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        up1 = F.interpolate(up1, size=enc1.shape[2:], mode="trilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final(dec1)