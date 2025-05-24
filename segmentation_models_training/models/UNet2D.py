import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2D, self).__init__()
        self.encoder1 = self.double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = self.double_conv(512, 1024)

        self.upconv4 = self.upsampling(1024, 512)
        self.decoder4 = self.double_conv(1024, 512)
        self.upconv3 = self.upsampling(512, 256)
        self.decoder3 = self.double_conv(512, 256)
        self.upconv2 = self.upsampling(256, 128)
        self.decoder2 = self.double_conv(256, 128)
        self.upconv1 = self.upsampling(128, 64)
        self.decoder1 = self.double_conv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upsampling(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))

        up4 = self.upconv4(bottleneck)
        up4 = F.interpolate(up4, size=encoder4.shape[2:], mode="trilinear", align_corners=False)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))
        up3 = self.upconv3(dec4)
        up3 = F.interpolate(up3, size=encoder3.shape[2:], mode="trilinear", align_corners=False)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))
        up2 = self.upconv2(decoder3)
        up2 = F.interpolate(up2, size=encoder2.shape[2:], mode="trilinear", align_corners=False)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))
        up1 = self.upconv1(decoder2)
        up1 = F.interpolate(up1, size=encoder1.shape[2:], mode="trilinear", align_corners=False)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

        return self.final(dec1)