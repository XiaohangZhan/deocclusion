# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import inconv, down, up, outconv
from .. import resnet

class UNetResNet(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2):
        super(UNetResNet, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(128 * w))
        self.image_encoder = resnet.resnet18(pretrained=True)
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(self.image_encoder.out_dim, 128 * w, kernel_size=1),
            nn.BatchNorm2d(128 * w),
            nn.ReLU(inplace=True))
        self.up1 = up(int(384 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

    def forward(self, x, rgb):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        img_feat = self.image_encoder(rgb)
        img_feat = self.reduce_dim(img_feat)
        img_feat = F.interpolate(
            img_feat, size=(x5.size(2), x5.size(3)), mode='bilinear', align_corners=True)
        cat = torch.cat((x5, img_feat), dim=1) # 256 + 128 * w
        x = self.up1(cat, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

def unet05res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=0.5, **kwargs)

def unet025res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=0.25, **kwargs)

def unet1res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=1, **kwargs)

def unet2res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=2, **kwargs)

def unet4res(in_channels, **kwargs):
    return UNetResNet(in_channels, w=4, **kwargs)
