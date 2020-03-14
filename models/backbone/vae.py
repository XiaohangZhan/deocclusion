import torch
import torch.nn as nn

class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class AE(nn.Module):

    def __init__(self, in_channels=3, w=4, latent_dim=256, n_classes=2):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            inconv(in_channels, 16 * w),
            down(16 * w, 32 * w),
            down(32 * w, 64 * w),
            down(64 * w, 64 * w),
            nn.MaxPool2d(2)
        )
        self.enc_linear = nn.Sequential(
            nn.Linear(16384 * w, latent_dim),
            nn.ReLU(inplace=True)
        )
        self.dec_linear = nn.Sequential(
            nn.Linear(latent_dim, 16384 * w),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            up(64 * w, 32 * w),
            up(32 * w, 16 * w),
            up(16 * w, n_classes)
        )
        self.head_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.enc_linear(x)
        x = self.dec_linear(x)
        x = x.view(x.size(0), -1, 16, 16)
        x = self.decoder(x)
        x = self.head_up(x)
        return x


class VAE(nn.Module):

    def __init__(self, in_channels=3, w=4, latent_dim=256, n_classes=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            inconv(in_channels, 16 * w),
            down(16 * w, 32 * w),
            down(32 * w, 64 * w),
            down(64 * w, 64 * w),
            nn.MaxPool2d(2)
        )
        self.mean_linear = nn.Linear(16384 * w, latent_dim)
        self.var_linear = nn.Linear(16384 * w, latent_dim)

        self.dec_linear = nn.Sequential(
            nn.Linear(latent_dim, 16384 * w),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            up(64 * w, 32 * w),
            up(32 * w, 16 * w),
            up(16 * w, n_classes)
        )
        self.head_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.eps = torch.randn([1, latent_dim]).cuda()

    def forward(self, x):
        x = self.encoder(x)
        bs = x.size(0)
        x = x.view(x.size(0), -1)
        mean = self.mean_linear(x)
        logvar = self.var_linear(x)
        std = torch.exp(0.5 * logvar)
        eps = self.eps.repeat(bs, 1)
        eps.normal_()
        z = eps * std + mean
        x = self.dec_linear(z)
        x = x.view(x.size(0), -1, 16, 16)
        x = self.decoder(x)
        x = self.head_up(x)
        return x, mean, logvar


def AE256(in_channels, **kwargs):
    return AE(in_channels, w=1, latent_dim=256, **kwargs)

def AE32(in_channels, **kwargs):
    return AE(in_channels, w=1, latent_dim=32, **kwargs)

def VAE32(in_channels, **kwargs):
    return VAE(in_channels, w=1, latent_dim=32, **kwargs)
