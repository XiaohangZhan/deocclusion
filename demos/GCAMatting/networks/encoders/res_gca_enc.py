import torch.nn as nn
import torch.nn.functional as F

from   ...utils import CONFIG
from   ..encoders.resnet_enc import ResNet_D
from   ..ops import GuidedCxtAtten, SpectralNorm


class ResGuidedCxtAtten(ResNet_D):

    def __init__(self, block, layers, norm_layer=None, late_downsample=False):
        super(ResGuidedCxtAtten, self).__init__(block, layers, norm_layer, late_downsample=late_downsample)
        first_inplane = 3 + CONFIG.model.trimap_channel
        self.shortcut_inplane = [first_inplane, self.midplanes, 64, 128, 256]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))

        self.guidance_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(16),
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(32),
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(32, 128, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(128)
        )

        self.gca = GuidedCxtAtten(128, 128)

        # initialize guidance head
        for layers in range(len(self.guidance_head)):
            m = self.guidance_head[layers]
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes)
        )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.activation(out) # N x 32 x 256 x 256
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.activation(out)

        im_fea = self.guidance_head(x[:,:3,...]) # downsample origin image and extract features
        if CONFIG.model.trimap_channel == 3:
            unknown = F.interpolate(x[:,4:5,...], scale_factor=1/8, mode='nearest')
        else:
            unknown = F.interpolate(x[:,3:,...].eq(1.).float(), scale_factor=1/8, mode='nearest')

        x2 = self.layer1(out) # N x 64 x 128 x 128
        x3= self.layer2(x2) # N x 128 x 64 x 64
        x3, offset = self.gca(im_fea, x3, unknown) # contextual attention
        x4 = self.layer3(x3) # N x 256 x 32 x 32
        out = self.layer_bottleneck(x4) # N x 512 x 16 x 16

        fea1 = self.shortcut[0](x) # input image and trimap
        fea2 = self.shortcut[1](x1)
        fea3 = self.shortcut[2](x2)
        fea4 = self.shortcut[3](x3)
        fea5 = self.shortcut[4](x4)

        return out, {'shortcut':(fea1, fea2, fea3, fea4, fea5),
                     'image_fea':im_fea,
                     'unknown':unknown,
                     'offset_1':offset}


if __name__ == "__main__":
    from networks.encoders.resnet_enc import BasicBlock
    m = ResGuidedCxtAtten(BasicBlock, [3, 4, 4, 2])
    for m in m.modules():
        print(m)
