from   ..ops import GuidedCxtAtten, SpectralNorm
from   .res_shortcut_dec import ResShortCut_D_Dec


class ResGuidedCxtAtten_Dec(ResShortCut_D_Dec):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False):
        super(ResGuidedCxtAtten_Dec, self).__init__(block, layers, norm_layer, large_kernel)
        self.gca = GuidedCxtAtten(128, 128)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        im = mid_fea['image_fea']
        x = self.layer1(x) + fea5 # N x 256 x 32 x 32
        x = self.layer2(x) + fea4 # N x 128 x 64 x 64
        x, offset = self.gca(im, x, mid_fea['unknown']) # contextual attention
        x = self.layer3(x) + fea3 # N x 64 x 128 x 128
        x = self.layer4(x) + fea2 # N x 32 x 256 x 256
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x = self.conv2(x)

        alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha, {'offset_1': mid_fea['offset_1'], 'offset_2': offset}

