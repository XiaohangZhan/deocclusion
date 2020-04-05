import torch
from   torch import nn
from   torch.nn import Parameter
from   torch.autograd import Variable
from   torch.nn import functional as F


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
    and add _noupdate_u_v() for evaluation
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        # if torch.is_grad_enabled() and self.module.training:
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)


class GuidedCxtAtten(nn.Module):
    # based on https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/blob/a6fe298fec502bfd9cbc64eb01e39f78a3262a59/models/DeepFill_Models/ops.py#L210
    def __init__(self, out_channels, guidance_channels, rate=2):
        super(GuidedCxtAtten, self).__init__()
        self.rate = rate
        self.padding = nn.ReflectionPad2d(1)
        self.up_sample = nn.Upsample(scale_factor=self.rate, mode='nearest')

        self.guidance_conv = nn.Conv2d(in_channels=guidance_channels, out_channels=guidance_channels//2,
                                       kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
            )

        nn.init.xavier_uniform_(self.guidance_conv.weight)
        nn.init.constant_(self.guidance_conv.bias, 0)
        nn.init.xavier_uniform_(self.W[0].weight)
        nn.init.constant_(self.W[1].weight, 1e-3)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, f, alpha, unknown=None, ksize=3, stride=1, fuse_k=3, softmax_scale=1., training=True):

        f = self.guidance_conv(f)
        # get shapes
        raw_int_fs = list(f.size()) # N x 64 x 64 x 64
        raw_int_alpha = list(alpha.size()) # N x 128 x 64 x 64

        # extract patches from background with stride and rate
        kernel = 2*self.rate
        alpha_w = self.extract_patches(alpha, kernel=kernel, stride=self.rate)
        alpha_w = alpha_w.permute(0, 2, 3, 4, 5, 1)
        alpha_w = alpha_w.contiguous().view(raw_int_alpha[0], raw_int_alpha[2] // self.rate, raw_int_alpha[3] // self.rate, -1)
        alpha_w = alpha_w.contiguous().view(raw_int_alpha[0], -1, kernel, kernel, raw_int_alpha[1])
        alpha_w = alpha_w.permute(0, 1, 4, 2, 3)

        f = F.interpolate(f, scale_factor=1/self.rate, mode='nearest')

        fs = f.size() # B x 64 x 32 x 32
        f_groups = torch.split(f, 1, dim=0) # Split tensors by batch dimension; tuple is returned

        # from b(B*H*W*C) to w(b*k*k*c*h*w)
        int_fs = list(fs)
        w = self.extract_patches(f)
        w = w.permute(0, 2, 3, 4, 5, 1)
        w = w.contiguous().view(raw_int_fs[0], raw_int_fs[2] // self.rate, raw_int_fs[3] // self.rate, -1)
        w = w.contiguous().view(raw_int_fs[0], -1, ksize, ksize, raw_int_fs[1])
        w = w.permute(0, 1, 4, 2, 3)
        # process mask

        if unknown is not None:
            unknown = unknown.clone()
            unknown = F.interpolate(unknown, scale_factor=1/self.rate, mode='nearest')
            assert unknown.size(2) == f.size(2), "mask should have same size as f at dim 2,3"
            unknown_mean = unknown.mean(dim=[2,3])
            known_mean = 1 - unknown_mean
            unknown_scale = torch.clamp(torch.sqrt(unknown_mean / known_mean), 0.1, 10).to(alpha)
            known_scale = torch.clamp(torch.sqrt(known_mean / unknown_mean), 0.1, 10).to(alpha)
            softmax_scale = torch.cat([unknown_scale, known_scale], dim=1)
        else:
            unknown = torch.ones([fs[0], 1, fs[2], fs[3]]).to(alpha)
            softmax_scale = torch.FloatTensor([softmax_scale, softmax_scale]).view(1,2).repeat(fs[0],1).to(alpha)

        m = self.extract_patches(unknown)

        m = m.permute(0, 2, 3, 4, 5, 1)
        m = m.contiguous().view(raw_int_fs[0], raw_int_fs[2]//self.rate, raw_int_fs[3]//self.rate, -1)
        m = m.contiguous().view(raw_int_fs[0], -1, ksize, ksize)

        m = self.reduce_mean(m) # smoothing, maybe
        # mask out the
        mm = m.gt(0.).float() # (N, 32*32, 1, 1)

        # the correlation with itself should be 0
        self_mask = F.one_hot(torch.arange(fs[2] * fs[3]).view(fs[2], fs[3]).contiguous().to(alpha).long(),
                              num_classes=int_fs[2] * int_fs[3])
        self_mask = self_mask.permute(2, 0, 1).view(1, fs[2] * fs[3], fs[2], fs[3]).float() * (-1e4)

        w_groups = torch.split(w, 1, dim=0) # Split tensors by batch dimension; tuple is returned
        alpha_w_groups = torch.split(alpha_w, 1, dim=0) # Split tensors by batch dimension; tuple is returned
        mm_groups = torch.split(mm, 1, dim=0)
        scale_group = torch.split(softmax_scale, 1, dim=0)
        y = []
        offsets = []
        k = fuse_k
        y_test = []
        for xi, wi, alpha_wi, mmi, scale in zip(f_groups, w_groups, alpha_w_groups, mm_groups, scale_group):
            # conv for compare
            wi = wi[0]
            escape_NaN = Variable(torch.FloatTensor([1e-4])).to(alpha)
            wi_normed = wi / torch.max(self.l2_norm(wi), escape_NaN)
            xi = F.pad(xi, (1,1,1,1), mode='reflect')
            yi = F.conv2d(xi, wi_normed, stride=1, padding=0) # yi => (B=1, C=32*32, H=32, W=32)
            y_test.append(yi)
            # conv implementation for fuse scores to encourage large patches
            yi = yi.permute(0, 2, 3, 1)
            yi = yi.contiguous().view(1, fs[2], fs[3], fs[2] * fs[3])
            yi = yi.permute(0, 3, 1, 2) # (B=1, C=32*32, H=32, W=32)

            # softmax to match
            # scale the correlation with predicted scale factor for known and unknown area
            yi = yi * (scale[0,0] * mmi.gt(0.).float() + scale[0,1] * mmi.le(0.).float()) # mmi => (1, 32*32, 1, 1)
            # mask itself, self-mask only applied to unknown area
            yi = yi + self_mask * mmi  # self_mask: (1, 32*32, 32, 32)
            # for small input inference
            yi = F.softmax(yi, dim=1)

            _, offset = torch.max(yi, dim=1) # argmax; index
            offset = torch.stack([offset // fs[3], offset % fs[3]], dim=1)

            wi_center = alpha_wi[0]

            if self.rate == 1:
                left = (kernel) // 2
                right = (kernel - 1) // 2
                yi = F.pad(yi, (left, right, left, right), mode='reflect')
                wi_center = wi_center.permute(1, 0, 2, 3)
                yi = F.conv2d(yi, wi_center, padding=0) / 4. # (B=1, C=128, H=64, W=64)
            else:
                yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4. # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0) # back to the mini-batch
        y.contiguous().view(raw_int_alpha)
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view([int_fs[0]] + [2] + int_fs[2:])

        # # case1: visualize optical flow: minus current position
        # h_add = Variable(torch.arange(0,float(fs[2]))).to(alpha).view([1, 1, fs[2], 1])
        # h_add = h_add.expand(fs[0], 1, fs[2], fs[3])
        # w_add = Variable(torch.arange(0,float(fs[3]))).to(alpha).view([1, 1, 1, fs[3]])
        # w_add = w_add.expand(fs[0], 1, fs[2], fs[3])
        #
        # offsets = offsets - torch.cat([h_add, w_add], dim=1).long()

        # case2: visualize absolute position
        offsets = offsets - torch.Tensor([fs[2]//2, fs[3]//2]).view(1,2,1,1).to(alpha).long()

        y = self.W(y) + alpha

        return y, (offsets, softmax_scale)

    @staticmethod
    def extract_patches(x, kernel=3, stride=1):
        left =(kernel - stride + 1) // 2
        right =(kernel - stride) // 2
        x = F.pad(x, (left, right, left, right), mode='reflect')
        all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)

        return all_patches

    @staticmethod
    def reduce_mean(x):
        for i in range(4):
            if i <= 1:
                continue
            x = torch.mean(x, dim=i, keepdim=True)
        return x

    @staticmethod
    def l2_norm(x):
        def reduce_sum(x):
            for i in range(4):
                if i == 0:
                    continue
                x = torch.sum(x, dim=i, keepdim=True)
            return x

        x = x**2
        x = reduce_sum(x)
        return torch.sqrt(x)