import numpy as np
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
from . import backbone, InpaintingLoss

class PartialCompletionContent(nn.Module):

    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(PartialCompletionContent, self).__init__()
        self.params = params
        self.with_modal = params.get('with_modal', False)

        # model
        self.model = backbone.__dict__[params['backbone_arch']](**params['backbone_param'])
        if load_pretrain is not None:
            assert load_pretrain.endswith('.pth'), "load_pretrain should end with .pth"
            utils.load_weights(load_pretrain, self.model)

        self.model.cuda()

        if dist_model:
            self.model = utils.DistModule(self.model)
            self.world_size = dist.get_world_size()
        else:
            self.model = backbone.FixModule(self.model)
            self.world_size = 1

        # optim
        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=params['lr'])

        # loss
        self.criterion = InpaintingLoss(backbone.VGG16FeatureExtractor()).cuda()

        cudnn.benchmark = True

    def set_input(self, rgb, modal, visible_mask, rgb_gt=None):
        self.rgb = rgb.cuda()
        self.modal = modal.cuda()
        self.visible_mask3 = visible_mask.repeat(
            1, 3, 1, 1).cuda()
        if self.with_modal:
            self.visible_mask4 = visible_mask.repeat(
                1, 4, 1, 1).cuda()
        if rgb_gt is not None:
            self.rgb_gt = rgb_gt.cuda()

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            if self.with_modal:
                output, _ = self.model(torch.cat([self.rgb, self.modal], dim=1),
                                       self.visible_mask4)
            else:
                output, _ = self.model(self.rgb, self.visible_mask3)
            if output.shape[2] != self.rgb.shape[2]:
                output = nn.functional.interpolate(
                    output, size=self.rgb.shape[2:4],
                    mode="bilinear", align_corners=True)
            output_comp = self.visible_mask3 * self.rgb + (1 - self.visible_mask3) * output
        ret_tensors = {'common_tensors': [self.rgb, output_comp, self.rgb_gt],
                       'mask_tensors': [self.modal, self.visible_mask3]}
        if ret_loss:
            loss_dict = self.criterion(self.rgb, self.visible_mask3, output, self.rgb_gt)
            for k in loss_dict.keys():
                loss_dict[k] /= self.world_size
            return ret_tensors, loss_dict
        else:
            return ret_tensors

    def step(self):
        if self.with_modal:
            output, _ = self.model(torch.cat([self.rgb, self.modal], dim=1),
                                   self.visible_mask4)
        else:
            output, _ = self.model(self.rgb, self.visible_mask3)
        if output.shape[2] != self.rgb.shape[2]:
            output = nn.functional.interpolate(
                output, size=self.rgb.shape[2:4],
                mode="bilinear", align_corners=True)
        loss_dict = self.criterion(self.rgb, self.visible_mask3, output, self.rgb_gt)
        for k in loss_dict.keys():
            loss_dict[k] /= self.world_size
        loss = 0.0
        for key, coef in self.params['lambda_dict'].items():
            value = coef * loss_dict[key]
            loss += value
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return loss_dict

    def load_state(self, path, Iter, resume=False):
        path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))

        if resume:
            utils.load_state(path, self.model, self.optim)
        else:
            utils.load_state(path, self.model)

    def save_state(self, path, Iter):
        path = os.path.join(path, "ckpt_iter_{}.pth.tar".format(Iter))
        torch.save({
            'step': Iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}, path)

    def switch_to(self, phase):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
