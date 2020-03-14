import numpy as np

import torch
import torch.nn as nn

import utils
import inference as infer
from . import SingleStageModel
from . import MaskWeightedCrossEntropyLoss


class Supervised(SingleStageModel):

    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(Supervised, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params.get("use_rgb", False)

        # loss
        self.criterion = nn.CrossEntropyLoss()

    def set_input(self, rgb=None, mask=None, target=None):
        self.rgb = rgb.cuda()
        self.mask = mask.cuda()
        self.target = target.cuda()

    def evaluate(self, image, inmodal, category, bboxes, amodal, gt_order_matrix, input_size):
        # amodal
        amodal_patches_pred = infer.infer_amodal_sup(
            self, image, inmodal, category, bboxes,
            use_rgb=self.use_rgb,
            th=self.params['inference']['positive_th_amodal'],
            input_size=input_size,
            min_input_size=16, interp=self.params['inference']['amodal_interp'])
        amodal_pred = infer.patch_to_fullimage(
            amodal_patches_pred, bboxes,
            image.shape[0], image.shape[1],
            interp=self.params['inference']['amodal_interp'])
        intersection = ((amodal_pred == 1) & (amodal == 1)).sum()
        union = ((amodal_pred == 1) | (amodal == 1)).sum()
        target = (amodal == 1).sum()

        return 0, 1, 0, 1, intersection, union, target

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            if self.use_rgb:
                output = self.model(self.mask, self.rgb)
            else:
                output = self.model(self.mask)
            if output.shape[2] != self.mask.shape[2]:
                output = nn.functional.interpolate(
                    output, size=self.mask.shape[2:4],
                    mode="bilinear", align_corners=True)
        comp = output.argmax(dim=1, keepdim=True).float()

        vis_target = self.target.cpu().clone().float()
        if vis_target.max().item() == 255:
            vis_target[vis_target == 255] = 0.5
        vis_target = vis_target.unsqueeze(1)
        if self.use_rgb:
            cm_tensors = [self.rgb]
        else:
            cm_tensors = []
        ret_tensors = {'common_tensors': cm_tensors,
                       'mask_tensors': [self.mask, comp, vis_target]}
        if ret_loss:
            loss = self.criterion(output, self.target) / self.world_size
            return ret_tensors, {'loss': loss}
        else:
            return ret_tensors

    def step(self):
        if self.use_rgb:
            output = self.model(self.mask, self.rgb)
        else:
            output = self.model(self.mask)
        loss = self.criterion(output, self.target) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}
