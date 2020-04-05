import numpy as np

import torch
import torch.nn as nn

import utils
import inference as infer
from . import SingleStageModel
from . import MaskWeightedCrossEntropyLoss

import pdb

class PartialCompletionMask(SingleStageModel):

    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(PartialCompletionMask, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params['use_rgb']

        # loss
        self.criterion = MaskWeightedCrossEntropyLoss(
            inmask_weight=params['inmask_weight'],
            outmask_weight=1.)

    def set_input(self, rgb=None, mask=None, eraser=None, target=None):
        self.rgb = rgb.cuda()
        self.mask = mask.cuda()
        self.eraser = eraser.cuda()
        self.target = target.cuda()

    def evaluate(self, image, inmodal, category, bboxes, amodal, gt_order_matrix, input_size):
        order_method = self.params.get('order_method', 'ours')
        # order
        if order_method == 'ours':
            order_matrix = infer.infer_order2(
                self, image, inmodal, category, bboxes,
                use_rgb=self.use_rgb,
                th=self.params['inference']['positive_th_order'],
                dilate_kernel=self.params['inference'].get('dilate_kernel_order', 0),
                input_size=input_size,
                min_input_size=16,
                interp=self.params['inference']['order_interp'])
        elif order_method == 'hull':
            order_matrix = infer.infer_order_hull(inmodal)
        elif order_method == 'area':
            order_matrix = infer.infer_order_area(inmodal, above=self.params['above'])
        elif order_method == 'yaxis':
            order_matrix = infer.infer_order_yaxis(inmodal)
        else:
            raise Exception("No such method: {}".format(order_method))

        gt_order_matrix = infer.infer_gt_order(inmodal, amodal)
        allpair_true, allpair, occpair_true, occpair, show_err = infer.eval_order(
            order_matrix, gt_order_matrix)

        # amodal
        amodal_method = self.params.get('amodal_method', 'ours')
        if amodal_method == 'ours':
            amodal_patches_pred = infer.infer_amodal(
                self, image, inmodal, category, bboxes,
                order_matrix, use_rgb=self.use_rgb,
                th=self.params['inference']['positive_th_amodal'],
                dilate_kernel=self.params['inference'].get('dilate_kernel_amodal', 0),
                input_size=input_size,
                min_input_size=16, interp=self.params['inference']['amodal_interp'],
                order_grounded=self.params['inference']['order_grounded'])
            amodal_pred = infer.patch_to_fullimage(
                amodal_patches_pred, bboxes,
                image.shape[0], image.shape[1],
                interp=self.params['inference']['amodal_interp'])
        elif amodal_method == 'hull':
            amodal_pred = np.array(infer.infer_amodal_hull(
                inmodal, bboxes, order_matrix,
                order_grounded=self.params['inference']['order_grounded']))
        elif amodal_method == 'raw':
            amodal_pred = inmodal # evaluate raw
        else:
            raise Exception("No such method: {}".format(amodal_method))

        intersection = ((amodal_pred == 1) & (amodal == 1)).sum()
        union = ((amodal_pred == 1) | (amodal == 1)).sum()
        target = (amodal == 1).sum()

        return allpair_true, allpair, occpair_true, occpair, intersection, union, target

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            if self.use_rgb:
                output = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
            else:
                output = self.model(torch.cat([self.mask, self.eraser], dim=1))
            if output.shape[2] != self.mask.shape[2]:
                output = nn.functional.interpolate(
                    output, size=self.mask.shape[2:4],
                    mode="bilinear", align_corners=True)
        comp = output.argmax(dim=1, keepdim=True).float()
        comp[self.eraser == 0] = (self.mask > 0).float()[self.eraser == 0]

        vis_combo = (self.mask > 0).float()
        vis_combo[self.eraser == 1] = 0.5
        vis_target = self.target.cpu().clone().float()
        if vis_target.max().item() == 255:
            vis_target[vis_target == 255] = 0.5
        vis_target = vis_target.unsqueeze(1)
        if self.use_rgb:
            cm_tensors = [self.rgb]
        else:
            cm_tensors = []
        ret_tensors = {'common_tensors': cm_tensors,
                       'mask_tensors': [self.mask, vis_combo, comp, vis_target]}
        if ret_loss:
            loss = self.criterion(output, self.target, self.eraser.squeeze(1)) / self.world_size
            return ret_tensors, {'loss': loss}
        else:
            return ret_tensors

    def step(self):
        if self.use_rgb:
            output = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
        else:
            output = self.model(torch.cat([self.mask, self.eraser], dim=1))
        loss = self.criterion(output, self.target, self.eraser.squeeze(1)) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}
