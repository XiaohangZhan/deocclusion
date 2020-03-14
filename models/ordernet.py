import numpy as np

import torch
import torch.nn as nn

import utils
import inference as infer
from . import SingleStageModel


class OrderNet(SingleStageModel):

    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(OrderNet, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params.get("use_rgb", False)

        # loss
        self.criterion = nn.CrossEntropyLoss()

    def set_input(self, rgb, modal1, modal2, target=None):
        self.rgb = rgb.cuda()
        self.modal1 = modal1.cuda()
        self.modal2 = modal2.cuda()
        self.target = target.cuda()

    def evaluate(self, image, inmodal, category, bboxes, amodal, gt_order_matrix, input_size):
        order_matrix = infer.infer_order_sup(
            self, image, inmodal, bboxes, use_rgb=self.use_rgb)
        gt_order_matrix = infer.infer_gt_order(inmodal, amodal)
        allpair_true, allpair, occpair_true, occpair, show_err = infer.eval_order(
            order_matrix, gt_order_matrix)

        return allpair_true, allpair, occpair_true, occpair, 0, 0, 0

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            if self.use_rgb:
                output = self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1))
            else:
                output = self.model(torch.cat([self.modal1, self.modal2], dim=1))

        acc = utils.accuracy(output, self.target, topk=(1,))[0]
        ret_tensors = {'common_tensors': [self.rgb],
                       'mask_tensors': [self.modal1, self.modal2]}
        if ret_loss:
            loss = self.criterion(output, self.target) / self.world_size
            acc = acc / self.world_size
            return ret_tensors, {'loss': loss, 'acc': acc}
        else:
            return ret_tensors

    def step(self):
        if self.use_rgb:
            output = self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1))
        else:
            output = self.model(torch.cat([self.modal1, self.modal2], dim=1))
        loss = self.criterion(output, self.target) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}
