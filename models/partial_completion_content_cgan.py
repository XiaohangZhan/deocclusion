import numpy as np
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
from . import backbone, InpaintingLoss, AdversarialLoss

class PartialCompletionContentCGAN(nn.Module):

    def __init__(self, params, load_pretrain=None, dist_model=False, demo=False):
        super(PartialCompletionContentCGAN, self).__init__()
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

        self.demo = demo
        if demo:
            return

        # optim
        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=params['lr'])

        # netD
        self.netD = backbone.__dict__[params['discriminator']](**params['discriminator_params'])
        self.netD.cuda()
        if dist_model:
            self.netD = utils.DistModule(self.netD)
        else:
            self.netD = backbone.FixModule(self.netD)
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=params['lr'] * params['d2g_lr'], betas=(0.0, 0.9))

        # loss
        self.criterion = InpaintingLoss(backbone.VGG16FeatureExtractor()).cuda()
        self.gan_criterion = AdversarialLoss(type=params['gan_type']).cuda()

        cudnn.benchmark = True

    def set_input(self, rgb, visible_mask, modal, rgb_gt=None):
        self.rgb = rgb.cuda()
        if self.with_modal:
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

        if self.with_modal:
            mask_tensors = [self.modal, self.visible_mask3]
        else:
            mask_tensors = [self.visible_mask3]
        ret_tensors = {'common_tensors': [self.rgb, output_comp, self.rgb_gt],
                       'mask_tensors': mask_tensors}
        if ret_loss:
            loss_dict = self.criterion(self.rgb, self.visible_mask3, output, self.rgb_gt)
            for k in loss_dict.keys():
                loss_dict[k] /= self.world_size
            return ret_tensors, loss_dict
        else:
            return ret_tensors

    def step(self):
        # output
        if self.with_modal:
            output, _ = self.model(torch.cat([self.rgb, self.modal], dim=1),
                                   self.visible_mask4)
        else:
            output, _ = self.model(self.rgb, self.visible_mask3)
        if output.shape[2] != self.rgb.shape[2]:
            output = nn.functional.interpolate(
                output, size=self.rgb.shape[2:4],
                mode="bilinear", align_corners=True)

        # discriminator loss
        dis_input_real = self.rgb_gt
        dis_input_fake = output.detach()
        if self.with_modal:
            dis_real, _ = self.netD(torch.cat([dis_input_real, self.modal], dim=1))
            dis_fake, _ = self.netD(torch.cat([dis_input_fake, self.modal], dim=1))
        else:
            dis_real, _ = self.netD(dis_input_real)
            dis_fake, _ = self.netD(dis_input_fake)
        dis_real_loss = self.gan_criterion(dis_real, True, True) / self.world_size
        dis_fake_loss = self.gan_criterion(dis_fake, False, True) / self.world_size
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_loss = 0
        gen_input_fake = output
        if self.with_modal:
            gen_fake, _ = self.netD(torch.cat([gen_input_fake, self.modal], dim=1))
        else:
            gen_fake, _ = self.netD(gen_input_fake)
        gen_gan_loss = self.gan_criterion(gen_fake, True, False) * \
            self.params['adv_loss_weight'] / self.world_size
        gen_loss += gen_gan_loss

        # other losses
        loss_dict = self.criterion(self.rgb, self.visible_mask3, output, self.rgb_gt)
        for k in loss_dict.keys():
            loss_dict[k] /= self.world_size
        for key, coef in self.params['lambda_dict'].items():
            value = coef * loss_dict[key]
            gen_loss += value

        # create loss dict
        loss_dict['dis'] = dis_loss
        loss_dict['adv'] = gen_gan_loss

        # update
        self.optimD.zero_grad()
        dis_loss.backward()
        utils.average_gradients(self.netD)
        self.optimD.step()

        self.optim.zero_grad()
        gen_loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()

        return loss_dict

    def load_model_demo(self, path):
        utils.load_state(path, self.model)

    def load_state(self, root, Iter, resume=False):
        path = os.path.join(root, "ckpt_iter_{}.pth.tar".format(Iter))
        netD_path = os.path.join(root, "D_iter_{}.pth.tar".format(Iter))

        if resume:
            utils.load_state(path, self.model, self.optim)
            utils.load_state(netD_path, self.netD, self.optimD)
        else:
            utils.load_state(path, self.model)
            utils.load_state(netD_path, self.netD)

    def save_state(self, root, Iter):
        path = os.path.join(root, "ckpt_iter_{}.pth.tar".format(Iter))
        netD_path = os.path.join(root, "D_iter_{}.pth.tar".format(Iter))

        torch.save({
            'step': Iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()}, path)

        torch.save({
            'step': Iter,
            'state_dict': self.netD.state_dict(),
            'optimizer': self.optimD.state_dict()}, netD_path)

    def switch_to(self, phase):
        if phase == 'train':
            self.model.train()
            self.netD.train()
        else:
            self.model.eval()
            if not self.demo:
                self.netD.eval()
