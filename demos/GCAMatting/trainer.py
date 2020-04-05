import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.backends.cudnn as cudnn
from   torch.nn import SyncBatchNorm
import torch.optim.lr_scheduler as lr_scheduler
from   torch.nn.parallel import DistributedDataParallel

import utils
from   utils import CONFIG
import networks


class Trainer(object):

    def __init__(self,
                 train_dataloader,
                 test_dataloader,
                 logger,
                 tb_logger):

        # Save GPU memory.
        cudnn.benchmark = False

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.tb_logger = tb_logger

        self.model_config = CONFIG.model
        self.train_config = CONFIG.train
        self.log_config = CONFIG.log
        self.loss_dict = {'rec': None,
                          'comp': None,
                          'smooth_l1':None,
                          'grad':None,
                          'gabor':None}
        self.test_loss_dict = {'rec': None,
                               'smooth_l1':None,
                               'mse':None,
                               'sad':None,
                               'grad':None,
                               'gabor':None}

        self.grad_filter = torch.tensor(utils.get_gradfilter()).cuda()
        self.gabor_filter = torch.tensor(utils.get_gaborfilter(16)).cuda()

        self.build_model()
        self.resume_step = None
        self.best_loss = 1e+8

        utils.print_network(self.G, CONFIG.version)
        if self.train_config.resume_checkpoint:
            self.logger.info('Resume checkpoint: {}'.format(self.train_config.resume_checkpoint))
            self.restore_model(self.train_config.resume_checkpoint)

        if self.model_config.imagenet_pretrain and self.train_config.resume_checkpoint is None:
            self.logger.info('Load Imagenet Pretrained: {}'.format(self.model_config.imagenet_pretrain_path))
            if self.model_config.arch.encoder == "vgg_encoder":
                utils.load_VGG_pretrain(self.G, self.model_config.imagenet_pretrain_path)
            else:
                utils.load_imagenet_pretrain(self.G, self.model_config.imagenet_pretrain_path)


    def build_model(self):

        self.G = networks.get_generator(encoder=self.model_config.arch.encoder, decoder=self.model_config.arch.decoder)
        self.G.cuda()

        if CONFIG.dist:
            self.logger.info("Using pytorch synced BN")
            self.G = SyncBatchNorm.convert_sync_batchnorm(self.G)

        self.G_optimizer = torch.optim.Adam(self.G.parameters(),
                                            lr = self.train_config.G_lr,
                                            betas = [self.train_config.beta1, self.train_config.beta2])

        if CONFIG.dist:
            # SyncBatchNorm only supports DistributedDataParallel with single GPU per process
            self.G = DistributedDataParallel(self.G, device_ids=[CONFIG.local_rank], output_device=CONFIG.local_rank)
        else:
            self.G = nn.DataParallel(self.G)

        self.build_lr_scheduler()

    def build_lr_scheduler(self):
        """Build cosine learning rate scheduler."""
        self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                          T_max=self.train_config.total_step
                                                                - self.train_config.warmup_step)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.G_optimizer.zero_grad()


    def restore_model(self, resume_checkpoint):
        """
        Restore the trained generator and discriminator.
        :param resume_checkpoint: File name of checkpoint
        :return:
        """
        pth_path = os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(resume_checkpoint))
        checkpoint = torch.load(pth_path, map_location = lambda storage, loc: storage.cuda(CONFIG.gpu))
        self.resume_step = checkpoint['iter']
        self.logger.info('Loading the trained models from step {}...'.format(self.resume_step))
        self.G.load_state_dict(checkpoint['state_dict'], strict=True)

        if not self.train_config.reset_lr:
            if 'opt_state_dict' in checkpoint.keys():
                try:
                    self.G_optimizer.load_state_dict(checkpoint['opt_state_dict'])
                except ValueError as ve:
                    self.logger.error("{}".format(ve))
            else:
                self.logger.info('No Optimizer State Loaded!!')

            if 'lr_state_dict' in checkpoint.keys():
                try:
                    self.G_scheduler.load_state_dict(checkpoint['lr_state_dict'])
                except ValueError as ve:
                    self.logger.error("{}".format(ve))
        else:
            self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                              T_max=self.train_config.total_step - self.resume_step - 1)

        if 'loss' in checkpoint.keys():
            self.best_loss = checkpoint['loss']


    def train(self):
        data_iter = iter(self.train_dataloader)

        if self.train_config.resume_checkpoint:
            start = self.resume_step + 1
        else:
            start = 0

        moving_max_grad = 0
        moving_grad_moment = 0.999
        max_grad = 0

        for step in range(start, self.train_config.total_step + 1):
            try:
                image_dict = next(data_iter)
            except:
                data_iter = iter(self.train_dataloader)
                image_dict = next(data_iter)

            image, alpha, trimap = image_dict['image'], image_dict['alpha'], image_dict['trimap']
            image = image.cuda()
            alpha = alpha.cuda()
            trimap = trimap.cuda()
            # train() of DistributedDataParallel has no return
            self.G.train()
            log_info = ""
            loss = 0

            """===== Update Learning Rate ====="""
            if step < self.train_config.warmup_step and self.train_config.resume_checkpoint is None:
                cur_G_lr = utils.warmup_lr(self.train_config.G_lr, step + 1, self.train_config.warmup_step)
                utils.update_lr(cur_G_lr, self.G_optimizer)

            else:
                self.G_scheduler.step()
                cur_G_lr = self.G_scheduler.get_lr()[0]

            """===== Forward G ====="""
            alpha_pred, info_dict = self.G(image, trimap) # info_dict: intermediate feature of networks like attention
            weight = utils.get_unknown_tensor(trimap)

            """===== Calculate Loss ====="""
            if self.train_config.rec_weight > 0:
                self.loss_dict['rec'] = self.regression_loss(alpha_pred, alpha, loss_type='l1', weight=weight) \
                                        * self.train_config.rec_weight
            if self.train_config.smooth_l1_weight > 0:
                self.loss_dict['smooth_l1'] = self.smooth_l1(alpha_pred, alpha, weight=weight) \
                                              * self.train_config.smooth_l1_weight
            if self.train_config.comp_weight > 0:
                self.loss_dict['comp'] = self.composition_loss(alpha_pred, image_dict['fg'].cuda(),
                                                               image_dict['bg'].cuda(), image, weight=weight) \
                                         * self.train_config.comp_weight
            if self.train_config.grad_weight > 0:
                self.loss_dict['grad'] = self.grad_loss(alpha_pred, alpha, weight=weight, grad_filter=self.grad_filter) \
                                         * self.train_config.grad_weight
            if self.train_config.gabor_weight > 0:
                self.loss_dict['gabor'] = self.gabor_loss(alpha_pred, alpha, weight=weight, gabor_filter=self.gabor_filter) \
                                          * self.train_config.gabor_weight


            for loss_key in self.loss_dict.keys():
                if self.loss_dict[loss_key] is not None and loss_key in ['rec', 'comp', 'smooth_l1', 'grad', 'gabor']:
                    loss += self.loss_dict[loss_key]

            """===== Back Propagate ====="""
            self.reset_grad()

            loss.backward()

            """===== Clip Large Gradient ====="""
            if self.train_config.clip_grad:
                if moving_max_grad == 0:
                    moving_max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 1e+6)
                    max_grad = moving_max_grad
                else:
                    max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 2 * moving_max_grad)
                    moving_max_grad = moving_max_grad * moving_grad_moment + max_grad * (
                                1 - moving_grad_moment)

            """===== Update Parameters ====="""
            self.G_optimizer.step()

            """===== Write Log and Tensorboard ====="""
            # stdout log
            if step % self.log_config.logging_step == 0:
                # reduce losses from GPUs
                if CONFIG.dist:
                    self.loss_dict = utils.reduce_tensor_dict(self.loss_dict, mode='mean')
                    loss = utils.reduce_tensor(loss)
                # create logging information
                for loss_key in self.loss_dict.keys():
                    if self.loss_dict[loss_key] is not None:
                        log_info += loss_key.upper() + ": {:.4f}, ".format(self.loss_dict[loss_key])

                self.logger.debug("Image tensor shape: {}. Trimap tensor shape: {}".format(image.shape, trimap.shape))
                log_info = "[{}/{}], ".format(step, self.train_config.total_step) + log_info
                log_info += "lr: {:6f}".format(cur_G_lr)
                self.logger.info(log_info)

                # tensorboard
                if step % self.log_config.tensorboard_step == 0 or step == start:  # and step > start:
                    self.tb_logger.scalar_summary('Loss', loss, step)

                    # detailed losses
                    for loss_key in self.loss_dict.keys():
                        if self.loss_dict[loss_key] is not None:
                            self.tb_logger.scalar_summary('Loss_' + loss_key.upper(),
                                                          self.loss_dict[loss_key], step)

                    self.tb_logger.scalar_summary('LearnRate', cur_G_lr, step)

                    if self.train_config.clip_grad:
                        self.tb_logger.scalar_summary('Moving_Max_Grad', moving_max_grad, step)
                        self.tb_logger.scalar_summary('Max_Grad', max_grad, step)

                # write images to tensorboard
                if step % self.log_config.tensorboard_image_step == 0 or step == start:
                    if self.model_config.trimap_channel == 3:
                        trimap = trimap.argmax(dim=1, keepdim=True)
                    alpha_pred[trimap==2] = 1
                    alpha_pred[trimap==0] = 0
                    image_set = {'image': (utils.normalize_image(image[-1, ...]).data.cpu().numpy()
                                           * 255).astype(np.uint8),
                                 'trimap': (trimap[-1, ...].data.cpu().numpy() * 127).astype(np.uint8),
                                 'alpha': (alpha[-1, ...].data.cpu().numpy() * 255).astype(np.uint8),
                                 'alpha_pred': (alpha_pred[-1, ...].data.cpu().numpy() * 255).astype(np.uint8)}


                    if info_dict is not None:
                        for key in info_dict.keys():
                            if key.startswith('offset'):
                                image_set[key] = utils.flow_to_image(info_dict[key][0][-1,...].data.cpu()
                                                                     .numpy()).transpose([2, 0, 1]).astype(np.uint8)
                                # write softmax_scale to offset image
                                scale = info_dict[key][1].cpu()
                                image_set[key] = utils.put_text(image_set[key], 'unknown: {:.2f}, known: {:.2f}'
                                                                .format(scale[-1,0].item(), scale[-1,1].item()))
                            else:
                                image_set[key] = (utils.normalize_image(info_dict[key][-1,...]).data.cpu().numpy()
                                                  * 255).astype(np.uint8)
                    self.tb_logger.image_summary(image_set, step)

            """===== TEST ====="""
            if ((step % self.train_config.val_step) == 0 or step == self.train_config.total_step):# and step > start:
                self.G.eval()
                test_loss = 0
                log_info = ""

                self.test_loss_dict['mse'] = 0
                self.test_loss_dict['sad'] = 0
                for loss_key in self.loss_dict.keys():
                    if loss_key in self.test_loss_dict and self.loss_dict[loss_key] is not None:
                        self.test_loss_dict[loss_key] = 0

                with torch.no_grad():
                    for image_dict in self.test_dataloader:
                        image, alpha, trimap = image_dict['image'], image_dict['alpha'], image_dict['trimap']
                        alpha_shape = image_dict['alpha_shape']
                        image = image.cuda()
                        alpha = alpha.cuda()
                        trimap = trimap.cuda()

                        alpha_pred, info_dict = self.G(image, trimap)

                        h, w = alpha_shape
                        alpha_pred = alpha_pred[..., :h, :w]
                        trimap = trimap[..., :h, :w]
                        weight = utils.get_unknown_tensor(trimap)
                        # value of MSE/SAD here is different from test.py and matlab version
                        self.test_loss_dict['mse'] += self.mse(alpha_pred, alpha, weight)
                        self.test_loss_dict['sad'] += self.sad(alpha_pred, alpha, weight)

                        if self.train_config.rec_weight > 0:
                            self.test_loss_dict['rec'] += self.regression_loss(alpha_pred, alpha, weight=weight) \
                                                          * self.train_config.rec_weight
                        if self.train_config.smooth_l1_weight > 0:
                            self.test_loss_dict['smooth_l1'] += self.smooth_l1(alpha_pred, alpha, weight=weight) \
                                                               * self.train_config.smooth_l1_weight
                        if self.train_config.grad_weight > 0:
                            self.test_loss_dict['grad'] = self.grad_loss(alpha_pred, alpha, weight=weight,
                                                                         grad_filter=self.grad_filter) \
                                                          * self.train_config.grad_weight
                        if self.train_config.gabor_weight > 0:
                            self.test_loss_dict['gabor'] = self.gabor_loss(alpha_pred, alpha, weight=weight,
                                                                           gabor_filter=self.gabor_filter) \
                                                          * self.train_config.gabor_weight
                # reduce losses from GPUs
                if CONFIG.dist:
                    self.test_loss_dict = utils.reduce_tensor_dict(self.test_loss_dict, mode='mean')

                """===== Write Log and Tensorboard ====="""
                # stdout log
                for loss_key in self.test_loss_dict.keys():
                    if self.test_loss_dict[loss_key] is not None:
                        self.test_loss_dict[loss_key] /= len(self.test_dataloader)
                        # logging
                        log_info += loss_key.upper()+": {:.4f} ".format(self.test_loss_dict[loss_key])
                        self.tb_logger.scalar_summary('Loss_'+loss_key.upper(),
                                                      self.test_loss_dict[loss_key], step, phase='test')

                        if loss_key in ['rec', 'smooth_l1', 'grad', 'gabor']:
                            test_loss += self.test_loss_dict[loss_key]

                self.logger.info("TEST: LOSS: {:.4f} ".format(test_loss)+log_info)
                self.tb_logger.scalar_summary('Loss', test_loss, step, phase='test')

                if self.model_config.trimap_channel == 3:
                    trimap = trimap.argmax(dim=1, keepdim=True)
                alpha_pred[trimap==2] = 1
                alpha_pred[trimap==0] = 0
                image_set = {'image': (utils.normalize_image(image[-1, ...]).data.cpu().numpy()
                                       * 255).astype(np.uint8),
                             'trimap': (trimap[-1, ...].data.cpu().numpy() * 127).astype(np.uint8),
                             'alpha': (alpha[-1, ...].data.cpu().numpy() * 255).astype(np.uint8),
                             'alpha_pred': (alpha_pred[-1, ...].data.cpu().numpy() * 255).astype(np.uint8)}

                if info_dict is not None:
                    for key in info_dict.keys():
                        if key.startswith('offset'):
                            image_set[key] = utils.flow_to_image(info_dict[key][0][-1,...].data.cpu()
                                                                 .numpy()).transpose([2, 0, 1]).astype(np.uint8)
                            # write softmax_scale to offset image
                            scale = info_dict[key][1].cpu()
                            image_set[key] = utils.put_text(image_set[key], 'unknown: {:.2f}, known: {:.2f}'
                                                            .format(scale[-1,0].item(), scale[-1,1].item()))
                        else:
                            image_set[key] = (utils.normalize_image(info_dict[key][-1,...]).data.cpu().numpy()
                                          * 255).astype(np.uint8)
                self.tb_logger.image_summary(image_set, step, phase='test')

                """===== Save Model ====="""
                if (step % self.log_config.checkpoint_step == 0 or step == self.train_config.total_step) \
                        and CONFIG.local_rank == 0 and (step > start):
                    self.logger.info('Saving the trained models from step {}...'.format(iter))
                    self.save_model("latest_model", step, loss)
                    if self.test_loss_dict['mse'] < self.best_loss:
                        self.best_loss = self.test_loss_dict['mse']
                        self.save_model("best_model", step, loss)


    def save_model(self, checkpoint_name, iter, loss):
        """Restore the trained generator and discriminator."""
        torch.save({
            'iter': iter,
            'loss': loss,
            'state_dict': self.G.state_dict(),
            'opt_state_dict': self.G_optimizer.state_dict(),
            'lr_state_dict': self.G_scheduler.state_dict()
        }, os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(checkpoint_name)))


    @staticmethod
    def regression_loss(logit, target, loss_type='l1', weight=None):
        """
        Alpha reconstruction loss
        :param logit:
        :param target:
        :param loss_type: "l1" or "l2"
        :param weight: tensor with shape [N,1,H,W] weights for each pixel
        :return:
        """
        if weight is None:
            if loss_type == 'l1':
                return F.l1_loss(logit, target)
            elif loss_type == 'l2':
                return F.mse_loss(logit, target)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
        else:
            if loss_type == 'l1':
                return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            elif loss_type == 'l2':
                return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))


    @staticmethod
    def smooth_l1(logit, target, weight):
        loss = torch.sqrt((logit * weight - target * weight)**2 + 1e-6)
        loss = torch.sum(loss) / (torch.sum(weight) + 1e-8)
        return loss


    @staticmethod
    def mse(logit, target, weight):
        # return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        return Trainer.regression_loss(logit, target, loss_type='l2', weight=weight)

    @staticmethod
    def sad(logit, target, weight):
        return F.l1_loss(logit * weight, target * weight, reduction='sum') / 1000

    @staticmethod
    def composition_loss(alpha, fg, bg, image, weight, loss_type='l1'):
        """
        Alpha composition loss
        """
        merged = fg * alpha + bg * (1 - alpha)
        return Trainer.regression_loss(merged, image, loss_type=loss_type, weight=weight)

    @staticmethod
    def gabor_loss(logit, target, gabor_filter, loss_type='l2', weight=None):
        """ pass """
        gabor_logit = F.conv2d(logit, weight=gabor_filter, padding=2)
        gabor_target = F.conv2d(target, weight=gabor_filter, padding=2)

        return Trainer.regression_loss(gabor_logit, gabor_target, loss_type=loss_type, weight=weight)

    @staticmethod
    def grad_loss(logit, target, grad_filter, loss_type='l1', weight=None):
        """ pass """
        grad_logit = F.conv2d(logit, weight=grad_filter, padding=1)
        grad_target = F.conv2d(target, weight=grad_filter, padding=1)
        grad_logit = torch.sqrt((grad_logit * grad_logit).sum(dim=1, keepdim=True) + 1e-8)
        grad_target = torch.sqrt((grad_target * grad_target).sum(dim=1, keepdim=True) + 1e-8)

        return Trainer.regression_loss(grad_logit, grad_target, loss_type=loss_type, weight=weight)