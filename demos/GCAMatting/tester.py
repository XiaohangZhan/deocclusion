import os
import cv2
import logging
import numpy as np

import torch

import utils
from   utils import CONFIG
import networks
from   utils import comput_sad_loss, compute_connectivity_error, \
    compute_gradient_loss, compute_mse_loss


class Tester(object):

    def __init__(self, test_dataloader):

        self.test_dataloader = test_dataloader
        self.logger = logging.getLogger("Logger")

        self.model_config = CONFIG.model
        self.test_config = CONFIG.test
        self.log_config = CONFIG.log
        self.data_config = CONFIG.data

        self.build_model()
        self.resume_step = None

        utils.print_network(self.G, CONFIG.version)

        if self.test_config.checkpoint:
            self.logger.info('Resume checkpoint: {}'.format(self.test_config.checkpoint))
            self.restore_model(self.test_config.checkpoint)

    def build_model(self):
        self.G = networks.get_generator(encoder=self.model_config.arch.encoder, decoder=self.model_config.arch.decoder)
        if not self.test_config.cpu:
            self.G.cuda()

    def restore_model(self, resume_checkpoint):
        """
        Restore the trained generator and discriminator.
        :param resume_checkpoint: File name of checkpoint
        :return:
        """
        pth_path = os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(resume_checkpoint))
        checkpoint = torch.load(pth_path)
        self.G.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    def test(self):
        self.G = self.G.eval()
        mse_loss = 0
        sad_loss = 0
        conn_loss = 0
        grad_loss = 0

        test_num = 0

        with torch.no_grad():
            for image_dict in self.test_dataloader:
                image, alpha, trimap = image_dict['image'], image_dict['alpha'], image_dict['trimap']
                alpha_shape, name = image_dict['alpha_shape'], image_dict['image_name']
                if not self.test_config.cpu:
                    image = image.cuda()
                    alpha = alpha.cuda()
                    trimap = trimap.cuda()
                alpha_pred, _ = self.G(image, trimap)

                if self.model_config.trimap_channel == 3:
                    trimap = trimap.argmax(dim=1, keepdim=True)

                alpha_pred[trimap == 2] = 1
                alpha_pred[trimap == 0] = 0

                trimap[trimap==2] = 255
                trimap[trimap==1] = 128

                for cnt in range(image.shape[0]):

                    h, w = alpha_shape
                    test_alpha = alpha[cnt, 0, ...].data.cpu().numpy() * 255
                    test_pred = alpha_pred[cnt, 0, ...].data.cpu().numpy() * 255
                    test_pred = test_pred.astype(np.uint8)
                    test_trimap = trimap[cnt, 0, ...].data.cpu().numpy()

                    test_pred = test_pred[:h, :w]
                    test_trimap = test_trimap[:h, :w]

                    if self.test_config.alpha_path is not None:
                        cv2.imwrite(os.path.join(self.test_config.alpha_path, os.path.splitext(name[cnt])[0] + ".png"),
                                    test_pred)

                    mse_loss += compute_mse_loss(test_pred, test_alpha, test_trimap)
                    print(name, comput_sad_loss(test_pred, test_alpha, test_trimap)[0])
                    sad_loss += comput_sad_loss(test_pred, test_alpha, test_trimap)[0]
                    if not self.test_config.fast_eval:
                        conn_loss += compute_connectivity_error(test_pred, test_alpha, test_trimap, 0.1)
                        grad_loss += compute_gradient_loss(test_pred, test_alpha, test_trimap)

                    test_num += 1

        self.logger.info("TEST NUM: \t\t {}".format(test_num))
        self.logger.info("MSE: \t\t {}".format(mse_loss / test_num))
        self.logger.info("SAD: \t\t {}".format(sad_loss / test_num))
        if not self.test_config.fast_eval:
            self.logger.info("GRAD: \t\t {}".format(grad_loss / test_num))
            self.logger.info("CONN: \t\t {}".format(conn_loss / test_num))
