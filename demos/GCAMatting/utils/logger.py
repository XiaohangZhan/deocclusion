import os
import cv2
import torch
import logging
import datetime
import numpy as np
from   pprint import pprint
from   . import util
from   .config import CONFIG
from   tensorboardX import SummaryWriter


LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def make_color_wheel():
    # from https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u,v):
    # from https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = COLORWHEEL
    # colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img

def flow_to_image(flow):
    # part from https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
    maxrad = -1
    u = flow[0, :, :]
    v = flow[1, :, :]
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(maxrad, np.max(rad))
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)
    img = compute_color(u, v)

    return img


def put_text(image, text, position=(10, 20)):
    image = cv2.resize(image.transpose([1, 2, 0]), (512, 512), interpolation=cv2.INTER_NEAREST)
    return cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, thickness=2).transpose([2, 0, 1])


class TensorBoardLogger(object):
    def __init__(self, tb_log_dir, exp_string):
        """
        Initialize summary writer
        """
        self.exp_string = exp_string
        self.tb_log_dir = tb_log_dir
        self.val_img_dir = os.path.join(self.tb_log_dir, 'val_image')

        if CONFIG.local_rank == 0:
            util.make_dir(self.tb_log_dir)
            util.make_dir(self.val_img_dir)

            self.writer = SummaryWriter(self.tb_log_dir+'/' + self.exp_string)
        else:
            self.writer = None

    def scalar_summary(self, tag, value, step, phase='train'):
        if CONFIG.local_rank == 0:
            sum_name = '{}/{}'.format(phase.capitalize(), tag)
            self.writer.add_scalar(sum_name, value, step)

    def image_summary(self, image_set, step, phase='train', save_val=True):
        """
        Record image in tensorboard
        The input image should be a numpy array with shape (C, H, W) like a torch tensor
        :param image_set: dict of images
        :param step:
        :param phase:
        :param save_val: save images in folder in validation or testing
        :return:
        """
        if CONFIG.local_rank == 0:
            for tag, image_numpy in image_set.items():
                sum_name = '{}/{}'.format(phase.capitalize(), tag)
                image_numpy = image_numpy.transpose([1, 2, 0])

                image_numpy = cv2.resize(image_numpy, (360, 360), interpolation=cv2.INTER_NEAREST)

                if len(image_numpy.shape) == 2:
                    image_numpy = image_numpy[None, :,:]
                else:
                    image_numpy = image_numpy.transpose([2, 0, 1])
                self.writer.add_image(sum_name, image_numpy, step)

            if (phase=='test') and save_val:
                tags = list(image_set.keys())
                image_pack = self._reshape_rgb(image_set[tags[0]])
                image_pack = cv2.resize(image_pack, (512, 512), interpolation=cv2.INTER_NEAREST)

                for tag in tags[1:]:
                    image = self._reshape_rgb(image_set[tag])
                    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
                    image_pack = np.concatenate((image_pack, image), axis=1)

                cv2.imwrite(os.path.join(self.val_img_dir, 'val_{:d}'.format(step)+'.png'), image_pack)

    @staticmethod
    def _reshape_rgb(image):
        """
        Transform RGB/L -> BGR for OpenCV
        """
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose([1, 2, 0])
            image = image[...,::-1]
        elif len(image.shape) == 3 and image.shape[0] == 1:
            image = image.transpose([1, 2, 0])
            image = np.repeat(image, 3, axis=2)
        elif len(image.shape) == 2:
            # image = image.transpose([1,0])
            image = np.stack((image, image, image), axis=2)
        else:
            raise ValueError('Image shape {} not supported to save'.format(image.shape))
        return image

    def __del__(self):
        if self.writer is not None:
            self.writer.close()


class MyLogger(logging.Logger):
    """
    Only write log in the first subprocess
    """
    def __init__(self, *args, **kwargs):
        super(MyLogger, self).__init__(*args, **kwargs)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        if CONFIG.local_rank == 0:
            super()._log(level, msg, args, exc_info, extra, stack_info)


def get_logger(log_dir=None, tb_log_dir=None, logging_level="DEBUG"):
    """
    Return a default build-in logger if log_file=None and tb_log_dir=None
    Return a build-in logger which dump stdout to log_file if log_file is assigned
    Return a build-in logger and tensorboard summary writer if tb_log_dir is assigned
    :param log_file: logging file dumped from stdout
    :param tb_log_dir: tensorboard dir
    :param logging_level:
    :return: Logger or [Logger, TensorBoardLogger]
    """
    level = LEVELS[logging_level.upper()]
    exp_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    logging.setLoggerClass(MyLogger)
    logger = logging.getLogger('Logger')
    logger.setLevel(level)
    # create formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m-%d %H:%M:%S')

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)

    # create file handler
    if log_dir is not None and CONFIG.local_rank == 0:
        log_file = os.path.join(log_dir, exp_string)
        fh = logging.FileHandler(log_file+'.log', mode='w')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        pprint(CONFIG, stream=fh.stream)

    # create tensorboard summary writer
    if tb_log_dir is not None:
        tb_logger = TensorBoardLogger(tb_log_dir=tb_log_dir, exp_string=exp_string)
        return logger, tb_logger
    else:
        return logger


def normalize_image(image):
    """
    normalize image array to 0~1
    """
    image_flat = torch.flatten(image, start_dim=1)
    return (image - image_flat.min(dim=1, keepdim=False)[0].view(3,1,1)) / (
                image_flat.max(dim=1, keepdim=False)[0].view(3,1,1) - image_flat.min(dim=1, keepdim=False)[0].view(3,1,1) + 1e-8)
