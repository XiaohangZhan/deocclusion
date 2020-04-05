import numpy as np
try:
    import mc
except Exception:
    pass
import cv2
import os
import io
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
from . import reader


class InpaintDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        if self.dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(config['{}_annot_file'.format(phase)])
        else:
            self.data_reader = reader.KINSLVISDataset(
                self.dataset, config['{}_annot_file'.format(phase)])

        self.img_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['crop_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config['data_mean'], config['data_std'])
        ])
        self.phase = phase

        self.config = config

        self.memcached = config.get('memcached', False)
        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)
        self.memcached = self.memcached_client is not None

    def __len__(self):
        return self.data_reader.get_image_length()

    def _init_memcached(self):
        if not self.initialized:
            assert self.memcached_client is not None, "Please specify the path of your memcached_client"
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _load_image(self, fn):
        if self.memcached:
            try:
                img_value = mc.pyvector()
                self.mclient.Get(fn, img_value)
                img_value_str = mc.ConvertBuffer(img_value)
                img = utils.pil_loader(img_value_str)
            except:
                print('Read image failed ({})'.format(fn))
                raise Exception("Exit")
            else:
                return img
        else:
            return Image.open(fn).convert('RGB')

    def _get_eraser(self, idx):
        modal, bbox, category, imgfn, _ = self.data_reader.get_instance(idx)
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = self.config['crop_size']

        # shift & scale aug
        centerx += np.random.uniform(-0.5, 0.5) * size
        centery += np.random.uniform(-0.5, 0.5) * size
        size /= np.random.uniform(0.8, 1.2)

        # crop
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        modal = cv2.resize(utils.crop_padding(modal, new_bbox, pad_value=(0,)),
            (self.config['crop_size'], self.config['crop_size']), interpolation=cv2.INTER_NEAREST)

        # flip
        if np.random.rand() > 0.5:
            modal = modal[:, ::-1]
        return modal

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()
        imgfn = self.data_reader.images_info[idx]['file_name']
        rgb = self._load_image(os.path.join(
            self.config['{}_image_root'.format(self.phase)], imgfn))
        rgb = self.img_transform(rgb)

        eraser_num = np.random.randint(1, self.config['max_eraser_num'])
        erasers = np.concatenate([self._get_eraser(
            np.random.choice(len(self.data_reader.annot_info)))[np.newaxis,:,:] \
            for _ in range(eraser_num)], axis=0)
        eraser = erasers.sum(axis=0) > 0 # union

        # get mask 
        visible_mask = (~eraser).astype(np.float32)[np.newaxis,:,:]

        # convert to tensors
        visible_mask_tensor = torch.from_numpy(visible_mask)
        rgb_erased = rgb.clone()
        rgb_erased = rgb * visible_mask_tensor # erase rgb
        return rgb_erased, visible_mask_tensor, -1, rgb
