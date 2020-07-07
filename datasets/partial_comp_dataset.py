import numpy as np
try:
    import mc
except Exception:
    pass
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
from . import reader


class PartialCompDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        if self.dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(config['{}_annot_file'.format(phase)])
        elif self.dataset == 'Mapillary':
            self.data_reader = reader.MapillaryDataset(
                config['{}_root'.format(phase)], config['{}_annot_file'.format(phase)])
        else:
            self.data_reader = reader.KINSLVISDataset(
                self.dataset, config['{}_annot_file'.format(phase)])
        if config['load_rgb']:
            self.img_transform = transforms.Compose([
                transforms.Normalize(config['data_mean'], config['data_std'])
            ])
        self.eraser_setter = utils.EraserSetter(config['eraser_setter'])
        self.sz = config['input_size']
        self.eraser_front_prob = config['eraser_front_prob']
        self.phase = phase

        self.config = config

        self.memcached = config.get('memcached', False)
        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)

    def __len__(self):
        return self.data_reader.get_instance_length()

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

    def _get_inst(self, idx, load_rgb=False, randshift=False):
        modal, bbox, category, imgfn, _ = self.data_reader.get_instance(idx)
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * self.config['enlarge_box']), bbox[2] * 1.1, bbox[3] * 1.1])
        if size < 5 or np.all(modal == 0):
            return self._get_inst(
                np.random.choice(len(self)), load_rgb=load_rgb, randshift=randshift)

        # shift & scale aug
        if self.phase  == 'train':
            if randshift:
                centerx += np.random.uniform(*self.config['base_aug']['shift']) * size
                centery += np.random.uniform(*self.config['base_aug']['shift']) * size
            size /= np.random.uniform(*self.config['base_aug']['scale'])

        # crop
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        modal = cv2.resize(utils.crop_padding(modal, new_bbox, pad_value=(0,)),
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        # flip
        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            flip = True
            modal = modal[:, ::-1]
        else:
            flip = False

        if load_rgb:
            rgb = np.array(self._load_image(os.path.join(
                self.config['{}_image_root'.format(self.phase)], imgfn))) # uint8
            rgb = cv2.resize(utils.crop_padding(rgb, new_bbox, pad_value=(0,0,0)),
                (self.sz, self.sz), interpolation=cv2.INTER_CUBIC)
            if flip:
                rgb = rgb[:, ::-1, :]
            rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)
            rgb = self.img_transform(rgb) # CHW

        if load_rgb:
            return modal, category, rgb
        else:
            return modal, category, None

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()
        randidx = np.random.choice(len(self))
        modal, category, rgb = self._get_inst(
            idx, load_rgb=self.config['load_rgb'], randshift=True) # modal, uint8 {0, 1}
        if not self.config.get('use_category', True):
            category = 1
        eraser, _, _ = self._get_inst(randidx, load_rgb=False, randshift=False)
        eraser = self.eraser_setter(modal, eraser) # uint8 {0, 1}

        # erase
        erased_modal = modal.copy().astype(np.float32)
        if np.random.rand() < self.eraser_front_prob:
            erased_modal[eraser == 1] = 0 # eraser above modal
        else:
            eraser[modal == 1] = 0 # eraser below modal
        erased_modal = erased_modal * category

        # shrink eraser
        max_shrink_pix = self.config.get('max_eraser_shrink', 0)
        if max_shrink_pix > 0:
            shrink_pix = np.random.choice(np.arange(max_shrink_pix + 1))
            if shrink_pix > 0:
                shrink_kernel = shrink_pix * 2 + 1
                eraser = 1 - cv2.dilate(
                    1 - eraser, np.ones((shrink_kernel, shrink_kernel), dtype=np.uint8),
                    iterations=1)
        eraser_tensor = torch.from_numpy(eraser.astype(np.float32)).unsqueeze(0) # 1HW
        # erase rgb
        if rgb is not None:
            rgb = rgb * (1 - eraser_tensor)
        else:
            rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32) # 3HW
        erased_modal_tensor = torch.from_numpy(
            erased_modal.astype(np.float32)).unsqueeze(0) # 1HW
        target = torch.from_numpy(modal.astype(np.int)) # HW
        return rgb, erased_modal_tensor, eraser_tensor, target
