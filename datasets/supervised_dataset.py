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
import inference as infer

class SupCompDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        if self.dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(config['{}_annot_file'.format(phase)])
        else:
            self.data_reader = reader.KINSLVISDataset(
                self.dataset, config['{}_annot_file'.format(phase)])

        self.img_transform = transforms.Compose([
            transforms.Normalize(config['data_mean'], config['data_std'])
        ])
        self.sz = config['input_size']
        self.phase = phase

        self.config = config

        self.memcached = config.get('memcached', False)
        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)
        self.memcached = self.memcached_client is not None

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
        modal, bbox, category, imgfn, amodal = self.data_reader.get_instance(idx, with_gt=True)
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
        amodal = cv2.resize(utils.crop_padding(amodal, new_bbox, pad_value=(0,)),
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        # flip
        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            flip = True
            modal = modal[:, ::-1]
            amodal = amodal[:, ::-1]
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
            return modal, amodal, rgb
        else:
            return modal, amodal, None

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()
        modal, amodal, rgb = self._get_inst(
            idx, load_rgb=self.config['load_rgb'], randshift=True) # modal, uint8 {0, 1}

        if rgb is None:
            rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32) # 3HW
        modal_tensor = torch.from_numpy(
            modal.astype(np.float32)).unsqueeze(0) # 1HW, float
        target = torch.from_numpy(amodal.astype(np.int)) # HW, int
        return rgb, modal_tensor, target


class SupOrderDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        if self.dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(config['{}_annot_file'.format(phase)])
        else:
            self.data_reader = reader.KINSLVISDataset(
                self.dataset, config['{}_annot_file'.format(phase)])

        self.img_transform = transforms.Compose([
            transforms.Normalize(config['data_mean'], config['data_std'])
        ])
        self.sz = config['input_size']
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

    def _get_pair(self, modal, bboxes, idx1, idx2, imgfn, load_rgb=False, randshift=False):
        bbox = utils.combine_bbox(bboxes[(idx1, idx2), :] )
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * 2.), bbox[2] * 1.1, bbox[3] * 1.1])

        # shift & scale aug
        if self.phase  == 'train':
            if randshift:
                centerx += np.random.uniform(*self.config['base_aug']['shift']) * size
                centery += np.random.uniform(*self.config['base_aug']['shift']) * size
            size /= np.random.uniform(*self.config['base_aug']['scale'])

        # crop
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        modal1 = cv2.resize(utils.crop_padding(modal[idx1], new_bbox, pad_value=(0,)),
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)
        modal2 = cv2.resize(utils.crop_padding(modal[idx2], new_bbox, pad_value=(0,)),
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        # flip
        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            flip = True
            modal1 = modal1[:, ::-1]
            modal2 = modal2[:, ::-1]
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
            return modal1, modal2, rgb
        else:
            return modal1, modal2, None

    def _get_pair_ind(self, idx):
        modal, category, bboxes, amodal, image_fn = self.data_reader.get_image_instances(
            idx, with_gt=True)
        gt_order_matrix = infer.infer_gt_order(modal, amodal)
        pairs = np.where(gt_order_matrix == 1)
        if len(pairs[0]) == 0:
            return self._get_pair_ind(np.random.choice(len(self)))
        return modal, bboxes, image_fn, pairs

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()

        modal, bboxes, image_fn, pairs = self._get_pair_ind(idx)
        randidx = np.random.choice(len(pairs[0]))
        idx1 = pairs[0][randidx]
        idx2 = pairs[1][randidx]

        # get pair
        modal1, modal2, rgb = self._get_pair(
            modal, bboxes, idx1, idx2, image_fn,
            load_rgb=self.config['load_rgb'], randshift=True)

        if rgb is None:
            rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32) # 3HW
        modal_tensor1 = torch.from_numpy(
            modal1.astype(np.float32)).unsqueeze(0) # 1HW, float
        modal_tensor2 = torch.from_numpy(
            modal2.astype(np.float32)).unsqueeze(0) # 1HW, float
        if np.random.rand() < 0.5:
            return rgb, modal_tensor1, modal_tensor2, 1
        else:
            return rgb, modal_tensor2, modal_tensor1, 0
