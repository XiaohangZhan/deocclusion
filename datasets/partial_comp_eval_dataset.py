import numpy as np
try:
    import mc
except Exception:
    pass
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
from . import reader

class PartialCompEvalDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        if self.dataset == 'SAS':
            self.data_reader = reader.SASDataset(config['{}_annot_file'.format(phase)])
        else:
            self.data_reader = reader.KINSLVISDataset(
                self.dataset, config['{}_annot_file'.format(phase)])

        if config['load_rgb']:
            self.img_transform = transforms.Compose([
                transforms.Normalize(config['data_mean'], config['data_std'])
            ])
        self.image_path = config['{}_image_root'.format(phase)]

        self.with_gt = True
        self.config = config

        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)
        self.memcached = self.memcached_client is not None

    def __len__(self):
        if self.config['eval_num'] == -1:
            return self.data_reader.get_image_length()
        else:
            return self.config['eval_num']

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

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()
        modal, category, bboxes, amodal, image_fn = self.data_reader.get_image_instances(
            idx, with_gt=self.with_gt)
        if not self.config.get('use_category', True):
            category[:] = 1
        if self.config.get('has_gt_ordering', False):
            gt_order_matrix = self.data_reader.get_gt_ordering(idx)
        else:
            gt_order_matrix = np.array([-2])
        if self.config['load_rgb']:
            rgb = np.array(self._load_image(os.path.join(self.image_path, image_fn)))
            rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)
            rgb = self.img_transform(rgb).numpy().transpose((1,2,0)) # HWC
        else:
            rgb = np.zeros((modal.shape[1], modal.shape[2], 3), dtype=np.uint8)

        new_bboxes = []
        for bbox in bboxes:
            centerx = bbox[0] + bbox[2] / 2.
            centery = bbox[1] + bbox[3] / 2.
            size = max([np.sqrt(bbox[2] * bbox[3] * self.config['enlarge_box']), bbox[2] * 1.1, bbox[3] * 1.1])
            new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
            new_bboxes.append(new_bbox)
        new_bboxes = np.array(new_bboxes)
        return rgb, modal, category, new_bboxes, amodal, gt_order_matrix
