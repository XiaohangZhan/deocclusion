import numpy as np
try:
    import mc
except Exception:
    pass
import os
from PIL import Image

import torch
from torch.utils.data import Dataset


class ImageRawDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        self.use_rgb = config.get('use_rgb', False)
        assert self.dataset in ["Image", "Npy"]
        if self.dataset == "Image":
            with open(config['{}_list'.format(phase)], 'r') as f:
                lines = f.readlines()
            self.fns = [os.path.join(config['{}_root'.format(phase)],
                l.strip()) for l in lines]
            if self.use_rgb:
                self.rgb_fns = [os.path.join(config['{}_rgb_root'.format(phase)],
                    l.strip()) for l in lines]
            self.num = len(self.fns)
        else:
            if self.use_rgb:
                self.rgb = np.fromfile(config['{}_rgb_file'.format(phase)],
                    dtype=np.uint8).reshape(-1, 256, 256, 3)
            self.data = np.fromfile(config['{}_file'.format(phase)],
                dtype=np.uint8).reshape(-1, 256, 256)
            self.num = self.data.shape[0]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.dataset == "Image":
            fn = self.fns[idx]
            img = np.array(Image.open(fn).convert('RGB'))
            if self.use_rgb:
                rgb = np.array(Image.open(self.rgb_fns[idx]).convert('RGB')) # HW3
        else:
            img = self.data[idx, :, :]
            if self.use_rgb:
                rgb = self.rgb[idx, ...] # HW3

        img[img == 128] = 0
        if self.use_rgb:
            masked_rgb = rgb.astype(np.float32) / 255. * (
                np.tile(img.astype(np.float32)[:,:,np.newaxis], [1,1,3]) / 255.)
            input = torch.from_numpy(masked_rgb.transpose(2, 0, 1)) # 3HW
        else:
            input = torch.from_numpy(img.astype(np.float32)).unsqueeze(0) / 255
        return input
