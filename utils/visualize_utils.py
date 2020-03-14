import numpy as np

import torch

def visualize_tensor(tensors_dict, mean, div):
    together = []
    for ct in tensors_dict['common_tensors']:
        ct = unormalize(ct.detach().cpu(), mean, div)
        ct *= 255
        ct = torch.clamp(ct, 0, 255)
        together.append(ct)
    for mt in tensors_dict['mask_tensors']:
        if mt.size(1) == 1:
            mt = mt.repeat(1,3,1,1)
        mt = mt.float().detach().cpu() * 255
        together.append(mt)
    if len(together) == 0:
        return None
    together = torch.cat(together, dim=3)
    together = together.permute(1,0,2,3).contiguous()
    together = together.view(together.size(0), -1, together.size(3))
    return together

def unormalize(tensor, mean, div):
    for c, (m, d) in enumerate(zip(mean, div)):
        tensor[:,c,:,:].mul_(d).add_(m)
    return tensor
