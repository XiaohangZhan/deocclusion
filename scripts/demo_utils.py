import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
sys.path.append('..')
import cv2
import yaml
import numpy as np

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import models
import utils
from datasets import reader

class ArgObj(object):
    def __init__(self):
        pass


class DemoPCNetM(object):
    
    def __init__(self, config_file, load_model):
        args = ArgObj()
        with open(config_file, 'r') as f:
            config = yaml.load(f)
        for k, v in config.items():
            setattr(args, k, v)
        
        if not hasattr(args, 'exp_path'):
            args.exp_path = os.path.dirname(config_file)
        
        self.model = models.__dict__[args.model['algo']](args.model, dist_model=False)
        self.model.load_state(load_model)
        
        self.model.switch_to('eval')
        self.use_rgb = args.model['use_rgb']
        
        self.args = args


class DemoPCNetC(object):
    
    def __init__(self, config_file, load_model):
        args = ArgObj()
        with open(config_file, 'r') as f:
            config = yaml.load(f)
        for k, v in config.items():
            setattr(args, k, v)
        
        if not hasattr(args, 'exp_path'):
            args.exp_path = os.path.dirname(config_file)
        
        self.model = models.__dict__[args.model['algo']](args.model, dist_model=False, demo=True)
        self.model.load_model_demo(load_model)
        
        self.model.switch_to('eval')
        self.img_transform = transforms.Compose([
            transforms.Normalize(args.data['data_mean'], args.data['data_std'])])
        
        self.args = args

    def inference(self, rgb, modal, category, amodal, dilate=0, with_modal=True):
        rgb = self.img_transform(torch.from_numpy(
            rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)).unsqueeze(0)
        invisible_mask = ((amodal == 1) & (modal == 0)).astype(np.uint8)
        if dilate > 0:
            invisible_mask = cv2.dilate(invisible_mask, np.ones((dilate,dilate),np.uint8), iterations=1)
        visible_mask = (1 - invisible_mask).astype(np.float32)[np.newaxis,np.newaxis,:,:]
        visible_mask = torch.from_numpy(visible_mask) # 11HW
        modal = torch.from_numpy(modal.astype(np.float32)).unsqueeze(0).unsqueeze(0) * float(category) # 11HW
        rgb_erased = rgb * visible_mask.repeat(1,3,1,1)
        if with_modal:
            visible_mask4 = visible_mask.repeat(1, 4, 1, 1)
        else:
            visible_mask3 = visible_mask.repeat(1, 3, 1, 1)
        with torch.no_grad():
            if with_modal:
                output, _ = self.model.model(torch.cat([rgb_erased.cuda(), modal.cuda()], dim=1), visible_mask4.cuda())
            else:
                output, _ = self.model.model(rgb_erased.cuda(), visible_mask3.cuda())
            output_comp = visible_mask * rgb + (1 - visible_mask) * output.detach().cpu()
        result = torch.clamp(utils.unormalize(
            output_comp, self.args.data['data_mean'], self.args.data['data_std']) * 255, 0, 255)
        rgb_erased = torch.clamp(utils.unormalize(
            rgb_erased, self.args.data['data_mean'], self.args.data['data_std']) * 255, 0, 255)

        return result.numpy().astype(np.uint8).squeeze().transpose((1,2,0)), \
               rgb_erased.numpy().astype(np.uint8).squeeze().transpose((1,2,0)), \
               visible_mask.numpy().squeeze()
        
def show(inputs, scale=1.0, cols=-1):
    num = len(inputs)
    if cols == -1:
        cols = num
    rows = int(np.ceil(num / float(cols)))
    plt.figure(figsize=(12*scale, 12*scale/cols*rows))
    for i in range(num):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(inputs[i])
        plt.axis('off')
    plt.show()
    
def combine(inst, eraser):
    inst = inst.copy().astype(np.float32)
    inst[eraser == 1] = 0.5
    return inst

def colormap(mask):
    show = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for c in range(3):
        show[:,:,c][mask == 1] = 255
        show[:,:,c][mask == 0.5] = 128
    return show
    
def draw_graph(matrix, ind=None, pos=None):
    edges = np.where(matrix == 1)
    if ind is not None:
        from_idx = []
        to_idx = []
        for i in range(len(edges[0])):
            if edges[0][i] in ind and edges[1][i] in ind:
                from_idx.append(edges[0][i])
                to_idx.append(edges[1][i])
    else:
        from_idx = edges[0].tolist()
        to_idx = edges[1].tolist()
        
    from_node = [str(i+1) for i in from_idx]
    to_node = [str(i+1) for i in to_idx]

    G = nx.DiGraph()
    G.add_edges_from(list(zip(from_node, to_node)))
    if pos is None:
#         pos = nx.kamada_kawai_layout(G)
        pos = nx.spring_layout(G,k=0.5,iterations=20)
#         pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_color='w')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=2)
    return pos

def get_square_bbox(h, w):
    if h > w:
        return [-(h-w) // 2, 0, h, h]
    else:
        return [0, -(w-h) // 2, w, w]

def expand_bbox(bboxes, enlarge_ratio, single_ratio=1.1):
    new_bboxes = []
    for bbox in bboxes:
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * enlarge_ratio), bbox[2] * single_ratio, bbox[3] * single_ratio])
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        new_bboxes.append(new_bbox)
    return np.array(new_bboxes)

def gettrimap(mask, k):
    """
    Compute matting trimap from given mask.
    :param mask: binary ground truth mask
    :param k: number of extended pixels
    :return: matting trimap. 255 for groundtruth foreground, 127 for uncertain area, 0 for ground truth background
    """
    assert mask.max() == 1 and mask.min() == 0
    kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.uint8)
    dilate_mask = cv2.dilate(mask, kernel, iterations=1)
    erode_mask = 1 - cv2.dilate(1 - mask, kernel, iterations=1)
    trimap = np.zeros(mask.shape, dtype=np.uint8)
    trimap[dilate_mask == 1] = 127
    trimap[erode_mask == 1] = 255

    if trimap.max() != 255 or trimap.min() != 0:
        raise Exception('matting trimap failed.')
    return trimap

def matting(rgb, mask):
    trimapi = gettrimap(mask, 5)
    alphamapi = global_matting(rgb, trimapi)
    alphamapi = guided_filter(rgb, trimapi, alphamapi, 10, 1e-5)
    return alphamapi

def recover_image_patch(patch, bbox, h, w, pad_value, interp='cubic'):
    interp = {'cubic': cv2.INTER_CUBIC, 'linear': cv2.INTER_LINEAR,
              'nearest': cv2.INTER_NEAREST}[interp]
    size = bbox[2]
    patch = cv2.resize(patch, (size, size), interpolation=interp)
    woff, hoff = bbox[0], bbox[1]
    newbbox = [-woff, -hoff, w, h]
    return utils.crop_padding(patch, newbbox, pad_value=pad_value)

def polygon_drawing(masks, selidx, color_source, bbox, thickness=2):
    polygons = []
    colors = []
    if bbox is not None:
        l,u,r,b = bbox
        masks = masks[:, u:b, l:r]
    for i,am in enumerate(masks[selidx,...]):
        pts_list = reader.mask_to_polygon(am)
        for pts in pts_list:
            pts = np.array(pts).reshape(-1, 2)
            polygons.append(Polygon(pts))
            colors.append(color_source[i])
    pface = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.4)
    pedge = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=thickness)
    return pface, pedge

def image_resize(image, short_size=None, long_size=None):
    '''
    Resize image by specify short_size or long_size
    img: numpy.ndarray
    '''
    assert (short_size is None) ^ (long_size is None)
    h, w = image.shape[:2]
    if short_size is not None:
        if w < h:
            neww = short_size
            newh = int(short_size / float(w) * h)
        else:
            neww = int(short_size / float(h) * w)
            newh = short_size
    else:
        if w < h:
            neww = int(long_size / float(h) * w)
            newh = long_size
        else:
            neww = long_size
            newh = int(long_size / float(w) * h)
    image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_LINEAR)
    return image

def gammaCorrection(img_original, gamma=1.0):
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img_original, lookUpTable)
    return res

def topological_sort(order_matrix):
    """
    Kahn ('62) topological sort.
    :param DAG: directed acyclic graph
    :type DAG: dict
    """
    N = order_matrix.shape[0]
    DAG = {}
    for i in range(N):
        DAG[i] = [j for j in range(N) if order_matrix[i, j] == 1]
    L = []
    S = [k for k, v in DAG.items() if not v]
    while S:
        n = S.pop(0)
        L.append(n)
        for m in (k for k, v in DAG.items() if n in v):
            DAG[m] = list(set(DAG[m]).difference([n]))
            if not DAG[m]:
                S.append(m)
    if any([bool(v) for v in DAG.values()]):
        raise Exception("Not a DAG, cannot perform topogical sort.")
    return np.array(L)
