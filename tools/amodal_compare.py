import numpy as np
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import yaml
from PIL import Image
import sys
sys.path.append('.')
import tqdm
import pdb

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import models
import inference as infer
from datasets import KINSLVISDataset, SASDataset, reader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--phase', type=str, default='val')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num', type=str, default=-1)
    return parser.parse_args()

class ArgObj(object):
    def __init__(self):
        pass

class Tester(object):
    
    def __init__(self, config_file, load_iter):
        args = ArgObj()
        with open(config_file, 'r') as f:
            config = yaml.load(f)
        for k, v in config.items():
            setattr(args, k, v)
        
        if not hasattr(args, 'exp_path'):
            args.exp_path = os.path.dirname(config_file)
        
        self.model = models.__dict__[args.model['algo']](args.model, dist_model=False)
        self.model.load_state("{}/checkpoints".format(args.exp_path), load_iter)
        
        self.model.switch_to('eval')
        self.use_rgb = args.model['use_rgb']
        
        self.args = args

def expand_bbox(bboxes, enlarge_ratio):
    new_bboxes = []
    for bbox in bboxes:
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * enlarge_ratio), bbox[2] * 1.1, bbox[3] * 1.1])
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        new_bboxes.append(new_bbox)
    return np.array(new_bboxes)


def polygon_drawing(masks, color_source, bbox):
    polygons = []
    colors = []
    if bbox is not None:
        l,u,r,b = bbox
        masks = masks[:, u:b, l:r]
    for i,am in enumerate(masks):
        pts_list = reader.mask_to_polygon(am)
        for pts in pts_list:
            pts = np.array(pts).reshape(-1, 2)
            polygons.append(Polygon(pts))
            colors.append(color_source[i])
    pface = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.4)
    pedge = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
    return pface, pedge


def main(args):
    if args.dataset == 'KINS':
        if args.phase == 'train':
            img_root = "data/KINS/2D-Det/training/image_2"
        else:
            img_root = "data/KINS/2D-Det/testing/image_2"
        gt_path = "data/KINS/instances_{}.json".format(args.phase)
        data_reader = KINSLVISDataset('KINS', gt_path)
        config = 'experiments/KINS/partial_unet2_in5_front0.8/config.yaml'
        tester = Tester(config, 32000)
    elif args.dataset == 'SAS':
        img_root = 'data/SAS/sas_{}'.format(args.phase)
        annot_fn = 'data/SAS/annotations/COCO_amodal_{}2014.json'.format(args.phase)
        data_reader = SASDataset(annot_fn)
        config = 'experiments/SAS/partial_unet2_in5_front0.8/config.yaml'
        tester = Tester(config, 56000)
    else:
        raise Exception("No such dataset: {}".format(args.dataset))

    out_path = "{}/{}_{}".format(args.output, args.dataset, args.phase)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    if args.num == -1:
        total = data_reader.get_image_length()
    else:
        total = args.num
    for img_idx in tqdm.tqdm(range(args.start, total), total=total-args.start):
        modal, category, ori_bboxes, amodal_gt, image_fn = \
            data_reader.get_image_instances(img_idx, with_gt=True)
        num = modal.shape[0]
        if args.dataset == 'SAS':
            img_path = os.path.join(img_root, "{:04d}.jpg".format(img_idx))
        else:
            img_path = os.path.join(img_root, image_fn)
        img = Image.open(img_path)
        image = np.array(img)
        height, width = img.height, img.width

        bboxes = expand_bbox(ori_bboxes, enlarge_ratio=3.)

        # convex
        order_matrix_cvx = infer.infer_order_hull(modal)
        amodal_pred_cvx = np.array(infer.infer_amodal_hull(modal, bboxes, order_matrix_cvx))

        # ours
        order_matrix_ours = infer.infer_order2(
            tester.model, image, modal, category, bboxes,
            use_rgb=tester.use_rgb, th=0.1, dilate_kernel=0,
            input_size=256, min_input_size=16, interp='nearest', debug_info=False)

        amodal_patches_pred = infer.infer_amodal(
            tester.model, image, modal, category, bboxes, order_matrix_ours,
            use_rgb=tester.use_rgb, th=0.2, dilate_kernel=0,
            input_size=256, min_input_size=16, interp='linear', debug_info=False)

        amodal_pred_ours = infer.patch_to_fullimage(
            amodal_patches_pred, bboxes, image.shape[0], image.shape[1], interp='linear')

        # drawing
        colors = [(np.random.random((1, 3))*0.6+0.4).tolist()[0] for i in range(num)]

        if args.dataset == 'SAS':
            rows, cols = 2, 2
        else:
            rows, cols = 4, 1
        plt.figure(figsize=(16, 16./ width * height * rows / cols))
        show = [modal, amodal_pred_cvx, amodal_pred_ours, amodal_gt]
        title = ['modal', 'convex', 'ours', 'gt']
        for i in range(4):
            plt.subplot(rows, cols, i + 1)
            ax = plt.gca()
            plt.imshow(image)
            plt.axis('off')
            plt.text(0, -10, title[i])
            pface, pedge = polygon_drawing(show[i], colors, None)
            ax.add_collection(pface)
            ax.add_collection(pedge)

        plt.savefig('{}/{}_{}/{:04d}.png'.format(args.output, args.dataset, args.phase, img_idx))


if __name__ == "__main__":
    args = parse_args()
    main(args)
