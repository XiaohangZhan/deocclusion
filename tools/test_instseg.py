import argparse
import yaml
import os
import json
import copy
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
sys.path.append('.')
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

from datasets import reader
import models
import inference as infer
import utils

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--load-iter', required=True, type=int)
    parser.add_argument('--annot-fn', required=True, type=str)
    parser.add_argument('--image-root', required=True, type=str)
    parser.add_argument('--gt-annot', default=None, type=str)
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--test-num', default=-1, type=int)
    parser.add_argument('--th', default=0.5, type=float)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    tester = Tester(args)
    tester.run()

class Tester(object):
    def __init__(self, args):
        self.args = args
        self.prepare_data()

    def prepare_data(self):
        config = self.args.data
        dataset = config['dataset']
        self.dataset = dataset
        self.data_root = self.args.image_root
        if dataset == 'SAS':
            self.data_reader = reader.SASDataset(self.args.annot_fn)
            self.img_ids = list(range(self.data_reader.get_image_length()))
        else:
            self.data_reader = reader.KINSLVISDataset(
                dataset, self.args.annot_fn)
            self.img_ids = self.data_reader.img_ids

        if self.args.test_num != -1:
            self.img_ids = self.img_ids[:self.args.test_num]
        self.output_path = self.args.output_path
        self.gt_annot = self.args.gt_annot
        self.img_transform = transforms.Compose([
            transforms.Normalize(config['data_mean'], config['data_std'])
        ])

    def prepare_model(self):
        self.model = models.__dict__[self.args.model['algo']](self.args.model, dist_model=False)
        self.model.load_state("{}/checkpoints".format(self.args.exp_path), self.args.load_iter)
        self.model.switch_to('eval')

    def expand_bbox(self, bboxes):
        new_bboxes = []
        for bbox in bboxes:
            centerx = bbox[0] + bbox[2] / 2.
            centery = bbox[1] + bbox[3] / 2.
            size = max([np.sqrt(bbox[2] * bbox[3] * self.args.data['enlarge_box']),
                        bbox[2] * 1.1, bbox[3] * 1.1])
            new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
            new_bboxes.append(new_bbox)
        return np.array(new_bboxes)

    def run(self):
        if not os.path.isfile(self.output_path) or args.force:
            self.prepare_model()
            self.infer()
        if self.dataset != "SAS":
            self.evaluate()

    def infer(self):
        segm_json_results = []
        count = 0
        for i,imgid in tqdm(enumerate(self.img_ids), total=len(self.img_ids)):
            modal, category, bboxes, _, image_fn, anns = self.data_reader.get_image_instances(
                i, with_gt=False, with_anns=True) # only use modal, category, and bboxes
            #
            image = np.array(Image.open(os.path.join(self.data_root, image_fn)).convert('RGB'))
            image_tensor = torch.from_numpy(image.astype(np.float32).transpose((2, 0, 1)) / 255.)
            image_transformed = self.img_transform(image_tensor).numpy().transpose((1,2,0))

            h, w = image_transformed.shape[:2]
            new_bboxes = self.expand_bbox(bboxes)

            # infer
            if False: # crf
                seg_patches_pred = infer.infer_instseg(
                    self.model, image_transformed, category, bboxes, new_bboxes, input_size=256,
                    th=self.args.th, rgb=image.copy())
            else:
                seg_patches_pred = infer.infer_instseg(
                    self.model, image_transformed, category, bboxes, new_bboxes, input_size=256,
                    th=self.args.th)
            seg_pred = infer.patch_to_fullimage(
                seg_patches_pred, new_bboxes, h, w, interp='linear' )
            
            # encode
            if self.dataset == "SAS":
                data = copy.deepcopy(anns)
                for i in range(seg_pred.shape[0]):
                    rle = maskUtils.encode(
                        np.array(seg_pred[i, :, :, np.newaxis], order='F'))[0]
                    if isinstance(rle['counts'], bytes):
                        rle['counts'] = rle['counts'].decode()
                    data['regions'][i]['visible_mask'] = rle
                segm_json_results.append(data)
            else:
                for i in range(seg_pred.shape[0]):
                    data = dict()
                    rle = maskUtils.encode(
                        np.array(seg_pred[i, :, :, np.newaxis], order='F'))[0]
                    data['image_id'] = imgid
                    data['category_id'] = category[i].item()
                    if isinstance(rle['counts'], bytes):
                        rle['counts'] = rle['counts'].decode()
                    data['segmentation'] = rle
                    data['bbox'] = utils.mask_to_bbox(seg_pred[i, :, :])
                    data['area'] = float(data['bbox'][2] * data['bbox'][3])
                    data['iscrowd'] = 0
                    data['score'] = 1.
                    data['id'] = anns[i]['id']
                    segm_json_results.append(data)
                    count += 1

        if self.dataset == 'SAS':
            output_dict = {'images': self.data_reader.images_info,
                           'annotations': segm_json_results}
            with open(self.output_path, 'w') as f:
                json.dump(output_dict, f)
        else:
            with open(self.output_path, 'w') as f:
                json.dump(segm_json_results, f)

    def evaluate(self):
        annType = 'segm'
        cocoGt = COCO(self.gt_annot)
        cocoDt = cocoGt.loadRes(self.output_path)
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        #cocoEval.params.imgIds = cocoGt.getImgIds()
        cocoEval.params.imgIds = self.img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

if __name__ == "__main__":
    args = parse_args()
    main(args)
