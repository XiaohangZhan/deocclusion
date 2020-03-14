import argparse
import yaml
import os
import json
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
sys.path.append('.')
from datasets import reader
import models
import inference as infer
import utils
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--load-iter', required=True, type=int)
    parser.add_argument('--method', required=True, type=str)
    parser.add_argument('--modal-res', required=True, type=str)
    parser.add_argument('--image-root', required=True, type=str)
    parser.add_argument('--gt-annot', required=True, type=str)
    parser.add_argument('--test-num', default=-1, type=int)
    parser.add_argument('--output', default=None, type=str)
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
        self.data_root = self.args.image_root
        if dataset == 'SAS':
            self.data_reader = reader.SASDataset(self.args.modal_res)
        else:
            self.data_reader = reader.KINSLVISDataset(
                dataset, self.args.modal_res)
        self.img_ids = self.data_reader.img_ids
        if self.args.test_num != -1:
            self.img_ids = self.img_ids[:self.args.test_num]
        self.gt_annot = self.args.gt_annot

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
        if not os.path.isfile(self.args.output) or args.force:
            self.prepare_model()
            self.infer()
        self.evaluate()

    def infer(self):
        order_th = self.args.model['inference']['positive_th_order']
        amodal_th = self.args.model['inference']['positive_th_amodal']

        segm_json_results = []
        count = 0
        
        for i,imgid in tqdm(enumerate(self.img_ids), total=len(self.img_ids)):
            modal, category, bboxes, amodal_gt, image_fn = self.data_reader.get_image_instances(
                i, with_gt=True)

            # data
            image = np.array(Image.open(os.path.join(self.data_root, image_fn)))
            h, w = image.shape[:2]
            bboxes = self.expand_bbox(bboxes)

            # infer
            if self.args.method == 'ours':
                order_matrix = infer.infer_order2(
                    self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=order_th, dilate_kernel=0,
                    input_size=256, min_input_size=16, interp='nearest', debug_info=False)
    
                amodal_patches_pred = infer.infer_amodal(
                    self.model, image, modal, category, bboxes, order_matrix,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, dilate_kernel=0,
                    input_size=256, min_input_size=16, interp='linear', debug_info=False)
    
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.method == 'ours_nog':
                order_matrix = infer.infer_order2(
                    self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=order_th, dilate_kernel=0,
                    input_size=256, min_input_size=16, interp='nearest', debug_info=False)
                amodal_patches_pred = infer.infer_amodal(
                    self.model, image, modal, category, bboxes, order_matrix,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, dilate_kernel=0,
                    input_size=256, min_input_size=16, interp='linear',
                    order_grounded=False, debug_info=False)
    
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.method == 'sup':
                amodal_patches_pred = infer.infer_amodal_sup(
                    self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, input_size=256,
                    min_input_size=16, interp='linear')
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.method == 'complexity':
                order_matrix = infer.infer_gt_order(modal, amodal_gt) # use gt order
                amodal_pred = np.array(infer.infer_amodal_hull(
                    modal, bboxes, order_matrix, order_grounded=True))

            elif self.args.method == 'convex':
                amodal_pred = np.array(infer.infer_amodal_hull(
                    modal, bboxes, None, order_grounded=False))

            elif self.args.method == 'convexr':
                order_matrix = infer.infer_order_hull(modal)
                amodal_pred = np.array(infer.infer_amodal_hull(
                    modal, bboxes, order_matrix, order_grounded=True))

            else:
                raise Exception("No such method: {}".format(self.args.method))

            # encode
            for i in range(amodal_pred.shape[0]):
                data = dict()
                rle = maskUtils.encode(
                    np.array(amodal_pred[i, :, :, np.newaxis], order='F'))[0]
                data['image_id'] = imgid
                data['category_id'] = category[i].item()
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode()
                data['segmentation'] = rle
                data['bbox'] = utils.mask_to_bbox(amodal_pred[i, :, :])
                data['area'] = float(data['bbox'][2] * data['bbox'][3])
                data['iscrowd'] = 0
                data['score'] = 1.
                data['id'] = count
                segm_json_results.append(data)
                count += 1

        with open(self.args.output, 'w') as f:
            json.dump(segm_json_results, f)

    def evaluate(self):
        annType = 'segm'
        cocoGt = COCO(self.gt_annot)
        cocoDt = cocoGt.loadRes(self.args.output)
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.params.imgIds = self.img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

if __name__ == "__main__":
    args = parse_args()
    main(args)
