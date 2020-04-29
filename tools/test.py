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
    parser.add_argument('--load-model', required=True, type=str)
    parser.add_argument('--order-method', required=True, type=str)
    parser.add_argument('--amodal-method', required=True, type=str)
    parser.add_argument('--order-th', default=0.1, type=float)
    parser.add_argument('--amodal-th', default=0.2, type=float)
    parser.add_argument('--annotation', required=True, type=str)
    parser.add_argument('--image-root', required=True, type=str)
    parser.add_argument('--test-num', default=-1, type=int)
    parser.add_argument('--output', default=None, type=str)
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
        if dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(self.args.annotation)
        else:
            self.data_reader = reader.KINSLVISDataset(
                dataset, self.args.annotation)
        self.data_length = self.data_reader.get_image_length()
        self.dataset = dataset
        if self.args.test_num != -1:
            self.data_length = self.args.test_num

    def prepare_model(self):
        self.model = models.__dict__[self.args.model['algo']](self.args.model, dist_model=False)
        self.model.load_state(self.args.load_model)
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
        self.prepare_model()
        self.infer()

    def infer(self):
        order_th = self.args.order_th
        amodal_th = self.args.amodal_th

        segm_json_results = []
        self.count = 0
        
        allpair_true_rec = utils.AverageMeter()
        allpair_rec = utils.AverageMeter()
        occpair_true_rec = utils.AverageMeter()
        occpair_rec = utils.AverageMeter()
        intersection_rec = utils.AverageMeter()
        union_rec = utils.AverageMeter()
        target_rec = utils.AverageMeter()

        for i in tqdm(range(self.data_length), total=self.data_length):
            modal, category, bboxes, amodal_gt, image_fn = self.data_reader.get_image_instances(
                i, with_gt=True)

            # data
            image = np.array(Image.open(os.path.join(self.data_root, image_fn)).convert('RGB'))
            h, w = image.shape[:2]
            bboxes = self.expand_bbox(bboxes)

            # gt order
            gt_order_matrix = infer.infer_gt_order(modal, amodal_gt)

            # infer order
            if self.args.order_method == 'area':
                order_matrix = infer.infer_order_area(
                    modal, above='smaller' if self.args.data['dataset'] == 'COCOA' else 'larger')

            elif self.args.order_method == 'yaxis':
                order_matrix = infer.infer_order_yaxis(modal)

            elif self.args.order_method == 'convex':
                order_matrix = infer.infer_order_convex(modal)

            elif self.args.order_method == 'ours':
                order_matrix = infer.infer_order(
                    self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=order_th, dilate_kernel=0,
                    input_size=256, min_input_size=16, interp='nearest', debug_info=False)
            else:
                raise Exception('No such order method: {}'.format(self.args.order_method))

            # infer amodal
            if self.args.amodal_method == 'raw':
                amodal_pred = modal.copy()

            elif self.args.amodal_method == 'ours_nog':
                amodal_patches_pred = infer.infer_amodal(
                    self.model, image, modal, category, bboxes, order_matrix,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, dilate_kernel=0,
                    input_size=256, min_input_size=16, interp='linear',
                    order_grounded=False, debug_info=False)
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.amodal_method == 'ours':
                amodal_patches_pred = infer.infer_amodal(
                    self.model, image, modal, category, bboxes, order_matrix,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, dilate_kernel=0,
                    input_size=256, min_input_size=16, interp='linear',
                    order_grounded=True, debug_info=False)
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.amodal_method == 'sup': # supervised
                amodal_patches_pred = infer.infer_amodal_sup(
                    self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, input_size=256,
                    min_input_size=16, interp='linear')
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.amodal_method == 'convex':
                amodal_pred = np.array(infer.infer_amodal_hull(
                    modal, bboxes, None, order_grounded=False))

            elif self.args.amodal_method == 'convexr':
                order_matrix = infer.infer_order_hull(modal)
                amodal_pred = np.array(infer.infer_amodal_hull(
                    modal, bboxes, order_matrix, order_grounded=True))

            else:
                raise Exception("No such method: {}".format(self.args.method))

            # eval
            allpair_true, allpair, occpair_true, occpair, _ = infer.eval_order(
                order_matrix, gt_order_matrix)
            allpair_true_rec.update(allpair_true)
            allpair_rec.update(allpair)
            occpair_true_rec.update(occpair_true)
            occpair_rec.update(occpair)

            intersection = ((amodal_pred == 1) & (amodal_gt == 1)).sum()
            union = ((amodal_pred == 1) | (amodal_gt == 1)).sum()
            target = (amodal_gt == 1).sum()
            intersection_rec.update(intersection)
            union_rec.update(union)
            target_rec.update(target)

            # make output
            if self.dataset == 'KINS':
                segm_json_results.extend(self.make_KINS_output(i, amodal_pred, category))

        # print results
        acc_allpair = allpair_true_rec.sum / float(allpair_rec.sum) # accuracy for all pairs
        acc_occpair = occpair_true_rec.sum / float(occpair_rec.sum) # accuray for occluded pairs
        miou = intersection_rec.sum / (union_rec.sum + 1e-10) # mIoU
        pacc = intersection_rec.sum / (target_rec.sum + 1e-10) # pixel accuracy
        print("Evaluation results. acc_allpair: {:.5g}, acc_occpair: {:.5g} \
              mIoU: {:.5g}, pAcc: {:.5g}".format(acc_allpair, acc_occpair, miou, pacc))

        # save
        if not os.path.isdir(os.path.dirname(self.args.output)):
            os.makedirs(os.path.dirname(self.args.output))
        with open(self.args.output, 'w') as f:
            json.dump(segm_json_results, f)

    def make_KINS_output(self, idx, amodal_pred, category):
        results = []
        for i in range(amodal_pred.shape[0]):
            data = dict()
            rle = maskUtils.encode(
                np.array(amodal_pred[i, :, :, np.newaxis], order='F'))[0]
            if hasattr(self.data_reader, 'img_ids'):
                data['image_id'] = self.data_reader.img_ids[idx]
            data['category_id'] = category[i].item()
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode()
            data['segmentation'] = rle
            data['bbox'] = utils.mask_to_bbox(amodal_pred[i, :, :])
            data['area'] = float(data['bbox'][2] * data['bbox'][3])
            data['iscrowd'] = 0
            data['score'] = 1.
            data['id'] = self.count
            results.append(data)
            self.count += 1
        return results


if __name__ == "__main__":
    args = parse_args()
    main(args)
