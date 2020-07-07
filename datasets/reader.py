import json
import numpy as np
import sys
from PIL import Image
sys.path.append('.')

import cvbase as cvb
import pycocotools.mask as maskUtils
import utils

def read_KINS(ann):
    modal = maskUtils.decode(ann['inmodal_seg']) # HW, uint8, {0, 1}
    bbox = ann['inmodal_bbox'] # luwh
    category = ann['category_id']
    if 'score' in ann.keys():
        score = ann['score']
    else:
        score = 1.
    return modal, bbox, category, score

def read_LVIS(ann, h, w):
    segm = ann["segmentation"]
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann["segmentation"]
    bbox = ann['bbox'] # luwh
    category = ann['category_id']
    return maskUtils.decode(rle), bbox, category

def read_COCOA(ann, h, w):
    if 'visible_mask' in ann.keys():
        rle = [ann['visible_mask']]
    else:
        rles = maskUtils.frPyObjects([ann['segmentation']], h, w)
        rle = maskUtils.merge(rles)
    modal = maskUtils.decode(rle).squeeze()
    if np.all(modal != 1):
        # if the object if fully occluded by others,
        # use amodal bbox as an approximated location,
        # note that it will produce random amodal results.
        amodal = maskUtils.decode(maskUtils.merge(
            maskUtils.frPyObjects([ann['segmentation']], h, w)))
        bbox = utils.mask_to_bbox(amodal)
    else:
        bbox = utils.mask_to_bbox(modal)
    return modal, bbox, 1 # category as constant 1


class COCOADataset(object):

    def __init__(self, annot_fn):
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']

        self.indexing = []
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann['regions'])):
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_image_length(self):
        return len(self.images_info)

    def get_gt_ordering(self, imgidx):
        num = len(self.annot_info[imgidx]['regions'])
        gt_order_matrix = np.zeros((num, num), dtype=np.int)
        order_str = self.annot_info[imgidx]['depth_constraint']
        if len(order_str) == 0:
            return gt_order_matrix
        order_str = order_str.split(',')
        for o in order_str:
            idx1, idx2 = o.split('-')
            idx1, idx2 = int(idx1) - 1, int(idx2) - 1
            gt_order_matrix[idx1, idx2] = 1
            gt_order_matrix[idx2, idx1] = -1
        return gt_order_matrix # num x num

    def get_instance(self, idx, with_gt=False):
        imgidx, regidx = self.indexing[idx]
        # img
        img_info = self.images_info[imgidx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        # region
        reg = self.annot_info[imgidx]['regions'][regidx]
        modal, bbox, category = read_COCOA(reg, h, w)
        if with_gt:
            amodal = maskUtils.decode(maskUtils.merge(
                maskUtils.frPyObjects([reg['segmentation']], h, w)))
        else:
            amodal = None
        return modal, bbox, category, image_fn, amodal

    def get_image_instances(self, idx, with_gt=False, with_anns=False, ignore_stuff=False):
        ann_info = self.annot_info[idx]
        img_info = self.images_info[idx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        for reg in ann_info['regions']:
            if ignore_stuff and reg['isStuff']:
                continue
            modal, bbox, category = read_COCOA(reg, h, w)
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            if with_gt:
                amodal = maskUtils.decode(maskUtils.merge(
                    maskUtils.frPyObjects([reg['segmentation']], h, w)))
                ret_amodal.append(amodal)
        if with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn, ann_info
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn


class KINSLVISDataset(object):

    def __init__(self, dataset, annot_fn):
        self.dataset = dataset
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']
        self.category_info = data['categories']

        # make dict
        self.imgfn_dict = dict([(a['id'], a['file_name']) for a in self.images_info])
        self.size_dict = dict([(a['id'], (a['width'], a['height'])) for a in self.images_info])
        self.anns_dict = self.make_dict()
        self.img_ids = list(self.anns_dict.keys())

    def get_instance_length(self):
        return len(self.annot_info)

    def get_image_length(self):
        return len(self.img_ids)

    def get_instance(self, idx, with_gt=False):
        ann = self.annot_info[idx]
        # img
        imgid = ann['image_id']
        w, h = self.size_dict[imgid]
        image_fn = self.imgfn_dict[imgid]
        # instance
        if self.dataset == 'KINS':
            modal, bbox, category, _ = read_KINS(ann)
        elif self.dataset == 'LVIS':
            modal, bbox, category = read_LVIS(ann, h, w)
        else:   
            raise Exception("No such dataset: {}".format(self.dataset))
        if with_gt:
            amodal = maskUtils.decode(
                maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
        else:
            amodal = None
        return modal, bbox, category, image_fn, amodal

    def make_dict(self):
        anns_dict = {}
        for ann in self.annot_info:
            image_id = ann['image_id']
            if not image_id in anns_dict:
                anns_dict[image_id] = [ann]
            else:
                anns_dict[image_id].append(ann)
        return anns_dict # imgid --> anns

    def get_image_instances(self, idx, with_gt=False, with_anns=False):
        imgid = self.img_ids[idx]
        image_fn = self.imgfn_dict[imgid]
        w, h = self.size_dict[imgid]
        anns = self.anns_dict[imgid]
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        #ret_score = []
        for ann in anns:
            if self.dataset == 'KINS':
                modal, bbox, category, score = read_KINS(ann)
            elif self.dataset == 'LVIS':
                modal, bbox, category = read_LVIS(ann, h, w)
            else:
                raise Exception("No such dataset: {}".format(self.dataset))
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            #ret_score.append(score)
            if with_gt:
                amodal = maskUtils.decode(
                    maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
                ret_amodal.append(amodal)
        if with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn, anns
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn


class MapillaryDataset(object):

    def __init__(self, root, annot_fn):
        with open(annot_fn, 'r') as f:
            annot = json.load(f)
        self.categories = annot['categories']
        self.annot_info = annot['images']
        self.root = root # e.g., "data/manpillary/training"
        self.indexing = []
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann['regions'])):
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_image_length(self):
        return len(self.annot_info)

    def get_instance(self, idx, with_gt=False):
        assert not with_gt, \
            "Mapillary Vista has no ground truth for ordering or amodal masks."
        imgidx, regidx = self.indexing[idx]
        # img
        image_id = self.annot_info[imgidx]['image_id']
        image_fn = image_id + ".jpg"
        # region
        instance_map = np.array(
            Image.open("{}/instances/{}.png".format(
            self.root, image_id)), dtype=np.uint16)
        h, w = instance_map.shape[:2]
        reg_info = self.annot_info[imgidx]['regions'][regidx]
        modal = (instance_map == reg_info['instance_id']).astype(np.uint8)
        category = reg_info['category_id']
        bbox = np.array(utils.mask_to_bbox(modal))
        return modal, bbox, category, image_fn, None

    def get_image_instances(self, idx, with_gt=False, with_anns=False, ignore_stuff=False):
        assert not with_gt
        assert not ignore_stuff
        # img
        image_id = self.annot_info[idx]['image_id']
        image_fn = image_id + ".jpg"
        # region
        instance_map = np.array(
            Image.open("{}/instances/{}.png".format(
            self.root, image_id)), dtype=np.uint16)
        h, w = instance_map.shape[:2]
        instance_ids = np.unique(instance_map)
        category = instance_ids // 256
        num_instance = len(instance_ids)
        instance_ids_tensor = np.zeros((num_instance, h, w), dtype=np.uint16)   
        instance_ids_tensor[...] = instance_ids[:, np.newaxis, np.newaxis]
        modal = (instance_ids_tensor == instance_map).astype(np.uint8)
        bboxes = []
        for i in range(modal.shape[0]):
            bboxes.append(utils.mask_to_bbox(modal[i,...]))
        return modal, category, np.array(bboxes), None, image_fn


def mask_to_polygon(mask, tolerance=1.0, area_threshold=1):
    """Convert object's mask to polygon [[x1,y1, x2,y2 ...], [...]]
    Args:
        mask: object's mask presented as 2D array of 0 and 1
        tolerance: maximum distance from original points of polygon to approximated
        area_threshold: if area of a polygon is less than this value, remove this small object
    """
    from skimage import measure
    polygons = []
    # pad mask with 0 around borders
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_mask, 0.5)
    # Fix coordinates after padding
    contours = np.subtract(contours, 1)
    for contour in contours:
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) > 2:
            contour = np.flip(contour, axis=1)
            reshaped_contour = []
            for xy in contour:
                reshaped_contour.append(xy[0])
                reshaped_contour.append(xy[1])
            reshaped_contour = [point if point > 0 else 0 for point in reshaped_contour]

            # Check if area of a polygon is enough
            rle = maskUtils.frPyObjects([reshaped_contour], mask.shape[0], mask.shape[1])
            area = maskUtils.area(rle)
            if sum(area) > area_threshold:
                polygons.append(reshaped_contour)
    return polygons


if __name__ == '__main__':
    phase = 'validation'
    root = '../data/mapillary'
    image_root = "{}/{}".format(root, phase)
    annot_path = "../data/mapillary/meta/{}.json".format(phase)
    data_reader = MapillaryDataset(root, annot_path)
    modal, category, ori_bboxes, _, image_fn = data_reader.get_image_instances(0, with_gt=False)
