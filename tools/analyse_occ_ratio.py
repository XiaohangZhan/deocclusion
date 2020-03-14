import multiprocessing as mp
import tqdm
import numpy as np
import json
import sys
import pycocotools.mask as maskUtils

def read_annot(ann, h, w):
    modal = maskUtils.decode(ann['inmodal_seg'])
    amodal = maskUtils.decode(maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
    return modal, amodal

def task(ann, data, size_dict):
    w, h = size_dict[ann['image_id']]
    amp = maskUtils.decode(data['segmentation']).astype(np.bool)
    m, amg = read_annot(ann, h, w)
    return [((amp == 1) & (amg == 1)).sum(),
            ((amp == 1) | (amg == 1)).sum(),
            m.sum(), amg.sum()]

def helper(args):
    return task(*args)

def compute(data, annot_data, size_dict):
    num = len(data)
    pool = mp.Pool(16)
    args = zip(annot_data, data, [size_dict] * num)
    ret = list(tqdm.tqdm(pool.imap(helper, args), total=num))
    return np.array(ret) # Nx4

if __name__ == "__main__":
    fold = "data/KINS/amodal_res_val"
    method = sys.argv[1]
    if method == 'raw':
        res_fn = 'data/KINS/annot_cocofmt/instances_inmodal_test.json'
        res_data = json.load(open(res_fn, 'r'))['annotations']
    else:
        res_fn = '{}/amodalcomp_test_{}.json'.format(fold, method)
        res_data = json.load(open(res_fn, 'r'))
    annot_fn = 'data/KINS/instances_val.json'

    annot_data = json.load(open(annot_fn, 'r'))
    size_dict = dict([(a['id'], (a['width'], a['height'])) for a in annot_data['images']])

    ret = compute(res_data, annot_data['annotations'], size_dict)
    np.save("{}/stat_{}.npy".format(fold, method), ret)
