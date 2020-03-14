import multiprocessing as mp
import tqdm
import numpy as np
import json
import sys
import pycocotools.mask as maskUtils

def read_annot(ann, h, w):
    amodal = maskUtils.decode(maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
    return amodal

def task(ann, cplx, data, size_dict):
    w, h = size_dict[ann['image_id']]
    amp = maskUtils.decode(data['segmentation']).astype(np.bool)
    amc = maskUtils.decode(cplx['segmentation']).astype(np.bool)
    amg = read_annot(ann, h, w)
    return [((amp == 1) & (amg == 1)).sum(),
            ((amp == 1) | (amg == 1)).sum(),
            ((amc == 1) & (amg == 1)).sum(),
            ((amc == 1) | (amg == 1)).sum()]

def helper(args):
    return task(*args)

def compute(data, annot_data, cplx_data, size_dict):
    num = len(data)
    pool = mp.Pool(16)
    args = zip(annot_data, cplx_data, data, [size_dict] * num)
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
    annot_cplx_fn = '{}/amodalcomp_test_complexity.json'.format(fold)

    annot_data = json.load(open(annot_fn, 'r'))
    annot_cplx_data = json.load(open(annot_cplx_fn, 'r'))
    size_dict = dict([(a['id'], (a['width'], a['height'])) for a in annot_data['images']])

    ret = compute(res_data, annot_data['annotations'], annot_cplx_data, size_dict)
    np.save("{}/stat_cplx_{}.npy".format(fold, method), ret)
