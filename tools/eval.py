import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('res', type=str)
    parser.add_argument('gt', type=str)
    args = parser.parse_args()
    return args

def main(args):
    evaluate(args.res, args.gt)

def evaluate(res_file, gt_file):
    annType = 'segm'
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(res_file)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    args = parse_args()
    main(args)
