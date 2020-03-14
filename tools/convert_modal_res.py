import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('res', type=str)
    parser.add_argument('ann', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.res, 'r') as f:
        res_data = json.load(f)
    with open(args.ann, 'r') as f:
        annot = json.load(f)

    for r,data in tqdm(zip(res_data, annot['annotations']), total=len(res_data)):
        assert r['id'] == data['id']
        data['inmodal_seg'] = r['segmentation']

    with open(args.output, 'w') as f:
        json.dump(annot, f)

if __name__ == '__main__':
    main()
