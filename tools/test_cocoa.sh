#!/bin/bash
DATA="data/COCOA"
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config experiments/COCOA/pcnet_m/config.yaml \
    --load-model "released/COCOA_pcnet_m.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours" \
    --order-th 0.1 \
    --amodal-th 0.2 \
    --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
    --image-root $DATA/val2014 \
    --test-num -1 \
    --output $DATA/amodal_results/amodalcomp_val_ours.json
