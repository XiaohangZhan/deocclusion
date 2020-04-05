#!/bin/bash
DATA="data/COCOA"

python tools/test.py \
    --config experiments/COCOA/pcnet_m/config.yaml \
    --load-model "released/COCOA_pcnet_m.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours" \
    --annotation "data/COCOA/annotations/COCO_amodal_val2014.json" \
    --image-root $DATA/val2014 \
    --test-num -1 \
    --output $DATA/amodal_results/amodalcomp_val_ours.json
