#!/bin/bash
DATA="data/KINS"
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config experiments/KINS/pcnet_m/config.yaml \
    --load-model "released/KINS_pcnet_m.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours" \
    --order-th 0.1 \
    --amodal-th 0.2 \
    --annotation "data/KINS/instances_val.json" \
    --image-root $DATA/testing/image_2 \
    --test-num -1 \
    --output $DATA/amodal_results/amodalcomp_val_ours.json
