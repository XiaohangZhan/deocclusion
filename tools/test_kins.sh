#!/bin/bash
DATA="data/KINS"

python tools/test.py \
    --config experiments/KINS/pcnet_m/config.yaml \
    --load-model "released/KINS_pcnet_m.pth.tar"\
    --order-method "ours" \
    --amodal-method "ours" \
    --annotation "data/KINS/instances_val.json" \
    --image-root $DATA/testing/image_2 \
    --test-num -1 \
    --output $DATA/amodal_results/amodalcomp_val_ours.json
