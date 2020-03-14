#!/bin/bash
DATA="/mnt/lustre/share/zhanxiaohang/data/KINS"


exp=sup
srun -p VI_UC_1080TI --gres gpu:1 -n1 \
    python tools/test.py \
        --config experiments/KINS/supervised/config.yaml \
        --load-iter 56000 \
        --method $exp \
        --modal-res $DATA/instances_val.json \
        --image-root $DATA/2D-Det/testing/image_2 \
        --test-num -1 \
        --gt-annot $DATA/annot_cocofmt/instances_amodal_test.json \
        --output data/KINS/amodal_res_val/amodalcomp_test_${exp}.json \
        --force
