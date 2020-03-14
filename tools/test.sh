#!/bin/bash
DATA="/mnt/lustre/share/zhanxiaohang/data/KINS"


exp="hull_order"
srun -p VI_UC_1080TI --gres gpu:1 -n1 \
    python tools/test.py \
        --config experiments/KINS/partial_unet2_in5_front0.8/config.yaml \
        --load-iter 32000 \
        --method $exp \
        --modal-res $DATA/instances_train.json \
        --image-root $DATA/2D-Det/training/image_2 \
        --test-num -1 \
        --gt-annot $DATA/annot_cocofmt/instances_amodal_train.json \
        --output $DATA/amodal_results/amodalcomp_train_${exp}.json \
        --force

python tools/convert_to_ann.py \
    $DATA/amodal_results/amodalcomp_train_${exp}.json \
    $DATA/annot_cocofmt/instances_amodal_train.json \
    $DATA/amodal_results/amodalcomp_train_${exp}_annot.json
