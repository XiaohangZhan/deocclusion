#!/bin/bash
work_path=$(dirname $0)
partition=$1
ITER=$2
NGPU=8
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $partition -n$NGPU -x SH-IDC1-10-5-30-62 \
    --gres=gpu:$NGPU --ntasks-per-node=$NGPU \
    python -u main.py \
        --config $work_path/config.yaml --launcher slurm \
        --load-iter $ITER --evaluate
