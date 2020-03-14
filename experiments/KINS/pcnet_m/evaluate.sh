#!/bin/bash
work_path=$(dirname $0)
partition=$1
ITER=$2
NGPU=8
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $partition -n$NGPU \
    --gres=gpu:$NGPU --ntasks-per-node=$NGPU \
    python -u main.py \
        --config $work_path/config.yaml --launcher slurm \
        --load-iter $ITER --evaluate
