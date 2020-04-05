#!/usr/bin/env bash
echo Which PYTHON: `which python`
OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=4 main.py \
--config=config/gca-dist.toml