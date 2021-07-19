#!/bin/bash

#--master_addr ='10.0.3.29'
NUM_GPUS_YOU_HAVE=2
DATASET=pascal_bdd_day_night
LABELED=1000

CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS_YOU_HAVE --master_port=9901 train_ssd.py --dataset $DATASET --batch_size 4 --mu 1 --eval-step 4096 --num-labeled $LABELED --image-width 300 --image-height 300 --out /nethome/jbang36/k_amoeba/fixmatch/results/amoeba_scenario/$DATASET/$LABELED