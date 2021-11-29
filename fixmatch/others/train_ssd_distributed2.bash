#!/bin/bash

#--master_addr ='10.0.3.29'
NUM_GPUS_YOU_HAVE=1
#DATASET=pascal_bdd_day_night
#DATASET=pascal_bdd_day_night
DATASET=pascal_bdd_res_city
#AUGMENTATION=aug_reduced
AUGMENTATION=aug_sharpness_only
#DATASET=pascal_bdd_day_all
LABELED=3000

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS_YOU_HAVE --master_port=9901 train_ssd.py --dataset $DATASET --batch_size 4 --mu 1 --eval-step 4096 --num-labeled $LABELED --image-width 300 --image-height 300 --out /srv/data/jbang36/checkpoints/k_amoeba/amoeba_scenario/$DATASET/$LABELED/$AUGMENTATION