#!/bin/bash


DATASET=pascal_bdd_day_night_cyclegan
LABELED=32000
CUDA_VISIBLE_DEVICES=0,1 python train_ssd.py --dataset $DATASET --batch_size 4 --mu 0 --eval-step 4096 --num-labeled $LABELED --image-width 300 --image-height 300 --out /nethome/jbang36/k_amoeba/results/$DATASET/$LABELED ## 'wideresnet', 'resnext', 'fcos', default batch-size 64
