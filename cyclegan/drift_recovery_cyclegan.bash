#!/bin/bash

#### day to night training
#MACHINEDIR=/nethome/jbang36
#DATAROOT=/data/bdd/cyclegan/day_night_scenario

#python benchmark.py --dataset=bdd --cyclegan-output=$DATAROOT/converted
#python convert.py --dataroot=$DATAROOT --name=bdd_day_night --gpu_ids=1 --checkpoints_dir=$MACHINEDIR/cyclegan/checkpoints --results_dir=$DATAROOT/converted

### the line below trains the SSD network

#DATASET=pascal_bdd_day_night_cyclegan
#LABELED=32000
#CUDA_VISIBLE_DEVICES=0,1 python ../fixmatch/drift_recovery_fixmatch.py --dataset $DATASET --batch_size 4 --mu 0 --eval-step 4096 --num-labeled $LABELED --image-width 300 --image-height 300 --out /nethome/jbang36/k_amoeba/results/$DATASET/$LABELED ## 'wideresnet', 'resnext', 'fcos', default batch-size 64



DATAROOT=/data/bdd/cyclegan/res_city_scenario
SCENARIO=bdd_res_city

#python benchmark.py --dataset=bdd --cyclegan-output=$DATAROOT/converted
#python convert.py --dataroot=$DATAROOT --name=$SCENARIO --gpu_ids=0 --checkpoints_dir=$DATAROOT/checkpoints --results_dir=$DATAROOT/converted

### the line below trains the SSD network

DATASET=pascal_bdd_res_city_cyclegan
LABELED=8000
CUDA_VISIBLE_DEVICES=0 python ../fixmatch/train_ssd_no_u.py --dataset $DATASET --batch_size 4  --lambda-u 0 --eval-step 4096 --num-labeled $LABELED --image-width 300 --image-height 300 --out /nethome/jbang36/k_amoeba/results/$DATASET/$LABELED ## 'wideresnet', 'resnext', 'fcos', default batch-size 64


