#!/bin/bash



### for cyclegan, we will run everything on ada-01 for now....

#MACHINEDIR=/srv/data/jbang36
MACHINEDIR=/nethome/jbang36

python train.py --dataroot=$MACHINEDIR/cyclegan/data --name=bdd_day_night --gpu_ids=0 --checkpoints_dir=$MACHINEDIR/cyclegan/checkpoints