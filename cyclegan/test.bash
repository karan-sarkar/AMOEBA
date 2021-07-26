#!/bin/bash

MACHINEDIR=/nethome/jbang36


python move_images.py
python convert.py --dataroot=$MACHINEDIR/cyclegan/data --name=bdd_day_night --gpu_ids=1 --checkpoints_dir=$MACHINEDIR/cyclegan/checkpoints --results_dir=$MACHINEDIR/cyclegan/results