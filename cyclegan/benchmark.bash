#!/bin/bash

MACHINEDIR=/nethome/jbang36
DATAROOT=/data/bdd/cyclegan/day_night_scenario

python benchmark.py --dataset=bdd --cyclegan-output=$DATAROOT/converted
python convert.py --dataroot=$DATAROOT --name=bdd_day_night --gpu_ids=1 --checkpoints_dir=$MACHINEDIR/cyclegan/checkpoints --results_dir=$DATAROOT/converted


