#!/bin/bash



### for cyclegan, we will run everything on ada-01 for now....

#MACHINEDIR=/srv/data/jbang36
#MACHINEDIR=/nethome/jbang36
SCENARIO_NAME=res_city_scenario
MACHINEDIR=/data/bdd/cyclegan/$SCENARIO_NAME


python train.py --dataroot=$MACHINEDIR --name=bdd_res_city --gpu_ids=1 --checkpoints_dir=$MACHINEDIR/checkpoints