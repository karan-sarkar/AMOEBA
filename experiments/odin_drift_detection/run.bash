#!/bin/bash

#MODEL=/srv/data/jbang36/checkpoints/eko/ua_detrac/checkpoint_ssd300.pth.tar
#VIDEO=
#OUTPUT=



#for DATASET in bdd jackson yesler ua_detrac cherry

for CATEGORY in overcast snowy clear
do
  #CUDA_VISIBLE_DEVICES=2 python aggregate.py --evaluation_dataset $DATASET
  CUDA_VISIBLE_DEVICES=1 python main.py --source rainy --target $CATEGORY
  echo Done with $CATEGORY
done
