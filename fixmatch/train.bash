#!/bin/bash


## In this file we write a bash script to compile and generate graphs

#CUDA_VISIBLE_DEVICES=3 python train.py --dataset bdd --num-labeled 100 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out models/bdd@100.5

#CUDA_VISIBLE_DEVICES=3 python train_fcos.py --dataset bdd --num-labeled 100 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out models/bdd_fcos@100.5

CUDA_VISIBLE_DEVICES=3 python train_fcos.py --batch_size 8 --mu 1 --num-labeled 5000 ## 'wideresnet', 'resnext', 'fcos', default batch-size 64

#CUDA_VISIBLE_DEVICES=3 python train_object_detection.py ## 'wideresnet', 'resnext', 'fcos', default batch-size 64