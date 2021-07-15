#!/bin/bash


## In this file we write a bash script to compile and generate graphs

#CUDA_VISIBLE_DEVICES=3 python train.py --dataset bdd --num-labeled 100 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out models/bdd@100.5

#CUDA_VISIBLE_DEVICES=3 python train_fcos.py --dataset bdd --num-labeled 100 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out models/bdd_fcos@100.5

#CUDA_VISIBLE_DEVICES=1 python train_ssd.py --batch_size 4 --mu 1 --eval-step 4096 --num-labeled 5000 --image-width 300 --image-height 300 ## 'wideresnet', 'resnext', 'fcos', default batch-size 64

#CUDA_VISIBLE_DEVICES=1 python train_ssd.py --dataset pascal_voc --batch_size 4 --mu 1 --eval-step 4096 --num-labeled 10000 --image-width 300 --image-height 300 --out /nethome/jbang36/j_amoeba/benchmark/fixmatch/models/save/ssd/pascal_voc/no_lossu ## 'wideresnet', 'resnext', 'fcos', default batch-size 64

#CUDA_VISIBLE_DEVICES=1 python train_ssd.py --dataset bdd_fcos --batch_size 4 --mu 1 --eval-step 4096 --num-labeled 1000 --image-width 300 --image-height 300 --out /nethome/jbang36/j_amoeba/benchmark/fixmatch/models/save/ssd/bdd/no_lossu ## 'wideresnet', 'resnext', 'fcos', default batch-size 64

###CUDA_VISIBLE_DEVICES=1 python train_ssd.py --dataset pascal_bdd --batch_size 4 --mu 1 --eval-step 4096 --num-labeled 1000 --image-width 300 --image-height 300 --out /nethome/jbang36/k_amoeba/benchmark/fixmatch/models/save/ssd/bdd/no_lossu ## 'wideresnet', 'resnext', 'fcos', default batch-size 64
