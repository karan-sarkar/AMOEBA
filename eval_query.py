import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import tqdm
import time
import argparse

from bdd import *
from util import *
from ssd import *


EPOCHS = 1000
NUM_CLASSES = 10
root_anno_path = "bdd100k_labels_detection20"

parser = argparse.ArgumentParser()
parser.add_argument('--attribute', type=str, default = 'timeofday')
parser.add_argument('--source', type=str, default = 'daytime')
parser.add_argument('--target', type=str, default = 'night')
parser.add_argument('--batch_size', type=int, default = 32)
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--momentum', type=float, default = 0.9)
parser.add_argument('--decay', type=float, default = 0.0005)
parser.add_argument('--clipping', type=float, default = 5.0)
parser.add_argument('--ckpt', type=int, default = -1)
parser.add_argument('--source_sample', type=int, default = -1)

args = parser.parse_args()

ATTRIBUTE = args.attribute
SOURCE_FLAG = args.source
TARGET_FLAG = args.target

BATCH_SIZE = args.batch_size
lr = args.lr
momentum = args.momentum
weight_decay = args.decay
clipping = args.clipping
mod = args.ckpt
source_sample = args.source_sample
path = str((SOURCE_FLAG, source_sample))


root_img_path = "../fcos-pytorch/bdd100k/images/100k"
root_anno_path = "bdd100k_labels_detection20/bdd100k/labels/detection20"

train_img_path = root_img_path + "/train/"
val_img_path = root_img_path + "/val/"

train_anno_json_path = root_anno_path + "/det_v2_train_release.json"
val_anno_json_path = root_anno_path + "/det_v2_val_release.json"

with open(train_anno_json_path, "r") as file:
    train_data = json.load(file)
print(len(train_data))
with open(val_anno_json_path, "r") as file:
    test_data = json.load(file)
print(len(test_data))

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def make_dataset(train, flag):
    if train:
        data = train_data
        json_file = train_anno_json_path
        header = train_img_path
    else:
        data = test_data
        json_file = val_anno_json_path
        header = val_img_path
    
    img_list = []
    idx = []
    for i in tqdm.tqdm(range(len(data))):
        if (flag == 'all' or data[i]['attributes'][ATTRIBUTE] == flag) and data[i]['labels'] != None:
            img_list.append(header + data[i]['videoName'] + '.jpg')
            idx.append(i)
    if source_sample > 0 and flag == SOURCE_FLAG and train:
        np.random.seed(0)
        perm = np.random.choice(len(idx), source_sample)
        idx = [idx[p] for p in perm]
        img_list = [img_list[p] for p in perm]
    dset = BDD(img_list, idx, json_file, train)
    return dset

source_train = make_dataset(True, SOURCE_FLAG)
source_test = make_dataset(False, SOURCE_FLAG)
target_train = make_dataset(True, TARGET_FLAG)
target_test = make_dataset(False, TARGET_FLAG)

def load(dset, sample):
    return torch.utils.data.DataLoader(dset,batch_size=BATCH_SIZE,shuffle=True, collate_fn=dset.collate_fn)

def get_model(num_classes):
    model = SSD300(num_classes)
    return model.to(device)
        
        
jm = get_model(NUM_CLASSES)

if mod >= 0:
    jm = torch.load('baseline_bdd100k-9_' + path + str(mod) + ".pth")

        
params = list(jm.parameters()) 
opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)

crit = MultiBoxLoss(priors_cxcy=jm.priors_cxcy.clone()).to(device)

filter_obj = 'truck'
filter_limit = 1
query_obj = 'car'
counts = []


def train(train_loader, test_loader, model, crit, optimizer, epoch, print_freq):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout
    criterion = MultiBoxLoss(priors_cxcy=crit.priors_cxcy.clone()).to(device)   
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    dlosses = AverageMeter()  # loss
    blosses = AverageMeter()
    tlosses = AverageMeter()
    start = time.time()

    # Batches
    test = iter(load(test_loader, False))
    train = load(train_loader, False)
    for i, (source_images, source_boxes, source_labels) in enumerate(train):
        print(source_labels)
        objects = [rev_label_map[ob] for ob in source_labels]
        
        counter = Counter()
        for ob in det_labels:
            counter[ob] += 1
    
        if counter[filter_obj] >= filter_limit:
            counts.append(counter[query_obj])
                                                      

def test(test_loader, model, criterion, epoch):
    model.eval()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    tlosses = AverageMeter()
    start = time.time()
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    
    i = 1
    with torch.no_grad():# Batches
        for (target_images, target_boxes, target_labels) in tqdm.tqdm(load(test_loader, False)):
            #if i > 5:
            #    break
            target_images = target_images.to(device)  # (batch_size (N), 3, 300, 300)
            target_boxes = [b.to(device) for b in target_boxes]
            target_labels = [l.to(device) for l in target_labels]
            
            predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2 = model(target_images)
            
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_target_locs1, predicted_target_scores1,
                                                                                           min_score=0.01, max_overlap=0.45,
                                                                                           top_k=200)
            
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(target_boxes)
            true_labels.extend(target_labels)
            
            del predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2
            del target_images, target_boxes, target_labels
            del det_boxes_batch, det_labels_batch, det_scores_batch
            i += 1
        
    
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
    print(APs)
    print(mAP)
                                                              
train(source_train, target_train, jm, crit, opt, epoch, 1)
print(sum(counts) / len(counts))

        
    
