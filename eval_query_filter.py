import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import math
import json
import tqdm
import time
import argparse
from collections import Counter


from bdd import *
from util import *
from ssd import *


EPOCHS = 1000
NUM_CLASSES = 10
root_anno_path = "bdd100k_labels_detection20"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str
parser.add_argument('--ckpt', type=int, default = -1)
parser.add_argument('--attribute', type=str, default = 'timeofday')
parser.add_argument('--source', type=str, default = 'daytime')
parser.add_argument('--target', type=str, default = 'night')
parser.add_argument('--sample', type=int, default = -1)
parser.add_argument('--source_sample', type=int, default = -1)
args = parser.parse_args()

ATTRIBUTE = args.attribute
SOURCE_FLAG = args.source
TARGET_FLAG = args.target
BATCH_SIZE = 1
sample = args.sample
source_sample = args.source_sample
mod = args.ckpt

checkpoint = args.model
if 'mcd' in checkpoint:
    model,_ = torch.load(checkpoint)
else:
    model = torch.load(checkpoint)
model = model.to(device)
if not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

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
        

        

filter_obj = 'truck'
filter_limit = 1
query_obj = 'car'
counts = []
path = str((filter_obj, filter_limit, query_obj))




min_score=0.2
max_overlap=0.45
top_k=200

class Filter(nn.Module):

    def __init__(self):
        super(Filter, self).__init__()
        self.base = VGGBase()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.base(x)[1]
        x = self.conv1(self.pool1(x)).relu()
        x = self.conv2(self.pool2(x)).relu()
        x = self.conv3(self.pool3(x)).relu()
        x = x.view(x.size(0), -1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

filter = Filter().to(device)
filter = nn.DataParallel(filter)
opt = optim.SGD(filter.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
bceloss = nn.BCEWithLogitsLoss

if mod >= 0:
     (filter, model, opt) = torch.load('filter_bdd100k-9_' + path +  str(epoch + mod + 1) + '.pth')

def train(train_loader, test_loader, model):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    dlosses = AverageMeter()  # loss
    blosses = AverageMeter()
    tlosses = AverageMeter()
    start = time.time()
    acc = []
    losses = []
    # Batches
    test = iter(load(test_loader, False))
    train = load(train_loader, False)
    pbar = tqdm.tqdm(train)
    for i, (source_images, source_boxes, source_labels) in enumerate(pbar):
        predicted_locs, predicted_scores,_,_ = model(source_images)
        x = filter(source_images)

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.module.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                     max_overlap=max_overlap, top_k=top_k)
        #print(det_labels[0])
        objects = [rev_label_map[ob] for ob in det_labels[0].detach().cpu().numpy() if ob > 0]
        
        counter = Counter()
        for ob in objects:
            counter[ob] += 1
        ans = int(x.sigmoid().ge(0.5))
        target = 0
        if counter[filter_obj] >= filter_limit:
            counts.append(counter[query_obj])
            target = 1
        acc.append(1 - abs(ans - target))
        s = 'ACC: ' + str(sum(acc)/len(acc))
        y = torch.ones(1, 1).to(device) * target
        opt.zero_grad()
        loss = bceloss(x, y)
        losses.append(float(loss))
        s += 'Loss: ' + str(sum(losses)/len(losses))
        s += 'Target: ' + target
        s += 'Answer: ' + str(float(x.view(-1)))
        loss.backward()
        opt.step()
        
        pbar.set_description(s
        if i % 5 == 0:
            torch.save((filter, model, optimizer), 'filter_bdd100k-9_' + path +  str(epoch + mod + 1) + '.pth')
                                                      
        
                                                      
for epoch in range(EPOCHS):                                                             
    train(source_train, target_train, model)
    print(sum(counts) / len(counts))
    counts = []

        
    
