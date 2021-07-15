import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
import random
from fixmatch.utils.ssd_sgr_utils import *
from fixmatch.dataset.boxlist import BoxList
from fixmatch.dataset.transforms import TransformFixMatch



def get_pascal_bdd_day_night(args, data_dir):
    day_data_folder = '/srv/data/jbang36/bdd/ssd_sgr/day'
    night_data_folder = '/srv/data/jbang36/bdd/ssd_sgr/night'

    keep_difficult = True  # use objects considered difficult to detect?
    train_cut = args.num_labeled

    day_labeled_dataset = PascalVOCFixmatch(day_data_folder,
                                        split='train',
                                        keep_difficult=keep_difficult,
                                        transform=transform_train)

    night_labeled_dataset = PascalVOCFixmatch(night_data_folder,
                                            split='train',
                                            keep_difficult=keep_difficult,
                                            transform=transform_train)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size = (args.image_height, args.image_width)
    tfm = TransformFixMatch(mean=mean, std=std, image_size=image_size)

    day_unlabeled_dataset = PascalVOCFixmatch(day_data_folder,
                                          split='train',
                                          keep_difficult=keep_difficult,
                                          transform=tfm)

    night_unlabeled_dataset = PascalVOCFixmatch(night_data_folder,
                                          split='train',
                                          keep_difficult=keep_difficult,
                                          transform=tfm)


    day_test_dataset = PascalVOCFixmatch(day_data_folder,
                                     split='val',
                                     keep_difficult=keep_difficult,
                                     transform=transform_val)

    night_test_dataset = PascalVOCFixmatch(night_data_folder,
                                     split='val',
                                     keep_difficult=keep_difficult,
                                     transform=transform_val)


    day_dataset_len = len(day_labeled_dataset)
    night_dataset_len = len(night_labeled_dataset)


    labeled_indices = np.arange(train_cut).astype(np.int32)
    day_unlabeled_indices = np.arange(train_cut, day_dataset_len)
    night_unlabeled_indices = np.arange(train_cut, night_dataset_len)


    day_labeled_set = torch.utils.data.Subset(day_labeled_dataset, labeled_indices)
    night_labeled_set = torch.utils.data.Subset(night_labeled_dataset, labeled_indices)
    labeled_set = torch.utils.data.ConcatDataset([day_labeled_set, night_labeled_set])


    day_unlabeled_set = torch.utils.data.Subset(day_unlabeled_dataset, day_unlabeled_indices)
    night_unlabeled_set = torch.utils.data.Subset(night_unlabeled_dataset, night_unlabeled_indices)

    unlabeled_set = torch.utils.data.ConcatDataset([day_unlabeled_set, night_unlabeled_set])


    #### let's shorten the test_dataset, we will use 1000 examples
    test_len = 500
    test_indices = np.arange(test_len).astype(np.int32)
    day_test_dataset = torch.utils.data.Subset(day_test_dataset, test_indices)
    night_test_dataset = torch.utils.data.Subset(night_test_dataset, test_indices)

    return labeled_set, unlabeled_set, (day_test_dataset, night_test_dataset)





def get_pascal_bdd(args, data_dir):
    data_folder = '/srv/data/jbang36/bdd/ssd_sgr'

    keep_difficult = True  # use objects considered difficult to detect?
    train_cut = args.num_labeled

    labeled_dataset = PascalVOCFixmatch(data_folder,
                                        split='train',
                                        keep_difficult=keep_difficult,
                                        transform=transform_train)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size = (args.image_height, args.image_width)
    tfm = TransformFixMatch(mean=mean, std=std, image_size=image_size)

    unlabeled_dataset = PascalVOCFixmatch(data_folder,
                                          split='train',
                                          keep_difficult=keep_difficult,
                                          transform=tfm)

    test_dataset = PascalVOCFixmatch(data_folder,
                                     split='val',
                                     keep_difficult=keep_difficult,
                                     transform=transform_val)

    dataset_len = len(labeled_dataset)

    labeled_indices = np.arange(train_cut).astype(np.int32)
    unlabeled_indices = np.arange(train_cut, dataset_len)

    labeled_set = torch.utils.data.Subset(labeled_dataset, labeled_indices)
    unlabeled_set = torch.utils.data.Subset(unlabeled_dataset, unlabeled_indices)


    #### let's shorten the test_dataset, we will use 1000 examples
    test_len = 200
    test_indices = np.arange(test_len).astype(np.int32)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    return labeled_set, unlabeled_set, test_dataset



def get_pascal_voc(args, data_dir):
    data_folder = '/srv/data/jbang36/VOCdevkit/ssd_sgr'  # folder with data files
    keep_difficult = True  # use objects considered difficult to detect?
    train_cut = args.num_labeled

    labeled_dataset = PascalVOCFixmatch(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult,
                                     transform = transform_train)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size = (args.image_height, args.image_width)
    tfm = TransformFixMatch(mean=mean, std=std, image_size=image_size)

    unlabeled_dataset = PascalVOCFixmatch(data_folder,
                                       split='train',
                                       keep_difficult=keep_difficult,
                                        transform = tfm)

    test_dataset = PascalVOCFixmatch(data_folder,
                                    split='test',
                                    keep_difficult=keep_difficult,
                                    transform = transform_val)


    dataset_len = len(labeled_dataset)

    labeled_indices = np.arange(train_cut).astype(np.int32)
    unlabeled_indices = np.arange(train_cut, dataset_len)

    labeled_set = torch.utils.data.Subset(labeled_dataset, labeled_indices)
    unlabeled_set = torch.utils.data.Subset(unlabeled_dataset, unlabeled_indices)

    return labeled_set, unlabeled_set, test_dataset



class PascalVOCFixmatch(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False, transform = None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST', 'VAL'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)
        self.transform = transform

    def __getitem__(self, index):
        # Read image
        image = Image.open(self.images[index], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[index]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        target = BoxList(boxes, image.size, mode='xyxy')

        classes = torch.tensor(labels)
        difficulties = torch.tensor(difficulties)
        target.fields['labels'] = classes
        target.fields['difficulties'] = difficulties
        target.clip(remove_empty=True)

        img = image
        if self.transform is not None:
            if type(self.transform) == TransformFixMatch:
                (img1, t1), (img2, t2) = self.transform(img, target)
                img = img1
                target = img2
            else:
                img, target = self.transform(img, target)

        return img, target, index


    def __len__(self):
        return len(self.images)




#################################################################################
#################################################################################
#################################################################################
########################### TRANSFORMS ###########################
#################################################################################
#################################################################################
#################################################################################




def transform_train(image, target):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = target.box
    new_labels = target.fields['labels']
    new_difficulties = target.fields['difficulties']
    # Skip the following operations for evaluation/testing
    # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
    new_image = photometric_distort(new_image)

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
    # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
    if random.random() < 0.5:
        new_image, new_boxes = expand(new_image, new_boxes, filler=mean)

    # Randomly crop image (zoom in)
    new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                     new_difficulties)

    # Convert Torch tensor to PIL image
    new_image = FT.to_pil_image(new_image)

    # Flip image with a 50% chance
    if random.random() < 0.5:
        new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    target.box = new_boxes
    target.fields['labels'] = new_labels
    target.fields['difficulties'] = new_difficulties

    return new_image, target


def transform_val(image, target):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = target.box
    new_labels = target.fields['labels']
    new_difficulties = target.fields['difficulties']
    # Skip the following operations for evaluation/testing

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    target.box = new_boxes
    target.fields['labels'] = new_labels
    target.fields['difficulties'] = new_difficulties

    return new_image, target

