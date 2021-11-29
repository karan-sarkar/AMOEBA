import logging

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset



import sys
sys.path.append('/nethome/jbang36/k_amoeba')

from fixmatch.dataset.transforms import *

from fixmatch.dataset.bdd_coco import BDDDataset



from .randaugment import RandAugmentMC
import torch



logger = logging.getLogger(__name__)


normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)



def get_object_detection_datasets(image_size = (300,400)):
    normal_mean = (0.5, 0.5, 0.5)
    normal_std = (0.5, 0.5, 0.5)
    path1 = 'daytime'

    transform_labeled = Compose([
        Resize(*image_size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = Compose([
        Resize(*image_size),
        ToTensor(),
        Normalize(mean=normal_mean, std=normal_std)
    ])

    train_set = BDDDataset(path1, 'train', transform = transform_labeled)
    valid_set = BDDDataset(path1, 'val', transform = transform_val)



    return train_set, valid_set



#### There is a high chance that subset / concat dataset is making the evaluation malfunction...
#### let's divide the train set into 2 -- labeled / unlabeled
#### let's keep the val set as is....
def get_bdd_fcos_new(args, data_dir):
    num_labeled = args.num_labeled
    transform_labeled = get_transform_labeled(image_size = (args.image_height, args.image_width))
    transform_heavy = TransformFixMatch(mean=normal_mean, std=normal_std, image_size = (args.image_height, args.image_width))

    transform_val = get_transform_val(image_size = (args.image_height, args.image_width))

    unlabeled_day_set = BDDDataset(args.path, 'train', transform=transform_heavy)
    unlabeled_night_set = BDDDataset(args.path2, 'train', transform=transform_heavy)
    labeled_day_set = BDDDataset(args.path, 'train', transform=transform_labeled)
    labeled_night_set = BDDDataset(args.path2, 'train', transform=transform_labeled)
    test_day_set = BDDDataset(args.path, 'val', transform=transform_val)
    test_night_set = BDDDataset(args.path2, 'val', transform=transform_val)

    train_cut = num_labeled ### 100
    train_datasets = [(labeled_day_set, unlabeled_day_set), (labeled_night_set, unlabeled_night_set)]
    new_train_datasets = []
    for i in range(2):
        labeled_set, unlabeled_set = train_datasets[i]
        dataset_len = len(labeled_set)
        labeled_indices = np.arange(train_cut).astype(np.int32)
        unlabeled_indices = np.arange(train_cut, dataset_len)

        labeled_set = torch.utils.data.Subset(labeled_set, labeled_indices)
        unlabeled_set = torch.utils.data.Subset(unlabeled_day_set, unlabeled_indices)

        new_train_datasets.append( (labeled_set, unlabeled_set) )

    (labeled_day_set, unlabeled_day_set), (labeled_night_set, unlabeled_night_set) = new_train_datasets

    labeled_set = torch.utils.data.ConcatDataset([labeled_day_set, labeled_night_set])
    unlabeled_set = torch.utils.data.ConcatDataset([unlabeled_day_set, unlabeled_night_set])

    return labeled_set, unlabeled_set, (test_day_set, test_night_set)




### TODO: Need to create an implementation of BDD Dataset,
### Get the indices, split into labeled / unlabeled images
def get_bdd_fcos(args, data_dir):
    transform_labeled = get_transform_labeled()
    transform_val = get_transform_val()

    transform_heavy = TransformFixMatch(mean=normal_mean, std=normal_std)

    labeled_day_set = BDDDataset(args.path, 'val', transform=transform_labeled)
    labeled_night_set = BDDDataset(args.path2, 'val', transform=transform_labeled)
    unlabeled_day_set = BDDDataset(args.path, 'train', transform=transform_heavy)
    unlabeled_night_set = BDDDataset(args.path2, 'train', transform=transform_heavy)

    ### how about we cut the datset into 2 subs, create a new instance with new transform, cut that as well
    day_len = len(labeled_day_set)
    ### number of labeled examples that we want is 100
    train_cut = 100
    day_train_indices = np.arange(train_cut).astype(np.int32)
    day_test_indices = np.arange(train_cut, day_len)
    labeled_day_set = torch.utils.data.Subset(labeled_day_set, day_train_indices)
    night_len = len(labeled_night_set)
    night_train_indices = np.arange(train_cut).astype(np.int32)
    night_test_indices = np.arange(train_cut, night_len)
    labeled_night_set = torch.utils.data.Subset(labeled_night_set, night_train_indices)

    test_day_set = BDDDataset(args.path, 'val', transform=transform_val)
    test_night_set = BDDDataset(args.path2, 'val', transform=transform_val)

    test_day_set = torch.utils.data.Subset(test_day_set, day_test_indices)
    test_night_set = torch.utils.data.Subset(test_night_set, night_test_indices)

    #test_dataset = torch.amoeba_utils.data.ConcatDataset([test_day_set, test_night_set])

    labeled_set = torch.utils.data.ConcatDataset([labeled_day_set, labeled_night_set])
    unlabeled_set = torch.utils.data.ConcatDataset([unlabeled_day_set, unlabeled_night_set])


    return labeled_set, unlabeled_set, (test_day_set, test_night_set)


##### we need to define new collate_fn functions



### this is basically the collate function for TransformFixMatch
def collate_fn2(config):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], config.size_divisible)
        targets = image_list(batch[1], config.size_divisible)
        indices = batch[2]

        return imgs, targets, indices

    return collate_data


### how to collate the bdd_fcos dataset
def collate_fn(config):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], config.size_divisible)
        targets = batch[1]
        indices = batch[2]

        return imgs, targets, indices

    return collate_data


def image_list(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = (max_size[1] | (stride - 1)) + 1
        max_size[2] = (max_size[2] | (stride - 1)) + 1
        max_size = tuple(max_size)

    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]

    return ImageList(batch, sizes)


class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)

###########################
####### simple BDD ########
###########################


class BDDSSL(Dataset):

    def __init__(self, raw_data:np.array, targets:np.array, indexs:np.array,
                 transform = None, target_transform = None):

        ### define the data and the targets
        self.data = raw_data[indexs]
        self.targets = targets[indexs]
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index:int):
        img, target = self.data[index],self.targets[index]
        img = Image.fromarray(img)
        #### we need to transform the img to pil array
        ### img will be in the form of... np array
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


