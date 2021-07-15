"""
In this file, we implement the utils functions to generate the JSON files like pascal voc.
Hence, we can just use the PASCALVOC Dataset object to load the data and train stuff

"""
import os
import json
import torch

import sys
sys.path.append('/nethome/jbang36/k_amoeba')
from fixmatch.utils.ssd_sgr_utils import find_jaccard_overlap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


label_map = {'traffic light': 1,
                 'traffic sign': 2,
                 'car': 3,
                 'pedestrian': 4,
                 'bus': 5,
                 'truck': 6,
                 'rider': 7,
                 'bicycle': 8,
                 'motorcycle': 9,
                 'train': 10,
                 'other vehicle': 11,
                 'other person': 12,
                 'trailer': 13,
                 'background': 0}

rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping


bdd_attributes = {'daytime'  : 'timeofday', 'dawn/dusk': 'timeofday',
                        'night'    : 'timeofday', 'undefined': 'timeofday',
                        'clear'    : 'weather', 'rainy': 'weather',
                        'undefined': 'weather', 'snowy': 'weather',
                        'overcast' : 'weather', 'partly cloudy': 'weather',
                        'foggy'    : 'weather'}


def calculate_mAP_bdd(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision



def create_data_lists(output_folder):

    splits = ['train', 'val']
    for split in splits:
        #generate_images(output_folder, split)
        #generate_objects(output_folder, split)
        generate_images_and_objects(output_folder, split)


    generate_label_map(output_folder)

def create_data_lists_day_night(output_folder):

    splits = ['train', 'val']
    for split in splits:
        generate_images_and_objects_category(output_folder, split, group='daytime')
        generate_images_and_objects_category(output_folder, split, group='night')



def determine_difficulty(obj):
    is_occluded = obj['attributes']['occluded']
    is_truncated = obj['attributes']['truncated']

    return int(is_occluded or is_truncated)

def determine_box_condition(box):
    x1, y1, x2, y2 = box
    if x1 >= x2 - 5:
        # print(f"x1 >= x2 error: {x1, x2}")
        return False
    if y1 >= y2 - 5:
        # print(f"y1 >= y2 error: {y1, y2}")
        return False
    if x1 < 0:
        # print(f" x1 < 0 error: {x1}")
        return False
    if y1 < 0:
        # print(f" y1 < 0 error: {y1}")
        return False

    return True


##########3


def generate_images_and_objects_category(output_folder:str, split:str, group:str):
    split = split.lower()
    assert(split == 'train' or split == 'val')
    labels_dir = f'/srv/data/jbang36/bdd/labels/bdd_{split}.json'
    json_file = _load_json(labels_dir)
    image_folder = os.path.join('/srv/data/jbang36/bdd/images/100k', split)

    filenames = []
    objects_info = []

    category = bdd_attributes[group]

    for frame_content in json_file:
        if group not in bdd_attributes:
            print(f'group: {group} not in attribution dict: {bdd_attributes}')
            raise ValueError

        frame_attribute_val = frame_content['attributes'][category]
        if frame_attribute_val == group:

            info_dict = {}
            info_dict['boxes'] = []
            info_dict['labels'] = [] ### we will use label_map to convert to numbers
            info_dict['difficulties'] = []
            if frame_content['labels'] is not None:
                ### we must also eliminate labels that have weird boxes

                boxes = []
                labels = []
                difficulties = []
                for obj in frame_content['labels']:
                    box = [int(obj['box2d']['x1']), int(obj['box2d']['y1']), int(obj['box2d']['x2']), int(obj['box2d']['y2'])]
                    if determine_box_condition(box):
                        boxes.append( box )
                        labels.append( label_map[obj['category']] )
                        difficulties.append( determine_difficulty(obj) )

                if len(boxes) > 0: ### we are including the labels in the set
                    filenames.append(os.path.join(image_folder, frame_content['name']))
                    info_dict['boxes'] = boxes
                    info_dict['labels'] = labels
                    info_dict['difficulties'] = difficulties

                    objects_info.append(info_dict)

    train_objects = objects_info
    split = split.upper()

    base = os.path.join(output_folder, group)
    os.makedirs(base, exist_ok=True)

    with open(os.path.join(output_folder, group, f'{split}_objects.json'), 'w') as j:
        json.dump(train_objects, j)

    train_images = filenames
    split = split.upper()
    with open(os.path.join(output_folder, group, f'{split}_images.json'), 'w') as j:
        json.dump(train_images, j)




def generate_images_and_objects(output_folder:str, split:str):
    split = split.lower()
    assert(split == 'train' or split == 'val')
    labels_dir = f'/srv/data/jbang36/bdd/labels/bdd_{split}.json'
    json_file = _load_json(labels_dir)
    image_folder = os.path.join('/srv/data/jbang36/bdd/images/100k', split)

    filenames = []
    objects_info = []

    for frame_content in json_file:
        info_dict = {}
        info_dict['boxes'] = []
        info_dict['labels'] = [] ### we will use label_map to convert to numbers
        info_dict['difficulties'] = []
        if frame_content['labels'] is not None:
            ### we must also eliminate labels that have weird boxes

            boxes = []
            labels = []
            difficulties = []
            for obj in frame_content['labels']:
                box = [int(obj['box2d']['x1']), int(obj['box2d']['y1']), int(obj['box2d']['x2']), int(obj['box2d']['y2'])]
                if determine_box_condition(box):
                    boxes.append( box )
                    labels.append( label_map[obj['category']] )
                    difficulties.append( determine_difficulty(obj) )

            if len(boxes) > 0:
                filenames.append(os.path.join(image_folder, frame_content['name']))
                info_dict['boxes'] = boxes
                info_dict['labels'] = labels
                info_dict['difficulties'] = difficulties

                objects_info.append(info_dict)

    train_objects = objects_info
    split = split.upper()
    with open(os.path.join(output_folder, f'{split}_objects.json'), 'w') as j:
        json.dump(train_objects, j)

    train_images = filenames
    split = split.upper()
    with open(os.path.join(output_folder, f'{split}_images.json'), 'w') as j:
        json.dump(train_images, j)




def generate_objects(output_folder, split:str):
    split = split.lower()
    assert (split == 'train' or split == 'val')
    labels_dir = f'/srv/data/jbang36/bdd/labels/bdd_{split}.json'
    json_file = _load_json(labels_dir)

    objects_info = []
    ### for each frame we need a dict
    for frame_content in json_file:
        info_dict = {}
        info_dict['boxes'] = []
        info_dict['labels'] = [] ### we will use label_map to convert to numbers
        info_dict['difficulties'] = []
        if frame_content['labels'] is not None:
            boxes = []
            labels = []
            difficulties = []
            for obj in frame_content['labels']:
                boxes.append( [int(obj['box2d']['x1']), int(obj['box2d']['y1']), int(obj['box2d']['x2']), int(obj['box2d']['y2'])] )
                labels.append( label_map[obj['category']] )
                difficulties.append( determine_difficulty(obj) )

            info_dict['boxes'] = boxes
            info_dict['labels'] = labels
            info_dict['difficulties'] = difficulties

            objects_info.append(info_dict)

    train_objects = objects_info
    split = split.upper()
    with open(os.path.join(output_folder, f'{split}_objects.json'), 'w') as j:
        json.dump(train_objects, j)



def generate_images(output_folder, split:str):
    split = split.lower()
    assert(split == 'train' or split == 'val')
    labels_dir = f'/srv/data/jbang36/bdd/labels/bdd_{split}.json'
    json_file = _load_json(labels_dir)
    image_folder = os.path.join('/srv/data/jbang36/bdd/images/100k', split)

    filenames = []
    for frame_content in json_file:
        if frame_content['labels'] is not None:
            filenames.append( os.path.join(image_folder, frame_content['name']) )


    train_images = filenames
    split = split.upper()
    with open(os.path.join(output_folder, f'{split}_images.json'), 'w') as j:
        json.dump(train_images, j)


def generate_label_map(output_folder):
    """
    we have extracted the label map data from the json train annotations so we are 100 percent sure this is correct
    """

    ###
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too


def _load_json(path_list_idx):
    with open(path_list_idx, "r") as file:
        data = json.load(file)
    print(len(data))
    return data




if __name__ == "__main__":
    output_folder = '/srv/data/jbang36/bdd/ssd_sgr'
    #create_data_lists(output_folder)
    create_data_lists_day_night(output_folder)