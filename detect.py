from torchvision import transforms
from util import *
from PIL import Image, ImageDraw, ImageFont
from os import listdir
from os.path import isfile, join
from random import sample
import tqdm
import argparse
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
args = parser.parse_args()

# Load model checkpoint
checkpoint = args.ckpt
model = torch.load(checkpoint)
model = model.to(device)
if not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

filter_obj = 'truck'
filter_limit = 1
query_obj = 'car'
counts = []

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores,_,_ = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.module.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    rev_label_map[0] = 'background'
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    
    counter = Counter()
    for ob in det_labels:
        counter[ob] += 1
    
    if counter[filter_obj] >= filter_limit:
        counts.append(counter[query_obj])
    
    

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("arial.ttf", 15, encoding="unic")

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    mypath = '../fcos-pytorch/bdd100k/images/100k/val'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    random.seed(10)
    for img_path in tqdm.tqdm(sample(onlyfiles,100)):
        original_image = Image.open(os.path.join(mypath, img_path), mode='r')
        original_image = original_image.convert('RGB')
        
        detect(original_image, min_score=0.2, max_overlap=0.45, top_k=200).save(img_path[:-3] + str(checkpoint) + '.jpg')
    print(sum(counts) / len(counts))
        
        
