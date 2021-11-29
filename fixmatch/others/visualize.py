
#### I will rewrite the visualization script....
### will take in a ImageList, will take in series of Boxlist, will also take in an index,


import numpy as np


from benchmark.fixmatch.dataset.bdd import ImageList
from benchmark.fixmatch.dataset.boxlist import BoxList

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import PIL

import torch
import torchvision.transforms as transforms

def visualize_tensor(image:torch.Tensor, boxes:BoxList):
    t = transforms.ToPILImage()
    img = t(image)
    box = boxes.box
    draw_boxes(img, box)

def visualize_pil(image:PIL.Image.Image, boxes:BoxList):
    box = boxes.box
    draw_boxes(image, box)



def visualize_normalized(input_imgs:ImageList, input_boxes:list, index:int):
    ### we will visualize the input
    ### 1. get the input_imgs
    image = input_imgs.tensors[index]
    image = image.permute(1,2,0)
    image = image.numpy()
    image *= 255
    image = image.astype(np.uint8)

    ### now we work on the input boxes
    boxes = input_boxes[index].box ### the type of this is boxlist
    boxes = boxes.cpu().numpy()
    ### multiply the boxes by image size
    image_width, image_height = input_imgs.sizes[index]
    boxes[:, 0] *= image_width
    boxes[:, 2] *= image_width
    boxes[:, 1] *= image_height
    boxes[:, 3] *= image_height
    boxes = boxes.astype(np.int32)
    #### this should be a 2d array, let's draw all the boxes given

    draw_boxes(image, boxes)

    return image, boxes




def visualize(input_imgs:ImageList, input_boxes:list, index:int):
    ### we will visualize the input
    ### 1. get the input_imgs
    image = input_imgs.tensors[index]
    image = image.permute(1,2,0)
    image = image.numpy()
    image *= 255
    image = image.astype(np.uint8)

    ### now we work on the input boxes
    boxes = input_boxes[index].box ### the type of this is boxlist
    boxes = boxes.cpu().numpy()
    #### this should be a 2d array, let's draw all the boxes given

    draw_boxes(image, boxes)

    return image, boxes


def draw_boxes(input_image:np.array, boxes:np.array):
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(input_image)

    # Create a Rectangle patch
    for i in range(boxes.shape[0]):
        box = boxes[i,:]
        ### need to convert the xyxy coordinates to something that rectangel takes
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


