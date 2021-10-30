"""
We need a bdd loader to create the appropriate .json files for image loading
"""


import json
import sys
sys.path.append('/nethome/jbang36/k_amoeba')
import os

from tqdm import tqdm
from PIL import Image
import numpy as np


BASE_DIRECTORY = '/srv/data/jbang36/bdd/odin'



category_list = ['clear', 'rainy', 'snowy', 'overcast', 'partly cloudy', 'foggy',
                     'daytime', 'dawn/dusk', 'night',
                     'city street', 'highway', 'residential', 'parking lot', 'tunnel', 'gas stations',
                     'undefined']

category2key = {
    'clear': 'weather', 'rainy': 'weather', 'snowy': 'weather',
    'overcast': 'weather', 'partly cloudy': 'weather', 'foggy': 'weather',
    'daytime': 'timeofday', 'dawn/dusk': 'timeofday', 'night': 'timeofday',
    'city street': 'scene', 'highway': 'scene', 'residential':'scene',
    'parking lot': 'scene', 'tunnel': 'scene', 'gas stations': 'scene'
}


adjusted_category_names = {
    'partly cloudy': 'partly_cloudy',
    'dawn/dusk': 'dawn_dusk',
    'city street': 'city_street',
    'parking lot': 'parking_lot',
    'gas stations': 'gas_stations'
}



def load_annotations_train():
    file_directory = '/srv/data/jbang36/bdd/labels/bdd_train.json'

    with open(file_directory, 'r') as f:
        annotations = json.load(f)

    return annotations


def load_images(category:str):
    ### we search the proper json and load all the files

    if category in adjusted_category_names:
        category = adjusted_category_names[category]

    filename = os.path.join(BASE_DIRECTORY, category+"_images.json")

    with open(filename, 'r') as f:
        image_filenames = json.load(f)

    images = []

    for image_filename in tqdm(image_filenames):
        im = load_image(image_filename)
        images.append( im )

    images = np.stack(images)
    print(images.shape)
    assert(images.ndim == 4)

    return images


def load_image(filename):
    im = Image.open(filename)
    return np.array(im)


def create_partition(category:str):
    """
    {'attributes':
        {'weather': 'clear', 'rainy', 'snowy', 'overcast', 'partly cloudy', 'foggy',
         'timeofday': 'daytime', 'dawn/dusk', 'night',
         'scene':     'city street', 'highway', 'residential', 'parking lot', 'tunnel', 'gas stations'
    """



    if category not in category_list:
        raise ValueError(f"category given: {category}, available categories: {category_list}")

    ### get the directory+imagefiles from
    image_directory = '/srv/data/jbang36/bdd/images/100k/train'

    image_files = []

    annotations_train = load_annotations_train()
    key = category2key[category]
    print(f"Searching: {category}...")

    for anno in tqdm(annotations_train):
        #print(f"key: {key}, cate: {category}")
        if anno['attributes'][key] == category:
            ### we need to append to the list
            image_files.append( os.path.join(image_directory, anno['name']) )

    ### save to the file
    if category == 'dawn/dusk':
        category = 'dawn_dusk'

    FILENAME = os.path.join(BASE_DIRECTORY, category + '_images.json')

    with open(FILENAME, 'w') as f:
        json.dump(image_files, f)


if __name__ == "__main__":
    category_list = ['clear', 'rainy', 'snowy', 'overcast', 'partly cloudy', 'foggy',
                     'daytime', 'dawn/dusk', 'night',
                     'city street', 'highway', 'residential', 'parking lot', 'tunnel', 'gas stations'
                     ]


    for category in category_list:
        create_partition(category)
