"""
In this file, we perform experiments on domain drift detection
"""

import sys
sys.path.append('/nethome/jbang36/k_amoeba')


import argparse

parser = argparse.ArgumentParser(description='Which Dataset?')
parser.add_argument('--source', type=str,
                    choices=['rainy'])
parser.add_argument('--target', type=str,
                    choices=['overcast', 'snowy', 'clear'])

args, unknown = parser.parse_known_args()


from odin.loaders.bdd_loader import load_images
import os
import numpy as np
from tqdm import tqdm
import torch

from odin.models import vaegan_model_builder
from amoeba_utils.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind
import json


from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t

import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data(images):
    train_dataset = InferenceDataset(images)
    batch_size = 16
    num_workers = 1

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=False)

    return train_dataloader



def load_dagan(load_file = None):

    architecture = 'VAEGAN'
    image_size = 32
    model_builder = vaegan_model_builder
    model_args = {"channels":3}
    latent_dimensions = 128

    vaegan_model = model_builder( arch = architecture,
                                  base = image_size,
                                  latent_dimensions = latent_dimensions,
                                  **model_args)

    if load_file:
        ## user supplies the directory to load the model...
        ## first check if the directory exists, if it does load from the directory
        if os.path.exists(load_file):
            ### we also expect the file to end i
            ### in the directory, we choose the last epoch?
            vaegan_model.load_state_dict(torch.load(load_file))


    return vaegan_model


def get_features(images, model):
    ### for the given number of images, we extract the features using the model
    ### we first need to make the images 4d


    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)


    dataloader = prepare_data(images)


    all_features = []
    with torch.no_grad():
        for data,_ in tqdm(dataloader):
            data = data.to(DEVICE)
            features = model(data)
            features = features.cpu()

            #### we need to convert the features to numpy
            features = features.numpy()

            all_features.append(features)

    all_features = np.vstack(all_features)
    print(all_features.shape)
    assert(all_features.ndim == 2)

    return all_features

def get_distances(source_features, target_features):
    ### basically, we apply the density bands based approach to calculate the distance
    ### for the source features, we first compute the mean then compute the l2 distance
    ### for the target features, we just compute the distances

    mean_feature = np.mean(source_features, axis = 0)
    print(f'mean feature shape is: {mean_feature.shape}')

    #### now we compute the distances
    source_distances = np.linalg.norm(source_features - mean_feature, axis = 1)
    target_distances = np.linalg.norm(target_features - mean_feature, axis = 1)

    print(f"distances shape: {source_distances.shape}, {target_distances.shape}")

    return source_distances, target_distances



def ttest_custom(data1, data2, alpha):


    # calculate means
    mean1, mean2 = mean(data1), mean(data2)
    # calculate standard errors
    se1, se2 = sem(data1), sem(data2)
    # standard error on the difference between the samples
    sed = sqrt(se1 ** 2.0 + se2 ** 2.0)
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = len(data1) + len(data2) - 2
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p


def detect_drift(source_distances, target_distances):
    target_distribution = []
    p_values = []
    max_count = 200
    #### try sampling instead
    for i, target_distance in enumerate(target_distances):
        target_distribution.append(target_distance)
        target_distances__ = np.array(target_distribution)
        #print(f'taget dist numpy shape: {target_dist_numpy.shape}')
        statistics, p_value = ttest_ind(source_distances, target_distances__,
                                        equal_var=False)  ### we might need to change this to numpy array

        #### we need to convert the given p_value to something else
        p_values.append(float(p_value))
        if i >= max_count:

            break

    return p_values



def get_model():
    ### after loading the data, we load the dagan model and extract the features
    model_directory = os.path.join('/srv/data/jbang36/checkpoints/odin/bdd/rainy',
                                   'checkpoint_epoch115.pth')

    vaegan_model = load_dagan(model_directory)
    encoder = vaegan_model.Encoder

    encoder = encoder.to(DEVICE)
    return encoder



if __name__ == "__main__":
    """
    steps:
    1. load the dataset
    2. extract the features
    3. feed it to detect drift function
    """
    source_name = args.source
    target_name = args.target

    #### load the bdd partition....
    source_data = load_images(source_name)
    target_data = load_images(target_name)

    model = get_model()

    source_features = get_features(source_data, model)
    target_features = get_features(target_data, model)

    #### after getting the features, we need to compute the distances
    source_distances, target_distances = get_distances(source_features, target_features)

    p_values = detect_drift(source_distances, target_distances)
    result_dict = {'p_values': p_values}


    ### we need to save the p_values array
    experiment_name = os.path.basename( os.getcwd() )
    BASE_DIRECTORY = '/srv/data/jbang36/amoeba/experiments'

    ### save directory
    os.makedirs(os.path.join(BASE_DIRECTORY, experiment_name), exist_ok=True)
    save_directory = os.path.join(BASE_DIRECTORY, experiment_name, f'{source_name}_{target_name}.json')

    with open(save_directory, 'w') as f:
        json.dump(result_dict, f)






