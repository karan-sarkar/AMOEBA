"""
We implement ODIN-DETECTOR and perform experiments on
speed - what is the the total throughput of the system
speed - how many samples are needed to fully determine that there has been drift
accuracy - how accurate is the division between source and target domain images
"""


from odin.models import vaegan_model_builder
import os
import torch
import numpy as np
from utils.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances

import sys
sys.path.append('/nethome/jbang36/k_amoeba')

from tqdm import tqdm

from odin.mlep_odin_main.mlep.mlep.text.DataCharacteristics.L2NormDataCharacteristics import L2NormDataCharacteristics
from odin.mlep_odin_main.mlep.mlep.drift_detector.UnlabeledDriftDetector.KullbackLeibler import KullbackLeibler

#### okay, I am going to have to implement my own distribution and implement my own KL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




class Distribution:

    def __init__(self):
        self.stats = {}

    def compute_centroid(self, features):
        return np.mean(features, axis = 0)

    def compute_distances(self, centroid, features):
        distances = pairwise_distances(centroid, features)
        return distances

    def compute_bounds(self, distances:np.array):
        delta = 0.5
        #### well there must be a zero, so we must first order the distances, and then compute the lower and upper
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        ### find the index of value zero
        start = -1
        for i in range(len(sorted_distances)):
            if sorted_distances[i] == 0:
                start = i

        if start == -1:
            raise ValueError

        index_lower, index_upper = start, start

        while True:
            ### update the bounds
            if index_lower - 1 < 0:
                index_upper += 1

            if index_upper + 1 == len(sorted_distances):
                index_lower -= 1

            else:
                if abs(sorted_distances[index_upper+1]) < abs(sorted_distances[index_lower-1]):
                    index_upper += 1
                else:
                    index_lower -= 1

            ### calculate the delta
            if index_upper - index_lower >= delta:
                break

        delta_lower = sorted_distances[index_lower]
        delta_upper = sorted_distances[index_upper]

        ### relevant indices gives the indices for features that fall between delta_lower, delta_upper
        ### we need this to compute the pdf during kl divergence
        relevant_indices = sorted_indices[index_lower:index_upper]

        return delta_lower, delta_upper, relevant_indices



class DriftDetector:

    def __init__(self):
        self.delta = 0.5
        self.clusters = []
        ## we create our first cluster

    def start(self, cluster0_features):

        cluster0 = Distribution()
        centroid = cluster0.compute_centroid(cluster0_features)
        distances = cluster0.compute_distances(centroid, cluster0_features)
        delta_low, delta_high, relevant_indices = cluster0.compute_bounds(distances)

        cluster_info = {}
        cluster_info['features'] = cluster0_features
        cluster_info['delta_low'] = delta_low
        cluster_info['delta_high'] = delta_high
        cluster_info['centroid'] = centroid
        cluster_info['relevant_features'] = cluster0_features[relevant_indices]
        #### compute distances between centroid and other features
        self.clusters.append(cluster_info)

        self.temp_cluster = {}


    def kl_divergence(self, p:np.array, q:np.array):
        """
        We first need to convert the arrays to have non zeros
        :param p:
        :param q:
        :return:
        """

        #p = np.nan_to_num(p, copy = False)
        #q = np.nan_to_num(q, copy = False)
        p_zero_indices = p <= 0
        q_zero_indices = q <= 0
        p[p_zero_indices] = 0.0001
        q[q_zero_indices] = 0.0001

        result = np.sum(p * np.log(np.divide(p, q)))
        #result = np.nan_to_num(result, 0)

        return result

    def update_one(self, feature):
        ### we assume an update on 1 feature
        ### we simplify the logic to just work on a cluster right before it
        ### this will work since we don't really..... train multiple models
        if len(self.clusters) == 0:
            raise ValueError(f"We need at least one cluster to make updates to the new one")
        previous_cluster_info = self.clusters[-1]
        previous_centroid = previous_cluster_info['centroid']

        temp_dist = Distribution()
        distance = temp_dist.compute_distances(previous_centroid, feature)
        delta_low, delta_high = previous_cluster_info['delta_low'], previous_cluster_info['delta_high']
        if distance >= delta_low and distance <= delta_high:
            ## if it falls under the previous category, we don't do anything
            return

        else:
            ### if it goes into a new cluster, we have to update the stats -- compute kl divergence
            previous_temp_cluster_info = self.temp_cluster

            temp_features = np.append(self.temp_cluster['features'], feature, axis = 0)
            ### after adding the feature, we need to recompute everything
            cluster = Distribution()
            centroid = cluster.compute_centroid(temp_features)
            distances = cluster.compute_distances(centroid, temp_features)
            delta_low, delta_high, relevant_indices = cluster.compute_bounds(distances)

            cluster_info = {}
            cluster_info['features'] = temp_features
            cluster_info['delta_low'] = delta_low
            cluster_info['delta_high'] = delta_high
            cluster_info['centroid'] = centroid
            cluster_info['relevant_features'] = temp_features[relevant_indices]
            self.temp_cluster = cluster_info

            ### after updating everything, we need to compute the kl divergence
            previous_features = previous_temp_cluster_info['relevant_features']
            current_features = cluster_info['relevant_features']

            score = self.kl_divergence(previous_features, current_features)


            return score


    def update(self, features):
        """
        steps:
        1. if datapoint falls under the previous delta low, delta high bounds,
           then the datapoint goes into original cluster
        2. Since the original cluster is stabilized, we don't make updates to delta low, delta high

        3. If the datapoint does not fall between delta low, delta high, a new cluster is made.
        4. For this cluster, we do the following process for each datapoint given
            a. keep copy of previous stats
            b. put the new datapoint in and calculate new centroid, new bounds
            c. compute the KL divergence between previous pdf and current pdf
            d. once kl divergence value gives 0, we finalize drift
        """

        for feature in features:
            score = self.update_one(feature)


            ### we have updated everything
            ### if score is none, there has been no drift
            ### if the score is 0, then we have finalized drift
            ### if the score is somewhere in between, we need more points to determine drift
            if score is None:
                ### the added point belongs to the original cluster, there is no drift
                pass
            if score == 0:
                ## we need to initialize a new cluster,

                total_number_of_points_needed = len(self.temp_cluster['features'])
                ### start making the result dict

                result_dict = {}
                result_dict['total_samples_needed'] = total_number_of_points_needed
                #### we also need information on the final labels for all the points (source and target)

                break



        return result_dict










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
    for data in tqdm(dataloader):
        data = data.to(DEVICE)
        features = model(data)
        features = features.cpu()

        #### we need to convert the features to numpy
        features = features.numpy()

        all_features.append(features)

    all_features = np.stack(all_features)
    print(all_features.shape)
    assert(all_features.ndim == 2)

    return all_features


def prepare_data(images):
    train_dataset = InferenceDataset(images)
    batch_size = 16
    num_workers = 1

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=False)

    return train_dataloader

if __name__ == "__main__":
    """
    steps:
    1. load the dagan
    2. load the drift detector
    3. profile the measurements
    """

    ## 1. load the dagan
    model_directory = os.path.join('/srv/data/jbang36/checkpoints/odin/bdd/daytime',
                                   'checkpoint_epoch115.pth')


    vaegan_model = load_dagan(model_directory)
    encoder = vaegan_model.Encoder

    encoder = encoder.to(DEVICE)

    ## 2. get the data
    #### we will load day, dawn, night
    from odin.loaders.bdd_loader import load_images

    dataset_name = 'bdd'
    dataset_split = 'daytime'
    day_images = load_images(dataset_split)

    dataset_split = 'dawn/dusk'
    dawn_images = load_images(dataset_split)

    dataset_split = 'night'
    night_images = load_images(dataset_split)

    ## 3. now get features on all these images

    day_features = get_features(day_images, encoder)
    dawn_features = get_features(dawn_images, encoder)
    night_features = get_features(night_images, encoder)

    ## 4. use the drift detector to perform drift detection









