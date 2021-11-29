"""
We implement ODIN-DETECTOR and perform experiments on
speed - what is the the total throughput of the system
speed - how many samples are needed to fully determine that there has been drift
accuracy - how accurate is the division between source and target domain images

"""


import sys
sys.path.append('/nethome/jbang36/k_amoeba')


from odin.models import vaegan_model_builder
import os
import torch
import numpy as np
from amoeba_utils.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances

from tqdm import tqdm
import json

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
        """
        We compute the distance between the designated point, and rest of the points
        :param points:
        :param solution_index:
        :return:
        """

        if features.ndim == 2:

            distances = np.ndarray(shape = (features.shape[0]))
            for i in range(len(features)):
                distance = np.linalg.norm(centroid - features[i])
                distances[i] = distance
            return distances
        else:
            distance = np.linalg.norm(centroid - features)
            return distance

    def compute_bounds(self, distances:np.array):
        delta = 0.5
        #### well there must be a zero, so we must first order the distances, and then compute the lower and upper
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        ### find the index of value zero
        ### for the sorted_distances, we just want to find the minimum
        ### that's where we will start
        start = np.argmin(sorted_distances)

        index_lower, index_upper = start, start
        print(f'index lower, index upper: {index_lower, index_upper}')

        while True:
            ### edge cases, only one element in list
            if len(sorted_distances) == 1:
                break

            ### update the bounds
            if index_lower - 1 < 0:
                index_upper += 1

            elif index_upper + 1 == len(sorted_distances):
                index_lower -= 1

            else:
                if abs(sorted_distances[index_upper+1]) < abs(sorted_distances[index_lower-1]):
                    index_upper += 1
                else:
                    index_lower -= 1

            ### calculate the delta
            if index_upper - index_lower >= len(distances) * delta:
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
        cluster_info['cluster'] = cluster0
        #### compute distances between centroid and other features
        self.clusters.append(cluster_info)

        self.temp_cluster = {}
        self.temp_cluster['features'] = []

    def init_temp(self, cluster1_features):
        cluster1 = Distribution()
        centroid = cluster1.compute_centroid(cluster1_features)
        distances = cluster1.compute_distances(centroid, cluster1_features)
        delta_low, delta_high, relevant_indices = cluster1.compute_bounds(distances)

        cluster_info = {}
        cluster_info['features'] = cluster1_features
        cluster_info['delta_low'] = delta_low
        cluster_info['delta_high'] = delta_high
        cluster_info['centroid'] = centroid
        cluster_info['relevant_features'] = cluster1_features[relevant_indices]
        cluster_info['cluster'] = cluster1
        #### compute distances between centroid and other features

        self.temp_cluster = cluster_info


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

        #### cut the p, q sizes to match
        min_length = min(p.shape[0], q.shape[0])
        p = p[len(p) - min_length:]
        q = q[len(q) - min_length:]

        result = np.sum(p * np.log(np.divide(p, q)))
        #result = np.nan_to_num(result, 0)

        return result



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
        result_dict = {}
        result_dict['scores'] = []
        result_dict['drift'] = False
        print(f'features shape: {features.shape}')

        features_start = features[:10]
        features = features[10:]

        self.init_temp(features_start)

        for i in range(features.shape[0]):
            print(f"i: {i}")
            feature = features[i]
            print(f"feature shape: {feature.shape}")
            score = self.update_one(feature)
            print(f"----------------- score is: {score}")
            result_dict['scores'].append(score)

            ### we have updated everything
            ### if score is none, there has been no drift
            ### if the score is 0, then we have finalized drift
            ### if the score is somewhere in between, we need more points to determine drift
            if score is None:
                ### the added point belongs to the original cluster, there is no drift
                result_dict['scores'].append(-1)

            if score == 0:
                ## we need to initialize a new cluster,

                total_number_of_points_needed = len(self.temp_cluster['features'])
                ### start making the result dict

                result_dict['drift'] = True
                result_dict['scores'].append(score)
                #### we also need information on the final labels for all the points (source and target)
                break


        result_dict['total_sampled_needed'] = len(self.temp_cluster['features'])

        return result_dict


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
        print(f'distance: {distance}, delta_low: {delta_low}, delta_high: {delta_high}')

        if distance >= delta_low and distance <= delta_high:
            ## if it falls under the previous category, we don't do anything
            print(f"----point feel between previous cluster bounds... no drift...")
            return

        else:
            ### if it goes into a new cluster, we have to update the stats -- compute kl divergence
            previous_temp_cluster_info = self.temp_cluster
            print(f"new cluster features: {len(self.temp_cluster['features'])}")

            feature = feature.reshape(1, -1)
            temp_features = np.append(self.temp_cluster['features'], feature, axis = 0)
            #print(f'temp features: {temp_features}')
            ### after adding the feature, we need to recompute everything
            cluster = Distribution()
            centroid = cluster.compute_centroid(temp_features)
            print(f"centroid: {centroid.shape}")
            print(f"temp features len: {temp_features.shape}")
            distances = cluster.compute_distances(centroid, temp_features)
            print(f"distance between centroid and temp features: {distances}")
            delta_low, delta_high, relevant_indices = cluster.compute_bounds(distances)

            cluster_info = {}
            cluster_info['features'] = temp_features
            cluster_info['delta_low'] = delta_low
            cluster_info['delta_high'] = delta_high
            cluster_info['centroid'] = centroid
            cluster_info['relevant_features'] = temp_features[relevant_indices]
            self.temp_cluster = cluster_info

            ### after updating everything, we need to compute the kl divergence
            #previous_features = previous_temp_cluster_info['relevant_features']
            #current_features = cluster_info['relevant_features']

            #score = self.kl_divergence(previous_features, current_features)

            #### new method of computing the score
            previous_delta_low = previous_temp_cluster_info['delta_low']
            previous_delta_high = previous_temp_cluster_info['delta_high']

            if previous_delta_low == delta_low and previous_delta_high == delta_high:
                score = 0
            else:
                score = 1


            return score


    def update2(self, features, confidence):
        """
        we make modifications to update as follows:
        (1) for the temporary cluster, we do not make updates if the new point is within bounds
        (2) we only make updates if the new point is not within bounds.
        (3) we conclude the drift experiment when the number of streaming points meets the confidence we give

        confidence should be a number between 0 and 1, once the included frames num / total given frames num reaches confidence,
        we exit the drift updates...

        """
        result_dict = {}
        result_dict['scores'] = []
        result_dict['drift'] = False
        print(f"features shape: {features.shape}")

        features_start = features[:10]
        features = features[10:]

        self.init_temp(features_start)

        #### how about we do the most recent 10? ###
        recent_ten = []
        total_count = 0


        for i in range(features.shape[0]):
            total_count += 1
            feature = features[i]
            within_bounds = self.update_one2(feature)
            if within_bounds is not None:


                recent_ten.append(within_bounds)

                if len(recent_ten) > 10:
                    recent_ten = recent_ten[1:]


                if len(recent_ten) == 10:
                    print(recent_ten)
                    confidence_computed = sum(recent_ten) / len(recent_ten)

                    if confidence_computed >= confidence:
                        break


        return total_count


    def update_one2(self, feature):
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
        print(f'distance: {distance}, delta_low: {delta_low}, delta_high: {delta_high}')

        if distance >= delta_low and distance <= delta_high:
            ## if it falls under the previous category, we don't do anything
            print(f"----point feel between previous cluster bounds... no drift...")
            within_bounds = None


        else:
            ### if it goes into a new cluster, we have to update the stats -- compute kl divergence

            delta_low, delta_high = self.temp_cluster['delta_low'], self.temp_cluster['delta_high']
            centroid = self.temp_cluster['centroid']
            cluster = self.temp_cluster['cluster']

            distance = cluster.compute_distances(centroid, feature)

            if distance >= delta_low and distance <= delta_high:
                ####
                within_bounds = True
                ### we simply update the relevant features and that's it....actually we don't even need that
                feature = feature.reshape(1, -1)
                self.temp_cluster['features'] = np.append(self.temp_cluster['features'], feature, axis = 0)


            else:
                within_bounds = False

                ##### we need to recompute the bounds

                feature = feature.reshape(1, -1)
                temp_features = np.append(self.temp_cluster['features'], feature, axis = 0)
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
                cluster_info['cluster'] = cluster
                self.temp_cluster = cluster_info


        return within_bounds





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
    
    
    #### we need to do the following experiments:
    1. rainy -> overcast
    2. rainy -> snowy
    3. rainy -> clear
    
    """

    ## 1. load the dagan
    model_directory = os.path.join('/srv/data/jbang36/checkpoints/odin/bdd/rainy',
                                   'checkpoint_epoch115.pth')

    vaegan_model = load_dagan(model_directory)
    encoder = vaegan_model.Encoder

    encoder = encoder.to(DEVICE)

    from odin.loaders.bdd_loader import load_images

    f2s = ['overcast', 'snowy', 'clear']
    drifts = ['rainy2overcast', 'rainy2snowy', 'rainy2clear']

    dataset_name = 'bdd'
    dataset_split = 'rainy'
    rainy_images = load_images(dataset_split)
    rainy_features = get_features(rainy_images, encoder)

    all_results = {}

    for i in range(3):
        dataset_split = f2s[i]
        f2_images = load_images(dataset_split)
        f2_features = get_features(f2_images, encoder)
        drift_detector = DriftDetector()
        drift_detector.start(rainy_features)
        all_results[drifts[i]] = drift_detector.update(f2_features)

    directory = '/srv/data/jbang36/checkpoints/odin/detector'
    result_file = os.path.join(directory, 'results.json')

    with open(result_file, 'w') as f:
        json.dump(all_results, f)










