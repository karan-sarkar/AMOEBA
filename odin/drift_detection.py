

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


import sys
sys.path.append('/nethome/jbang36/k_amoeba')





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

    #### now we use the inference dataset


if __name__ == "__main__":
    """
    steps:
    1. load the dagan
    2. load the drift detector
    3. profile the measurements
    """

    ### how do we load the model?? we just need the encoder right?
    model_directory = os.path.join('/srv/data/jbang36/checkpoints/odin/bdd/daytime',
                                   'checkpoint_epoch115.pth')


    vaegan_model = load_dagan(model_directory)
    encoder = vaegan_model.Encoder




