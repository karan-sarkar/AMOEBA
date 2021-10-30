"""
In this file, we load an already trained da-gan

Specifically the model will have components we can access
"""

from odin.models import vaegan_model_builder
import os
import torch


def image_convert32(images):
    """
    We change the image format from the bdd_loader.load_images to tensors
    required to execute on the dagan

    """
    images = images[:,::9,::9,:]
    images = images[:,:32,:32,:]
    tensors = torch.Tensor(images)
    tensors = tensors.permute(0,3,1,2)

    ### now it should have the shape N, 3, 32, 32

    return tensors






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

