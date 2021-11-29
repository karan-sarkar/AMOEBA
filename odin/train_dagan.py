"""
This file is dedicated to training dagan.

Naturally, the input should be the dataset.
Everything else should be readily configured.

"""

import sys
sys.path.append('/nethome/jbang36/k_amoeba')

import os

import numpy as np
import shutil
import torch
import json

from odin.models import vaegan_model_builder
from odin.optimizer.VAEGANOptimizerBuilder import VAEGANOptimizerBuilder
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from odin.trainer import VAEGANTrainer
from odin import utils
from amoeba_utils.inference_dataset import InferenceDataset







def train_dagan(dataset:np.array, dataset_name:str, dataset_split:str):
    """
    we expect the dataset to be numpy array of 4 dimensions
    dataset_name should be something like bdd
    dataset_split should be something like daytime, dawn, night, rainy, clear etc....

    """
    BASE_DIRECTORY = '/srv/data/jbang36/checkpoints/odin'
    MODEL_SAVE_NAME = 'checkpoint'
    MODEL_SAVE_FOLDER = os.path.join(BASE_DIRECTORY, dataset_name, dataset_split)
    CHECKPOINT_DIRECTORY = MODEL_SAVE_FOLDER
    LOGGER_SAVE_NAME = os.path.join(CHECKPOINT_DIRECTORY, 'train_dagan.log')

    config_dict = load_config()
    NORMALIZATION_MEAN = config_dict['NORMALIZATION_MEAN']
    NORMALIZATION_STD = config_dict['NORMALIZATION_STD']
    RANDOM_ERASE_VALUE = config_dict['RANDOM_ERASE_VALUE']
    TRAINDATA_KWARGS = {'rea_value': config_dict['RANDOM_ERASE_VALUE']}

    #### now we need to load the train_dataset (dataloader), and test_dataset (dataloader)

    ### we can either see which format these generators give
    train_generator, test_generator = get_data(dataset, config_dict) #### I guess we just need the dataloader

    model_filename = os.path.join(CHECKPOINT_DIRECTORY, MODEL_SAVE_NAME)
    vaegan_model = load_dagan(model_filename)


    vaegan_model.cuda()

    ### initiate other parts of the trainer i.e. optimizer, scheduler, trainer
    optimizer_builder = VAEGANOptimizerBuilder(base_lr = config_dict['BASE_LR'])
    optimizer = optimizer_builder.build(vaegan_model, config_dict['OPTIMIZER_NAME'])

    scheduler = {}
    for base_model in ["Encoder", "Decoder", "Discriminator", "Autoencoder", "LatentDiscriminator"]:
        #config_dict['LR_KWARGS'] = {"step_size": 30, "gamma": 0.25}
        scheduler[base_model] = StepLR(optimizer[base_model], last_epoch = -1, **config_dict['LR_KWARGS'])


    os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
    logger = utils.generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME)

    loss_stepper = VAEGANTrainer(model = vaegan_model,
                            loss_fn = None,
                            optimizer = optimizer,
                            scheduler = scheduler,
                            train_loader = train_generator,
                            test_loader = test_generator,
                            epochs = config_dict['EPOCHS'],
                            batch_size = config_dict['BATCH_SIZE'],
                            latent_size = config_dict['LATENT_DIMENSIONS'],
                            logger = logger)

    loss_stepper.setup(step_verbose = config_dict['STEP_VERBOSE'],
                       save_frequency=config_dict["SAVE_FREQUENCY"],
                       test_frequency=config_dict["TEST_FREQUENCY"],
                       save_directory=MODEL_SAVE_FOLDER,
                       save_backup=None,
                       backup_directory=CHECKPOINT_DIRECTORY,
                       gpus=1,
                       fp16=config_dict["FP16"],
                       model_save_name=MODEL_SAVE_NAME,
                       logger_file=LOGGER_SAVE_NAME
                       )

    previous_stop = -1
    loss_stepper.train(continue_epoch=previous_stop)





def compute_mean_and_std(dataset, cached_mean_directory, cached_std_directory):
    if os.path.exists(cached_mean_directory):
        NORMALIZATION_MEAN = np.load(cached_mean_directory)
    else:
        NORMALIZATION_MEAN = np.mean(dataset, axis=(0, 1, 2))
        np.save(cached_mean_directory, NORMALIZATION_MEAN)

    if os.path.exists(cached_std_directory):
        NORMALIZATION_STD = np.load(cached_std_directory)
    else:
        NORMALIZATION_STD = np.std(dataset, axis=(0, 1, 2))
        np.save(cached_std_directory, NORMALIZATION_STD)

    return NORMALIZATION_MEAN, NORMALIZATION_STD


def load_config():
    ### give things in a dictionary format??
    config_dict = {}
    config_dict['NORMALIZATION_MEAN'] = 0.5
    config_dict['NORMALIZATION_STD'] = 0.5
    config_dict['NORMALIZATION_SCALE'] = 255.0
    config_dict['H_FLIP'] = 0.5
    config_dict['T_CROP'] = True
    config_dict['RANDOM_ERASE'] = True
    config_dict['RANDOM_ERASE_VALUE'] = 0.5
    config_dict['CHANNELS'] = 3
    config_dict['BATCH_SIZE'] = 32
    config_dict['NUM_WORKERS'] = 1

    config_dict['OPTIMIZER_NAME'] = "Adam"

    config_dict['OPTIMIZER_KWARGS'] = {"betas":[0.5, 0.999]}
    config_dict['BASE_LR'] =  0.0001
    config_dict['LR_BIAS_FACTOR'] = 1.0
    config_dict['WEIGHT_DECAY'] = 0.0005
    config_dict['WEIGHT_BIAS_FACTOR'] =  0.0005
    config_dict['FP16'] = True

    config_dict['LR_KWARGS'] = {"step_size":30, "gamma":0.25}

    config_dict['EPOCHS'] =  120
    config_dict['TEST_FREQUENCY'] = 5
    config_dict['LATENT_DIMENSIONS'] = 128

    config_dict['STEP_VERBOSE'] = 100
    config_dict['SAVE_FREQUENCY'] = 5


    return config_dict



def save_dagan(model, model_filename):
    """
    filename should end with something like checkpoint.pth.tar
    """
    dictionary = {'state_dict': model.state_dict()}

    torch.save(dictionary, model_filename)


def load_dagan(model_filename):

    architecture = 'VAEGAN'
    image_size = 32
    model_builder = vaegan_model_builder
    model_args = {"channels":3}
    latent_dimensions = 128

    vaegan_model = model_builder( arch = architecture,
                                  base = image_size,
                                  latent_dimensions = latent_dimensions,
                                  **model_args)


    return vaegan_model




def get_data(dataset: np.array, config: dict):
    """
    we will bs about the labels since we don't need any
    we will process the numpy array into torch tensors
    """
    ### first we will divide the dataset into train and test
    ### then we will apply the Dataset -> DataLoader coating to both partitions

    length = len(dataset)
    division = int(length * 0.8)
    train_images = dataset[:division]
    test_images = dataset[division:]

    train_dataset = InferenceDataset(train_images)
    test_dataset = InferenceDataset(test_images)

    train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                  num_workers=config['NUM_WORKERS'], shuffle=True,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                 num_workers=config['NUM_WORKERS'], shuffle=True,
                                 pin_memory=True)

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    """
    we need to perform 3 types of experiments 
    1. (day, dusk_dawn, night)
    2. (
    
    
    """
    ### first let's train the images on day
    from odin.loaders.bdd_loader import load_images
    dataset_name = 'bdd'
    dataset_split = 'rainy'
    day_dataset = load_images(dataset_split)
    print(f'dataset shape: {day_dataset.shape}')


    train_dagan(day_dataset, dataset_name, dataset_split)