
import sys
sys.path.append('/nethome/jbang36/j_amoeba')


import os, shutil, glob, re, pdb, json
import kaptan
import click
import benchmark.odin.utils as odin_utils
import torch, torchsummary, torchvision
from benchmark.eko.loaders.bdd_loader import BDDLoader, BDD_Odin
import numpy as np



def main(config, mode, weights):
    # Generate configuration

    config = get_config_string()
    cfg = kaptan.Kaptan(handler='yaml')
    config = cfg.import_config(config)

    # Generate logger
    MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY = odin_utils.generate_save_names(config)

    ###



    logger = odin_utils.generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME)

    logger.info("*" * 40)
    logger.info("")
    logger.info("")
    logger.info("Using the following configuration:")
    logger.info(config.export("yaml", indent=4))
    logger.info("")
    logger.info("")
    logger.info("*" * 40)

    """ SETUP IMPORTS """
    # from crawlers import ReidDataCrawler
    # from generators import SequencedGenerator

    # from loss import LossBuilder

    NORMALIZATION_MEAN, NORMALIZATION_STD, RANDOM_ERASE_VALUE = odin_utils.fix_generator_arguments(config)
    TRAINDATA_KWARGS = {"rea_value": config.get("TRANSFORMATION.RANDOM_ERASE_VALUE")}


    NUM_GPUS = 1
    if NUM_GPUS > 1:
        raise RuntimeError("Not built for multi-GPU. Please start with single-GPU.")
    logger.info("Found %i GPUs" % NUM_GPUS)

    # --------------------- BUILD GENERATORS ------------------------
    # Supported integrated data sources --> MNIST, CIFAR
    # For BDD or others need a crawler and stuff...but we;ll deal with it later
    if config.get("EXECUTION.DATASET_NAME") == 'bdd':
        loader = BDDLoader()
        cutoff = 60000
        img_paths = loader.get_image_filenames()

        idx = np.arange(cutoff)  ### max idx would be 600000
        anno_path = '/srv/data/jbang36/bdd/labels/bdd_train.json'
        train = True
        img_paths = img_paths[:cutoff]

        size = config.get("DATASET.SHAPE")[0]
        print(f'size of input will be {size}')
        train_dataset = BDD_Odin(img_paths, idx, anno_path, train, size)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size = config.get('TRANSFORMATION.BATCH_SIZE'),
                                                       shuffle = True, pin_memory = True, collate_fn = train_dataset.collate_fn)

    else:


        from benchmark.odin.generators import ClassedGenerator

        load_dataset = config.get("EXECUTION.DATASET_PRELOAD")
        if load_dataset in ["MNIST", "CIFAR10", "CIFAR100"]:
            crawler = load_dataset
            # dataset = torchvision.datasets.MNIST(root="./MNIST", train=True,)
            logger.info("No crawler necessary when using %s dataset" % crawler)
        else:
            raise NotImplementedError()

        train_generator = ClassedGenerator(gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                                           normalization_mean=NORMALIZATION_MEAN, normalization_std=NORMALIZATION_STD,
                                           normalization_scale=1. / config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                                           h_flip=config.get("TRANSFORMATION.H_FLIP"),
                                           t_crop=config.get("TRANSFORMATION.T_CROP"),
                                           rea=config.get("TRANSFORMATION.RANDOM_ERASE"),
                                           **TRAINDATA_KWARGS)
        train_generator.setup(crawler, preload_classes=config.get("EXECUTION.DATASET_PRELOAD_CLASS"), \
                              mode='train', batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), \
                              workers=config.get("TRANSFORMATION.WORKERS"))
        logger.info("Generated training data generator")

        test_generator = ClassedGenerator(gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                                          normalization_mean=NORMALIZATION_MEAN, normalization_std=NORMALIZATION_STD,
                                          normalization_scale=1. / config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                                          h_flip=config.get("TRANSFORMATION.H_FLIP"),
                                          t_crop=config.get("TRANSFORMATION.T_CROP"),
                                          rea=config.get("TRANSFORMATION.RANDOM_ERASE"),
                                          **TRAINDATA_KWARGS)
        test_generator.setup(crawler, preload_classes=config.get("EXECUTION.DATASET_TEST_PRELOAD_CLASS"), \
                             mode='test', batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), \
                             workers=config.get("TRANSFORMATION.WORKERS"))
        logger.info("Generated testing data generator")

    # --------------------- INSTANTIATE MODEL ------------------------
    model_builder = __import__("odin.models", fromlist=["*"])
    model_builder = getattr(model_builder, config.get("EXECUTION.MODEL_BUILDER"))
    logger.info("Loaded {} from {} to build VAEGAN model".format(config.get("EXECUTION.MODEL_BUILDER"), "models"))

    vaegan_model = model_builder(arch=config.get("MODEL.ARCH"), base=config.get("MODEL.BASE"), \
                                 latent_dimensions=config.get("MODEL.LATENT_DIMENSIONS"), \
                                 **json.loads(config.get("MODEL.MODEL_KWARGS")))
    logger.info("Finished instantiating model")

    if mode == "test":
        vaegan_model.load_state_dict(torch.load(weights))
        vaegan_model.cuda()
        vaegan_model.eval()
    else:
        vaegan_model.cuda()
        # logger.info(torchsummary.summary(vaegan_model, input_size=(config.get("TRANSFORMATION.CHANNELS"), *config.get("DATASET.SHAPE"))))
        logger.info(torchsummary.summary(vaegan_model.Encoder, input_size=(
        config.get("TRANSFORMATION.CHANNELS"), *config.get("DATASET.SHAPE"))))
        logger.info(torchsummary.summary(vaegan_model.Decoder, input_size=(config.get("MODEL.LATENT_DIMENSIONS"), 1)))
        logger.info(torchsummary.summary(vaegan_model.LatentDiscriminator,
                                         input_size=(config.get("MODEL.LATENT_DIMENSIONS"), 1)))
        logger.info(torchsummary.summary(vaegan_model.Discriminator, input_size=(
        config.get("TRANSFORMATION.CHANNELS"), *config.get("DATASET.SHAPE"))))

    # --------------------- INSTANTIATE LOSS ------------------------
    # ----------- NOT NEEDED. VAEGAN WILL USE BCE LOSS THROUGHOUT
    # loss_function = LossBuilder(loss_functions=config.get("LOSS.LOSSES"), loss_lambda=config.get("LOSS.LOSS_LAMBDAS"), loss_kwargs=config.get("LOSS.LOSS_KWARGS"), **{"logger":logger})
    # logger.info("Built loss function")
    # --------------------- INSTANTIATE OPTIMIZER ------------------------
    optimizer_builder = __import__("odin.optimizer", fromlist=["*"])
    optimizer_builder = getattr(optimizer_builder, config.get("EXECUTION.OPTIMIZER_BUILDER"))
    logger.info(
        "Loaded {} from {} to build VAEGAN model".format(config.get("EXECUTION.OPTIMIZER_BUILDER"), "optimizer"))

    OPT = optimizer_builder(base_lr=config.get("OPTIMIZER.BASE_LR"))
    optimizer = OPT.build(vaegan_model, config.get("OPTIMIZER.OPTIMIZER_NAME"),
                          **json.loads(config.get("OPTIMIZER.OPTIMIZER_KWARGS")))
    logger.info("Built optimizer")
    # --------------------- INSTANTIATE SCHEDULER ------------------------
    try:
        scheduler = __import__('torch.optim.lr_scheduler', fromlist=['lr_scheduler'])
        scheduler_ = getattr(scheduler, config.get("SCHEDULER.LR_SCHEDULER"))
    except (ModuleNotFoundError, AttributeError):
        scheduler_ = config.get("SCHEDULER.LR_SCHEDULER")
        scheduler = __import__("odin.scheduler." + scheduler_, fromlist=[scheduler_])
        scheduler_ = getattr(scheduler, scheduler_)
    scheduler = {}
    for base_model in ["Encoder", "Decoder", "Discriminator", "Autoencoder", "LatentDiscriminator"]:
        scheduler[base_model] = scheduler_(optimizer[base_model], last_epoch=-1,
                                           **json.loads(config.get("SCHEDULER.LR_KWARGS")))
        logger.info("Built scheduler for {}".format(base_model))

    # --------------------- SETUP CONTINUATION  ------------------------
    fl_list = glob.glob(os.path.join(MODEL_SAVE_FOLDER, "*.pth"))
    _re = re.compile(r'.*epoch([0-9]+)\.pth')
    previous_stop = [int(item[1]) for item in [_re.search(item) for item in fl_list] if item is not None]
    if len(previous_stop) == 0:
        previous_stop = 0
    else:
        previous_stop = max(previous_stop) + 1
        logger.info("Previous stop detected. Will attempt to resume from epoch %i" % previous_stop)

    # --------------------- INSTANTIATE TRAINER  ------------------------
    Trainer = __import__("odin.trainer", fromlist=["*"])
    Trainer = getattr(Trainer, config.get("EXECUTION.TRAINER"))
    logger.info("Loaded {} from {} to build VAEGAN model".format(config.get("EXECUTION.TRAINER"), "trainer"))

    loss_stepper = Trainer(model=vaegan_model, loss_fn=None, optimizer=optimizer, scheduler=scheduler,
                           train_loader=train_dataloader, test_loader=train_dataloader,
                           epochs=config.get("EXECUTION.EPOCHS"), batch_size=config.get("TRANSFORMATION.BATCH_SIZE"),
                           latent_size=config.get("MODEL.LATENT_DIMENSIONS"), logger=logger)
    loss_stepper.setup(step_verbose=config.get("LOGGING.STEP_VERBOSE"),
                       save_frequency=config.get("SAVE.SAVE_FREQUENCY"),
                       test_frequency=config.get("EXECUTION.TEST_FREQUENCY"), save_directory=MODEL_SAVE_FOLDER,
                       save_backup=None, backup_directory=CHECKPOINT_DIRECTORY, gpus=NUM_GPUS,
                       fp16=config.get("OPTIMIZER.FP16"), model_save_name=MODEL_SAVE_NAME, logger_file=LOGGER_SAVE_NAME)
    if mode == 'train':
        loss_stepper.train(continue_epoch=previous_stop)
    elif mode == 'test':
        loss_stepper.evaluate()
    else:
        raise NotImplementedError()


def get_config_string():
    string = """
    EXECUTION:
      MODEL_SERVING: None
      EPOCHS: 120
      TEST_FREQUENCY: 5
      OPTIMIZER_BUILDER: VAEGANOptimizerBuilder
      MODEL_BUILDER: vaegan_model_builder
      TRAINER: VAEGANTrainer
      DATASET_NAME: 'bdd'

    SAVE:
      MODEL_VERSION: 1
      MODEL_CORE_NAME: "vagan" ##"ablation_reid"
      MODEL_BACKBONE: "res50" ##"res50"
      MODEL_QUALIFIER: "all"
      DRIVE_BACKUP: False
      SAVE_FREQUENCY: 5 # Epoch

    DATASET:
      ROOT_DATA_FOLDER: "bdd"
      TRAIN_FOLDER: "train"
      TEST_FOLDER: "test"
      SHAPE: [64,64]

    TRANSFORMATION:
      NORMALIZATION_MEAN: 0.5
      NORMALIZATION_STD: 0.5
      NORMALIZATION_SCALE: 255.0
      H_FLIP: 0.5
      T_CROP: True
      RANDOM_ERASE: True
      RANDOM_ERASE_VALUE: 0.5
      CHANNELS: 3
      BATCH_SIZE: 32
      WORKERS: 1

    MODEL:
      BASE: 64
      ARCH: "VAEGAN"
      LATENT_DIMENSIONS: 128
      MODEL_KWARGS: '{"channels":3}'

    OPTIMIZER:
      OPTIMIZER_NAME: "Adam"
      OPTIMIZER_KWARGS: '{"betas":[0.5, 0.999]}'
      BASE_LR: 0.0001
      LR_BIAS_FACTOR: 1.0
      WEIGHT_DECAY: 0.0005
      WEIGHT_BIAS_FACTOR: 0.0005
      FP16: True

    SCHEDULER:
      LR_SCHEDULER: 'StepLR'
      LR_KWARGS: '{"step_size":30, "gamma":0.25}'

    LOGGING:
      STEP_VERBOSE: 100 # Batch"""

    return string

if __name__ == "__main__":
    ### modify this so that we implement with bdd instead

    config = '/nethome/jbang36/j_amoeba/benchmark/odin/bdd.yml'
    mode = 'train'
    weights = '/nethome/jbang36/j_amoeba/benchmark/odin/data/models'

    main(config, mode, weights) ###config, mode, weights


