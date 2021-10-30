"""
This file implements the model pipelines needed to create the benchmark for
CycleGAN + SSD

The process we need to do is as follows:
1. Train the CycleGAN (assume this is done through train.py)
2. Use the trained CycleGAN to transfer all DAY images to NIGHT
3. Train SSD on transferred NIGHt images with labels
4. Evaluate the real NIGHT images against GT labels

"""
import os
import argparse
import json
import shutil
from tqdm import tqdm


#SCENARIO_NAME = 'day_night_scenario'
#SOURCE = 'day'
#TARGET = 'night'

SCENARIO_NAME = 'res_city_scenario'
SOURCE = 'res'
TARGET = 'city'


def copy_images(filenames, test_folder_name):
    """
    Conditions: set the filenames exactly the same, this is necessary for loading the annotations correctly
    """
    ### we need to create a temporary directory with the filenames
    dst = f'/data/bdd/cyclegan/{SCENARIO_NAME}/{test_folder_name}'
    if len(filenames) == 0:
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)

    os.makedirs(dst, exist_ok=True)

    for filename in tqdm(filenames):
        name = os.path.basename(filename)
        final_dst = os.path.join(dst, name)
        shutil.copy(filename, final_dst)

    return


def transfer_images(args):
    print(f"transferring images to testA, testB folders")
    files = [SOURCE, TARGET]
    tests = ['testA', 'testB']
    for i, file in enumerate(files):
        train_file = os.path.join(f'/data/bdd/ssd_sgr/{file}/TRAIN_images.json')
        with open(train_file, 'r') as j:
            image_files = json.load(j)

            copy_images(image_files, tests[i])




def prepare_bdd(args):
    transfer_images(args)
    image_output_folder = args.cyclegan_output
    base = '/data/bdd/ssd_sgr/'

    ### do TRAIN_images.json
    directory = os.path.join(f'/data/bdd/ssd_sgr/{SOURCE}', 'TRAIN_images.json')
    new_image_files = []
    with open(directory, 'r') as j:
        image_files = json.load(j)

    ## change the beginning part
    ### because transfer has taken place, TRAIN_images should be in args.cyclegan_output -- ~~~~/converted/
    ### we save this TRAIN_iamges.json to the ~~~~~
    for image_file in image_files:

        filename = os.path.basename(image_file)
        new_image_file = os.path.join(image_output_folder, filename)
        new_image_files.append(new_image_file)
    ### we need to save the file

    output_folder = os.path.dirname(image_output_folder)
    destination = os.path.join(output_folder, 'TRAIN_images.json')
    with open(destination, 'w') as j:
        json.dump(new_image_files, j)

    ### do TRAIN_objects.json
    ### we just need to copy this
    output_folder = os.path.dirname(image_output_folder)
    directory = os.path.join(f'{base}/{SOURCE}', 'TRAIN_objects.json')
    destination = os.path.join(output_folder, 'TRAIN_objects.json')
    shutil.copy(directory, destination)

    ### do VAL_images.json
    directory = os.path.join(f'{base}/{TARGET}', 'VAL_images.json')
    destination = os.path.join(output_folder, 'VAL_images.json')
    shutil.copy(directory, destination)

    ### do VAL_objects.json
    directory = os.path.join(f"{base}/{TARGET}", 'VAL_objects.json')
    destination = os.path.join(output_folder, 'VAL_objects.json')
    shutil.copy(directory, destination)












############################################################
############################################################
############################################################
############### One time run ##############################
############################################################


def one_time_run():
    print(f'converting _images.json files from ada-03 format to ada-01 format')
    import json
    ## we need to convert the filenames in day / night folder to fit ada-01 instead of ada-03
    files = ['TRAIN', 'VAL']
    partitions = [SOURCE, TARGET]
    base = '/data/bdd/ssd_sgr/'
    image_file_location = '/data/bdd/images/100k/'
    for partition in partitions:
        for file in tqdm(files):
            directory = os.path.join(f'/data/bdd/ssd_sgr/{partition}', f'{file}_images.json')
            new_image_files = []
            with open(directory, 'r') as j:
                image_files = json.load(j)


            ## change the beginning part
            for image_file in image_files:
                filename = os.path.basename(image_file)
                file_lower = file.lower()
                new_image_file = os.path.join(image_file_location, file_lower, filename)
                new_image_files.append(new_image_file)
            ### we need to save the file
            with open(directory, 'w') as j:
                json.dump(new_image_files, j)


def bug_fix():
    ### currently, ssd_sgr/day/TRAIN_images.json is buggy
    ### the filenames refer to the wrong thing....

    directory = os.path.join(f'/data/bdd/ssd_sgr/{SOURCE}', 'TRAIN_images.json')
    image_output_folder = '/data/bdd/images/100k/train'
    new_image_files = []
    with open(directory, 'r') as j:
        image_files = json.load(j)

    ## change the beginning part
    for image_file in image_files:
        filename = os.path.basename(image_file)
        new_image_file = os.path.join(image_output_folder, filename)
        new_image_files.append(new_image_file)
    ### we need to save the file


    with open(directory, 'w') as j:
        json.dump(new_image_files, j)


def reshape_converted(old_directory, new_directory):
    """
    After the converted images have been generated, we need to reshape the images
    This is because the output of cyclegan gives us 416x416 image whereas
    the images we work with are 720x1280.
    """
    from PIL import Image
    ### first load all the images in the old directory
    files = os.listdir(old_directory)
    files = sorted(files)
    new_filenames = []

    file_names = []
    for file in files:
        file_names.append(os.path.join(old_directory, file))
        new_filenames.append(os.path.join(new_directory, file))

    images = []
    n_images = len(file_names)


    for i in tqdm(range(n_images)):
        file_name = file_names[i]
        img = Image.open(file_name)
        img = img.resize((1280, 720)) ### we need to flip this because for PIL it's width, length

        img.save(new_filenames[i], "JPEG")




    return images






if __name__ == "__main__":
    one_time_run()
    #bug_fix()


    parser = argparse.ArgumentParser(description='Setting up directories for conversion...')
    parser.add_argument('--dataset', type = str, help='which dataset to work on')
    parser.add_argument('--cyclegan-output', type=str, help='directory to which the cyclegan outputs its images')

    args = parser.parse_args()

    if args.dataset == 'bdd':
        prepare_bdd(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not yet implemented")



