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


def move_images(filenames, test_folder_name):
    """
    Conditions: set the filenames exactly the same, this is necessary for loading the annotations correctly
    """
    ### we need to create a temporary directory with the filenames
    dst = f'/data/bdd/cyclegan/day_night_scenario/{test_folder_name}'
    if len(filenames) == 0:
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)

    os.makedirs(dst, exist_ok=True)

    for filename in filenames:
        name = os.path.basename(filename)
        final_dst = os.path.join(dst, name)
        shutil.copy(filename, final_dst)

    return

def prepare_bdd(args):
    ### this means we transfer the day images to night images
    ### hence we need the filenames for all the day images

    ### steps would be 1. load the json TRAIN images,
    ### we use move images to move them to a different location
    files = ['day', 'night']
    tests = ['testA', 'testB']
    for i, file in enumerate(files):
        train_file = os.path.join(f'/data/bdd/ssd_sgr/{file}/TRAIN_images.json')
        with open(train_file, 'r') as j:
            image_files = json.load(j)

            move_images(image_files, tests[i])


    ### when preparing the bdd, we need to create new
    # TRAIN_images.json / TRAIN_objects.json / VAL_images.json / VAL_objects.json

    #### TRAIN_images.json should be the converted files -- we need to know where the CYCLEGAN spits out the files
    #### TRAIN_objects.json should be a day annotations
    #### VAL_images.json should be night images
    #### VAL_objecst.json should be night annotations

    image_output_folder = args.cyclegan_output
    # /data/bdd/cyclegan/day_night_scenario

    ## we need to convert the filenames in day / night folder to fit ada-01 instead of ada-03
    base = '/data/bdd/ssd_sgr/'

    ### do TRAIN_images.json
    directory = os.path.join(f'/data/bdd/ssd_sgr/day', 'TRAIN_images.json')
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

    ### do TRAIN_objects.json
    ### we just need to copy this
    output_folder = os.path.dirname(image_output_folder)
    directory = os.path.join(f'{base}/day', 'TRAIN_objects.json')
    destination = os.path.join(output_folder, 'TRAIN_objects.json')
    shutil.copy(directory, destination)

    ### do VAL_images.json
    directory = os.path.join(f'{base}/night', 'VAL_images.json')
    destination = os.path.join(output_folder, 'VAL_images.json')
    shutil.copy(directory, destination)

    ### do VAL_objects.json
    directory = os.path.join(f"{base}/night", 'VAL_objects.json')
    destination = os.path.join(output_folder, 'VAL_objects.json')
    shutil.copy(directory, destination)












############################################################
############################################################
############################################################
############### One time run ##############################
############################################################


def one_time_run():
    import json
    ## we need to convert the filenames in day / night folder to fit ada-01 instead of ada-03
    files = ['TRAIN', 'VAL']
    partitions = ['day', 'night']
    base = '/data/bdd/ssd_sgr/'
    image_file_location = '/data/bdd/images/100k/'
    for partition in partitions:
        for file in files:
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



if __name__ == "__main__":
    #one_time_run()


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type = str, help='which dataset to work on')
    parser.add_argument('--cyclegan_output', type=str, help='directory to which the cyclegan outputs its images')

    args = parser.parse_args()

    if args.dataset == 'bdd':
        prepare_bdd(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not yet implemented")



