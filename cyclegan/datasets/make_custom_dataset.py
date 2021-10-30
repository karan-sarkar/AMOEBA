"""
In this file, we develop scripts to create custom datasets for cycleGAN training
"""
import os
import numpy as np
import cv2


def convertCIFAR10(original_data_path, categories_of_interest):
    """
    original_data_path: path where CIFAR10 is located
    categories_of_interest: categories that we are interested in extracting (any number suffices)

    cifar categories: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    These are some metadata that could be useful
    There are 5 batches for training for a total of 50000 images
    within a batch, these are the keys available: dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    b'batch_label' just gives info of b'training batch 1 of 5'
    b'labels' is a number array where labels are converted to numbers
    b'data' is a numpy array of shape (10000, 3072) where 3072 is 32x32x3 (height, width, channel)
    b'filenames' is actual filenames of the compressed images
    """

    ## make sure the directory exists and upload the corresponding paths
    if not os.path.exists(original_data_path):
        raise IOError("Path does not exists")

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    labelname_path = os.path.join(original_data_path,
                                  'batches.meta')  ## the labels will be in bytes we can convert via type(b'rabbit'.decode("utf-8"))
    labelnames = unpickle(labelname_path)
    labelnames = labelnames[b'label_names']
    labelnames = [x.decode('utf-8') for x in labelnames]
    ##convert all bytes to string
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    #test_batches = ['test_batch'] -- we actually don't need this because our make_dataset function separates train and test

    ## load all train data
    X_train = []
    y_train = []
    for train_batch in train_batches:
        full_path = os.path.join(original_data_path, train_batch)
        data_batch = unpickle(full_path)
        image_data = data_batch[b'data']
        X_train.append(image_data.reshape((-1, 32, 32, 3)))
        y_train.append(data_batch[b'labels'])

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    print(X_train.shape)
    print(y_train.shape)
    assert (X_train.shape == (50000, 32, 32, 3))
    assert (y_train.shape == (50000,))

    ### next, we need to start filtering with the categories of interest
    ### we need the indices of the categories of interest
    indices_of_interest = []
    print(labelnames)
    for category in categories_of_interest:
        indices_of_interest.append(labelnames.index(category))

    X_train_filtered = {}

    for i, ii in enumerate(indices_of_interest):
        indices = y_train == ii
        X_train_filtered[categories_of_interest[i]] = X_train[indices]

    ## we return a dict where it is like {'dog':np.ndarray(n,32,32,3), 'cat':np.ndarray(m,32,32,3)....}
    return X_train_filtered


def make_dataset(imagesA, imagesB, save_path):
    """
    imagesA: first set of data
    imagesB: second set of data
    save_path: the top most directory to save the files in
    """
    os.makedirs(save_path, exist_ok=True)
    ## we need to make testA, testB, trainA, trainB folders
    os.makedirs(os.path.join(save_path, 'testA'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'testB'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'trainA'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'trainB'), exist_ok=True)

    ## we need to save the images into corresponding paths... let's do a 4 / 1 split -- pretty much industry standard
    ### we don't need the numbers of imagesA, and number of images B to be the same
    divisionA = len(imagesA) // 4
    divisionB = len(imagesB) // 4
    trainA_indices = np.arange(len(imagesA))
    trainB_indices = np.arange(len(imagesB))
    np.random.shuffle(trainA_indices)
    np.random.shuffle(trainB_indices)

    testA_indices = trainA_indices[:divisionA]
    trainA_indices = trainA_indices[divisionA:]
    testB_indices = trainB_indices[:divisionB]
    trainB_indices = trainB_indices[divisionB:]

    counter = 0
    name2indices = {'trainA': trainA_indices, 'testA': testA_indices, 'trainB': trainB_indices, 'testB': testB_indices}
    for name, indices in name2indices.items():
        for ii in indices:
            filename = os.path.join(save_path, name, 'image_{}.jpg'.format(counter))
            cv2.imwrite(filename, imagesA[ii])
            counter += 1

    print(f"finished writing all images to {save_path}")
    return



def get_images(directory, count):
    day_file = open(directory, 'r')
    day_np = json.load(day_file)

    day_images = []
    #### let's just do 1000 images
    for i in range(count):
        image = Image.open(day_np[i], mode='r')
        image = image.convert('RGB')
        ### we don't need to do this, we just save the images to the save directory
        image = np.array(image)
        day_images.append(image)

    day_images = np.stack(day_images, axis=0)

    assert (day_images.ndim == 4)
    assert (day_images.shape[0] == count)


    return day_images



if __name__ == "__main__":
    #### we load the day images
    import json
    from PIL import Image
    SOURCE = 'res'
    TARGET = 'city'

    day_filename = f'/srv/data/jbang36/bdd/ssd_sgr/{SOURCE}/TRAIN_images.json'
    save_path = '/srv/data/jbang36/cyclegan/data'
    count = 1000

    day_images = get_images(day_filename, count)
    print(f"loaded {len(day_images)} day images")

    night_filename = f'/srv/data/jbang36/bdd/ssd_sgr/{TARGET}/TRAIN_images.json'

    night_images = get_images(night_filename, count)
    print(f"loaded {len(night_images)} night images")

    make_dataset(day_images, night_images, save_path)

    print(f"saved custom dataset to {save_path}")
