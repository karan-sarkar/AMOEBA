"""
In this file, we take in a dataset and simulate it as a stream.
Basically, we sort the dataset by labels and send it out.

"""

import numpy as np


class StreamCreator:
    def __init__(self):
        self.sorted_labels = None
        self.sorted_indices = None
        self.reverse_mapping = None


        pass


    def generate_stream(self, images:np.ndarray, labels:list = None):
        ### if labels is given, we group images and labels
        organized_stream = []

        if labels is not None:

            ### we must make sure the length of images and labels are the same
            print(f'len of images: {len(images)}')
            print(f'len of labels: {len(labels)}')
            assert(len(images) == len(labels))
            sorted_labels = np.sort(labels)
            sorted_indices = np.argsort(labels)
            sorted_images = images[sorted_indices]
            self.sorted_labels = sorted_labels
            self.sorted_indices = sorted_indices
            self.reverse_mapping = np.argsort(self.sorted_indices)

            for i,label in enumerate(sorted_labels):
                image = sorted_images[i]
                organized_stream.append( (image, label) )

        else:
            organized_stream = images ### there is no ordering we can enforce onto the images
            
        return organized_stream

    def generate_stream_separate(self, images:np.ndarray, labels:list = None):
        if labels is not None:


            sorted_labels = np.sort(labels)
            sorted_indices = np.argsort(labels)
            sorted_images = images[sorted_indices]

            self.sorted_labels = sorted_labels
            self.sorted_indices = sorted_indices
            self.reverse_mapping = np.argsort(self.sorted_indices)

            return sorted_images, sorted_labels

        else: ## there is nothing to sort if we don't provide the labels
            return images


    def get_reverse_mapping(self):
        if self.reverse_mapping is not None:
            return self.reverse_mapping

        else:
            print(f'Must run generate_stream functions to get the reverse mapping')
            return None

