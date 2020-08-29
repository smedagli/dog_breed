"""
This module contains the functions to handle training, validation and test datasets
"""
import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_files
from keras.utils import np_utils

from dog_breed.common import paths


folders = paths.Folders()


def load_data(path: str):
    """ Loads file names and their labels (as categorical)
    Args:
        path:
    Returns:
        dog_files: list of filenames of dog images
        dog_targets: categorical (one hot encoded) labels for each file in the dog_files list
    """
    print(f'Loading from file: {path}')
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def _load_training(training_folder=folders.training_data): return load_data(training_folder)
def _load_test(test_folder=folders.test_data): return load_data(test_folder)
def _load_validation(valid_folder=folders.validation_data): return load_data(valid_folder)


def load_dataset(dataset='train'):
    """ Loads the specified dataset
    Args:
        dataset: select among ['test', 'train', 'valid']
    Returns:
        files: list of filenames of dog images
        labels: categorical (one hot encoded) labels for each file in the dog_files lis
    """
    if dataset == 'test':
        return _load_test()
    elif dataset == 'train':
        return _load_training()
    else:
        return _load_validation()


def get_dog_names(training_folder=folders.training_data) -> list:
    """ Return the list of dog breeds (from training) """
    if os.path.exists(training_folder):
        return list(map(lambda x: x.split('.')[-1], os.listdir(training_folder)))
    else:
        return pd.read_csv('data/dog_names.csv', header=None)[0].tolist()


def get_number_of_classes(training_folder=folders.training_data) -> int:
    return len(get_dog_names(training_folder))
