import numpy as np
import os

from sklearn.datasets import load_files
from keras.utils import np_utils

from dog_breed.common import paths

folders = paths.Folders()


def load_dataset(path: str):
    """
    Args:
        path:
    Returns:
    """
    print(f'Loading from file: {path}')
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def load_training(training_folder=folders.training_data): return load_dataset(training_folder)
def load_test(test_folder=folders.test_data): return load_dataset(test_folder)
def load_validation(valid_folder=folders.validation_data): return load_dataset(valid_folder)


def get_dog_names(training_folder=folders.training_data) -> list:
    """ Return the list of dog breeds (from training) """
    return list(map(lambda x: x.split('.')[-1], os.listdir(training_folder)))
