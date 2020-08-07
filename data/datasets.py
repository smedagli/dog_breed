import numpy as np
import os

from sklearn.datasets import load_files
from keras.utils import np_utils


def load_dataset(path: str):
    """
    Args:
        path:
    Returns:
    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def load_training(): return load_dataset('data/dogImages/train')
def load_test(): return load_dataset('data/dogImages/test')
def load_validation(): return load_dataset('data/dogImages/valid')


def get_dog_names() -> list:
    """ Return the list of dog breeds """
    return list(map(lambda x: x.split('.')[-1], os.listdir('data/dogImages/train/')))
