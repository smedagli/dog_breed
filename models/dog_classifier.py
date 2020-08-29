"""
@ TODO:
    - add header
"""
import os
import keras
import numpy as np

from dog_breed.common import paths
from dog_breed.data import datasets
from dog_breed.detectors import detectors
from dog_breed.models import build_network as bn
from dog_breed.models import bottleneck_features


def load_best_model() -> (keras.Sequential, str):
    """  Loads the best network for TL
    @ TODO:
        - write docstring
    Returns:
        model:
        pretrained_network: name of the pre-trained network used to compute bottleneck features of the model
    """
    best_net = 'data\\saved_models\\TL\\tl_20_A_weight.best.xception.hdf5'
    pretrained_network = best_net.split('.')[-2]

    model = bn.build_transfer_learning_netwok(input_shape=bottleneck_features.BOTTLENECK_SHAPES[pretrained_network],
                                              n_of_classes=datasets.get_number_of_classes())
    model.load_weights(best_net)
    return model, pretrained_network


def get_probl(img_path: str):
    """ Returns the probability to belong to each possible category
    Args:
        img_path: img_path: path to the image
    Returns:
    """
    model, pretrained_network = load_best_model()
    feature = bottleneck_features.path_to_bottleneck(img_path, pretrained_network)
    probl = model.predict(feature)
    return probl


def get_breed(img_path: str) -> str:
    """
    @ TODO:
        - write docstring
    Args:
        img_path: img_path: path to the image
    Returns:
    """
    probl = get_probl(img_path)
    return datasets.get_dog_names()[int(np.argmax(probl))]


def save_best_model() -> None:
    """    Saves the best model to 'data/saved_models/transfer_learning_model.h5'    """
    model, _ = load_best_model()
    model.save(os.path.join(paths.Folders().models, 'transfer_learning_model.h5'))


def get_output_message(image_file: str) -> str:
    """ Returns a different message if there is a human or a dog in the image.
    Args:
        image_file: img_path: path to the image
    Returns:
    """
    if detectors.is_dog(image_file):
        return 'The breed of the dog in the picture is:'
    elif detectors.is_human(image_file):
        return 'This person looks like a:'
    else:
        return "Not a dog nor a person was found in the image"
