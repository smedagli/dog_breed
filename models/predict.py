import requests
import io
import os
import numpy as np

from keras.callbacks import ModelCheckpoint

from dog_breed.models.bottleneck_features import extract_bottleneck_features
from dog_breed.data.datasets import get_dog_names


def load_bottleneck_features(network: str) -> dict:
    """ Loads the precomputed bottleneck features for a series of architectures
    @ TODO:
        save the bottleneck features in the 'data' folder
    Args:
        network: kind of network. Can be:
            * VGG19
            * Resnet50
            * InceptionV3
            * Xception
    Returns:
        the features before the output layer for state-of-the-art CNN (as dictionary)
    """
    url = f'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/Dog{network}Data.npz'
    response = requests.get(url)
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    return data


def load_network_weights(network, weight_file: str):
    network.load_weights(weight_file)


def train_network(network, bottleneck_network, training_data, training_target,
                  validation_data, validation_target, overwrite=0, prefix='mynet'):

    """ Trains the given network and saves the best weights
    Trains the network using the bottleneck features
    Args:
        network:
        bottleneck_network:
        training_data:
        training_target:
        validation_data:
        validation_target:
        overwrite: if 1, overwrites the saved weights
        prefix: prefix of the filename to save weights
    Returns:
    """
    model_weight_file = f'saved_models/{prefix}_weight.best.{bottleneck_network}.hdf5'
    if os.path.exists(model_weight_file) and not overwrite:
        print("Loading existing weights")
    else:
        checkpointer = ModelCheckpoint(filepath=model_weight_file,
                                       verbose=1, save_best_only=True)

        network.fit(training_data, training_target,
                    validation_data=(validation_data, validation_target),
                    epochs=10, batch_size=20, callbacks=[checkpointer], verbose=1)
    load_network_weights(network, model_weight_file)


def predict(network, image_path: str):
    """
    TODO:
        implement dictionary 'dog_names'
    Args:
        network:
        image_path:
    Returns:
    """
    dog_names = get_dog_names()
    bottleneck_features = extract_bottleneck_features(network, image_path)
    pred = network.predict(bottleneck_features)
    return dog_names[np.argmax(pred)]
