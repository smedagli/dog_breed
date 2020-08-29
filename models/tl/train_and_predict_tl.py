"""
@ TODO:
    - save history from train_network_tl()
    - add header description
"""
import io
import os
import requests
import numpy as np


from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from dog_breed.models.bottleneck_features import extract_bottleneck_features
from dog_breed.common import paths
from dog_breed.common import models_param
from dog_breed.data.datasets import get_dog_names


def load_bottleneck_features(network: str) -> dict:
    """ Loads the precomputed bottleneck features for a series of architectures
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


def train_network_tl(network, bottleneck_network: str, training_data, training_target,
                     validation_data, validation_target, overwrite=0, prefix='mynet',
                     data_augmentation=True, epochs=15, batch_size=20):
    """ Trains the given network and saves the best weights (and history) - Transfer Learning
    Args:
        network:
        bottleneck_network:
        training_data:
        training_target:
        validation_data:
        validation_target:
        overwrite: if 1, overwrites the saved weights
        prefix: prefix of the filename to save weights
        data_augmentation: if 1 uses data augmentation for training data
        epochs: number of epochs to train the model
        batch_size: batch size to train the model
    Returns:
    @ TODO:
        - fix data augmentation: augmentation must be done before computing the bottleneck features?
    """

    args_model_training = {'epochs': epochs,
                           'verbose': 1,
                           }

    model_weight_file = paths.get_weights_filename(bottleneck_network, prefix, epochs, data_augmentation)
    # model_hist_file = paths.get_hist_filename(bottleneck_network, prefix, epochs, data_augmentation)

    if os.path.exists(model_weight_file) and not overwrite:
        print("Loading existing weights")
        # hist = pickle.load(open(model_hist_file, 'rb'))
    else:
        checkpointer = ModelCheckpoint(filepath=model_weight_file,
                                       verbose=1, save_best_only=True)
        if not data_augmentation:
            hist = network.fit(training_data, training_target,
                               validation_data=(validation_data, validation_target),
                               callbacks=[checkpointer],
                               **args_model_training, batch_size=batch_size,
                               )
        else:
            datagen = ImageDataGenerator(**models_param.args_data_augmentation)
            datagen.fit(training_data)
            hist = network.fit(datagen.flow(training_data, training_target, batch_size=batch_size),
                               steps_per_epoch=training_data.shape[0] / batch_size,
                               validation_data=(validation_data, validation_target),
                               callbacks=[checkpointer],
                               **args_model_training,
                               )
        # pickle.dump(hist.history, open(model_hist_file, 'wb'))


def predict(network, image_path: str) -> str:
    """ Predicts the dog breed as string (and not as categorical).
    The output of network.predict is a category of dog breed.
    This function returns the name of the breed given the category.
    Args:
        network:
        image_path: path to the image
    Returns:
    """
    dog_names = get_dog_names()
    bottleneck_features = extract_bottleneck_features(network, image_path)
    pred = network.predict(bottleneck_features)
    return dog_names[int(np.argmax(pred))]
