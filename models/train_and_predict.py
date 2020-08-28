"""
@ TODO:
    - save history from train_network_tl()
"""
import io
import os
import pickle
import requests
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from dog_breed.models.bottleneck_features import extract_bottleneck_features
from dog_breed.common import paths
from dog_breed.data.datasets import get_dog_names

args_data_augmentation = {'width_shift_range': 0.3,
                          'height_shift_range': 0.3,
                          'horizontal_flip': True,
                          'rescale': True,
                          'rotation_range': 0.5,
                          }


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


def load_network_weights(network, weight_file: str):
    network.load_weights(weight_file)


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
            datagen = ImageDataGenerator(**args_data_augmentation)
            datagen.fit(training_data)
            hist = network.fit(datagen.flow(training_data, training_target, batch_size=batch_size),
                               steps_per_epoch=training_data.shape[0] / batch_size,
                               validation_data=(validation_data, validation_target),
                               callbacks=[checkpointer],
                               **args_model_training,
                               )
        # pickle.dump(hist.history, open(model_hist_file, 'wb'))

    load_network_weights(network, model_weight_file)
    # return hist


def predict(network, image_path: str) -> str:
    """ Predicts the dog breed as string (and not as categorical).
    The output of network.predict is a category of dog breed.
    This function returns the name of the breed given the category.
    Args:
        network:
        image_path:
    Returns:
    """
    dog_names = get_dog_names()
    bottleneck_features = extract_bottleneck_features(network, image_path)
    pred = network.predict(bottleneck_features)
    return dog_names[int(np.argmax(pred))]


def train_cnn(cnn, training_data, training_target,
              validation_data, validation_target, overwrite=0, prefix='CNN',
              data_augmentation=True, epochs=15, batch_size=20):

    args_model_training = {'epochs': epochs,
                           'verbose': 1,
                           }

    model_weight_file = paths.get_weights_filename('', prefix, epochs, data_augmentation,
                                                   transfer_learning=False).replace('..', '.')
    # model_hist_file = paths.get_hist_filename('', prefix, epochs, data_augmentation).replace('..', '.')

    if os.path.exists(model_weight_file) and not overwrite:
        print("Loading existing weights")
        # hist = pickle.load(open(model_hist_file, 'rb'))
    else:
        checkpointer = ModelCheckpoint(filepath=model_weight_file,
                                       verbose=1, save_best_only=True)
        if not data_augmentation:
            # model_weight_file = f'saved_models/{prefix}_weight.best.{bottleneck_network}.hdf5'
            cnn.fit(training_data, training_target,
                    validation_data=(validation_data, validation_target),
                    callbacks=[checkpointer],
                    **args_model_training, batch_size=batch_size,
                    )
        else:
            datagen = ImageDataGenerator(**args_data_augmentation)
            datagen.fit(training_data)
            cnn.fit(datagen.flow(training_data, training_target, batch_size=batch_size),
                    steps_per_epoch=training_data.shape[0] / batch_size,
                    validation_data=(validation_data, validation_target),
                    callbacks=[checkpointer],
                    **args_model_training,
                    )

    load_network_weights(cnn, model_weight_file)
