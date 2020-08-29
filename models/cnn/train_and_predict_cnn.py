"""
@ TODO:
    - add header description
"""
import os
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from dog_breed.common import tools as ct
from dog_breed.common import metrics
from dog_breed.common import paths
from dog_breed.common import models_param
from dog_breed.data import datasets
from dog_breed.preprocessing import preprocess
from dog_breed.models import build_network as bn


def load_training_datasets():
    """ Loads training and validation datasets (as tensor and one-hot encoded labels)
    Returns:
        (tensors_train, y_train), (tensors_valid, y_valid)
    """
    train_files, y_train = datasets.load_dataset(dataset='train')
    valid_files, y_valid = datasets.load_dataset(dataset='valid')

    tensors_train = np.vstack(list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255,
                                       ct.progr(train_files))))
    tensors_valid = np.vstack(list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255,
                                       ct.progr(valid_files))))

    return (tensors_train, y_train), (tensors_valid, y_valid)


def train_cnn_reduced(epochs=15, batch_size=20, data_augmentation=True, overwrite=0) -> None:
    """
    @ TODO:
        - add docstring
    Args:
        epochs:
        batch_size:
        data_augmentation:
        overwrite:

    Returns:

    """
    n_of_classes = datasets.get_number_of_classes()
    (tensors_train, y_train), (tensors_valid, y_valid) = load_training_datasets()

    model = bn.build_network_from_scratch(input_shape=tensors_train[0].shape, n_of_classes=n_of_classes)
    train_cnn(model, training_data=tensors_train, training_target=y_train,
              validation_data=tensors_valid, validation_target=y_valid,
              epochs=epochs, data_augmentation=data_augmentation, prefix='CNN',
              batch_size=batch_size, overwrite=overwrite,
              )


def train_cnn(cnn, training_data, training_target,
              validation_data, validation_target, overwrite=0, prefix='CNN',
              data_augmentation=True, epochs=15, batch_size=20) -> None:
    """
    @ TODO:
        - add docstring
    Args:
        cnn:
        training_data:
        training_target:
        validation_data:
        validation_target:
        overwrite:
        prefix:
        data_augmentation:
        epochs:
        batch_size:

    Returns:

    """

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
            datagen = ImageDataGenerator(**models_param.args_data_augmentation)
            datagen.fit(training_data)
            cnn.fit(datagen.flow(training_data, training_target, batch_size=batch_size),
                    steps_per_epoch=training_data.shape[0] / batch_size,
                    validation_data=(validation_data, validation_target),
                    callbacks=[checkpointer],
                    **args_model_training,
                    )


def eval_cnn(epochs: int, data_augmentation: bool, dataset='test', overwrite=0) -> float:
    """
    @ TODO:
        - add docstring
    Args:
        epochs:
        data_augmentation:
        dataset:
        overwrite:

    Returns:

    """
    model_file = paths.get_weights_filename('', "CNN", epochs, data_augmentation,
                                            transfer_learning=False).replace('..', '.')
    if os.path.exists(model_file) and not overwrite:
        print(f"Model already trained.\nWeights' file at\t{model_file}")
        files, labels = datasets.load_dataset(dataset=dataset)
        tensors = np.vstack(list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255,
                                     ct.progr(files))))
        model = bn.build_network_from_scratch(input_shape=tensors[0].shape, n_of_classes=133)
        model.load_weights(model_file)
        pred = model.predict(tensors)
        acc = metrics.get_accuracy(np.array([np.argmax(x) for x in pred]), np.array([np.argmax(y) for y in labels]))
        return acc
    else:
        train_cnn_reduced(epochs=epochs, data_augmentation=data_augmentation)
        return eval_cnn(epochs, data_augmentation, dataset)
