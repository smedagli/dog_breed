"""
This module contains the tool to generate model using transfer learning

train_transfer_learning_net():
    * loads the dataset (training and validation)
    * bottle neck features are computed from pre-trained networks
        - VGG16
        - VGG19
        - Resnet50
        - InceptionV3
        - Xception
    * trains a Sequential network made of 2 layers (see dog_breed.models.build_network.py):
        - GlobalAveragePooling2D layer
        - Dense layer (n_of_classes)
    * saves the weight in 'data/saved_models/' with the filename:
        if data augmentation is used ->     <prefix>_<epochs>_A_weight.best.<pre_trained_net>.hdf5
        else ->                             <prefix>_<epochs>_weight.best.<pre_trained_net>.hdf5

eval_performance():
    if the weights' file already exists, creates the basic network and loads the weights.
    Then computes the performance (by default, the accuracy on the test set).
    Otherwise calls train_transfer_learning_net() and then evaluates the performance.
"""
import os
import pickle
import numpy as np
from PIL import ImageFile

from dog_breed.data import datasets
from dog_breed.preprocessing import preprocess
from dog_breed.common import tools as ct
from dog_breed.common import paths
from dog_breed.common import metrics
from dog_breed.models import bottleneck_features as bf
from dog_breed.models import build_network as bn
from dog_breed.models import train_and_predict

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transfer_learning_net(pretrained_network: str, epochs: int, data_augmentation: bool,
                                prefix: str, overwrite=0) -> None:
    """ Computes bottleneck features for training and validation data and trains the Sequential network.
    Weights will be saved in 'data/saved_models/' with name
    if data augmentation is used ->     <prefix>_<epochs>_A_weight.best.<pre_trained_net>.hdf5
                            else ->     <prefix>_<epochs>_weight.best.<pre_trained_net>.hdf5
    Args:
        pretrained_network: pre trained network for bottleneck features.
                            Can be ['vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']
        epochs: number of epochs to train the model
        data_augmentation: use or not data augmentation
        prefix: prefix to save the
        overwrite: if True, trains and saves the weight file again, otherwise does nothing
    Returns:
        None, but weights will be saved is the preselected location
    """
    model_file = paths.get_weights_filename(pretrained_network, prefix, epochs, data_augmentation)
    if os.path.exists(model_file) and not overwrite:
        print(f"Model already trained.\nWeights' file at\t{model_file}")
    else:

        args_train = {'data_augmentation': data_augmentation,
                      'epochs': epochs,
                      'prefix': prefix,
                      'overwrite': overwrite,
                      'bottleneck_network': pretrained_network,
                      }
        # load train/validation file names and labels
        train_files, y_train = datasets.load_training()
        valid_files, y_valid = datasets.load_validation()
        # compute tensors
        tensors_train = list(map(preprocess.path_to_tensor, ct.progr(train_files)))
        tensors_valid = list(map(preprocess.path_to_tensor, ct.progr(valid_files)))
        # compute bottleneck features

        bottle_file = os.path.join(paths.Folders().bottleneck_features, f'bottleneck_{pretrained_network.lower()}.pkl')
        if os.path.exists(bottle_file):
            bottleneck_train, bottleneck_valid, _ = pickle.load(open(bottle_file, 'rb'))
        else:
            bottleneck_train = bf.extract_bottleneck_features_list(pretrained_network, tensors_train)
            bottleneck_valid = bf.extract_bottleneck_features_list(pretrained_network, tensors_valid)

        n_of_classes = len(datasets.get_dog_names())
        model = bn.build_transfer_learning_netwok(input_shape=bottleneck_train[0].shape, n_of_classes=n_of_classes)

        # _ = trainAndPredict.train_network_tl(network=model,
        #                                   training_data=bottleneck_train, training_target=y_train,
        #                                   validation_data=bottleneck_valid, validation_target=y_valid,
        #                                   **args_train,
        #                                   )
        train_and_predict.train_network_tl(network=model,
                                           training_data=bottleneck_train, training_target=y_train,
                                           validation_data=bottleneck_valid, validation_target=y_valid,
                                           **args_train,
                                           )

        model_file = paths.get_weights_filename(pretrained_network, prefix, epochs, data_augmentation)
        print(f"Weights saved at\t{model_file}")


def eval_performance(pretrained_network: str, epochs: int, data_augmentation: bool,
                     prefix: str, overwrite=False, dataset='test') -> float:
    """ Returns the accuracy of the model.
    If the model does not exist, trains a new model before computing the accuracy.
    Args:
        pretrained_network: pre trained network for bottleneck features.
                            Can be ['vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']
        epochs: number of epochs to train the model
        data_augmentation: use or not data augmentation
        prefix: prefix to save the
        overwrite: if True, trains and saves the weight file again, otherwise does nothing
        dataset: accuracy will be computed on training, validation or test set based on this input
                 Possible options in ['train', 'test', 'valid'].
    Returns:
        accuracy in [0, 1] interval
    """
    model_file = paths.get_weights_filename(pretrained_network, prefix, epochs, data_augmentation)
    if os.path.exists(model_file) and not overwrite:
        print(f"Model already trained.\nWeights' file at\t{model_file}")
        if dataset == 'test':
            files, labels = datasets.load_test()
        elif dataset == 'train':
            files, labels = datasets.load_training()
        else:
            files, labels = datasets.load_validation()

        tensors = list(map(preprocess.path_to_tensor, ct.progr(files)))

        bottle_file = os.path.join(paths.Folders().bottleneck_features, f'bottleneck_{pretrained_network.lower()}.pkl')
        if os.path.exists(bottle_file):
            bottleneck_train, bottleneck_valid, bottleneck_test = pickle.load(open(bottle_file, 'rb'))
            if dataset == 'test':
                bottleneck_features = bottleneck_test
            elif dataset == 'train':
                bottleneck_features = bottleneck_train
            else:
                bottleneck_features = bottleneck_valid
        else:
            bottleneck_features = bf.extract_bottleneck_features_list(pretrained_network, tensors)

        model = bn.build_transfer_learning_netwok(input_shape=bottleneck_features[0].shape, n_of_classes=133)
        train_and_predict.load_network_weights(model, model_file)
        pred = model.predict(bottleneck_features)
        acc = metrics.get_accuracy(np.array([np.argmax(x) for x in pred]),
                                   np.array([np.argmax(y) for y in labels]))
        return acc
    else:
        print("Model does not exist.\nTraining model")
        train_transfer_learning_net(pretrained_network, epochs, data_augmentation, prefix)
        return eval_performance(pretrained_network, epochs, data_augmentation, prefix, dataset=dataset)
