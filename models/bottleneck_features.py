"""
This module contains the functions to extract the bottleneck features from state-of-the-art networks:
    * VGG16
    * VGG19
    * Resnet50
    * InceptionV3
    * Xception
"""
import os
import pickle
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess

from dog_breed.common import tools
from dog_breed.preprocessing import preprocess


args_NN = {'weights': 'imagenet',
           'include_top': False,
           }


NETS = ['vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']  # pre-trained networks available
BOTTLENECK_SHAPES = {'vgg16': (7, 7, 512),
                     'vgg19': (7, 7, 512),
                     'resnet50': (7, 7, 2048),
                     'inceptionv3': (5, 5, 2048),
                     'xception': (7, 7, 2048),
                     }  # shape of the generic bottleneck feature for each pre-trained network


def extract_VGG16(tensor) -> np.ndarray: return VGG16(**args_NN).predict(vgg16_preprocess(tensor))
def extract_VGG19(tensor) -> np.ndarray: return VGG19(**args_NN).predict(vgg19_preprocess(tensor))
def extract_Resnet50(tensor) -> np.ndarray: return ResNet50(**args_NN).predict(resnet50_preprocess(tensor))
def extract_Xception(tensor) -> np.ndarray: return Xception(**args_NN).predict(xception_preprocess(tensor))
def extract_InceptionV3(tensor) -> np.ndarray: return InceptionV3(**args_NN).predict(inception_preprocess(tensor))


def extract_bottleneck_features(network: str, tensor) -> np.ndarray:
    """ Computes the bottleneck features of a tensors with the given network.
    This function is useful for transfer learning purposes.
    Args:
        network: state-of-the-art network. Can be
            * VGG16
            * VGG19
            * Resnet50
            * InceptionV3
            * Xception
        tensor:
    Returns:
        the features computes with the selected state-of-the-art network before its output layer
    See Also:
        extract_bottleneck_features_list()
    """
    lower_net = network.lower()
    if lower_net == 'vgg19':
        return extract_VGG19(tensor)
    if lower_net == 'vgg16':
        return extract_VGG16(tensor)
    if lower_net == 'resnet50':
        return extract_Resnet50(tensor)
    if lower_net == 'inceptionv3':
        return extract_InceptionV3(tensor)
    if lower_net == 'xception':
        return extract_Xception(tensor)


def extract_bottleneck_features_list(network: str, tensor_list: list) -> np.ndarray:
    """ Computes the bottleneck features of a list of tensors with the given network.
    Useful for transfer learning.
    Since this function loads the pre-built state-of-the-art network just once, is much better to use this function
    instead of mapping extract_bottleneck_features to a list of tensors.
    Args:
        network: state-of-the-art network. Can be
            * VGG16
            * VGG19
            * Resnet50
            * InceptionV3
            * Xception
        tensor_list:
    Returns:

    See Also:
        extract_bottleneck_features()
    """
    lower_net = network.lower()
    if lower_net == 'vgg19':
        preproc = vgg19_preprocess
        net = VGG19(**args_NN)
    elif lower_net == 'vgg16':
        preproc = vgg16_preprocess
        net = VGG16(**args_NN)
    elif lower_net == 'resnet50':
        preproc = resnet50_preprocess
        net = ResNet50(**args_NN)
    elif lower_net == 'inceptionv3':
        preproc = inception_preprocess
        net = InceptionV3(**args_NN)
    elif lower_net == 'xception':
        preproc = xception_preprocess
        net = Xception(**args_NN)
    return np.vstack(list(map(lambda x: net.predict(preproc(x)), tools.progr(tensor_list))))


def path_to_bottleneck(img_path: str, pre_trained_network):
    """ Returns the bottleneck features directly from the image path """
    tensor = preprocess.path_to_tensor(img_path)
    return extract_bottleneck_features(pre_trained_network, tensor)


def load_bottleneck(bottle_file, dataset='train'):
    """ Loads the correct bottleneck features for the .pkl file
    Args:
        bottle_file: .pkl file containing the bottleneck features
        dataset: can be ['train', 'valid', 'test']
    Returns:
    """
    if not os.path.exists(bottle_file): raise FileNotFoundError
    if dataset == 'test':
        return pickle.load(open(bottle_file, 'rb'))[2]
    elif dataset == 'train':
        return pickle.load(open(bottle_file, 'rb'))[0]
    elif dataset == 'valid':
        return pickle.load(open(bottle_file, 'rb'))[1]
