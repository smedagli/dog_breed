"""
This module contains the functions to extract the bottleneck features from state-of-the-art networks:
    * VGG16
    * VGG19
    * Resnet50
    * InceptionV3
    * Xception
"""
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input


args_NN = {'weights': 'imagenet',
           'include_top': False,
           }

def extract_VGG16(tensor): return VGG16(**args_NN).predict(preprocess_input(tensor))
def extract_VGG19(tensor): return VGG19(**args_NN).predict(preprocess_input(tensor))
def extract_Resnet50(tensor): return ResNet50(**args_NN).predict(preprocess_input(tensor))
def extract_Xception(tensor): return Xception(**args_NN).predict(preprocess_input(tensor))
def extract_InceptionV3(tensor): return InceptionV3(**args_NN).predict(preprocess_input(tensor))


def extract_bottleneck_features(network: str, tensor) -> np.ndarray:
    if network == 'VGG19':
        return extract_VGG19(tensor)
    elif network == 'VGG16':
        return extract_VGG16(tensor)
    elif network == 'Resnet50':
        return extract_Resnet50(tensor)
    elif network == 'InceptionV3':
        return extract_InceptionV3(tensor)
    elif network == 'Xception':
        return extract_Xception(tensor)


def extract_bottleneck_features_list(network: str, tensor_list: list) -> np.ndarray:
    if network == 'VGG19':
        net = VGG19(**args_NN)
    elif network == 'VGG16':
        net = VGG16(**args_NN)
    elif network == 'Resnet50':
        net = ResNet50(**args_NN)
    elif network == 'InceptionV3':
        net = InceptionV3(**args_NN)
    elif network == 'Xception':
        net = Xception(**args_NN)
    return np.vstack(list(map(lambda x: net.predict(preprocess_input(x)), tensor_list)))

