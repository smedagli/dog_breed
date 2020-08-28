"""
This module is used to define the network.
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPool2D


def build_transfer_learning_netwok(input_shape: tuple, n_of_classes: int) -> Sequential:
    """ Returns a neural network to use for transfer learning
    Returns a network just made of:
        * GlobalAveragePooling2D layer
        * Dense layer (n_of_classes)
    Args:
        input_shape:
        n_of_classes: number of classes (will also be the number of nodes at the output layer)
    Returns:

    """
    net = Sequential()
    net.add(GlobalAveragePooling2D(input_shape=input_shape))
    net.add(Dense(n_of_classes, activation='softmax'))
    net.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='rmsprop')
    return net


def build_network_from_scratch(input_shape: tuple, n_of_classes: int) -> Sequential:
    """ Returns a neural network for image classification (built from the sketch)
    To change the architecture of the network is necessary to change this function.
    Args:
        input_shape:
        n_of_classes:
    Returns:
    """
    base_conv_filter_number = 128  # the first Conv2D layer will have this number of filters, then 2 * prev_n_of_filters
    args_conv2d = {'strides': 3, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}
    args_maxpool = {'pool_size': 3}

    net = Sequential()
    net.add(Conv2D(**args_conv2d, input_shape=input_shape, name='Input', filters=base_conv_filter_number))
    net.add(MaxPool2D(**args_maxpool))
    net.add(Conv2D(**args_conv2d, filters=2 * base_conv_filter_number))
    net.add(MaxPool2D(**args_maxpool))
    # net.add(Conv2D(**args_conv2d, filters=4 * base_conv_filter_number))
    # net.add(MaxPool2D(**args_maxpool))
    net.add(GlobalAveragePooling2D())
    net.add(Dense(256))
    net.add(Dropout(.2))
    net.add(Dense(n_of_classes))
    net.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='rmsprop')
    return net
