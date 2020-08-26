"""
This module is used to define the network.
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D


def build_network(input_shape: tuple, n_of_classes: int) -> Sequential:
    """ Returns a neural network to

    Args:
        input_shape:
        n_of_classes:

    Returns:

    """
    net = Sequential()
    # net.add(Flatten(input_shape=input_shape))
    net.add(GlobalAveragePooling2D(input_shape=input_shape))
    net.add(Dense(512, activation='relu'))
    # net.add(Dropout(0.2))
    net.add(Dense(n_of_classes, activation='softmax'))

    net.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='rmsprop')
    return net
