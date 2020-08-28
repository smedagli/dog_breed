"""
Here are implemented functions
    * is_dog(): returns True if a dog is in the image
    * is_human(): returns True if a human face is in the image
"""

import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import cv2

from dog_breed.preprocessing import preprocess

ResNet50_model = ResNet50(weights='imagenet')


def detect_human(image_path: str) -> list:
    """ Implement the "face recognition"
    Args:
        image_path: path of the image file
    Returns:
        each element of the list is made of (x, y, w, h).
        x, y are the bottom left coordinate of a rectangle including the a face detected.
        w, h are the width and height of the rectangle
    """
    face_cascade = cv2.CascadeClassifier('detectors/saved_detectors/haarcascade_frontalface_alt.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return faces


def is_human(image_path: str) -> bool:
    """
    Args:
        image_path:
    Returns:
    Examples:
        >>> is_human('samples/sample_human_2.png')
        True
        >>> is_human('samples/sample_dog.jpg')
        False
    """
    return len(detect_human(image_path)) > 0


def is_dog(image_path: str) -> bool:
    """
    Args:
        image_path:
    Returns:
        Examples:
        >>> is_dog('samples/sample_human_2.png')
        False
        >>> is_dog('samples/sample_dog.jpg')
        True
    """
    img = preprocess_input(preprocess.path_to_tensor(image_path))
    pred = np.argmax(ResNet50_model.predict(img))
    return np.logical_and(pred <= 268, pred >= 151)
