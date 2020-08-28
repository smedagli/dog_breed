"""
This module contains tools for preprocessing of the images.
Images need to be preprocessed to be transformed to a tensor in order to feed a keras CNN.
"""
import numpy as np
import tqdm

from keras.preprocessing import image


IMG_SIZE = 224  # images loaded must be resized to this dimension (squared)


def _load_image_size(image_path: str, height=IMG_SIZE, width=IMG_SIZE):
    """ Loads an image imposing the input resolution w x h
    @ TODO:
        check the order of h and w
    Args:
        image_path: path to the image
        height: height of the image to load
        width: width of the image to load
    Returns:
    Examples:
        >>> _load_image_size('samples/sample_dog.jpg')
        <PIL.Image.Image image mode=RGB size=224x224 at 0x1FABF911B70>
    """
    img = image.load_img(image_path, target_size=(width, height))
    return img


def load_image(image_path: str, height=IMG_SIZE, width=IMG_SIZE):
    """ Loads the image with a given resolution and normalizes the pixel values
    Args:
        image_path: path to the image
        height: height of the image to load
        width: width of the image to load
    Returns:
    Examples:
        >>> load_image('samples/sample_dog.jpg')[0, 0, 0]
        array([0.5372549 , 0.4745098 , 0.27450982], dtype=float32)
    """
    img = _load_image_size(image_path, height, width)
    x = image.img_to_array(img)
    return np.vstack(x, axis=0)


def path_to_tensor(img_path: str) -> np.array:
    """ Returns the tensor (not normalized) representing the image file
    Args:
        img_path: path to the image
    Returns:
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths: list) -> np.array:
    """ Returns the tensors (not normalized) representing the image files
    Args:
        img_paths: list of paths
    Returns:
    See Also:
        path_to_tensor()
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm.tqdm(img_paths)]
    return np.vstack(list_of_tensors)
