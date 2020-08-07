import numpy as np
from keras.preprocessing import image
import cv2

img_size = 224 # images loaded must be resized to this dimension (squared)

def _load_image_size(image_path: str, h=img_size, w=img_size):
    """ Loads an image imposing the input resolution w x h
    @ TODO:
        check the order of h and w
    Args:
        image_path: path to the image
        h: height of the image to load
        w: width of the image to load
    Returns:
    """
    img = image.load_img(image_path, target_size=(w, h))
    return img

def load_image(image_path: str, h=img_size, w=img_size):
    """ Loads the image with a given resolution and normalizes the pixel values
    Args:
        image_path: path to the image
        h: height of the image to load
        w: width of the image to load
    Returns:
    """
    img = _load_image_size(image_path, h, w)
    x = np.expand_dims(image.img_to_array(img), axis=0)
    return x.astype('float32') / 255
