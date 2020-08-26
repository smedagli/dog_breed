import numpy as np

from dog_breed.common.tools import progr

from keras.preprocessing import image


img_size = 224  # images loaded must be resized to this dimension (squared)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(img_size, img_size))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = list(map(path_to_tensor, progr(img_paths)))
    # list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def _load_image_size(image_path: str, h=img_size, w=img_size):
    """ Loads an image imposing the input resolution w x h
    @ TODO:
        check the order of h and w
    Args:
        image_path: path to the image
        h: height of the image to load
        w: width of the image to load
    Returns:
    Examples:
        >>> _load_image_size('samples/sample_dog.jpg')
        <PIL.Image.Image image mode=RGB size=224x224 at 0x1FABF911B70>
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
    Examples:
        >>> load_image('samples/sample_dog.jpg')[0, 0, 0]
        array([0.5372549 , 0.4745098 , 0.27450982], dtype=float32)
    """
    img = _load_image_size(image_path, h, w)
    x = np.expand_dims(image.img_to_array(img), axis=0)
    return x.astype('float32') / 255
