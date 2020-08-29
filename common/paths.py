import os


def listdir(folder: str) -> list: return [os.path.join(folder, file) for file in os.listdir(folder)]


def is_hdf5(x: str) -> bool: return x.endswith('hdf5')
def is_weight_file(x: str) -> bool: return is_hdf5(x) and 'weight' in x


module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class Folders:
    """    This class contains the default folders for the project    """
    def __init__(self):
        self.data = os.path.join(module_path, 'data')
        self.models = os.path.join(self.data, 'saved_models')
        self.training_data = os.path.join(self.data, 'dogImages', 'train')
        self.test_data = os.path.join(self.data, 'dogImages', 'test')
        self.validation_data = os.path.join(self.data, 'dogImages', 'valid')
        self.bottleneck_features = os.path.join(self.data, 'bottleneck_features')


def get_weights_filename(bottleneck_network: str, prefix='mynet', epochs=5, data_augmentation=False,
                         transfer_learning=True):
    """ Returns the filename where the weights of the network will be saved """
    if transfer_learning:
        base_folder = os.path.join(Folders().models, 'TL')
    else:
        base_folder = os.path.join(Folders().models, 'CNN')
    if not data_augmentation:
        return os.path.join(base_folder, f'{prefix}_{epochs}_weight.best.{bottleneck_network}.hdf5')
    else:
        return get_weights_filename(bottleneck_network, prefix, epochs).replace('_weight', '_A_weight')


def get_hist_filename(bottleneck_network: str, prefix='mynet', epochs=5, data_augmentation=False,
                      transfer_learning=True):
    """ Returns the filename where the history of the network will be saved """
    if transfer_learning:
        base_folder = os.path.join(Folders().models, 'TL')
    else:
        base_folder = os.path.join(Folders().models, 'CNN')
    if not data_augmentation:
        return os.path.join(base_folder, f'{prefix}_{epochs}_hist.{bottleneck_network}.hdf5')
    else:
        return get_hist_filename(bottleneck_network, prefix, epochs).replace('_hist', '_A_hist')
