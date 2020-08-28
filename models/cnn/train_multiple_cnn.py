import numpy as np
from itertools import product
from PIL import ImageFile


from dog_breed.preprocessing import preprocess
from dog_breed.data import datasets
from dog_breed.models import build_network as bn
from dog_breed.common import tools as ct
from dog_breed.models import train_and_predict

ImageFile.LOAD_TRUNCATED_IMAGES = True


EPOCHS = [5, 10, 20, 50]
DATA_AUGMENTATION = [False]


if __name__ == '__main__':
    n_of_classes = len(datasets.get_dog_names())
    train_files, y_train = datasets.load_training()
    valid_files, y_valid = datasets.load_validation()

    tensors_train = np.vstack(list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255,
                                       ct.progr(train_files))))
    tensors_valid = np.vstack(list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255,
                                       ct.progr(valid_files))))

    input_shape = tensors_train[0].shape

    for i in product(EPOCHS, DATA_AUGMENTATION):
        print(i)
        epochs, data_augmentation = i
        model = bn.build_network_from_scratch(input_shape=tensors_train[0].shape, n_of_classes=n_of_classes)
        train_and_predict.train_cnn(model, training_data=tensors_train, training_target=y_train,
                                    validation_data=tensors_valid, validation_target=y_valid,
                                    epochs=epochs, data_augmentation=data_augmentation, prefix='CNN', batch_size=20,
                                    overwrite=0,
                                    )

    #
    # test_files, y_test = datasets.load_test()
    # tensors_test = list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255, ct.progr(test_files)))
