"""
@ TODO:
    - add header
"""
from itertools import product
from PIL import ImageFile

from dog_breed.data import datasets
from dog_breed.models import build_network as bn
from dog_breed.models.cnn import train_and_predict_cnn

ImageFile.LOAD_TRUNCATED_IMAGES = True


EPOCHS = [5, 10, 20, 50]
DATA_AUGMENTATION = [False]


if __name__ == '__main__':
    n_of_classes = datasets.get_number_of_classes()
    (tensors_train, y_train), (tensors_valid, y_valid) = train_and_predict_cnn.load_training_datasets()

    input_shape = tensors_train[0].shape

    for i in product(EPOCHS, DATA_AUGMENTATION):
        print(i)
        epochs, data_augmentation = i
        model = bn.build_network_from_scratch(input_shape=tensors_train[0].shape, n_of_classes=n_of_classes)
        train_and_predict_cnn.train_cnn(model, training_data=tensors_train, training_target=y_train,
                                        validation_data=tensors_valid, validation_target=y_valid,
                                        epochs=epochs, data_augmentation=data_augmentation, prefix='CNN', batch_size=20,
                                        overwrite=0,
                                        )

    #
    # test_files, y_test = datasets.load_dataset(dataset='test')
    # tensors_test = list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255, ct.progr(test_files)))
