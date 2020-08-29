"""
This module uses Transfer Learning to train multiple models.
The parameters are defined as factorial combination of NET, EPOCHS and DATA_AUGMENTATION.

Procedure:
    * bottle neck features are computed from state-of-the-art networks
        * VGG16
        * VGG19
        * Resnet50
        * InceptionV3
        * Xception
"""
import pickle
import os
from itertools import product
import numpy as np
from PIL import ImageFile


from dog_breed.models import build_network as bn
from dog_breed.data import datasets
from dog_breed.preprocessing import preprocess
from dog_breed.models.tl import train_and_predict_tl
from dog_breed.models import bottleneck_features as bf
from dog_breed.common import tools as ct
from dog_breed.common import metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True

NETS = ['vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'xception']
EPOCHS = [5, 10, 20]
DATA_AUGMENTATION = [False, True]

if __name__ == '__main__':
    # get all file names and labels
    train_files, y_train = datasets.load_dataset(dataset='train')
    valid_files, y_valid = datasets.load_dataset(dataset='valid')
    test_files, y_test = datasets.load_dataset(dataset='test')
    n_classes = len(datasets.get_dog_names())
    # compute the tensors
    tensors_train = list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255, ct.progr(train_files)))
    tensors_valid = list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255, ct.progr(valid_files)))
    tensors_test = list(map(lambda x: preprocess.path_to_tensor(x).astype('float32') / 255, ct.progr(test_files)))

    for soa_network in reversed(NETS):
        # compute bottleneck features
        bottle_file = f'data/bottleneck_features/bottleneck_{soa_network}.pkl'
        if not os.path.exists(bottle_file):
            bottleneck_train = bf.extract_bottleneck_features_list(soa_network, tensors_train)
            bottleneck_valid = bf.extract_bottleneck_features_list(soa_network, tensors_valid)
            bottleneck_test = bf.extract_bottleneck_features_list(soa_network, tensors_test)
            pickle.dump([bottleneck_train, bottleneck_valid, bottleneck_test], open(bottle_file, 'wb'))
        else:
            bottleneck_train, bottleneck_valid, bottleneck_test = pickle.load(open(bottle_file, 'rb'))

        for i in product(EPOCHS, DATA_AUGMENTATION):
            print(soa_network, i)
            epochs, data_augmentation = i

            model = bn.build_transfer_learning_netwok(input_shape=bottleneck_train[0].shape, n_of_classes=n_classes)
            model.summary()

            # hist = trainAndPredict.train_network_tl(network=model, bottleneck_network=soa_network,
            #                                      training_data=bottleneck_train, training_target=y_train,
            #                                      validation_data=bottleneck_valid, validation_target=y_valid,
            #                                      overwrite=0, prefix='tl',
            #                                      data_augmentation=data_augmentation,
            #                                      epochs=epochs,
            #                                      )
            train_and_predict_tl.train_network_tl(network=model, bottleneck_network=soa_network,
                                                  training_data=bottleneck_train, training_target=y_train,
                                                  validation_data=bottleneck_valid, validation_target=y_valid,
                                                  overwrite=0, prefix='tl', data_augmentation=data_augmentation,
                                                  epochs=epochs,
                                                  )

            pred_test_probl = model.predict(bottleneck_test)
            pred_test = np.array([np.argmax(x) for x in pred_test_probl])
            labels_test = np.array([np.argmax(y) for y in y_test])
            test_accuracy = metrics.get_accuracy(pred_test, labels_test) * 100
            print(f'Test accuracy {test_accuracy:3.2f}%')
