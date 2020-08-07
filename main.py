"""
This module uses Transfer Learning to classify the breed of dogs from the picture.
Procedure:
    * bottle neck features are computed from state-of-the-art networks
        * VGG16
        * VGG19
        * Resnet50
        * InceptionV3
        * Xception
"""
from PIL import ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

from dog_breed.models import build_network as bn
from dog_breed.data import datasets
from dog_breed.preprocessing import preprocess
from dog_breed.models import predict
from dog_breed.models import bottleneck_features as bf
from dog_breed.common import tools as ct

soa_network = 'Resnet50'

train_files, y_train = datasets.load_training()
valid_files, y_valid = datasets.load_validation()
test_files, y_test = datasets.load_test()

tensors_train = list(map(lambda x: preprocess.path_to_tensor(x), ct.progr(train_files)))
tensors_valid = list(map(lambda x: preprocess.path_to_tensor(x), ct.progr(valid_files)))
tensors_test = list(map(lambda x: preprocess.path_to_tensor(x), ct.progr(test_files)))


bottleneck_train = bf.extract_bottleneck_features_list(soa_network, tensors_train)
bottleneck_test = bf.extract_bottleneck_features_list(soa_network, tensors_test)
bottleneck_valid = bf.extract_bottleneck_features_list(soa_network, tensors_valid)

pickle.dump([bottleneck_train, bottleneck_valid,bottleneck_test],
            open(f'data/bottleneck_features/bottleneck_{soa_network}.pkl', 'wb'))

dog_names = datasets.get_dog_names()
n_classes = len(dog_names)

model = bn.build_network(input_shape=bottleneck_train[0].shape, n_of_classes=n_classes)
model.summary()

hist = predict.train_network(network=model, bottleneck_network=soa_network,
                             training_data=bottleneck_train, training_target=y_train,
                             validation_data=bottleneck_valid, validation_target=y_valid,
                             overwrite=0, prefix='mynet',
                             data_augmentation=True,
                             )
# model.fit(bottleneck_train, y_train, )
