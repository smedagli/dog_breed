"""
This module uses Transfer Learning to classify the breed of dogs from the picture.
Procedure:
    * bottle neck features are computed from state-of-the-art networks
        * VGG16
        * VGG19
        * Resnet50
        * InceptionV3
        * Xception
@ TODO:
    - tensors_train is computed just on the first 20 files, fix this (same for y_train)
"""
from dog_breed.models import build_network as bn
from dog_breed.data import datasets
from dog_breed.preprocessing import preprocess
from dog_breed.models import bottleneck_features as bf
from dog_breed.common import tools as ct


soa_network = 'Resnet50'

train_files, y_train = datasets.load_training()
valid_files, y_valid = datasets.load_validation()
test_files, y_test = datasets.load_test()

n_samples = 200
tensors_train = list(map(lambda x: preprocess.path_to_tensor(x), ct.progr(train_files[: n_samples])))
tensors_test = list(map(lambda x: preprocess.path_to_tensor(x), ct.progr(train_files[: n_samples])))

y_train = y_train[: n_samples]

bottleneck_train = bf.extract_bottleneck_features_list(soa_network, tensors_train)
bottleneck_test = bf.extract_bottleneck_features_list(soa_network, tensors_test)

dog_names = datasets.get_dog_names()
n_classes = len(dog_names)

model = bn.build_network(input_shape=bottleneck_train[0].shape, n_of_classes=n_classes)
model.summary()

model.fit(bottleneck_train, y_train)