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
import os
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

from dog_breed.models import build_network as bn
from dog_breed.data import datasets
from dog_breed.preprocessing import preprocess
from dog_breed.models import predict
from dog_breed.models import bottleneck_features as bf
from dog_breed.common import tools as ct
from dog_breed.common import metrics


soa_network = 'Resnet50'
# get all file names and labels
train_files, y_train = datasets.load_training()
valid_files, y_valid = datasets.load_validation()
test_files, y_test = datasets.load_test()
# compute the tensors
tensors_file = f'data/tensors.npy'
if not os.path.exists(tensors_file):
    # tensors_train = preprocess.paths_to_tensor(train_files).astype('float32') / 255
    # tensors_valid = preprocess.paths_to_tensor(valid_files).astype('float32') / 255
    # tensors_test = preprocess.paths_to_tensor(test_files).astype('float32') / 255

    tensors_train = list(map(lambda x: preprocess.path_to_tensor(x), ct.progr(train_files)))
    tensors_valid = list(map(lambda x: preprocess.path_to_tensor(x), ct.progr(valid_files)))
    tensors_test = list(map(lambda x: preprocess.path_to_tensor(x), ct.progr(test_files)))
    # np.save(tensors_file, [tensors_train, tensors_valid, tensors_test])
# else:
#     tensors_train, tensors_valid, tensors_test = np.load(tensors_file, allow_pickle=True)
# compute bottleneck features
bottle_file = f'data/bottleneck_features/bottleneck_{soa_network}.pkl'
if not os.path.exists(bottle_file):
    bottleneck_train = bf.extract_bottleneck_features_list(soa_network, tensors_train)
    bottleneck_valid = bf.extract_bottleneck_features_list(soa_network, tensors_valid)
    bottleneck_test = bf.extract_bottleneck_features_list(soa_network, tensors_test)
    pickle.dump([bottleneck_train, bottleneck_valid, bottleneck_test], open(bottle_file, 'wb'))
else:
    bottleneck_train, bottleneck_valid, bottleneck_test = pickle.load(open(bottle_file, 'rb'))


dog_names = datasets.get_dog_names()
n_classes = len(dog_names)

model = bn.build_network(input_shape=bottleneck_train[0].shape, n_of_classes=n_classes)
model.summary()

hist = predict.train_network(network=model, bottleneck_network=soa_network,
                             training_data=bottleneck_train, training_target=y_train,
                             validation_data=bottleneck_valid, validation_target=y_valid,
                             overwrite=1, prefix='mynet',
                             data_augmentation=True,
                             )

# prefix = 'mynet'
# epochs = 5
# bottleneck_network = soa_network
# model_weight_file = f'saved_models/{prefix}_{epochs}_A_weight.best.{bottleneck_network}.hdf5'
# model.load_weights(model_weight_file)

pred = model.predict(bottleneck_test)
[np.argmax(x) for x in pred]

metrics.get_accuracy(, y_test)


import matplotlib.pyplot as plt
h = hist[0]
plt.close('all')
plt.figure()
plt.subplot(211)
plt.title('Loss')
plt.plot(h.history['val_loss'])
plt.plot(h.history['loss'])
plt.xticks(range(len(h.history['loss'])), range(1, len(h.history['loss']) + 1))
plt.xlabel("Epoch")
plt.subplot(212)
plt.title('Accuracy')
plt.plot(h.history['val_accuracy'])
plt.plot(h.history['accuracy'])
plt.legend(['validation', 'training'])
plt.xticks(range(len(h.history['loss'])), range(1, len(h.history['loss']) + 1))
plt.xlabel("Epoch")
# model.fit(bottleneck_train, y_train, )
