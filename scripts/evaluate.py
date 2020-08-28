"""
Evaluates the performance (in terms of test accuracy) of the saved models.
@ Todo:
    - convert to executable
    - use dog_breed.models.transfer_learning methods
"""
import os
import pandas as pd
import numpy as np
import pickle

from dog_breed.common import paths
from dog_breed.data import datasets
from dog_breed.models import build_network as bn
from dog_breed.models import trainAndPredict
from dog_breed.models import bottleneck_features as bf
from dog_breed.preprocessing import preprocess
from dog_breed.models import transfer_learning as tl

pd.options.display.width = 2500
pd.options.display.max_columns = 25

n_classes = len(datasets.get_dog_names())

models_folder = paths.Folders().models
weight_file_list = list(filter(lambda x: paths.is_weight_file(x),
                               paths.listdir(models_folder)))  # all weight files
base_net = list(map(lambda x: x.rsplit('.', 2)[1], weight_file_list))  # name of the bottleneck network for each model
epochs = list(map(lambda x: x.rsplit('_')[3], weight_file_list))  # number of epochs for each model
basename = list(map(lambda x: os.path.basename(x).split('_')[0], weight_file_list))  # prefix for each model
augmented = list(map(lambda x: '_A_weight' in x, weight_file_list))

saved_models_df = pd.DataFrame(data=(base_net, basename, epochs, augmented, weight_file_list),
                               index=['bottleneck_features', 'prefix', 'epochs', 'augmented', 'path'],
                               ).T

saved_models_df.to_dict()

test_files, y_test = datasets.load_test()  # load the test set
tensors_test = preprocess.paths_to_tensor(test_files).astype('float32') / 255

if saved_models_df.empty:
    pass
else:
    for bottle_net, df in saved_models_df.groupby('bottleneck_features'):
        print(bottle_net)
        bottle_file = f'data/bottleneck_features/bottleneck_{bottle_net.lower()}.pkl'

        if os.path.exists(bottle_file):
            _, _, bottleneck_test = pickle.load(open(bottle_file, 'rb'))
        else:
            print("Computing bottleneck features")
            bottleneck_test = bf.extract_bottleneck_features_list(bottle_net, tensors_test)

        input_shape = bottleneck_test[0].shape

        for i in df.index:  # iter the different saved weights
            weight_file = df.loc[i, 'path']
            net = bn.build_transfer_learning_netwok(input_shape=input_shape, n_of_classes=n_classes)
            trainAndPredict.load_network_weights(net, weight_file)
            pred = net.predict(bottleneck_test)
            acc = (np.array([np.argmax(x) for x in pred]) == np.array([np.argmax(y) for y in y_test])).sum() / len(y_test)
            saved_models_df.loc[i, 'test_accuracy[%]'] = acc * 100

    print(saved_models_df.sort_values(by='test_accuracy[%]', ascending=False))
